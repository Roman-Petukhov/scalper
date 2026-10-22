"""Три идеи из спецификации модернизации, каждая против контроля (сделки бота как есть). Критерии заданы до запуска.

  ker — фильтр режима: эффективность Кауфмана KER = |c[t] − c[t−n]| / Σ|Δc| за n свечей до пробоя включительно
        (n = 10 / 20 / 40), сделки с KER < T (0.2 / 0.3 / 0.4) режутся. Принятие порога: отрезанные сделки убыточны в
        каждом периоде, где их >= 20, за всё время t по дням <= −3 (девять комбинаций сразу — Harvey, Liu, Zhu 2016),
        R в месяц с порогом не ниже, чем без него, в каждом периоде. Обратное направление (резать высокий KER) — справочно.
  ts  — линия по Тейлу–Сену вместо касательной по двум вершинам: наклон — медиана наклонов по всем парам вершин
        зигзага (опорная + подтверждённые позже, не меньше трёх, ни одна позже опоры не выше неё), линия сдвинута на
        крайнюю вершину (касательная, над ней вершин нет); пробой, фильтры, стоп и цель как у бота.
  gk  — ATR по Гарману–Классу вместо ATR Уайлдера во всём правиле (зигзаг, стоп, наклон): σ_GK по ewm 14 свечей,
        масштаб к ATR — по медиане отношения за прошлую историю монеты (без заглядывания), первые 250 свечей не торгуем
        ни в одном варианте.
  Принятие ts / gk (для каждого ТФ): плюс в каждом периоде, где >= 20 сделок, t по дням >= 3 и R в месяц не ниже, чем у
  контроля, в каждом периоде. 4h — ретест, обе стороны, без фильтра наклона; 15m — по рынку, шорты, наклон <= 0.025.

    python -m research.spec collect --symbols <монеты>      # SPEC_TFS=4h или 15m
    python -m research.spec report
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import noline, smc, tline
from .broad import adv30
from .oos import PER_ALL
from .ppr import day_t
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import ZZ_ANCHOR, ZZ_K, ZZ_LIFE, ZZ_SPAN, tf_frame, zigzag, zz_lines

TFS = tuple(os.environ.get("SPEC_TFS", "4h,15m").split(","))
ENTRY = {"4h": "retest", "15m": "market"}
MAX_SLOPE = 0.025
WARMUP = 250
KER_N = (10, 20, 40)
KER_T = (0.2, 0.3, 0.4)
VARIANTS = ("base", "ts", "gk")


def ker(c: np.ndarray, n: int) -> np.ndarray:
    s = pd.Series(c)
    path = s.diff().abs().rolling(n).sum()
    return ((s - s.shift(n)).abs() / path.where(path > 0)).to_numpy()


def gk_atr(d: pd.DataFrame, n: int = 14) -> pd.Series:
    """ATR-подобная шкала из σ Гармана–Класса; масштаб к ATR Уайлдера — expanding-медиана отношения (по прошлому)."""
    o, h, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    var = 0.5 * np.log(h / lo) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    sig = np.sqrt(pd.Series(np.maximum(var, 0.0), index=d.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean())
    raw = sig * d["close"]
    ratio = (_atr(d) / raw.where(raw > 0)).replace([np.inf, -np.inf], np.nan)
    scale = ratio.expanding(min_periods=WARMUP).median().shift(1)
    return (raw * scale).rename("atr")


def ts_lines(d: pd.DataFrame, k: float = ZZ_K) -> list[dict]:
    """Линии по Тейлу–Сену; формат записей как у zz_lines (t — бар пробоя, line_t — линия на нём)."""
    price = d["close"].to_numpy(dtype="float64")
    atr = tline._atr(d).to_numpy()
    m = len(price)
    hs, ls = zigzag(price, atr, k)
    out = []
    for side, piv in ((1, hs), (-1, ls)):
        seen = set()
        for a, conf_a in piv:
            w = price[max(0, a - ZZ_ANCHOR): a]
            if len(w) < ZZ_ANCHOR or side * (price[a] - (w.max() if side > 0 else w.min())) <= 0:
                continue
            later = piv[piv[:, 0] > a]
            li, rec, line = 0, None, None
            for t in range(conf_a + 1, min(a + ZZ_LIFE, m - 1)):
                changed = False
                while li < len(later) and later[li][1] <= t - 1:
                    li += 1
                    changed = True
                if changed:
                    xs = np.r_[a, later[:li, 0]]
                    ys = price[xs]
                    if len(xs) > 1 and np.any(side * (ys[1:] - ys[0]) >= 0):
                        break                                       # после опоры вершина не ниже неё — структура сломана
                    line = None
                    if len(xs) >= 3 and xs[-1] - a >= ZZ_SPAN:
                        i, j = np.triu_indices(len(xs), 1)
                        beta = float(np.median((ys[j] - ys[i]) / (xs[j] - xs[i])))
                        if side * beta < 0:
                            kk = side * np.max(side * (ys - beta * xs))
                            line = (beta, float(kk), int(xs[-1]))
                if line is None:
                    continue
                beta, kk, last = line
                lt, lp = beta * t + kk, beta * (t - 1) + kk
                if side * (price[t] - lt) > 0 and side * (price[t - 1] - lp) <= 0:
                    ln = beta * (t + 1) + kk
                    rec = {"side": side, "a": int(a), "b": last, "slope": beta, "t": t,
                           "tc": t + 1 if side * (price[t + 1] - ln) > 0 else -1, "line_t": float(lt),
                           "line_n": float(ln), "log": False}
                    break
            if rec is not None and rec["t"] not in seen:
                seen.add(rec["t"])
                out.append(rec)
    return out


@contextlib.contextmanager
def patched(variant: str):
    saved = (tline._atr, noline._atr, noline.zz_lines)
    try:
        if variant == "gk":
            tline._atr = noline._atr = gk_atr
        elif variant == "ts":
            noline.zz_lines = ts_lines
        yield
    finally:
        tline._atr, noline._atr, noline.zz_lines = saved


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str, variant: str) -> list[dict]:
    noline.TF = tf
    with patched(variant):
        x = noline.coin_trades(d, liq, sym, lines_only=True)
        if not len(x):
            return []
        x = x[x.entry == ENTRY[tf]]
        atr = (gk_atr(d) if variant == "gk" else _atr(d)).to_numpy()
        recs = {(int(r["t"]), int(r["side"])): r for r in (ts_lines(d) if variant == "ts" else zz_lines(d)) if r["t"] > 0}
    c = d["close"].to_numpy(dtype="float64")
    kers = {n: ker(c, n) for n in KER_N}
    rows = []
    for r in x.itertuples(index=False):
        t, side = d.index.get_loc(r.t), int(r.side)
        rec = recs.get((t, side))
        if rec is None or t < WARMUP or not (atr[t] > 0):
            continue
        if tf != "4h" and abs(float(rec["slope"])) / atr[t] > MAX_SLOPE:
            continue
        rows.append({"symbol": sym, "tf": tf, "variant": variant, "t": r.t, "side": side, "R": float(r.R3),
                     **{f"ker{n}": float(kers[n][t]) for n in KER_N}})
    return rows


def collect(root: Path, syms: list[str]) -> None:
    for tf in TFS:
        rows = []
        for s in mine(syms):
            try:
                if tf != "4h" and not (root / f"{s}-{tf}.parquet").exists():
                    continue
                d = tf_frame(root, s, tf)
                if d is None or len(d) < 500:
                    continue
                liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
                for v in VARIANTS:
                    rows += coin_rows(d, liq, s, tf, v)
            except Exception as e:
                print(f"  spec {s} {tf}: пропуск ({e})", flush=True)
        print(f"  spec {tf}: сделок {len(rows)}", flush=True)
        if rows:
            pd.DataFrame(rows).to_parquet(part_path(f"spec_{tf}"), index=False)


def months(a, b) -> float:
    return max(1.0, (pd.Timestamp(b) - pd.Timestamp(a)).days / 30.4)


def cut_verdicts(z: pd.DataFrame, col: str, thresholds=KER_T) -> dict[float, tuple[bool, str]]:
    """Порог T → (принят, строки по периодам): сделки с z[col] < T отрезаются."""
    out = {}
    for T in thresholds:
        cut = z[col] < T
        ok_cut, ok_month, cells = [], [], []
        for p, (a, b) in PER_ALL.items():
            m = z[(z.t >= a) & (z.t < b)]
            if m.empty:
                continue
            ct = m[m[col] < T]
            base_m, new_m = m.R.sum() / months(a, b), m.R[m[col] >= T].sum() / months(a, b)
            cells.append(f"{p}: отрезано {_cell(ct) if len(ct) else '—'}; R в месяц {base_m:+.1f} → {new_m:+.1f}")
            if len(ct) >= 20:
                ok_cut.append(ct.R.mean() < 0)
            ok_month.append(new_m >= base_m)
        t = day_t(z[cut], "R") if cut.sum() > 10 else float("nan")
        out[T] = (bool(ok_cut) and all(ok_cut) and all(ok_month) and t <= -3,
                  "\n".join(f"      {c}" for c in cells)
                  + f"\n      отрезанные за всё время: {z.R[cut].mean():+.3f}R на {int(cut.sum())}, t {t:.1f}")
    return out


def variant_verdict(v: pd.DataFrame, base: pd.DataFrame) -> tuple[bool, str]:
    pos, month, cells = [], [], []
    for p, (a, b) in PER_ALL.items():
        mv, mb = v[(v.t >= a) & (v.t < b)], base[(base.t >= a) & (base.t < b)]
        if mv.empty and mb.empty:
            continue
        rv, rb = mv.R.sum() / months(a, b), mb.R.sum() / months(a, b)
        cells.append(f"{p}: вариант {_cell(mv) if len(mv) else '—'}, R в месяц {rv:+.1f} | контроль "
                     f"{_cell(mb) if len(mb) else '—'}, {rb:+.1f}")
        if len(mv) >= 20:
            pos.append(mv.R.mean() > 0)
        month.append(rv >= rb)
    t = day_t(v, "R") if len(v) > 10 else float("nan")
    return bool(pos) and all(pos) and all(month) and t >= 3, "\n".join(f"      {c}" for c in cells) + \
        f"\n      всё время: вариант {_cell(v)} | контроль {_cell(base)}"


def report() -> None:
    print("\n===== SPEC: KER-фильтр, линия по Тейлу–Сену, ATR по Гарману–Классу — против сделок бота "
          "(t по дням, прибыльных, сделок в месяц) =====")
    passed = []
    for tf in TFS:
        parts = all_parts(f"spec_{tf}")
        if not parts:
            print(f"  spec {tf}: нет данных")
            continue
        x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
        x = x.dropna(subset=["R"])
        sides = (("обе стороны", 0),) if tf == "4h" else (("шорты", -1), ("обе стороны", 0))
        for label, sd in sides:
            sub = x if sd == 0 else x[x.side == sd]
            base = sub[sub.variant == "base"]
            print(f"\n  --- {tf}, вход {ENTRY[tf]}, {label}: контроль {len(base)} сделок ---")
            primary = (tf == "4h" and label == "обе стороны") or (tf == "15m" and label == "шорты")
            for n in KER_N:
                z = base.dropna(subset=[f"ker{n}"])
                for T, (ok, txt) in cut_verdicts(z, f"ker{n}").items():
                    print(f"    KER({n}) < {T:g} режем: {'принято' if ok and primary else 'нет'}\n{txt}")
                    if ok and primary:
                        passed.append(f"{tf} {label} KER({n})<{T:g}")
                hi = z[z[f'ker{n}'] >= 0.4]
                print(f"    справочно KER({n}) >= 0.4 (спецификация: «ламинарный тренд»): {_cell(hi)} против остальных "
                      f"{_cell(z[z[f'ker{n}'] < 0.4])}")
            for var, nm in (("ts", "линия Тейла–Сена"), ("gk", "ATR Гармана–Класса")):
                v = sub[sub.variant == var]
                if v.empty:
                    print(f"    {nm}: сделок нет")
                    continue
                ok, txt = variant_verdict(v, base)
                print(f"    {nm}: {'принято' if ok and primary else 'нет'}\n{txt}")
                if ok and primary:
                    passed.append(f"{tf} {label} {nm}")
    print("\n  ИТОГ: " + ("прошли — " + ", ".join(passed) if passed else "ничего из KER / Тейл–Сен / Гарман–Класс не принято"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [s for s in a.symbols.split(",") if s])
    else:
        report()
