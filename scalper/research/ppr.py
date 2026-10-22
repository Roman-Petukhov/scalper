"""Свечная формация PPR — Pivot Point Reversal (по просьбе владельца). Три свечи вокруг локального экстремума (Нил
Фуллер, Price Action): для лонга средняя свеча делает минимум ниже обеих соседних, третья закрывается выше максимума
средней (строгий PPR) или выше её закрытия (мягкий). Шорт — зеркально. Сигнал известен на закрытии третьей свечи.

Вход — по рынку на закрытии третьей свечи (taker); стоп — за экстремумом средней свечи ∓ 0.1 ATR (0.3–4 ATR, как у
бота); цель 2R или 3R; срок — как у правила бота на ТФ; издержки maker / taker 2 / 5.5 б.п. и funding. Монеты — с
оборотом за 30 дней не ниже порога бота. ТФ: 1h и 4h (с 2020), 15m (монеты с 15m-свечами).
Контекст (флаги к каждому сигналу): trend — по тренду старшего ТФ (close выше / ниже EMA50, как у бота); level —
экстремум средней свечи в пределах 0.3 ATR от последнего подтверждённого свинга (фрактал 5) — «у уровня»;
fbreak — средняя свеча проколола этот свинг, третья закрылась обратно за ним (ложный пробой).
Принятие (задано до запуска): вариант (ТФ × строгий / мягкий × контекст × сторона × цель) принят, если R на сделку
> 0 в каждом периоде 2020 / 2021 / IS / VAL / HO, где сделок ≥ 20, и за всё время t ≥ 3 (вариантов десятки —
Harvey, Liu, Zhu 2016). Отдельно, как фильтр бота: сделки бота (4h ретест, 15m пологие по рынку) с PPR в ту же сторону
на 5 свечах до пробоя против остальных; принятие — разница «с PPR − без» > 0 в каждом периоде и t ≥ 2.

    python -m research.ppr collect --symbols <монеты>      # PPR_TFS=1h,4h или 15m
    python -m research.ppr report
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import noline
from .broad import ADV_MIN, adv30
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import HOLD, PIV, STOP_BUF, TAKER, htf_trend, last_confirmed, pivots, tf_frame, two_targets, zz_lines

TFS = tuple(os.environ.get("PPR_TFS", "1h,4h,15m").split(","))
LEVEL_ATR = 0.3
BEFORE = 5                         # PPR на стольких свечах до пробоя бота
BOT = {"4h": ("retest", None), "15m": ("market", 0.025)}
CONTEXTS = ("all", "trend", "level", "fbreak", "trend_level")
TARGETS = (2.0, 3.0)


def find(o: np.ndarray, hi: np.ndarray, lo: np.ndarray, c: np.ndarray) -> list[tuple[int, int, bool]]:
    """(бар третьей свечи, сторона, строгий ли) — все PPR ряда."""
    out = []
    for t in range(2, len(c)):
        m = t - 1
        if lo[m] < lo[m - 1] and lo[m] < lo[t] and c[t] > c[m]:
            out.append((t, 1, bool(c[t] > hi[m])))
        if hi[m] > hi[m - 1] and hi[m] > hi[t] and c[t] < c[m]:
            out.append((t, -1, bool(c[t] < lo[m])))
    return out


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str) -> list[dict]:
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    a = _atr(d).to_numpy()
    trend = htf_trend(d, tf)
    sw_lo, sw_hi = last_confirmed(pivots(lo, PIV, False), len(c)), last_confirmed(pivots(hi, PIV, True), len(c))
    rows = []
    for t, side, strict in find(o, hi, lo, c):
        m = t - 1
        if t >= len(c) - 1 or not (a[t] > 0) or not (liq[t] >= ADV_MIN):
            continue
        ext = lo[m] if side > 0 else hi[m]
        stop = ext - side * STOP_BUF * a[t]
        risk = side * (c[t] - stop)
        if not (0.3 * a[t] <= risk <= 4.0 * a[t]):
            continue
        sw = sw_lo[m - 1] if side > 0 else sw_hi[m - 1]           # свинг, подтверждённый до средней свечи
        lvl = (lo[sw] if side > 0 else hi[sw]) if sw >= 0 else np.nan
        near = bool(np.isfinite(lvl) and abs(ext - lvl) <= LEVEL_ATR * a[t])
        fb = bool(np.isfinite(lvl) and side * (ext - lvl) < 0 and side * (c[t] - lvl) > 0)
        row = {"symbol": sym, "tf": tf, "t": d.index[t], "side": side, "strict": strict, "trend": bool(trend[t] == side),
               "level": near, "fbreak": fb}
        for k in TARGETS:
            row[f"R{k:g}"] = float(two_targets(o, hi, lo, c, f, t, side, c[t], stop, k, k, False, HOLD[tf], TAKER)[0])
        rows.append(row)
    return rows


def bot_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str) -> list[dict]:
    """Сделки бота с флагом «PPR в ту же сторону на BEFORE свечах до пробоя»."""
    entry, max_slope = BOT[tf]
    noline.TF = tf
    x = noline.coin_trades(d, liq, sym, lines_only=True)
    if not len(x):
        return []
    x = x[x.entry == entry]
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    marks = {(t, s) for t, s, _ in find(o, hi, lo, c)}
    a = _atr(d).to_numpy()
    slope = {(int(r["t"]), int(r["side"])): abs(float(r["slope"])) for r in zz_lines(d) if r["t"] > 0}
    rows = []
    for r in x.itertuples(index=False):
        t, side = d.index.get_loc(r.t), int(r.side)
        if max_slope is not None and not (a[t] > 0 and slope.get((t, side), np.inf) / a[t] <= max_slope):
            continue
        rows.append({"symbol": sym, "tf": tf, "t": r.t, "side": side, "R": float(r.R3),
                     "ppr": any((j, side) in marks for j in range(t - BEFORE, t + 1))})
    return rows


def collect(root: Path, syms: list[str]) -> None:
    for tf in TFS:
        rows, bots = [], []
        for s in mine(syms):
            try:
                if tf == "15m" and not (root / f"{s}-15m.parquet").exists():
                    continue
                d = tf_frame(root, s, tf)
                if d is None or len(d) < 500:
                    continue
                liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
                rows += coin_rows(d, liq, s, tf)
                if tf in BOT:
                    bots += bot_rows(d, liq, s, tf)
            except Exception as e:
                print(f"  ppr {s} {tf}: пропуск ({e})", flush=True)
        print(f"  ppr {tf}: формаций {len(rows)}, сделок бота {len(bots)}", flush=True)
        if rows:
            pd.DataFrame(rows).to_parquet(part_path(f"ppr_{tf}"), index=False)
        if bots:
            pd.DataFrame(bots).to_parquet(part_path(f"pprbot_{tf}"), index=False)


def tstat(a: np.ndarray) -> float:
    return float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))) if len(a) > 2 and a.std(ddof=1) > 0 else float("nan")


def context(x: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "all":
        return x
    if name == "trend_level":
        return x[x.trend & x.level]
    return x[x[name]]


def day_t(z: pd.DataFrame, col: str) -> float:
    """t по дням, как в _cell: формации одного дня на разных монетах — одна коррелированная группа."""
    r = z[col].to_numpy(dtype="float64")
    if len(r) < 10:
        return float("nan")
    g = pd.Series(r - r.mean()).groupby(z["t"].dt.floor("D").to_numpy()).sum()
    d = float(np.sqrt((g ** 2).sum()))
    return float(r.sum() / d) if d > 0 else float("nan")


def verdict(z: pd.DataFrame, col: str) -> tuple[bool, float]:
    ok = []
    for _, (a, b) in PER_ALL.items():
        m = z[(z.t >= a) & (z.t < b)]
        if len(m) >= 20:
            ok.append(m[col].mean() > 0)
    t = day_t(z, col)
    return bool(ok) and all(ok) and t >= 3, t


def report() -> None:
    print("\n===== PPR: Pivot Point Reversal — вход на закрытии третьей свечи, стоп за средней, цель 2R / 3R (t по дням, "
          "прибыльных, сделок в месяц) =====")
    passed = []
    for tf in TFS:
        parts = all_parts(f"ppr_{tf}")
        if not parts:
            print(f"  ppr {tf}: нет данных")
            continue
        x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
        print(f"\n  --- {tf}: формаций {len(x)} ---")
        for kind, sub in (("строгий", x[x.strict]), ("мягкий", x)):
            for side, nm in ((1, "лонг"), (-1, "шорт")):
                for ctx in CONTEXTS:
                    z = context(sub[sub.side == side], ctx)
                    if len(z) < 20:
                        continue
                    for k in TARGETS:
                        col = f"R{k:g}"
                        cells = "  ".join(f"{p}: {_cell(z[(z.t >= a) & (z.t < b)].assign(R=z[col]))}"
                                          for p, (a, b) in PER_ALL.items() if ((z.t >= a) & (z.t < b)).any())
                        ok, t = verdict(z, col)
                        print(f"    {kind} {nm} {ctx} {k:g}R: {cells}  всё время: {_cell(z.assign(R=z[col]))}"
                              f"{'  → принято' if ok else ''}")
                        if ok:
                            passed.append(f"{tf} {kind} {nm} {ctx} {k:g}R (t {t:.1f})")
        bparts = all_parts(f"pprbot_{tf}")
        if bparts:
            y = pd.concat([pd.read_parquet(p) for p in bparts], ignore_index=True)
            y["t"] = pd.to_datetime(y["t"], utc=True).dt.tz_localize(None)
            cells, ok = [], []
            for p, (a, b) in PER_ALL.items():
                m = y[(y.t >= a) & (y.t < b)]
                if m.empty:
                    continue
                w, wo = m[m.ppr], m[~m.ppr]
                cells.append(f"{p}: с PPR {_cell(w) if len(w) else '—'} / без {wo.R.mean():+.3f}R")
                if len(w) >= 20 and len(wo) >= 20:
                    ok.append(w.R.mean() > wo.R.mean())
            w, wo = y.R[y.ppr].to_numpy(), y.R[~y.ppr].to_numpy()
            se = np.sqrt(w.var(ddof=1) / len(w) + wo.var(ddof=1) / len(wo)) if len(w) > 2 and len(wo) > 2 else np.nan
            t = (w.mean() - wo.mean()) / se if se and se > 0 else np.nan
            acc = bool(ok) and all(ok) and t >= 2
            print(f"    фильтр бота ({BOT[tf][0]}): с PPR {len(w)} из {len(y)}; " + "; ".join(cells)
                  + f"; разница t {t:.1f}{'  → принято' if acc else ''}")
            if acc:
                passed.append(f"{tf} фильтр бота")
    print("\n  ИТОГ: " + ("прошли — " + "; ".join(passed) if passed else "ни один вариант PPR не принят"))


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
