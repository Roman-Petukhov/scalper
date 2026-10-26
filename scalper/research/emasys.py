"""EMA-системы чемпионов на крипте (документ «стратегии чемпионов», разделы 2.2, 4.1, 6.1) и чистые EMA-стратегии — по
просьбе владельца «пробой, отскок, возврат к средней, если цена ушла далеко от EMA».

Правила заданы до запуска, по истории не подбираются. Одна позиция на монету в варианте. Вход на закрытии сигнальной
свечи (taker), стоп и цель проверяются со следующей свечи, стоп раньше цели в одной свече; издержки и funding — как в
research/champs.py (walk). Оборот монеты — от $20M в день. Тренд — как у бота: закрытие последней закрытой свечи
старшего ТФ против её EMA50 (строка «по тренду» — сторона сделки совпала с ним).

  M  Унгер, возврат к средней. TF 4h и 1d. Закрытие дальше k ATR (k = 3 / 4 / 5) от EMA N (N = 20 / 50), первая такая
     свеча подряд; сделка против отклонения, стоп 1.5 ATR от входа. Выходы: «к EMA» (цель — значение EMA на сигнале),
     «к середине» (половина пути к EMA); срок 30 свечей (4h) / 15 (1d).
  G  Гаджала, внутридневной VCP + 9 / 21 EMA. TF 15m. Тренд: EMA9 выше EMA21 (шорт зеркально), обе растут за 3 свечи,
     закрытие выше EMA21. Сжатие: размах 8 свечей до сигнала <= 2.5 ATR и средний объём 4 свечей до сигнала <= 0.7 от
     объёма 20 свечей перед ними. RVOL: объём последних 24 ч >= 2× / 3× среднесуточного за 50 дней. Триггер: «тест EMA9»
     (минимум коснулся EMA9, закрытие над ней, за 8 свечей до того касаний не было) или «пробой коробки» (закрытие над
     максимумом 8 свечей на объёме от 1.5× среднего за 20). Стоп — минимум сигнальной свечи (для пробоя — минимум коробки)
     минус 0.1 ATR, не уже 0.3 ATR; шире 1.5% цены или 4 ATR — пропуск. Выходы: «3R», «закрытие за EMA21»; срок 96 свечей.
  E  Чистые EMA. TF 4h и 1d. «Пробой»: закрытие пересекло EMA N (N = 20 / 50 / 100 / 200), стоп 2 ATR; выходы «обратное
     пересечение», «3R»; срок 120 свечей (4h) / 60 (1d). «Отскок»: EMA N (20 / 50) растёт за 5 свечей (шорт зеркально),
     минимум коснулся EMA, закрытие над ней, за 5 свечей до того касаний не было; стоп — минимум сигнальной свечи минус
     0.1 ATR (0.3–4 ATR); выходы «закрытие за EMA», «3R».
Принятие (на строку, как у champs): t по дням >= 3 (около 60 строк — Harvey, Liu, Zhu), плюс в каждом периоде 2020 /
2021 / IS / VAL / HO, где сделок >= 20, плюс при издержках ×2.

    python -m research.emasys collect --symbols <монеты>      # EMASYS=M,G,E
    python -m research.emasys report
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30
from .champs import _row, line
from .masters import ema, sma, walk
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _atr
from .tline import htf_trend, tf_frame

SYSTEMS = tuple(os.environ.get("EMASYS", "M,G,E").split(","))
NAMES = {"M": "Унгер: возврат к средней от EMA", "G": "Гаджала: VCP + 9/21 EMA на 15m",
         "E": "Чистые EMA: пробой и отскок"}
M_N, M_K, M_STOP = (20, 50), (3.0, 4.0, 5.0), 1.5
M_HOLD = {"4h": 30, "1d": 15}
G_BOX, G_BOX_ATR, G_DRY, G_DRY_X, G_RVOL, G_STOP_PCT, G_HOLD = 8, 2.5, 4, 0.7, (2.0, 3.0), 0.015, 96
E_CROSS_N, E_BOUNCE_N, E_STOP = (20, 50, 100, 200), (20, 50), 2.0
E_HOLD = {"4h": 120, "1d": 60}


def _prep(d: pd.DataFrame) -> tuple:
    o, hi, lo, c, v = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close", "volume"))
    f = d["funding"].fillna(0.0).to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    return o, hi, lo, c, v, f, _atr(d).to_numpy()


def _first_true(mask: np.ndarray) -> np.ndarray:
    """Сигнал — первая подряд свеча mask: на предыдущей свече его не было."""
    prev = np.r_[False, mask[:-1]]
    return mask & ~prev


def _no_touch(touch: np.ndarray, n: int) -> np.ndarray:
    """За n свечей до текущей касаний не было."""
    s = pd.Series(touch.astype(float)).shift(1).rolling(n, min_periods=n).sum().to_numpy()
    return s == 0


def mean_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str) -> list[dict]:
    o, hi, lo, c, _, f, a = _prep(d)
    trend = htf_trend(d, tf)
    rows: list[dict] = []
    for n in M_N:
        e = ema(c, n)
        with np.errstate(invalid="ignore", divide="ignore"):
            dist = (c - e) / a
        for k in M_K:
            sig = _first_true(np.abs(dist) >= k) & (liq >= ADV_MIN) & (a > 0) & np.isfinite(e)
            busy: dict[str, int] = {}
            for t in np.flatnonzero(sig):
                if t >= len(c) - 1:
                    continue
                side = -1 if dist[t] > 0 else 1
                entry = c[t]
                stop = entry - side * M_STOP * a[t]
                for name, tgt in (("к EMA", e[t]), ("к середине", entry + (e[t] - entry) / 2)):
                    key = f"{n}/{k}/{name}"
                    if busy.get(key, -1) >= t or not side * (tgt - entry) > 0:
                        continue
                    r, ex = walk(o, hi, lo, c, f, t, side, entry, stop, M_HOLD[tf], tp=tgt)
                    if np.isfinite(r):
                        busy[key] = ex
                        tr = int(trend[t]) if np.isfinite(trend[t]) else 0
                        rows.append(_row("M", f"{tf}, EMA{n}, k {k:g} ATR", name, "perp", tf, sym, d.index[t], side, r,
                                         entry, stop, tr))
    return rows


def vcp_rows(d: pd.DataFrame, liq: np.ndarray, sym: str) -> list[dict]:
    o, hi, lo, c, v, f, a = _prep(d)
    tf = "15m"
    trend = htf_trend(d, tf)
    e9, e21 = ema(c, 9), ema(c, 21)
    s = pd.Series
    box_hi = s(hi).shift(1).rolling(G_BOX, min_periods=G_BOX).max().to_numpy()
    box_lo = s(lo).shift(1).rolling(G_BOX, min_periods=G_BOX).min().to_numpy()
    v4 = s(v).shift(1).rolling(G_DRY, min_periods=G_DRY).mean().to_numpy()
    v20 = s(v).shift(1 + G_DRY).rolling(20, min_periods=20).mean().to_numpy()
    v20b = sma(v, 20)
    day = s(v).rolling(96, min_periods=96).sum().to_numpy()
    base = s(v).rolling(96 * 50, min_periods=96 * 20).mean().shift(96).to_numpy() * 96
    with np.errstate(invalid="ignore", divide="ignore"):
        compact = (box_hi - box_lo) / a <= G_BOX_ATR
        dry = v4 <= G_DRY_X * v20
        rvol = day / base
    ok = compact & dry & (liq >= ADV_MIN) & (a > 0)
    rows: list[dict] = []
    for side in (1, -1):
        up = side * (e9 - e21) > 0
        rise = side * (e9 - np.r_[[np.nan] * 3, e9[:-3]]) > 0
        rise &= side * (e21 - np.r_[[np.nan] * 3, e21[:-3]]) > 0
        trendok = up & rise & (side * (c - e21) > 0)
        touch = (lo <= e9) if side > 0 else (hi >= e9)
        test = touch & (side * (c - e9) >= 0) & _no_touch(touch, G_BOX)
        brk = (side * (c - (box_hi if side > 0 else box_lo)) > 0) & (v >= 1.5 * np.r_[np.nan, v20b[:-1]])
        for trig, mask in (("тест EMA9", test), ("пробой коробки", brk)):
            sig = ok & trendok & mask
            for rv in G_RVOL:
                busy: dict[str, int] = {}
                for t in np.flatnonzero(sig & (rvol >= rv)):
                    if t >= len(c) - 1:
                        continue
                    entry = c[t]
                    ref = (lo[t] if side > 0 else hi[t]) if trig == "тест EMA9" else (box_lo[t] if side > 0 else box_hi[t])
                    stop = ref - side * 0.1 * a[t]
                    if side * (entry - stop) < 0.3 * a[t]:
                        stop = entry - side * 0.3 * a[t]
                    risk = side * (entry - stop)
                    if risk > G_STOP_PCT * entry or risk > 4 * a[t]:
                        continue
                    viol = (lambda j, s_=side: s_ * (c[j] - e21[j]) < 0)
                    for name, kw in (("3R", {"tp": entry + side * 3 * risk}), ("закрытие за EMA21", {"close_exit": viol})):
                        key = f"{trig}/{rv}/{name}/{side}"
                        if busy.get(key, -1) >= t:
                            continue
                        r, ex = walk(o, hi, lo, c, f, t, side, entry, stop, G_HOLD, **kw)
                        if np.isfinite(r):
                            busy[key] = ex
                            tr = int(trend[t]) if np.isfinite(trend[t]) else 0
                            rows.append(_row("G", f"{trig}, RVOL >= {rv:g}", name, "perp", tf, sym, d.index[t], side, r,
                                             entry, stop, tr))
    return rows


def pure_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str) -> list[dict]:
    o, hi, lo, c, _, f, a = _prep(d)
    trend = htf_trend(d, tf)
    base_ok = (liq >= ADV_MIN) & (a > 0)
    rows: list[dict] = []
    for n in sorted(set(E_CROSS_N) | set(E_BOUNCE_N)):
        e = ema(c, n)
        fin = np.isfinite(e)
        for side in (1, -1):
            prev_c, prev_e = np.r_[np.nan, c[:-1]], np.r_[np.nan, e[:-1]]
            cross = (side * (prev_c - prev_e) <= 0) & (side * (c - e) > 0)
            touch = (lo <= e) if side > 0 else (hi >= e)
            rising = side * (e - np.r_[[np.nan] * 5, e[:-5]]) > 0
            bounce = rising & touch & (side * (c - e) > 0) & _no_touch(touch, 5)
            viol = (lambda j, s_=side, e_=e: s_ * (c[j] - e_[j]) < 0)
            for kind, mask, names in (("пробой", cross, ("обратное пересечение", "3R")),
                                      ("отскок", bounce, ("закрытие за EMA", "3R"))):
                if (kind == "пробой" and n not in E_CROSS_N) or (kind == "отскок" and n not in E_BOUNCE_N):
                    continue
                busy: dict[str, int] = {}
                for t in np.flatnonzero(mask & base_ok & fin):
                    if t >= len(c) - 1:
                        continue
                    entry = c[t]
                    if kind == "пробой":
                        stop = entry - side * E_STOP * a[t]
                    else:
                        stop = (lo[t] if side > 0 else hi[t]) - side * 0.1 * a[t]
                        if side * (entry - stop) < 0.3 * a[t]:
                            stop = entry - side * 0.3 * a[t]
                    risk = side * (entry - stop)
                    if not 0 < risk <= 4 * a[t]:
                        continue
                    for name in names:
                        kw = {"tp": entry + side * 3 * risk} if name == "3R" else {"close_exit": viol}
                        key = f"{kind}/{n}/{name}/{side}"
                        if busy.get(key, -1) >= t:
                            continue
                        r, ex = walk(o, hi, lo, c, f, t, side, entry, stop, E_HOLD[tf], **kw)
                        if np.isfinite(r):
                            busy[key] = ex
                            tr = int(trend[t]) if np.isfinite(trend[t]) else 0
                            rows.append(_row("E", f"{tf}, {kind} EMA{n}", name, "perp", tf, sym, d.index[t], side, r,
                                             entry, stop, tr))
    return rows


def collect(root: Path, syms: list[str]) -> None:
    rows: list[dict] = []
    for s in mine(syms):
        try:
            liq1d = adv30(root, s)
            for tf in ("4h", "1d"):
                if not ({"M", "E"} & set(SYSTEMS)):
                    break
                d = tf_frame(root, s, tf)
                if d is None or len(d) < 500:
                    continue
                lv = liq1d.reindex(d.index.floor("D")).to_numpy()
                if "M" in SYSTEMS:
                    rows += mean_rows(d, lv, s, tf)
                if "E" in SYSTEMS:
                    rows += pure_rows(d, lv, s, tf)
            if "G" in SYSTEMS and (root / f"{s}-15m.parquet").exists():
                d = tf_frame(root, s, "15m")
                if d is not None and len(d) > 96 * 60:
                    rows += vcp_rows(d, liq1d.reindex(d.index.floor("D")).to_numpy(), s)
        except Exception as e:
            print(f"  emasys {s}: пропуск ({e})", flush=True)
    print(f"  emasys: сделок {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("emasys"), index=False)


def report() -> None:
    parts = all_parts("emasys")
    print("\n===== EMASYS: EMA-системы чемпионов и чистые EMA на крипте — Унгер (возврат к средней), Гаджала (VCP + "
          "9/21 EMA), пробой и отскок от EMA (t по дням, прибыльных, сделок в месяц; x2 — издержки вдвое) =====")
    if not parts:
        print("  emasys: нет данных")
        return
    x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
    x = x.dropna(subset=["R"])
    passed = []
    for sys_ in SYSTEMS:
        xs = x[x.sys == sys_]
        if xs.empty:
            print(f"\n  {NAMES[sys_]}: сделок нет")
            continue
        print(f"\n  --- {NAMES[sys_]} ---")
        for (var, ex), g in xs.groupby(["var", "exit"], sort=True):
            modes = [("обе стороны", g), ("по тренду", g[g.side == g.trend])]
            if sys_ == "G":
                modes.append(("шорты", g[g.side < 0]))
            for mode, z in modes:
                if len(z) < 20:
                    continue
                txt, ok, t = line(z, PER_ALL)
                print(f"    {var}, {ex}, {mode}: {txt}  → {'принято' if ok else 'нет'}")
                if ok:
                    passed.append(f"{sys_} {var} {ex} {mode} (t {t:.1f})")
    print("\n  ИТОГ: " + ("прошли — " + "; ".join(passed) if passed else "ни одна EMA-система не принята"))


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
