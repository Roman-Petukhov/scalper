"""Почему бот 4h почти одни шорты: воронка фильтров отдельно для лонгов и шортов.

Поводы — все пробои линий зигзага 3 ATR на 4h (research/noline.py, lines_only). Для каждого записаны флаги фильтров
бота и результат сделки «как у бота» (лимитка на линии 12 свечей, стоп за свингом PIV5 ± 0.1 ATR, 3R, долив 0.5 на
+1.5R — research/dconf.py), посчитанный при любых флагах, если стоп и риск допустимы и лимитка исполнилась:
    оборот      ADV30 от $20M
    тренд       дневное закрытие против EMA50 в сторону сделки
    агрессоры   доля тейкеров в сторону сделки ≥ 55%
    закрытие    закрытие в своей половине свечи
    стоп        есть свинг, риск 0.3–4 ATR
    ретест      лимитка на линии исполнилась
Отчёт: воронка по шагам (сколько поводов осталось и R на сделку), «бот без одного фильтра» по сторонам и периодам,
доля времени монет в дневном восходящем тренде по годам (свечи 4h с оборотом от $20M).

    python -m research.sidefunnel collect --symbols <1h-qualified>
    python -m research.sidefunnel report
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30
from .dconf import EXITS, simulate
from .noline import triggers
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import MAKER, PIV, RETEST_BARS, STOP_BUF, TAKER, htf_trend, last_confirmed, pivots, retest_fill, tf_frame

TF = "4h"
BOT_EXIT = EXITS["бот (3R, долив на +1.5R)"]
FILTERS = ("liq", "trend", "aggr", "loc", "stop", "fill")
NAMES = {"liq": "оборот от $20M", "trend": "дневной тренд", "aggr": "агрессоры ≥ 55%", "loc": "закрытие в своей половине",
         "stop": "стоп за свингом 0.3–4 ATR", "fill": "ретест исполнен"}


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str) -> tuple[list[dict], list[dict]]:
    """Поводы с флагами и R; и счётчики свечей в тренде по годам."""
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64")
    a = _atr(d).to_numpy()
    trend = htf_trend(d, TF)
    buy = (d["taker_buy_volume"] / d["volume"].replace(0, np.nan)).to_numpy()
    rng = hi - lo
    loc = (c - lo) / np.where(rng > 0, rng, np.nan)
    sw_lo, sw_hi = last_confirmed(pivots(lo, PIV, False), len(c)), last_confirmed(pivots(hi, PIV, True), len(c))
    rows = []
    for _, t, side, level in triggers(d, a, lines_only=True):
        if t >= len(c) - 1 or not (a[t] > 0):
            continue
        aggr = buy[t] if side > 0 else 1 - buy[t]
        cl = loc[t] if side > 0 else 1 - loc[t]
        row = {"symbol": sym, "t": d.index[t], "side": side, "liq": bool(liq[t] >= ADV_MIN),
               "trend": bool(trend[t] == side), "aggr": bool(aggr >= 0.55), "loc": bool(cl >= 0.5),
               "stop": False, "fill": False, "R": np.nan}
        sw = sw_lo[t] if side > 0 else sw_hi[t]
        if sw >= 0:
            stop = (lo[sw] - STOP_BUF * a[t]) if side > 0 else (hi[sw] + STOP_BUF * a[t])
            risk = side * (level - stop)
            if 0.3 * a[t] <= risk <= 4.0 * a[t]:
                row["stop"] = True
                fill = retest_fill(hi, lo, t, side, level, 3.0 * risk, RETEST_BARS)
                if fill >= 0:
                    row["fill"] = True
                    if (side > 0 and lo[fill] <= stop) or (side < 0 and hi[fill] >= stop):
                        row["R"] = (side * (stop - level) - (MAKER + TAKER) * level) / risk
                    else:
                        row["R"] = simulate(o, hi, lo, c, f, fill, side, level, stop, a[t], BOT_EXIT)
        rows.append(row)
    ok = liq >= ADV_MIN
    yr = d.index.year.to_numpy()
    time_rows = [{"symbol": sym, "year": int(y), "bars": int((ok & (yr == y)).sum()),
                  "up": int((ok & (yr == y) & (trend > 0)).sum())} for y in np.unique(yr)]
    return rows, time_rows


def collect(root: Path, syms: list[str]) -> None:
    rows, trows = [], []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, TF)
            if d is None or len(d) < 500:
                continue
            liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
            r, tr = coin_rows(d, liq, s)
            rows += r
            trows += tr
        except Exception as e:
            print(f"  sidefunnel {s}: пропуск ({e})", flush=True)
    print(f"  sidefunnel: поводов {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("sidefunnel"), index=False)
        pd.DataFrame(trows).to_parquet(part_path("sidefunnel_time"), index=False)


def _r(z: pd.DataFrame) -> str:
    r = z["R"].dropna()
    return f"{len(z)} (R {r.mean():+.2f} на {len(r)})" if len(r) else f"{len(z)}"


def report() -> None:
    parts = all_parts("sidefunnel")
    print("\n===== SIDEFUNNEL: почему бот 4h почти одни шорты — воронка фильтров по сторонам; R — сделка как у бота "
          "(ретест, 3R, долив на +1.5R) =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    sides = ((1, "лонги"), (-1, "шорты"))
    print("\n  Воронка (накопительно): поводов осталось (R на исполненных при этих фильтрах)")
    for s, nm in sides:
        z, line = df[df.side == s], [f"все пробои {len(df[df.side == s])}"]
        for k in FILTERS:
            z = z[z[k]]
            line.append(f"{NAMES[k]} → {_r(z)}")
        print(f"    {nm}: " + "; ".join(line))
    print("\n  Доля прошедших каждый фильтр отдельно (среди поводов с оборотом от $20M)")
    liq = df[df.liq]
    for s, nm in sides:
        z = liq[liq.side == s]
        print(f"    {nm} ({len(z)}): " + ", ".join(f"{NAMES[k]} {z[k].mean():.0%}" for k in FILTERS[1:]))
    z = liq[liq.trend]
    print("    при дневном тренде в сторону сделки — агрессоры ≥ 55%: " + ", ".join(
        f"{nm} {z[z.side == s].aggr.mean():.0%}" for s, nm in sides) + "; закрытие в своей половине: " + ", ".join(
        f"{nm} {z[z.side == s]['loc'].mean():.0%}" for s, nm in sides))
    periods = {**PER_ALL, "2026": ("2026-01-01", "2026-10-01")}
    print("\n  Бот и бот без одного фильтра, по периодам (R на сделку)")
    for drop in (None, "trend", "aggr", "loc"):
        keep = [k for k in FILTERS if k != drop]
        z0 = df[df[keep].all(axis=1)]
        print(f"    {'бот' if drop is None else 'без фильтра «' + NAMES[drop] + '»'}:")
        for s, nm in sides:
            z = z0[z0.side == s]
            print(f"      {nm} {len(z)}: " + "  ".join(
                f"{p}: {_cell(z[(z.t >= a) & (z.t < b)])}" for p, (a, b) in periods.items()))
    tp = all_parts("sidefunnel_time")
    if tp:
        tm = pd.concat([pd.read_parquet(p) for p in tp], ignore_index=True).groupby("year")[["bars", "up"]].sum()
        print("\n  Доля времени монет с оборотом от $20M выше дневной EMA50 (свечи 4h): " + ", ".join(
            f"{y} {r.up / r.bars:.0%}" for y, r in tm.iterrows() if r.bars > 0))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x])
    else:
        report()
