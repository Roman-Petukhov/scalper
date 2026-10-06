"""Нужна ли трендовая линия: правило бота 4h с другими поводами для входа вместо пробоя линии.

Все фильтры бота одинаковые: оборот от $20M, агрессоры в свече пробоя >= 55%, закрытие в верхней половине свечи
(для шорта — в нижней), сторона совпадает с дневным трендом (закрытие против EMA50), стоп за свингом PIV=5 ± 0.1 ATR
(0.3–4 ATR), лимитка ретеста на уровне пробоя 12 свечей, цель 3R, удержание 60 свечей. Меняется только повод:
  line  — пробой трендовой линии бота (зигзаг 3 ATR) — контроль;
  zzpiv — закрытие за последней подтверждённой впадиной (вершиной) того же зигзага 3 ATR — горизонтальный уровень;
  piv5  — закрытие за последним свингом PIV=5 по закрытиям (тот, что ближе);
  don20 — закрытие за минимумом (максимумом) закрытий 20 предыдущих свечей;
  bar   — закрытие за минимумом (максимумом) предыдущей свечи: повода почти нет, только фильтры.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import (HOLD, MAKER, PIV, RETEST_BARS, STOP_BUF, TAKER, ZZ_K, _years, htf_trend, last_confirmed, pivots,
                    retest_fill, tf_frame, two_targets, zigzag, zz_lines)

TF = "4h"
DON = 20
TRIGGERS = {"line": "трендовая линия (бот)", "zzpiv": "уровень зигзага 3 ATR", "piv5": "свинг PIV=5",
            "don20": "канал 20 свечей", "bar": "минимум/максимум прошлой свечи"}


def _level_series(c: np.ndarray, piv: np.ndarray) -> np.ndarray:
    """Для каждого бара — закрытие последнего подтверждённого (к закрытию бара) экстремума, NaN — нет."""
    idx = last_confirmed(piv, len(c))
    return np.where(idx >= 0, c[np.maximum(idx, 0)], np.nan)


def triggers(d: pd.DataFrame, a: np.ndarray) -> list[tuple[str, int, int, float]]:
    """(повод, бар пробоя, сторона, уровень для ретеста). Пробой — закрытие за уровнем, прошлое закрытие — нет."""
    c, hi, lo = (d[k].to_numpy(dtype="float64") for k in ("close", "high", "low"))
    out = [("line", r["t"], r["side"], r["line_t"]) for r in zz_lines(d) if r["t"] > 0]
    zh, zl = zigzag(c, a, ZZ_K)
    s = pd.Series(c)
    levels = {
        "zzpiv": (_level_series(c, zh), _level_series(c, zl)),
        "piv5": (_level_series(c, pivots(c, PIV, True)), _level_series(c, pivots(c, PIV, False))),
        "don20": (s.shift(1).rolling(DON).max().to_numpy(), s.shift(1).rolling(DON).min().to_numpy()),
        "bar": (np.r_[np.nan, hi[:-1]], np.r_[np.nan, lo[:-1]]),
    }
    for name, (up, dn) in levels.items():
        for side, lv in ((1, up), (-1, dn)):
            prev = np.r_[np.nan, lv[:-1]]
            hit = (side * (c - lv) > 0) & (side * (np.r_[np.nan, c[:-1]] - prev) <= 0)
            if name == "bar":
                hit = side * (c - lv) > 0
            for t in np.flatnonzero(hit):
                out.append((name, int(t), side, float(lv[t])))
    return out


def coin_trades(d: pd.DataFrame, liq: np.ndarray, sym: str) -> pd.DataFrame:
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    a = _atr(d).to_numpy()
    trend = htf_trend(d, TF)
    buy = (d["taker_buy_volume"] / d["volume"].replace(0, np.nan)).to_numpy()
    rng = hi - lo
    loc = (c - lo) / np.where(rng > 0, rng, np.nan)
    sw_lo, sw_hi = last_confirmed(pivots(lo, PIV, False), len(c)), last_confirmed(pivots(hi, PIV, True), len(c))
    rows = []
    for name, t, side, level in triggers(d, a):
        if t >= len(c) - 1 or not (a[t] > 0) or not (liq[t] >= ADV_MIN) or trend[t] != side:
            continue
        aggr = buy[t] if side > 0 else 1 - buy[t]
        cl = loc[t] if side > 0 else 1 - loc[t]
        if not (aggr >= 0.55 and cl >= 0.5):
            continue
        sw = sw_lo[t] if side > 0 else sw_hi[t]
        if sw < 0:
            continue
        stop = (lo[sw] - STOP_BUF * a[t]) if side > 0 else (hi[sw] + STOP_BUF * a[t])
        for entry in ("market", "retest"):
            if entry == "market":
                fill, px, fee = t, c[t], TAKER
            else:
                if not (side * (level - stop) > 0):
                    continue
                fill = retest_fill(hi, lo, t, side, level, 3 * side * (level - stop), RETEST_BARS)
                if fill < 0:
                    continue
                px, fee = level, MAKER
            risk = side * (px - stop)
            if not (0.3 * a[t] <= risk <= 4.0 * a[t]):
                continue
            if entry == "retest" and ((side > 0 and lo[fill] <= stop) or (side < 0 and hi[fill] >= stop)):
                r3 = (side * (stop - px) - (fee + TAKER) * px) / risk
            else:
                r3 = two_targets(o, hi, lo, c, f, fill, side, px, stop, 3.0, 3.0, False, HOLD[TF], fee)[0]
            rows.append({"symbol": sym, "trig": name, "t": d.index[t], "side": side, "entry": entry, "R3": r3,
                         "risk_pct": risk / px})
    return pd.DataFrame(rows)


def collect(root: Path, syms: list[str]) -> None:
    parts = []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, TF)
            if d is None or len(d) < 500:
                continue
            liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
            x = coin_trades(d, liq, s)
            if len(x):
                parts.append(x)
        except Exception as e:
            print(f"  noline {s}: пропуск ({e})", flush=True)
    print(f"  noline: монет со сделками {len(parts)}", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path("noline"), index=False)


def _months(p: str) -> float:
    a, b = PER_ALL[p]
    return (pd.Timestamp(b) - pd.Timestamp(a)).days / 30.4


def report() -> None:
    parts = all_parts("noline")
    print("\n===== NOLINE: нужна ли трендовая линия — правило бота 4h (оборот, агрессоры, свеча, тренд, стоп, 3R) с "
          "другими поводами для входа; ячейка — R на сделку (t по дням, прибыльных, сделок в месяц) =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["per"] = ""
    for p, (a, b) in PER_ALL.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    per_m = "R/мес " + " / ".join(PER_ALL)
    for entry in ("retest", "market"):
        rows = []
        for trig, nm in TRIGGERS.items():
            z = df[(df.trig == trig) & (df.entry == entry)]
            rows.append({"повод": nm, **{p: _cell(z.assign(R=z.R3)[z.per == p]) for p in PER_ALL},
                         per_m: " / ".join(f"{z[z.per == p].R3.sum() / _months(p):+.1f}" for p in PER_ALL)})
        print(f"\n  --- вход {'ретест' if entry == 'retest' else 'рынок'}, цель 3R ---")
        print(pd.DataFrame(rows).to_string(index=False))
    for trig in ("line", "zzpiv", "don20"):
        z = df[(df.trig == trig) & (df.entry == "retest")]
        print(f"  {trig}, ретест, по годам: {_years(z)}")
    # пересечение: сделки по линии, у которых в ±3 свечах (12 ч) был пробой уровня зигзага или канала в ту же сторону
    ln = df[(df.trig == "line") & (df.entry == "retest")]
    for trig in ("zzpiv", "don20", "piv5"):
        o = df[(df.trig == trig) & (df.entry == "market")][["symbol", "side", "t"]]
        m = ln[["symbol", "side", "t"]].reset_index().merge(o, on=["symbol", "side"], suffixes=("", "_o"))
        near = m[(m.t_o - m.t).abs() <= pd.Timedelta(hours=12)]["index"].unique()
        inside, alone = ln.loc[ln.index.isin(near)], ln.loc[~ln.index.isin(near)]
        print(f"  линия ∩ {trig} (±12 ч): {len(inside) / max(len(ln), 1):.0%} сделок линии; R на сделку: "
              f"совпали {inside.R3.mean():+.3f} ({len(inside)}), только линия {alone.R3.mean():+.3f} ({len(alone)})")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 400)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x])
    else:
        report()
