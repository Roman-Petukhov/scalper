"""Разбор одной монеты за окно: что сделал бот 4h и что дала бы трендовая система Дончиана на дневках.

Бот 4h — каждый пробой линии зигзага 3 ATR в окне (обе стороны) с проверкой всех фильтров бота по порядку: оборот от
$20M, дневной тренд (EMA50), агрессоры >= 55%, закрытие в своей половине свечи, свинг для стопа, риск 0.3–4 ATR, ретест.
Для каждого пробоя — результат сделки, как у бота (3R, долив 0.5 на +1.5R), даже если фильтр его отрезал: видно, что
потеряно. Вход бота — гибрид: по рынку, если свеча пробоя от 2.15 ATR, иначе лимитка на линии (ретест 12 свечей).
Дончиан на дневных закрытиях (одна позиция, лонг): вход на закрытии выше максимума закрытий N прошлых дней, выход на
закрытии ниже минимума M прошлых дней (или середины канала N — как в research/dtrend.py). R — от расстояния до уровня
выхода в день входа. Это пример, а не проверка правила: одна монета в одном тренде ничего не доказывает.

    CASE_SYMBOL=NEARUSDT CASE_FROM=2026-07-20 CASE_TO=2026-10-14 python -m research.case collect
    python -m research.case report
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
from .dconf import EXITS, simulate
from .noline import triggers
from .shard import all_parts, mine, part_path
from .smc import _atr
from .tline import MAKER, PIV, RETEST_BARS, STOP_BUF, TAKER, htf_trend, last_confirmed, pivots, retest_fill, tf_frame

SYMBOL = os.environ.get("CASE_SYMBOL", "NEARUSDT")
START = os.environ.get("CASE_FROM", "2026-07-20")
END = os.environ.get("CASE_TO", "2026-10-14")
BOT_EXIT = EXITS["бот (3R, долив на +1.5R)"]
HYBRID_ATR = 2.15
DONCHIAN = (("20 / 10", 20, 10), ("55 / 20", 55, 20), ("20 / середина канала", 20, 0))


def _r(o, hi, lo, c, f, t, side, level, stop, a, market: bool) -> float:
    if market:
        px, fill, fee = c[t], t, TAKER
    else:
        px, fee = level, MAKER
        fill = retest_fill(hi, lo, t, side, level, 3.0 * side * (level - stop), RETEST_BARS)
        if fill < 0:
            return np.nan
        if (side > 0 and lo[fill] <= stop) or (side < 0 and hi[fill] >= stop):
            return (side * (stop - level) - (MAKER + TAKER) * level) / (side * (level - stop))
    if not side * (px - stop) > 0:
        return np.nan
    return simulate(o, hi, lo, c, f, fill, side, px, stop, a, BOT_EXIT, fee)


def bot_rows(d: pd.DataFrame, liq: np.ndarray, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64")
    a = _atr(d).to_numpy()
    trend = htf_trend(d, "4h")
    buy = (d["taker_buy_volume"] / d["volume"].replace(0, np.nan)).to_numpy()
    rng = hi - lo
    loc = (c - lo) / np.where(rng > 0, rng, np.nan)
    sw_lo, sw_hi = last_confirmed(pivots(lo, PIV, False), len(c)), last_confirmed(pivots(hi, PIV, True), len(c))
    rows = []
    for _, t, side, level in triggers(d, a, lines_only=True):
        if not (start <= d.index[t] < end) or t >= len(c) - 1 or not (a[t] > 0):
            continue
        aggr = buy[t] if side > 0 else 1 - buy[t]
        cl = loc[t] if side > 0 else 1 - loc[t]
        sw = sw_lo[t] if side > 0 else sw_hi[t]
        stop = np.nan if sw < 0 else (lo[sw] - STOP_BUF * a[t]) if side > 0 else (hi[sw] + STOP_BUF * a[t])
        risk = side * (level - stop) / a[t]
        market = rng[t] / a[t] >= HYBRID_ATR
        failed = [nm for nm, ok in (("оборот", liq[t] >= ADV_MIN), ("тренд", trend[t] == side),
                                    ("агрессоры", aggr >= 0.55), ("закрытие", cl >= 0.5), ("свинг", sw >= 0),
                                    ("риск", 0.3 <= risk <= 4.0)) if not ok]
        r = np.nan if sw < 0 or not np.isfinite(risk) or risk <= 0 else _r(o, hi, lo, c, f, t, side, level, stop,
                                                                           a[t], market)
        if not failed and not market and not np.isfinite(r):
            failed.append("ретест не исполнился")
        rows.append({"t": d.index[t] + pd.Timedelta(hours=4), "side": side, "close": c[t], "line": level,
                     "aggr": aggr, "trend": bool(trend[t] == side), "loc": cl, "risk_atr": risk,
                     "risk_pct": side * (level - stop) / level if np.isfinite(stop) else np.nan,
                     "entry": "рынок" if market else "ретест", "failed": ", ".join(failed), "R": r})
    return rows


def donchian_trades(close: pd.Series, n: int, m: int) -> list[dict]:
    """Сделки одной системы на дневных закрытиях; m = 0 — выход ниже середины канала n."""
    hi = close.rolling(n, min_periods=n).max().shift(1)
    lo_n = close.rolling(n, min_periods=n).min().shift(1)
    out_lv = ((hi + lo_n) / 2) if m == 0 else close.rolling(m, min_periods=m).min().shift(1)
    trades, pos = [], None
    for day, px in close.items():
        if pos is None:
            if np.isfinite(hi[day]) and px >= hi[day] and np.isfinite(out_lv[day]) and out_lv[day] < px:
                pos = {"entry_t": day, "entry": px, "risk": px - out_lv[day]}
        elif np.isfinite(out_lv[day]) and px <= out_lv[day]:
            trades.append({**pos, "exit_t": day, "exit": px})
            pos = None
    if pos is not None:
        trades.append({**pos, "exit_t": pd.NaT, "exit": float(close.iloc[-1])})
    for x in trades:
        x["pct"] = x["exit"] / x["entry"] - 1 - 2 * TAKER
        x["R"] = (x["exit"] - x["entry"] - 2 * TAKER * x["entry"]) / x["risk"]
    return trades


def collect(root: Path) -> None:
    if not mine([SYMBOL]):
        return
    d = tf_frame(root, SYMBOL, "4h")
    if d is None:
        print(f"  case: нет данных {SYMBOL}", flush=True)
        return
    start, end = pd.Timestamp(START, tz="UTC"), pd.Timestamp(END, tz="UTC")
    liq = adv30(root, SYMBOL).reindex(d.index.floor("D")).to_numpy()
    bot = pd.DataFrame(bot_rows(d, liq, start, end))
    bot["symbol"] = SYMBOL
    bot.to_parquet(part_path("case_bot"), index=False)
    day = tf_frame(root, SYMBOL, "1d")["close"]
    don = []
    for nm, n, m in DONCHIAN:
        for x in donchian_trades(day, n, m):
            if x["exit_t"] is pd.NaT or x["exit_t"] >= start:
                don.append({"system": nm, **x})
    pd.DataFrame(don).to_parquet(part_path("case_don"), index=False)
    w = d[(d.index >= start) & (d.index < end)]
    pd.DataFrame({"symbol": [SYMBOL], "first": [w.index[0]], "last": [w.index[-1] + pd.Timedelta(hours=4)],
                  "low": [w["low"].min()], "low_t": [w["low"].idxmin()], "high": [w["high"].max()],
                  "high_t": [w["high"].idxmax()], "last_close": [w["close"].iloc[-1]]}).to_parquet(
        part_path("case_info"), index=False)
    print(f"  case {SYMBOL}: пробоев 4h {len(bot)}, сделок Дончиана {len(don)}", flush=True)


def _fmt_t(t) -> str:
    return "открыта" if pd.isna(t) else pd.Timestamp(t).strftime("%d.%m %H:%M")


def report() -> None:
    bp, dp, ip = all_parts("case_bot"), all_parts("case_don"), all_parts("case_info")
    print(f"\n===== CASE: {SYMBOL} {START} — {END}: бот 4h и Дончиан на дневках =====")
    if not bp or not ip:
        print("частей нет")
        return
    info = pd.read_parquet(ip[0]).iloc[0]
    print(f"данные {info['first']:%d.%m.%Y} — {info['last']:%d.%m.%Y %H:%M} UTC; минимум {info['low']:.4g} "
          f"({info['low_t']:%d.%m}), максимум {info['high']:.4g} ({info['high_t']:%d.%m}), ×{info['high'] / info['low']:.2f}; "
          f"последнее закрытие {info['last_close']:.4g}")
    bot = pd.read_parquet(bp[0])
    print("\n  --- бот 4h: пробои линий (время — закрытие свечи пробоя, UTC); R — сделка как у бота, даже если фильтр "
          "её отрезал ---")
    for side, nm in ((1, "лонги"), (-1, "шорты")):
        z = bot[bot.side == side]
        taken = z[z.failed == ""]
        print(f"\n  {nm}: пробоев {len(z)}, прошли все фильтры {len(taken)} → {taken.R.sum():+.2f}R; "
              f"все пробои без фильтров {z.R.sum():+.2f}R ({z.R.notna().sum()} сделок)")
        for _, x in z.iterrows():
            tag = "ВХОД" if x.failed == "" else f"отсев: {x.failed}"
            r = "нет сделки" if not np.isfinite(x.R) else f"{x.R:+.2f}R"
            print(f"    {_fmt_t(x.t)}  закр {x.close:.4g}  линия {x.line:.4g}  агр {x.aggr:.0%}  "
                  f"тренд {'да' if x.trend else 'нет'}  закр. в половине {x.loc:.0%}  риск {x.risk_atr:.1f} ATR "
                  f"({x.risk_pct:.1%})  {x.entry}  → {r}  [{tag}]")
        if len(z):
            cnt = pd.Series([f for s in z.failed if s for f in s.split(", ")]).value_counts()
            if len(cnt):
                print("    чаще всего отсекал: " + ", ".join(f"{k} {v}" for k, v in cnt.items()))
    if dp:
        don = pd.read_parquet(dp[0])
        print("\n  --- Дончиан на дневных закрытиях, лонг, одна позиция; R — от расстояния до уровня выхода при входе ---")
        for nm, _, _ in DONCHIAN:
            z = don[don.system == nm]
            print(f"\n  {nm}: сделок {len(z)}, всего {z.R.sum():+.2f}R")
            for _, x in z.iterrows():
                print(f"    вход {_fmt_t(x.entry_t)} по {x.entry:.4g} (выход-уровень {x.entry - x.risk:.4g}, "
                      f"риск {x.risk / x.entry:.1%})  →  выход {_fmt_t(x.exit_t)} по {x.exit:.4g}:  "
                      f"{x.pct:+.1%}, {x.R:+.2f}R")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    a = ap.parse_args()
    collect(Path(a.root).expanduser()) if a.mode == "collect" else report()
