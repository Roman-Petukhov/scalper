"""Разбор сигнала: как бот видел линии в каждой из последних свечей. Свечи Binance USDⓈ-M (как у бота, окно
HISTORY), на каждой свече — расчёт ровно как в сканере (detect на данных до этой свечи включительно): все линии
нужной стороны, которые построитель знает на этой свече (точки A и B, значение линии, закрытие), пробой и почему
сигнал отфильтрован (агрессоры, тренд старшего ТФ, место закрытия, стоп).

    python -m tools.sigtrace RVNUSDT 4h -1 2026-09-28
"""
from __future__ import annotations

import sys

import httpx
import numpy as np
import pandas as pd

from research.smc import _atr
from research.tline import htf_trend, zz_lines
from trader.domain.models import Settings, Timeframe
from trader.domain.strategy import _with_probe_bar, detect
from trader.infrastructure.binance_data import HISTORY, BinanceMarketData


def main(symbol: str, tf_s: str, side: int, since: str) -> None:
    tf = Timeframe(tf_s)
    params = {"symbol": symbol, "interval": tf.value, "limit": HISTORY}
    r = httpx.get("https://fapi.binance.com/fapi/v1/klines", params=params, timeout=30)
    if r.status_code != 200:                     # с серверов в США fapi закрыт (451) — спот с зеркала, свечи близкие
        print(f"fapi {r.status_code}: свечи спота data-api.binance.vision")
        r = httpx.get("https://data-api.binance.vision/api/v3/klines", params=params, timeout=30)
    rows = r.json()
    full = BinanceMarketData._frame(rows)
    full = full[full["close_time"] < pd.Timestamp.now(tz="UTC").value // 10**6].drop(columns="close_time")
    s = Settings()
    start = full.index.searchsorted(pd.Timestamp(since, tz="UTC"))
    for k in range(start, len(full)):
        d = full.iloc[max(0, k + 1 - HISTORY): k + 1]
        last = len(d) - 1
        c = d["close"].to_numpy()
        atr = _atr(d).to_numpy()
        trend = int(htf_trend(d, tf.value)[last])
        buy = d["taker_buy_volume"].iloc[last] / d["volume"].iloc[last]
        aggr = buy if side > 0 else 1 - buy
        rng = d["high"].iloc[last] - d["low"].iloc[last]
        loc = ((c[last] - d["low"].iloc[last]) if side > 0 else (d["high"].iloc[last] - c[last])) / rng if rng else 0
        x = _with_probe_bar(d)
        lines = [r for r in zz_lines(x) if r["side"] == side]
        live = []
        for r in lines:
            if r["t"] < 0 or r["t"] >= last - 6:              # живые и пробитые за последние 6 свечей
                lv = c[r["a"]] + r["slope"] * (last - r["a"])
                state = ("ПРОБОЙ на этой свече" if r["t"] == last else
                         f"пробита {d.index[r['t']]:%m-%d %H:%M}" if r["t"] >= 0 else "жива")
                live.append(f"A {d.index[r['a']]:%m-%d %H:%M} B {d.index[r['b']]:%m-%d %H:%M} линия {lv:.7g} "
                            f"(закрытие {side * (c[last] - lv) / atr[last]:+.2f} ATR за ней) — {state}")
        sig = detect(d, tf, symbol, s)
        print(f"{d.index[last]:%m-%d %H:%M} close {c[last]:.7g} агрессоры {aggr:.0%} тренд {trend:+d} "
              f"место {loc:.2f} сигнал {[int(x.side) for x in sig]}")
        for t in live:
            print("    " + t)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
