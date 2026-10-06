import numpy as np
import pandas as pd

from research.range4 import N, coin_trades, range_signals


def _wave(n=200, mid=100.0, amp=3.0, period=10):
    t = np.arange(n)
    c = mid + amp * np.sin(2 * np.pi * t / period)
    return c + 0.2, c - 0.2, c


def test_signals_in_flat_range_and_none_in_trend():
    hi, lo, c = _wave()
    a = np.full(len(c), 1.5)                                   # коробка 6.4 → 4.3 ATR, касаний много
    sig = range_signals(hi, lo, c, a)
    assert sig and {s for _, s, *_ in sig} == {1, -1}
    t, side, level, stop, far = sig[0]
    assert t >= N and (stop < level < far if side > 0 else far < level < stop)
    tr = np.linspace(100, 200, 200)
    assert range_signals(tr + 0.2, tr - 0.2, tr, np.full(200, 1.5)) == []     # тренд — смещение больше 1.5 ATR


def test_trades_have_costs_and_sane_r():
    hi, lo, c = _wave(600)
    idx = pd.date_range("2024-01-01", periods=len(c), freq="4h", tz="UTC")
    d = pd.DataFrame({"open": c, "high": hi, "low": lo, "close": c, "volume": 1.0, "quote_volume": 1e9, "count": 1,
                      "taker_buy_volume": 0.5, "taker_buy_quote_volume": 5e8, "funding": 0.0}, index=idx)
    x = coin_trades(d, np.full(len(c), 1e9), "T")
    assert len(x) and set(x.tgt) == {"mid", "opp"} and x.R.notna().all() and (x.R <= x.k + 1e-9).all()
