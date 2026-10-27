import numpy as np
import pandas as pd

from research import longlevels as L
from research.tline import MAKER, TAKER


def _ohlc(c):
    c = np.asarray(c, dtype="float64")
    o = np.r_[c[0], c[:-1]]
    return o, np.maximum(o, c), np.minimum(o, c), c


def test_bounce_fills_at_the_level_with_stop_below():
    c = [10, 9, 8, 10, 12, 13, 12, 7.9, 10, 12, 14, 16, 18]   # впадина 8, касание на 7-м баре, стоп 7.5 не задет
    o, hi, lo, cl = _ohlc(c)
    n = len(cl)
    a, trend, liq = np.full(n, 1.0), np.ones(n), np.full(n, 1e9)
    rows = L.bounce_rows(o, hi, lo, cl, np.zeros(n), a, trend, liq, np.array([[2, 4]]), 1, 100)
    assert len(rows) == 1 and rows[0][0] == 7
    assert abs(rows[0][1] - (1.5 - (MAKER + MAKER) * 8) / 0.5) < 1e-9           # цель 3R = 9.5, лимиткой


def test_false_break_enters_after_close_back_above():
    c = [10, 9, 8, 10, 12, 11, 7.8, 8.5, 9, 10, 12, 14, 16, 18, 20]
    o, hi, lo, cl = _ohlc(c)
    n = len(cl)
    a, trend, liq = np.full(n, 1.0), np.ones(n), np.full(n, 1e9)
    rows = L.false_break_rows(o, hi, lo, cl, np.zeros(n), a, trend, liq, np.array([[2, 4]]), 1, 100)
    assert len(rows) == 1 and rows[0][0] == 7 and rows[0][1] > 2.5


def test_trend_against_blocks_the_bounce():
    c = [10, 9, 8, 10, 12, 13, 12, 7.9, 10]
    o, hi, lo, cl = _ohlc(c)
    n = len(cl)
    rows = L.bounce_rows(o, hi, lo, cl, np.zeros(n), np.ones(n), -np.ones(n), np.full(n, 1e9), np.array([[2, 4]]), 1, 50)
    assert rows == []


def test_rows_on_random_walk_cover_modes_and_sides():
    rng = np.random.default_rng(8)
    n = 8000
    idx = pd.date_range("2021-01-01", periods=n, freq="4h", tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, n)))
    v = np.full(n, 1000.0)
    d = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": v,
                      "taker_buy_volume": v * rng.uniform(0.3, 0.8, n), "funding": 0.0}, index=idx)
    x = pd.DataFrame(L.coin_rows(d, np.full(n, 1e9), "RW", "4h"))
    assert set(x["mode"]) == {"пробой", "отбой", "ложный пробой"} and set(x.side) == {1, -1}
    assert x.R.notna().all() and (x.R >= -1.5).all()
