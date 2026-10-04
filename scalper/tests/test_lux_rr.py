import numpy as np
import pandas as pd

from research.lux import lux_signals, slope_series
from research.lux_rr import lux_breaks, simulate


def _df(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    h, l = c * (1 + rng.random(n) * 0.003), c * (1 - rng.random(n) * 0.003)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(h, o), "low": np.minimum(l, o), "close": c})


def test_breaks_match_reference_signals():
    df = _df()
    sl = slope_series(df, 14, 1.0, "atr")
    h, l, c = df.high.to_numpy(), df.low.to_numpy(), df.close.to_numpy()
    up, dn = lux_signals(h, l, c, sl, 14)
    side, lv, _ = lux_breaks(h, l, c, sl, 14)
    assert ((side == 1) == (up & ~dn)).all() and ((side == -1) == (dn & ~up)).all()
    k = np.flatnonzero(side == 1)
    assert (c[k] > lv[k]).all()


def _sim(o, h, l, c, retest=False, tp=5.0):
    n = len(c)
    side = np.zeros(n, np.int8); side[0] = 1
    lv = np.full(n, np.nan); lv[0] = 99.0
    ls = np.zeros(n)
    return simulate(side, lv, ls, np.ones(n, np.bool_), np.array(o, float), np.array(h, float), np.array(l, float),
                    np.array(c, float), np.ones(n), 14, retest, False, tp, 8, 50, 0.0, 0.0)


def test_market_take_profit_five_r():
    i, r, s = _sim([100, 100, 103], [100, 101, 106], [100, 99.5, 102], [100, 100, 105])
    assert np.isclose(r[0], 5.0)


def test_market_stop_first_when_both_hit():
    i, r, s = _sim([100, 100, 100], [100, 106, 100], [100, 98, 100], [100, 100, 100])
    assert np.isclose(r[0], -1.0)


def test_retest_fills_at_line_and_needs_trade_through():
    # линия 99: бар 1 касается 99.0 ровно (нет исполнения), бар 2 проходит до 98.9 — исполнение по 99, далее тейк 5R
    i, r, s = _sim([100, 100, 99.5, 101, 104], [100, 100.5, 99.8, 103, 105], [100, 99.0, 98.9, 100.5, 103],
                   [100, 99.6, 99.4, 102, 104.5], retest=True)
    assert np.isclose(r[0], 5.0)


def test_frozen_price_with_zero_atr_is_skipped():
    n = 5
    side = np.zeros(n, np.int8); side[0] = 1
    lv = np.full(n, 99.0); ls = np.zeros(n)
    p = np.array([100, 100, 100, 150, 150], float)
    i, r, s = simulate(side, lv, ls, np.ones(n, np.bool_), p, p, p, p, np.zeros(n), 14, False, False, 5.0, 8, 50,
                       0.0, 0.0)
    assert len(i) == 0
