import numpy as np
import pandas as pd

from research import retest2 as R


def test_nearest_swing_takes_latest_unbroken_after_anchor():
    hi = np.array([10, 9, 12, 8, 11, 7, 9.5, 6], dtype=float)
    piv = np.array([[0, 2], [2, 4], [4, 6], [6, 7]])
    assert R.nearest_swing(hi, piv, 1, 7, 1, 9.0) == 9.5        # pivot 6 confirmed at 7, above close
    assert R.nearest_swing(hi, piv, 1, 6, 1, 9.0) == 11.0       # pivot 6 not confirmed yet
    assert R.nearest_swing(hi, piv, 1, 7, 1, 12.5) != R.nearest_swing(hi, piv, 1, 7, 1, 12.5)   # nan: none above
    assert np.isnan(R.nearest_swing(hi, piv, 4, 6, 1, 12.5))


def test_fill_tp_moving_line_and_cancels():
    c = np.array([10, 11, 11.5, 11.2, 11.0, 12.0])
    hi, lo, o = c + 0.2, c - 0.2, c
    line = lambda j: 10.0 - 0.5 * (j - 1)                        # descending line
    j, p = R._fill_tp(o, hi, lo, c, 1, 1, 5, line, stop=9.0)
    assert j == -1                                                # never touched (line falls away)
    j, p = R._fill_tp(o, hi, lo, c, 1, 1, 5, lambda j: 11.0, stop=10.0)
    assert (j, p) == (1, 11.0)
    j, _ = R._fill_tp(o, hi, lo, c, 1, 1, 5, lambda j: 10.5, stop=10.3)   # 3R target 11.1 reached first
    assert j == -1
    j, _ = R._fill_tp(o, hi, lo, c, 1, 1, 5, lambda j: 10.5, stop=10.0, cancel_line=lambda j: 11.4)
    assert j == -1                                                # close 11.0 < line at bar 1 → cancel


def test_random_walk_has_no_edge():
    out = []
    for seed in range(12):
        rng = np.random.default_rng(seed)
        n = 15000
        idx = pd.date_range("2018-01-01", periods=n, freq="4h", tz="UTC")
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
        o = np.r_[c[0], c[:-1]]
        hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, n)))
        lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, n)))
        vol = np.full(n, 1000.0)
        d = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": vol,
                          "taker_buy_volume": vol * rng.uniform(0.3, 0.8, n)}, index=idx)
        out += R.coin_rows(d, np.full(n, 1e9), "RW", "4h")
    rows = pd.DataFrame(out)
    assert len(rows) > 100
    for v in R.VARIANTS:
        assert rows[v].mean() < 0.1, (v, rows[v].mean(), len(rows))     # no look-ahead edge
