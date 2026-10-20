import numpy as np
import pandas as pd

from research import tlcanon as T
from research.tline import zz_lines


def _rw(seed: int, n: int = 8000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, n)))
    vol = np.full(n, 1000.0)
    idx = pd.date_range("2018-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": vol,
                         "taker_buy_volume": vol * rng.uniform(0.3, 0.8, n)}, index=idx)


def test_touches_counts_separate_approaches():
    y = np.full(40, 90.0)
    y[[0, 10, 11, 12, 25]] = [100.0, 99.0, 98.0, 99.0, 98.5]      # line 100 - 0.05·j: touches at 0, 10-12 (one), 25
    atr = np.full(40, 2.0)
    assert T.touches(y, atr, 0, 40, 1, 100.0, -0.05) == 3
    assert T.touches(y, atr, 0, 20, 1, 100.0, -0.05) == 2


def test_wick_lines_built_on_highs_and_break_on_close():
    d = _rw(1)
    hi, c = d.high.to_numpy(), d.close.to_numpy()
    lines = [r for r in zz_lines(d, wicks=True) if r["t"] > 0 and r["side"] == 1]
    assert lines
    for r in lines[:20]:
        a, t, sl = r["a"], r["t"], r["slope"]
        assert abs(r["line_t"] - (hi[a] + sl * (t - a))) < 1e-9
        assert c[t] > r["line_t"] and c[t - 1] <= hi[a] + sl * (t - 1 - a)


def test_rows_have_features_and_no_random_walk_edge():
    rows = []
    for seed in range(6):
        d = _rw(seed)
        for wicks in (False, True):
            rows += T.coin_rows(d, np.full(len(d), 1e9), "RW", "4h", wicks)
    x = pd.DataFrame(rows)
    assert {"close", "wick"} <= set(x.kind) and (x.touches >= 1).all() and (x.depth_atr > 0).all()
    for k, z in x[x.entry == "retest"].groupby("kind"):
        assert z.R.mean() < 0.15, (k, z.R.mean(), len(z))
