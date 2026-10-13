import numpy as np
import pandas as pd

from research.topcrowd import coin_rows, split, top_dev


def test_top_dev_is_relative_to_the_coins_own_month():
    idx = pd.date_range("2023-01-01", periods=600, freq="4h", tz="UTC")
    top = pd.Series(np.r_[np.full(400, 2.0), np.full(200, 1.0)], index=idx)
    dev = top_dev(top, "4h")
    assert abs(dev[300]) < 1e-12 and dev[450] < 0                    # стали меньше в лонг, чем обычно


def test_rows_and_split():
    idx = pd.date_range("2023-01-01", periods=900, freq="4h", tz="UTC")
    rng = np.random.default_rng(5)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 900)))
    o = np.r_[c[0], c[:-1]]
    v = rng.uniform(1e5, 2e5, 900)
    d = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.006, "low": np.minimum(o, c) * 0.994, "close": c,
                      "volume": v, "taker_buy_volume": v * rng.uniform(0.4, 0.6, 900), "funding": 0.0}, index=idx)
    rows = pd.DataFrame(coin_rows(d, np.full(900, 1e9), "X", "4h", rng.normal(0, 1, 900)))
    assert len(rows) and set(rows.side) <= {-1, 1} and rows.R.notna().all()
    z = pd.DataFrame({"t": pd.date_range("2023-01-01", periods=40, freq="1D", tz="UTC"),
                      "dev": np.r_[np.full(20, 0.1), np.full(20, -0.1)], "R": np.r_[np.full(20, 1.0), np.full(20, -1.0)]})
    ok, _ = split(z, {"x": ("2023-01-01", "2024-01-01")}, True)
    assert ok == [True]
    ok, _ = split(z, {"x": ("2023-01-01", "2024-01-01")}, False)
    assert ok == [False]
