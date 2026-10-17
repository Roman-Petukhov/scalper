import numpy as np
import pandas as pd

from research.retest15 import WINDOWS, coin_rows


def test_rows_have_market_and_every_window():
    n = 6000
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(1)
    c = 100 * np.exp(np.cumsum(rng.normal(-0.0002, 0.004, n)))
    o = np.r_[c[0], c[:-1]]
    v = rng.uniform(1e5, 2e5, n)
    d = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.002, "low": np.minimum(o, c) * 0.998, "close": c,
                      "volume": v, "taker_buy_volume": v * rng.uniform(0.3, 0.5, n), "funding": 0.0}, index=idx)
    rows = pd.DataFrame(coin_rows(d, np.full(n, 1e9), "X"))
    assert len(rows)
    cols = ["market"] + [f"retest{w}" for w in WINDOWS]
    assert set(cols) <= set(rows.columns)
    r = rows[cols].replace(np.inf, np.nan)
    fills = rows[[f"retest{w}" for w in WINDOWS]].apply(lambda x: np.isfinite(x).mean())
    assert (np.diff(fills.to_numpy()) >= -1e-12).all()                 # окно длиннее — исполняется не реже
    assert r.notna().any().all()
