import numpy as np
import pandas as pd

from research.sidefunnel import FILTERS, coin_rows


def _frame(n=900, seed=5):
    idx = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    o = np.r_[c[0], c[:-1]]
    v = rng.uniform(1e5, 2e5, n)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.005, "low": np.minimum(o, c) * 0.995, "close": c,
                         "volume": v, "taker_buy_volume": v * rng.uniform(0.3, 0.7, n), "funding": np.zeros(n)},
                        index=idx)


def test_every_line_break_is_kept_with_consistent_flags():
    d = _frame()
    rows, trows = coin_rows(d, np.full(len(d), 1e9), "X")
    assert rows and {r["side"] for r in rows} <= {-1, 1}
    for r in rows:
        assert set(FILTERS) <= set(r)
        assert not r["fill"] or r["stop"]                  # ретест — только при допустимом стопе
        assert np.isfinite(r["R"]) == r["fill"]
        assert r["liq"]
    assert sum(t["bars"] for t in trows) == len(d)
    assert all(0 <= t["up"] <= t["bars"] for t in trows)


def test_illiquid_bars_are_flagged_and_not_counted_in_trend_time():
    d = _frame()
    rows, trows = coin_rows(d, np.zeros(len(d)), "X")
    assert rows and not any(r["liq"] for r in rows)
    assert sum(t["bars"] for t in trows) == 0
