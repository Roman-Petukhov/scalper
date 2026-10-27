import numpy as np
import pandas as pd

from research import majors as M


def _rw(seed: int, n: int, freq: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq=freq, tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.008, n)))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.003, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.003, n)))
    v = np.full(n, 1000.0)
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": v,
                         "taker_buy_volume": v * rng.uniform(0.3, 0.8, n), "funding": 0.0}, index=idx)


def test_rows_keep_both_sides_and_record_aggressors():
    rows = pd.DataFrame(M.coin_rows(_rw(7, 15000, "4h"), np.full(15000, 1e9), "BTCUSDT", "4h"))
    assert len(rows) > 5 and set(rows.side) <= {1, -1} and (rows.aggr < 0.55).any()


def test_majors_list_is_fixed_and_groups_reference_it():
    assert len(M.MAJORS) == 10 and M.GROUPS["BTC"] == ("BTCUSDT",)
    assert set(M.GROUPS["десятка"]) == set(M.MAJORS)
