import numpy as np
import pandas as pd

from research import formation as F


def _rw(seed: int, n: int = 30000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="15min", tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.002, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.002, n)))
    v = np.full(n, 1000.0)
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": v,
                         "taker_buy_volume": v * rng.uniform(0.3, 0.8, n), "funding": 0.0}, index=idx)


def test_features_of_a_built_line():
    c = np.array([10.0, 9.0, 8.0, 9.4, 7.0, 8.9, 9.5])
    atr = np.full(7, 0.5)
    rec = {"a": 0, "b": 3, "side": 1, "slope": -0.2}          # линия 10, 9.8, 9.6, 9.4, 9.2, 9.0, 8.8
    f = F.features(c, atr, rec, 6, 8.0, 9.5)
    assert (f["length"], f["span"], f["age"]) == (6, 3, 3)
    assert f["height"] == (9.2 - 7.0) / 0.5                     # самое далёкое закрытие под линией
    assert f["slope"] == 0.4 and f["stop"] == 3.0 and f["squeeze"] == 1.0 and f["touches"] >= 2


def test_rows_on_a_random_walk(monkeypatch):
    monkeypatch.setattr(F, "MAX_SLOPE", 1e9)
    rows = pd.DataFrame(F.coin_rows(_rw(1), np.full(30000, 1e9), "RW", "15m"))
    assert len(rows) > 20 and rows[list(F.FEATS)].notna().all().all()
    assert (rows.height >= 0).all() and (rows.age >= 0).all() and (rows.length > rows.span).all()


def test_is_edges_use_only_is_period():
    t = pd.to_datetime(["2023-01-01"] * 100 + ["2025-12-01"] * 100)
    z = pd.DataFrame({"t": t, "x": np.r_[np.arange(100.0), np.arange(100.0) + 1000]})
    assert F.is_edges(z, "x").max() < 100
