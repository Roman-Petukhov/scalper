import numpy as np
import pandas as pd

from research import longstop, reentry as R


def _rw(seed: int, n: int, freq: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq=freq, tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.006, n)))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.003, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.003, n)))
    v = np.full(n, 1000.0)
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": v,
                         "taker_buy_volume": v * rng.uniform(0.3, 0.8, n), "funding": 0.0}, index=idx)


def test_rows_are_shorts_and_reentries_follow_stops_only(monkeypatch):
    monkeypatch.setattr(R, "MAX_SLOPE_15M", 1e9)
    rows = pd.DataFrame(R.coin_rows(_rw(6, 30000, "15min"), np.full(30000, 1e9), "RW", "15m"))
    assert longstop.TF == "15m" and len(rows) > 10
    assert rows.loc[~rows.stopped, "re"].isna().all()


def test_verdict_needs_positive_periods_and_t():
    t = pd.date_range("2020-01-01", "2026-06-01", freq="D")
    good = pd.DataFrame({"t": t, "R": np.where(np.arange(len(t)) % 3 == 0, 3.0, -1.0) + 0.05})
    assert R.verdict(good)[0]
    assert not R.verdict(good.assign(R=-good.R))[0]
