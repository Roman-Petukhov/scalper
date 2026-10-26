import numpy as np
import pandas as pd
import pytest

from research import pullentry as P


def _rw(seed: int, n: int = 30000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="15min", tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n) * np.where(rng.random(n) < 0.05, 4, 1)))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.002, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.002, n)))
    v = np.full(n, 1000.0)
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": v,
                         "taker_buy_volume": v * rng.uniform(0.3, 0.8, n), "funding": 0.0}, index=idx)


def test_limit_r_fills_on_pullback_and_scores_zero_when_missed():
    o = np.array([100, 110, 108, 104, 115, 125, 130.0]); c = o.copy()
    hi = o + 1; lo = o - 1; f = np.zeros(7)
    r = P.limit_r(o, hi, lo, c, f, 1, 1, 105.0, 101.0, 2.0, "15m")    # откат к 105 на свече 3, цель 117 — достигнута
    assert r == pytest.approx((12 - 2 * P.MAKER * 105) / 4)                 # комиссии two_targets — от цены входа
    assert P.limit_r(o, hi, lo, c, f, 1, 1, 90.0, 85.0, 2.0, "15m") == 0.0          # до 90 не откатилась


def test_rows_only_for_long_candles_with_all_ways(monkeypatch):
    monkeypatch.setattr(P, "MAX_SLOPE", 1e9)
    monkeypatch.setattr(P, "LONG", 0.5)
    rows = pd.DataFrame(P.coin_rows(_rw(1), np.full(30000, 1e9), "RW", "15m"))
    assert len(rows) > 5 and (rows.range_atr >= 0.5).all() and (rows.side < 0).all()
    assert set(P.WAYS) <= set(rows.columns) and rows[list(P.WAYS)].notna().all().all()


def test_diff_t_sees_a_consistent_gain():
    rng = np.random.default_rng(0)
    t = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 1000, 3000), unit="D")
    z = pd.DataFrame({"t": t, "a": rng.normal(0.2, 1, 3000)})
    z["b"] = z.a - 0.2 + rng.normal(0, 0.3, 3000)
    assert P.diff_t(z, "a", "b") > 5
