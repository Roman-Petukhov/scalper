import numpy as np
import pandas as pd

from research import ema200 as E
from research import noline
from research.tline import htf_trend


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


def test_htf_sign_matches_bot_trend_for_ema50_after_warmup():
    d = _rw(1)
    a, b = E.htf_sign(d, "15m", 50), htf_trend(d, "15m")
    m = np.isfinite(a)
    assert m.mean() > 0.9 and (a[m] == b[m]).all()
    assert np.isnan(E.htf_sign(d, "15m", 200, "1D")[: 96 * 150]).all()      # дневной EMA200 — не раньше 200 дней


def test_pierce_measures_closes_beyond_the_line_before_the_break():
    c = np.array([10.0, 9.0, 8.5, 8.0, 8.4, 7.0, 7.5, 6.0, 7.0])
    rec = {"a": 0, "side": 1, "slope": -0.5, "log": False}       # линия 10, 9.5, 9, …; 8.4 над 8.0 и 7.5 над 7.0
    assert E.pierce(c, rec, 8, 1.0) == 0.5
    assert E.pierce(c, {**rec, "slope": -0.1}, 8, 1.0) == 0.0


def test_rows_cover_both_trend_sides_and_restore_the_filter(monkeypatch):
    monkeypatch.setattr(E, "MAX_SLOPE", 1e9)
    before = noline.htf_trend
    rows = pd.DataFrame(E.coin_rows(_rw(2), np.full(30000, 1e9), "RW", "15m"))
    assert noline.htf_trend is before
    assert len(rows) > 20 and (rows.ema50 > 0).any() and (rows.ema50 < 0).any()
    assert (rows.pierce >= 0).all() and rows.R.notna().all()
