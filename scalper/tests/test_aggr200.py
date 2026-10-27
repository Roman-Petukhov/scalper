import numpy as np
import pandas as pd

from research import aggr200 as A
from research import noline


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


def test_rows_drop_the_aggressor_filter_and_restore_the_trend(monkeypatch):
    monkeypatch.setattr(A, "MAX_SLOPE_15M", 1e9)
    before = noline.htf_trend
    rows = pd.DataFrame(A.coin_rows(_rw(2), np.full(30000, 1e9), "RW", "15m"))
    assert noline.htf_trend is before
    assert len(rows) > 20 and (rows.aggr < 0.55).any() and (rows.aggr >= 0.55).any()
    assert set(rows.e200.unique()) <= {-1.0, 0.0, 1.0} and (rows.e200 > 0).any() and (rows.e200 < 0).any()
    assert ((rows.ema50 > 0).any() and (rows.ema50 < 0).any()) and rows.R.notna().all()


def test_default_trades_keep_the_bot_aggressor_threshold():
    d = _rw(3)
    x = noline.coin_trades(d, np.full(len(d), 1e9), "RW", lines_only=True)
    assert len(x) and (x.aggr >= 0.55).all()


def test_pick_requires_every_condition_on_the_trade_side():
    z = pd.DataFrame({"ema50": [1, 1, -1, 1], "e200": [1, -1, 1, 1], "e200l": [1, 1, 1, -1]})
    assert list(A.pick(z, ("ema50", "e200")).index) == [0, 3]
    assert list(A.pick(z, ("ema50", "e200", "e200l")).index) == [0]


def test_verdict_rejects_a_variant_losing_per_trade():
    t = pd.date_range("2020-01-01", "2026-06-01", freq="D")
    base = pd.DataFrame({"t": t, "R": -0.5})
    v = pd.DataFrame({"t": t, "R": np.where(np.arange(len(t)) % 2, 0.4, -0.5)})
    assert not A.verdict(v, base)[0]
