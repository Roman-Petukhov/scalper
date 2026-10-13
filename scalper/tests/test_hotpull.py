import numpy as np
import pandas as pd
import pytest

from research.hotpull import EXITS, coin_rows, daily_table, label, ride
from research.tline import TAKER


def _frame(n=700, seed=4, drift=0.001):
    idx = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(drift, 0.015, n)))
    o = np.r_[c[0], c[:-1]]
    v = rng.uniform(1e5, 2e5, n)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.004, "low": np.minimum(o, c) * 0.996, "close": c,
                         "volume": v, "taker_buy_volume": v / 2, "funding": np.zeros(n)}, index=idx)


def test_ride_exits_on_stop_target_or_channel_mid():
    c = np.array([100, 101, 103, 99, 94.0])
    o = np.r_[100, c[:-1]]
    h, lo, f = np.maximum(o, c), np.minimum(o, c), np.zeros(5)
    mid = np.full(5, 50.0)
    r, j = ride(o, h, lo, c, f, mid, 0, 100.0, 95.0, None, 360)          # стоп на 95 в последней свече
    assert j == 4 and r == pytest.approx((95 - 100 - 2 * TAKER * 100) / 5)
    mid = np.array([0, 0, 0, 100, 0.0])
    r, j = ride(o, h, lo, c, f, mid, 0, 100.0, 95.0, None, 360)          # закрытие 99 ниже середины 100
    assert j == 3 and r == pytest.approx((99 - 100 - 2 * TAKER * 100) / 5)
    c2 = np.array([100, 108, 117.0])
    o2 = np.r_[100, c2[:-1]]
    r, j = ride(o2, np.maximum(o2, c2), np.minimum(o2, c2), c2, np.zeros(3), np.zeros(3), 0, 100.0, 95.0, 3.0, 60)
    assert j == 2 and r > 2.9


def test_positions_on_a_coin_do_not_overlap():
    rows = pd.DataFrame(coin_rows(_frame(), "X"))
    assert len(rows) and set(rows.exit) <= set(EXITS)
    for _, g in rows.groupby(["entry", "exit"]):
        g = g.sort_values("t")
        ends = g.t + pd.to_timedelta(g.bars * 4, unit="h")
        assert (g.t.iloc[1:].to_numpy() > ends.iloc[:-1].to_numpy()).all()


def test_hot_is_top_decile_smooth_uptrend_known_before_the_signal():
    idx = pd.date_range("2023-01-01", periods=120, freq="1D", tz="UTC")
    days = []
    for i in range(20):
        c = pd.Series(100 * np.exp(np.linspace(0, 0.02 * i, 120)), index=idx)
        d1 = pd.DataFrame({"close": c})
        days.append(daily_table(d1, pd.Series(1e9, index=idx), f"C{i}"))
    days = pd.concat(days, ignore_index=True)
    tr = pd.DataFrame({"symbol": ["C19", "C0"], "day": [idx[100], idx[100]], "R": [1.0, 1.0]})
    out = label(tr, days)
    assert out.hot.tolist() == [True, False] and out.trend.tolist() == [True, False]
