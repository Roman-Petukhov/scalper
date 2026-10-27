import math

import numpy as np
import pandas as pd
import pytest

from research import vrgrid as G
from research import vrp as V


def test_bs_put_call_parity():
    s, k, sg, t = 100.0, 95.0, 0.6, 7 / 365
    assert V.bs(s, k, sg, t, True) - V.bs(s, k, sg, t, False) == pytest.approx(s - k, abs=1e-9)


def test_condor_keeps_credit_when_price_stays_and_loss_is_bounded():
    win, ml = V.week_trade(100.0, 100.0, 0.6, 1.0, "кондор")
    assert win > 0 and ml > win
    loss, ml2 = V.week_trade(100.0, 300.0, 0.6, 1.0, "кондор")
    assert loss == pytest.approx(-ml2, rel=1e-9)                     # за крылом убыток = максимальный
    naked, inf = V.week_trade(100.0, 300.0, 0.6, 1.0, "стрэнгл")
    assert math.isinf(inf) and naked < loss


def test_double_costs_lower_the_result():
    assert V.week_trade(100.0, 101.0, 0.6, 1.0, "кондор", 2.0)[0] < V.week_trade(100.0, 101.0, 0.6, 1.0, "кондор")[0]


def test_variance_ratio_below_one_for_mean_reverting_series():
    rng = np.random.default_rng(1)
    n = 24 * 120
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.7 * x[i - 1] + rng.normal(0, 0.01)                  # возврат к среднему
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    mr = G.mean_reversion(pd.Series(100 * np.exp(x), index=idx))
    assert mr.vr.iloc[-1] < 0.5 and mr.ac.iloc[-1] < -0.05
    rw = G.mean_reversion(pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx))
    assert 0.7 < rw.vr.iloc[-1] < 1.3


def test_active_mask_filters():
    mr = pd.DataFrame({"vr": [0.7, 0.85, 1.0], "ac": [-0.1, -0.1, 0.0]})
    assert list(G.active_mask(mr, "VR < 0.9")) == [True, True, False]
    assert list(G.active_mask(mr, "VR < 0.8, AC")) == [True, False, False]


def test_coin_days_runs_on_random_5m_data():
    rng = np.random.default_rng(2)
    n = 288 * 70
    idx = pd.date_range("2023-01-01", periods=n, freq="5min", tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    o = np.r_[c[0], c[:-1]]
    b5 = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.001, "low": np.minimum(o, c) * 0.999, "close": c,
                       "funding": 0.0}, index=idx)
    out, act, pr = G.coin_days(b5, 2000)
    assert out.shape == (len(G.VARIANTS) * 2, 2000) and act.sum() >= 69 and np.isfinite(out).all()
