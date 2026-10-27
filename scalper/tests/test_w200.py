import numpy as np
import pandas as pd
import pytest

from research import w200 as W


def test_weekly_ratio_waits_for_200_weeks():
    d = pd.Series(np.linspace(1, 100, 7 * 260), index=pd.date_range("2015-01-01", periods=7 * 260, freq="D"))
    wk = W.weekly_ratio(d)
    assert wk["ema"].iloc[:W.SPAN - 1].isna().all() and wk["ema"].iloc[W.SPAN - 1:].notna().all()
    assert (wk["ratio"].dropna() > 1).all()                       # растущая цена выше средней


def test_episodes_and_further_drop():
    idx = pd.date_range("2020-01-05", periods=10, freq="W-SUN")
    wk = pd.DataFrame({"close": [10, 10, 8, 7, 9, 12, 12, 8, 12, 12.0], "ema": [9.0] * 10}, index=idx)
    wk["ratio"] = wk.close / wk.ema
    daily = wk["close"].resample("D").ffill()
    ep = W.episodes(wk, daily)
    assert len(ep) == 2
    assert ep.iloc[0]["недель"] == 2 and ep.iloc[0]["ещё вниз"] == pytest.approx(7 / 8 - 1)
    assert ep.iloc[1]["недель"] == 1


def test_monthly_state_uses_last_closed_week():
    idx = pd.date_range("2021-01-01", "2021-03-31", freq="D")
    daily = pd.Series(1.0, index=idx)
    wk = pd.DataFrame({"ratio": [0.5, 2.0]}, index=pd.to_datetime(["2021-01-31", "2021-02-28"]))
    st = W.monthly_state(daily, wk)
    assert list(st["ratio"]) == [0.5, 2.0, 2.0]


def test_nw_slope_t_matches_difference_of_means():
    rng = np.random.default_rng(0)
    x = np.r_[np.ones(50), np.zeros(150)]
    y = 0.5 * x + rng.normal(0, 0.1, 200)
    assert W.nw_slope_t(x, y, 0) > 20
    st = pd.DataFrame({"ratio": np.r_[np.full(50, 0.8), np.full(150, 1.2)]})
    d, t, n = W.diff_t(st, pd.Series(y), 0)
    assert d == pytest.approx(y[:50].mean() - y[50:].mean()) and n == 50


def test_alt_forward_haircuts_dead_coins():
    m = pd.date_range("2020-01-31", periods=14, freq="ME")
    cols = [f"C{i}USDT" for i in range(20)] + ["BTCUSDT"]
    c = pd.DataFrame(1.0, index=m, columns=cols)
    c.loc[m[12]:, "C0USDT"] = np.nan                                # умерла
    c.loc[m[12], "C1USDT"] = 2.0
    adv = pd.DataFrame(2e6, index=m, columns=cols)
    meta = pd.DataFrame({"first": pd.Timestamp("2019-01-01"), "last": pd.Timestamp("2030-01-01")}, index=cols)
    meta.loc["C0USDT", "last"] = m[11]
    r = W.alt_forward(c, adv, meta, 12)
    assert r.loc[m[0]] == pytest.approx(((1 - W.DEAD_HAIRCUT) - 1 + 1.0) / 20)
