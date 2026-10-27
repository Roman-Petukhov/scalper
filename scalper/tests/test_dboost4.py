import numpy as np
import pandas as pd

from research import dboost4 as D
from research.dboost2 import capped_pick
from trader.domain.trend import _mid_models


def _frame(seed=1, n=600, k=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    c = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.001, 0.03, (n, k)), axis=0)), index=idx,
                     columns=[f"C{i}USDT" for i in range(k)])
    return c


def test_intraday_models_match_close_models_when_high_is_the_close():
    c = _frame()
    st, gain = D.intraday_models(c, c, c.shift(1).fillna(c), 20)
    assert (st == _mid_models(c, 20)).all()
    assert (gain >= -1).all() and (gain[st == 0] == 0).all()


def test_intraday_entry_fills_at_the_level_and_earns_to_the_close():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    c = pd.DataFrame({"A": [10.0, 10, 10, 10, 10.5, 11]}, index=idx)
    h = pd.DataFrame({"A": [10.0, 10, 10, 10, 11.0, 11]}, index=idx)
    o = pd.DataFrame({"A": [10.0, 10, 10, 10, 10.0, 10.5]}, index=idx)
    c.iloc[:4, 0] = [9.0, 9.5, 10.0, 9.9]
    h.iloc[:4, 0] = c.iloc[:4, 0]
    st, gain = D.intraday_models(c, h, o, 3)
    assert st[4, 0] == 1.0 and abs(gain[4, 0] - (10.5 / 10.0 - 1)) < 1e-12   # уровень 10 (макс. 3 закрытий), вход по 10


def test_sector_cap_skips_the_third_meme():
    score = pd.DataFrame([[5.0, 4.0, 3.0, 2.0]], columns=["DOGEUSDT", "PEPEUSDT", "WIFUSDT", "BTCUSDT"])
    pick = capped_pick(score, 3, 2).iloc[0]
    assert list(pick.index[pick]) == ["DOGEUSDT", "PEPEUSDT", "BTCUSDT"]


def test_vol_scaling_keeps_gross_cap_and_capped_limits_weight():
    c = _frame()
    w = pd.DataFrame(0.8, index=c.index, columns=c.columns)
    r = pd.Series(np.random.default_rng(2).normal(0, 0.005, len(c)), index=c.index)
    s = D.vol_scaled(w, r)
    assert (s.sum(axis=1) <= D.GROSS_MAX + 1e-9).all()
    assert (D.capped(w, 1.0).sum(axis=1) <= 1 + 1e-9).all()


def test_spot_returns_skip_funding():
    c = _frame()
    w = pd.DataFrame(0.3, index=c.index, columns=c.columns)
    f = pd.DataFrame(0.001, index=c.index, columns=c.columns)
    assert (D.returns(w, c, f, funding=False) - D.returns(w, c, f)).iloc[2:].gt(0).all()
