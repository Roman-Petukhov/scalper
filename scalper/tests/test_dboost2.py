import numpy as np
import pandas as pd

from research.dboost2 import _cap, book, inflow_gate, returns, to_daily
from trader.domain.trend import RULES, adv30, weights


def _data(n=700, coins=25, seed=3):
    idx = pd.date_range("2022-01-01", periods=n, freq="1D")
    rng = np.random.default_rng(seed)
    close = pd.DataFrame({f"C{j}": 10 * np.exp(np.cumsum(rng.normal(0.001, 0.04, n))) for j in range(coins)}, index=idx)
    qv = pd.DataFrame({c: rng.uniform(3e7, 1e9) * np.ones(n) for c in close}, index=idx)
    return close, qv


def test_base_book_is_the_bot_rule():
    close, qv = _data()
    w = book(close, adv30(qv))
    assert np.allclose(w.to_numpy(), weights(close, qv, RULES[0]).to_numpy(), atol=1e-12)


def test_short_mirror_and_cap():
    close, qv = _data()
    adv = adv30(qv)
    both = _cap(book(close, adv) - book(1 / close, adv))
    assert (both < 0).any().any() and both.abs().sum(axis=1).max() <= 2 + 1e-9


def test_funding_cap_drops_expensive_coin():
    close, qv = _data()
    adv = adv30(qv)
    w = book(close, adv)
    hot = w.iloc[-1].idxmax()
    fund = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    fund[hot] = 0.005
    assert book(close, adv, fund, fund_cap=0.001).iloc[-1][hot] == 0


def test_returns_charge_short_funding_in_our_favour():
    idx = pd.date_range("2024-01-01", periods=3, freq="1D")
    close = pd.DataFrame({"A": [10.0, 10.0, 10.0]}, index=idx)
    w = pd.DataFrame({"A": [-1.0, -1.0, -1.0]}, index=idx)
    fund = pd.DataFrame({"A": [0.0, 0.001, 0.001]}, index=idx)
    r = returns(w, close, fund)
    assert r.iloc[2] == 0.001


def test_gate_and_daily_compounding():
    idx = pd.date_range("2024-01-01", periods=40, freq="1D")
    g = inflow_gate(pd.Series(np.arange(40.0), index=idx), idx)
    assert not g.iloc[29] and g.iloc[35]
    r = pd.Series([0.1, 0.1], index=pd.to_datetime(["2024-01-01 04:00", "2024-01-01 08:00"]))
    assert abs(to_daily(r).iloc[0] - 0.21) < 1e-12
