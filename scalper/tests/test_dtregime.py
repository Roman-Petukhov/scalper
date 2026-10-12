import numpy as np
import pandas as pd

from research.dtrend import portfolio
from research.dtregime import gates


def _panel(n=400, coins=25, drift=0.002):
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    cols = ["BTCUSDT"] + [f"C{i}USDT" for i in range(coins - 1)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(drift + rng.normal(0, 0.02, (n, coins)), axis=0)), index=idx, columns=cols)
    adv = pd.DataFrame(1e8, index=idx, columns=cols)
    return close, close * 0.0, adv


def test_gates_turn_on_in_a_broad_uptrend_and_off_in_a_downtrend():
    up = gates(*_panel()[::2])
    assert up["широта > 60%"].iloc[-50:].mean() > 0.9 and up["BTC > SMA200"].iloc[-50:].all()
    down = gates(*_panel(drift=-0.003)[::2])
    assert down["широта > 60%"].iloc[-50:].mean() < 0.1 and not down["оба"].iloc[-50:].any()


def test_gate_off_means_no_positions_and_no_pnl():
    close, fund, adv = _panel()
    off = pd.Series(False, index=close.index)
    assert (portfolio(close, fund, adv, gate=off) == 0).all()
    on = pd.Series(True, index=close.index)
    assert portfolio(close, fund, adv, gate=on).equals(portfolio(close, fund, adv))


def test_skip_selects_a_lower_turnover_tier():
    close, fund, adv = _panel(coins=30)
    adv = adv * np.arange(30, 0, -1)                     # BTC — самый ликвидный, дальше по убыванию
    top = portfolio(close, fund, adv, top=10)
    tier = portfolio(close, fund, adv, skip=10, top=20)
    assert not top.equals(tier)
    assert (portfolio(close, fund, adv, skip=30, top=40) == 0).all()      # мест в группе нет — позиций нет
