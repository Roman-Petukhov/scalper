import numpy as np
import pandas as pd

from research.dtrend import portfolio
from research.trendfam import mom_signal, signals, sma_signal


def _close(n=500, k=5, seed=2):
    idx = pd.date_range("2022-01-01", periods=n, freq="1D")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.001, 0.03, (n, k)), axis=0)), index=idx,
                        columns=[f"C{i}" for i in range(k)])


def test_signals_are_long_only_shares_and_use_only_the_past():
    c = _close()
    s = signals(c)
    for v in s.values():
        assert v.min().min() >= 0 and v.max().max() <= 1
    c2 = c.copy()
    c2.iloc[300:] *= 3
    for nm, v in signals(c2).items():
        pd.testing.assert_frame_equal(v.iloc[:300], s[nm].iloc[:300], check_names=False)


def test_rising_coin_is_fully_long_and_falling_coin_is_flat():
    idx = pd.date_range("2022-01-01", periods=400, freq="1D")
    c = pd.DataFrame({"UP": np.linspace(10, 50, 400), "DN": np.linspace(50, 10, 400)}, index=idx)
    for f in (sma_signal, mom_signal):
        s = f(c)
        assert s["UP"].iloc[-1] == 1.0 and s["DN"].iloc[-1] == 0.0


def test_portfolio_accepts_a_ready_signal():
    c = _close()
    f = c * 0.0
    adv = c * 0 + 1e9
    a = portfolio(c, f, adv, top=5)
    b = portfolio(c, f, adv, top=5, sig=signals(c)["Дончиан"])
    pd.testing.assert_series_equal(a, b)
