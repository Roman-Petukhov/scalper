import numpy as np
import pandas as pd
import pytest

from research.classic import divergences, indicators, signals


def _frame(close):
    c = pd.Series(close, index=pd.date_range("2024-01-01", periods=len(close), freq="4h", tz="UTC"))
    return pd.DataFrame({"open": c.shift(1).fillna(c.iloc[0]), "high": c + 0.5, "low": c - 0.5, "close": c})


def test_rsi_is_wilder_and_bounded():
    rng = np.random.default_rng(0)
    x = indicators(_frame(100 + np.cumsum(rng.normal(0, 1, 400))))
    r = x.rsi.dropna()
    assert r.between(0, 100).all()
    up = indicators(_frame(np.arange(100.0, 200.0)))
    assert up.rsi.iloc[-1] == pytest.approx(100.0)                      # одни рост — RSI 100


def test_bearish_divergence_new_high_with_lower_rsi():
    h = np.full(40, 10.0)
    lo = np.full(40, 9.0)
    rsi = np.full(40, 50.0)
    h[20], rsi[20] = 12.0, 80.0                                          # прежний максимум, RSI 80
    h[39], rsi[39] = 12.5, 65.0                                          # новый максимум, RSI ниже
    bull, bear = divergences(h, lo, rsi)
    assert bear[39] and not bull[39] and not bear[38]
    rsi[39] = 85.0
    assert not divergences(h, lo, rsi)[1][39]


def test_signals_are_causal_and_boolean():
    rng = np.random.default_rng(1)
    d = _frame(100 + np.cumsum(rng.normal(0, 1, 300)))
    full = signals(indicators(d), d["close"])
    cut = signals(indicators(d.iloc[:200]), d["close"].iloc[:200])
    for k in full:
        assert full[k][0].dtype == bool
        assert (full[k][0][:200] == cut[k][0]).all() and (full[k][1][:200] == cut[k][1]).all()
