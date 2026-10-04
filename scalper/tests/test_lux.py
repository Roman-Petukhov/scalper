import numpy as np
import pandas as pd

from research.lux import lux_signals, positions, slope_series


def _df(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"open": c, "high": c * 1.002, "low": c * 0.998, "close": c}, index=idx)


def test_signals_are_causal():
    df = _df()
    s = slope_series(df, 14, 1.0, "atr")
    up, dn = lux_signals(df.high.to_numpy(), df.low.to_numpy(), df.close.to_numpy(), s, 14)
    assert up.sum() > 20 and dn.sum() > 20
    for k in (800, 2999):
        u2, d2 = lux_signals(df.high.to_numpy()[:k], df.low.to_numpy()[:k], df.close.to_numpy()[:k], s[:k], 14)
        assert np.array_equal(u2, up[:k]) and np.array_equal(d2, dn[:k])


def test_linreg_slope_matches_pine_formula_on_a_line():
    idx = pd.date_range("2024-01-01", periods=100, freq="5min", tz="UTC")
    c = pd.Series(50 + 0.3 * np.arange(100.0), index=idx)
    df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c})
    s = slope_series(df, 14, 1.0, "linreg")
    assert np.isclose(s[-1], 0.3 / 2)                      # |наклон| / 2 * mult


def test_positions_filter_blocks_counter_trend_entries():
    up = np.array([1, 0, 0, 0, 0], bool); dn = np.array([0, 0, 1, 0, 0], bool)
    htf = np.array([1.0, 1.0, 1.0, -1.0, -1.0])
    p = positions(up, dn, htf, True, 0)
    # лонг по сигналу; сигнал вниз против фильтра — выход, а не шорт; разворот старшего ТФ — тоже выход
    assert list(p) == [1.0, 1.0, 0.0, 0.0, 0.0]
