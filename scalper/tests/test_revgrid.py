import numpy as np

from research.revgrid import simulate
from research.tline import TAKER


def _run(c, m=1.0, x=1, n=4, trend=None, use_trend=False, liq=1e9):
    c = np.asarray(c, dtype="float64")
    k = len(c)
    o = np.r_[c[0], c[:-1]]
    hi, lo = np.maximum(o, c), np.minimum(o, c)
    atr = np.ones(k)
    tr = np.ones(k) if trend is None else np.asarray(trend, dtype="float64")
    return simulate(o, hi, lo, c, atr, np.zeros(k), tr, np.full(k, liq), m, x, n, use_trend)


def test_steady_rise_pyramids_to_four_units_and_exits_on_pullback():
    path = [100, 100, 101, 102, 103, 104, 105, 106, 105.5, 104.5]
    out = _run(path)
    assert len(out) == 1 and out[0]["side"] == 1 and out[0]["units"] == 4
    # входы 101, 102, 103, 104; лучшая цена 106, выход на шаг ниже — 105
    gross = sum(105 - e for e in (101, 102, 103, 104))
    fees = TAKER * (101 + 102 + 103 + 104 + 4 * 105)
    assert abs(out[0]["r"] - (gross - fees)) < 1e-9 and out[0]["r2"] < out[0]["r"]


def test_chop_whipsaws_both_ways():
    path = [100, 100] + [101.2, 99.8] * 6
    out = _run(path)
    assert len(out) >= 4 and all(r["r"] < 0 for r in out)
    assert _run([100, 100, 98.8, 100.2, 98.8, 100.2])[0]["side"] == -1


def test_trend_filter_blocks_counter_trend_entries_and_low_liquidity():
    path = [100, 100, 99, 98, 97, 96, 97.5]
    assert _run(path)[0]["side"] == -1
    assert _run(path, trend=np.ones(len(path)), use_trend=True) == []
    assert _run(path, liq=1e6) == []


def test_gap_fills_at_open_and_single_unit_variant():
    c = np.array([100, 100, 100, 105, 105, 103.5])
    o = np.array([100, 100, 100, 104, 105, 105])
    hi, lo = np.maximum(o, c), np.minimum(o, c)
    k = len(c)
    out = simulate(o, hi, lo, c, np.ones(k), np.zeros(k), np.ones(k), np.full(k, 1e9), 1.0, 1, 1, False)
    assert out[0]["units"] == 1                                             # вход по открытию 104, не по уровню 101
    assert abs(out[0]["r"] - (0 - TAKER * (104 + 104))) < 1e-9             # лучшая цена 105, выход на шаг ниже — 104


def test_single_unit_trails_and_can_win():
    out = _run([100, 100, 101, 103, 105, 107, 105.5], n=1)
    assert out[0]["units"] == 1 and out[0]["r"] > 4.5                         # вход 101, выход 106


def test_fine_bars_use_last_closed_4h_values():
    import pandas as pd

    from research.revgrid import _fine
    i4 = pd.date_range("2024-01-01", periods=10, freq="4h", tz="UTC")
    i1 = pd.date_range("2024-01-01", periods=40, freq="1h", tz="UTC")
    a, t = _fine(pd.DataFrame(index=i4), pd.DataFrame(index=i1), np.arange(10.0), -np.arange(10.0))
    assert np.isnan(a[:4]).all() and (a[4:8] == 0).all() and a[8] == 1 and t[8] == -1
