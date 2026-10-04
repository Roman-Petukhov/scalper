import numpy as np

from research.trend_trail import run_exits


def _run(c, h=None, l=None, mode=0, stop_k=1.0):
    c = np.array(c, float)
    h = c if h is None else np.array(h, float)
    l = c if l is None else np.array(l, float)
    n = len(c)
    side = np.zeros(n, np.int8); side[0] = 1
    o = np.r_[c[0], c[:-1]]                          # open = прошлый close, без гэпов
    return run_exits(side, np.ones(n, np.bool_), o, h, l, c, np.ones(n), mode, 100, stop_k, 0.0, 0.0)


def test_tp5():
    i, r, b = _run([100, 102, 105, 103])
    assert np.isclose(r[0], 5.0)


def test_trail3_locks_profit_after_run():
    # цена до 110 (стоп тянется на 107), откат к 106 -> выход по 107 = +7R
    i, r, b = _run([100, 104, 108, 110, 106], l=[100, 104, 108, 110, 106], mode=1)
    assert np.isclose(r[0], 7.0)


def test_hybrid_half_at_2r_then_breakeven():
    # +2R: половина зафиксирована (+1R), стоп остатка в безубыток; возврат к 100 -> итог +1R
    i, r, b = _run([100, 102, 101, 100], mode=4)
    assert np.isclose(r[0], 1.0)


def test_struct_moves_stop_under_recent_lows():
    # после +1R стоп под минимум 10 баров (99.5), затем падение -> выход по 99.5 = -0.5R
    c = [100, 101, 102, 98]
    l = [100, 99.5, 101, 98]
    i, r, b = _run(c, l=l, mode=3)
    assert np.isclose(r[0], -0.5)
