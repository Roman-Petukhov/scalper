import numpy as np

from research.trend import trade_machine


def _run(o, h, lo, c, L, mode=0, stop_k=1.0):
    n = len(c)
    z = np.zeros(n, np.bool_)
    return trade_machine(np.array(o, float), np.array(h, float), np.array(lo, float), np.array(c, float),
                         np.ones(n), np.zeros(n), np.array(L, np.bool_), z, stop_k, mode, 60, 0.0)


def test_take_profit_at_five_r():
    c = [100, 101, 103, 106]
    i, r, d = _run(c, [100, 102, 104, 106], [100, 100.5, 102, 104], c, [1, 0, 0, 0])
    assert list(i) == [0] and np.isclose(r[0], 5.0) and d[0] == 3


def test_stop_counts_first_when_both_hit():
    i, r, _ = _run([100, 100], [100, 106], [100, 98], [100, 100], [1, 0])
    assert np.isclose(r[0], -1.0)


def test_trailing_moves_to_breakeven_after_one_r():
    # +1R на баре 1 (стоп -> 100), затем падение к 99: выход в безубыток
    i, r, _ = _run([100, 100.5, 100.2], [100, 101.5, 100.3], [100, 100.4, 99.0], [100, 101, 99.5], [1, 0, 0], mode=1)
    assert np.isclose(r[0], 0.0)
