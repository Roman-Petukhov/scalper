import numpy as np

from research.stops_exam import path_trade


def test_market_entry_without_stop_exits_at_last_close():
    o = h = lo = c = np.array([100.0, 99.0, 101.0])
    r, f, st = path_trade(1.0, 1.0, np.array([100.0]), 1, o, h + 0.5, lo - 0.5, c, np.inf, 0.0, 0.0)
    assert np.isclose(r, 0.01) and f == 1.0 and not st


def test_stop_hits_beyond_deepest_ladder_level():
    o = np.array([99.0, 97.5])
    h = np.array([99.5, 97.6])
    lo = np.array([98.4, 96.0])          # бар 0: исполнены 100 и 99; бар 1: исполнен 98, затем стоп 97
    c = np.array([98.8, 96.5])
    levels = np.array([100.0, 99.0, 98.0])
    r, f, st = path_trade(1.0, 1.0, levels, 0, o, h, lo, c, 1.0, 0.0, 0.0)
    assert st and np.isclose(f, 1.0)     # во втором баре исполнился и 98, затем стоп
    assert np.isclose(r, ((97 / 100 - 1) + (97 / 99 - 1) + (97 / 98 - 1)) / 3)
