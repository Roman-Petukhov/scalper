import numpy as np
import pandas as pd

from research.tline import signals, two_targets


def test_two_targets_ladder_and_breakeven():
    # лонг 100, стоп 98 (R=2): первая цель 106, вторая 110
    o = np.array([100, 101, 105, 104, 101.0])
    h = np.array([100, 103, 107, 105, 102.0])
    lo = np.array([100, 100, 104, 103, 99.5])
    c = np.array([100, 102, 105, 104, 100.0])
    r, ex = two_targets(o, h, lo, c, np.zeros(5), 0, 1, 100.0, 98.0, 3.0, 5.0, True, 100, 0.0)
    # половина на 106 (+1.5R), остаток выбит в безубыток на 100
    assert ex == 4 and np.isclose(r, 0.5 * (6 - 2e-4 * 100) / 2 + 0.5 * (0 - 5.5e-4 * 100) / 2)


def test_signal_on_descending_line_of_closes():
    c = np.array([8, 8, 9, 9, 10, 11, 12, 15, 12, 11, 10, 9, 10, 11, 13, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15, 16.0])
    d = pd.DataFrame({"close": c})
    sig = [s for s in signals(d) if s[2] == 1]
    assert sig, "нет пробоя нисходящей линии"
    t, tc, side, line_b, line_c, i1, i2 = sig[0]
    assert (i1, i2) == (7, 14)
    # линия через закрытия 15 (бар 7) и 13 (бар 14): наклон −2/7; пробой на баре 22, закрепление на 23
    assert (t, tc) == (22, 23)
    assert np.isclose(line_b, 13 - 2 / 7 * 8) and c[t] > line_b and c[t - 1] <= 13 - 2 / 7 * 7
