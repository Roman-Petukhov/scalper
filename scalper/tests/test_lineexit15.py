import numpy as np

from research.lineexit15 import line_exit


def _bars(closes):
    c = np.asarray(closes, dtype="float64")
    return c.copy(), c + 0.1, c - 0.1, c, np.zeros(len(c))


def test_short_exits_on_close_back_above_line_within_window():
    o, h, lo, c, f = _bars([100.0, 99.0, 99.2, 99.8, 99.0])       # линия 99.5, вход 99.0 на баре 1, стоп 101
    r, j, by_line = line_exit(o, h, lo, c, f, 1, -1, 99.0, 101.0, 3.0, 200, 0.0, 99.5, 1, 0.0, 0.0, 12)
    assert (j, by_line) == (3, 1) and r < 0                          # закрылась на 99.8 > 99.5
    r2, j2, by2 = line_exit(o, h, lo, c, f, 1, -1, 99.0, 101.0, 3.0, 200, 0.0, 99.5, 1, 0.0, 0.5, 12)
    assert by2 == 0                                                  # 99.8 не дальше 0.5 от линии — держим
    r3, j3, by3 = line_exit(o, h, lo, c, f, 1, -1, 99.0, 101.0, 3.0, 200, 0.0, 99.5, 1, 0.0, 0.0, 1)
    assert by3 == 0                                                  # окно 1 свеча: на баре 3 уже не выходим
