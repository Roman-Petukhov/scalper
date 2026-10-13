import numpy as np
import pytest

from research.liqmap import build_map, first_touch


def _run(prices_lo, oi, buy=0.8, near=0.2):
    n = len(oi)
    lo = np.array(prices_lo, dtype=float)
    c = lo + 0.5
    h = c + 0.5
    return build_map(h, lo, c, np.full(n, buy), np.array(oi, dtype=float), np.full(n, near))


def test_new_longs_sit_below_and_get_liquidated_when_price_falls_through():
    m = _run([100, 100, 100, 85], [1000, 2000, 2000, 2000])
    assert m["below"][1] > m["above"][1] > 0                  # покупатели перевешивали — больше лонгов
    assert m["cleared_long"][3] > 0                           # 10× лонги (~90.5) сняты падением до 85
    assert m["below"][3] < m["below"][2]


def test_falling_oi_shrinks_the_map_proportionally():
    m = _run([100, 100, 100], [1000, 2000, 1000])
    assert m["below"][2] * 1000 == pytest.approx(m["below"][1] * 2000 / 2)


def test_first_touch():
    h = np.array([10, 10.5, 11.5, 10])
    lo = np.array([10, 9.8, 10.2, 8])
    assert first_touch(h, lo, 0, 11.0, 9.0, 10) == 1
    assert first_touch(h, lo, 0, 12.0, 9.5, 10) == -1
    assert first_touch(h, lo, 0, 13.0, 7.0, 10) == 0
