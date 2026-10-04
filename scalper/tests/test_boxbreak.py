import numpy as np

from research.boxbreak import box_breaks


def test_break_below_twice_tested_box_floor():
    # боковик 100..101, дно 100 тестируется на барах 6 и 14 (окно 4..19), затем закрытие 99.7
    hi = np.full(22, 101.0)
    lo = np.full(22, 100.5)
    lo[6] = lo[14] = 100.0
    c = np.full(22, 100.7)
    c[20] = 99.7
    sigma = np.full(22, 0.05)
    side, h = box_breaks(hi, lo, c, sigma, 16, 1.0, 2)
    assert side[20] == -1 and np.isclose(h[20], 1.0)
    side3, _ = box_breaks(hi, lo, c, sigma, 16, 1.0, 3)
    assert side3[20] == 0                                   # трёх касаний не было


def test_wide_box_is_not_a_compression():
    hi = np.full(22, 105.0)
    lo = np.full(22, 100.0)
    c = np.full(22, 102.0)
    c[20] = 99.0
    side, _ = box_breaks(hi, lo, c, np.full(22, 0.02), 16, 1.0, 1)
    assert side[20] == 0                                    # 5% коробка при 2% дневной волатильности
