import numpy as np

from research.levels import sr_levels, sr_signals

N = 3


def _series():
    # плавный рост к вершине 110 на баре 10, откат до 100, затем пробой вверх на баре 25
    c = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 108, 106, 104, 102, 100,
                  100, 101, 100, 101, 102, 103, 104, 105, 106, 112, 113, 114, 115, 116], dtype=float)
    return c, c + 0.5, c - 0.5


def test_resistance_is_known_only_after_confirmation_bar():
    c, h, lo = _series()
    res, sup = sr_levels(h, lo, N)
    # вершина на баре 10 подтверждается на баре 13 (10 + N) и в fixnan(...[1]) доступна с бара 14
    assert np.isnan(res[13]) or res[13] != h[10]
    assert res[14] == h[10]


def test_break_B_fires_on_close_above_resistance_with_volume():
    c, h, lo = _series()
    o = np.concatenate([[c[0]], c[:-1]])                     # open = прошлый close: тела вверх без нижней тени
    res, sup = sr_levels(h, lo, N)
    osc = np.full(len(c), 50.0)
    up, dn = sr_signals(o, h, lo, c, osc, res, sup, 20.0, 0)
    assert up[25] and up.sum() == 1
    up_low_vol, _ = sr_signals(o, h, lo, c, np.zeros(len(c)), res, sup, 20.0, 0)
    assert not up_low_vol.any()                               # без всплеска объёма метки B нет


def test_fade_after_failed_breakout_goes_short():
    c, h, lo = _series()
    c = c.copy()
    c[26:] = [109.0, 108.0, 107.0, 106.0]                     # пробой на 25 и возврат под уровень на 26
    h, lo = c + 0.5, c - 0.5
    h[25] = 112.5
    o = np.concatenate([[c[0]], c[:-1]])
    res, sup = sr_levels(h, lo, N)
    up, dn = sr_signals(o, h, lo, c, np.full(len(c), 50.0), res, sup, 20.0, 2)
    assert dn[26] and not up.any()
