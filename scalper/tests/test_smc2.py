import numpy as np

from research.smc2 import ladder_tp


def test_ladder_tp_on_brownian_bars_is_zero_before_costs():
    rs = []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        nh = 4000
        m = 100 * np.exp(np.cumsum(rng.normal(0, 0.01 / np.sqrt(60), nh * 60))).reshape(nh, 60)
        o = np.r_[100, m[:-1, -1]]
        c = m[:, -1]
        h, lo = np.maximum(m.max(1), o), np.minimum(m.min(1), o)
        for t in range(100, nh - 300, 100):
            lvl = c[t] * 0.99
            stop = lvl * 0.987
            r, *_ = ladder_tp(o, h, lo, c, np.zeros(nh), t, 1, lvl, lvl, lvl, stop, lvl + 3 * (lvl - stop), 72, 120)
            if not np.isnan(r):
                rs.append(r)
    rs = np.array(rs)
    # без преимущества результат — минус издержки (~0.06R) в пределах статистической ошибки
    assert abs(rs.mean() + 0.06) < 4 * rs.std() / np.sqrt(len(rs))
