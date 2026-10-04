import numpy as np

from research.breakout.lines import trend_breaks, triple_barrier


def _walk(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    h = c * (1 + np.abs(rng.normal(0, 0.002, n)))
    l = c * (1 - np.abs(rng.normal(0, 0.002, n)))
    atr = np.full(n, c.mean() * 0.004)
    return h, l, c, atr


def test_events_do_not_depend_on_future_bars():
    h, l, c, atr = _walk()
    ev_full, f_full, r_full, s_full = trend_breaks(h, l, c, atr, 6)
    assert (ev_full != 0).sum() > 50
    for t in (500, 1234, 2999):
        ev_cut, f_cut, r_cut, s_cut = trend_breaks(h[: t + 1], l[: t + 1], c[: t + 1], atr[: t + 1], 6)
        assert np.array_equal(ev_cut, ev_full[: t + 1])
        assert np.allclose(np.nan_to_num(f_cut), np.nan_to_num(f_full[: t + 1]))
        assert np.allclose(np.nan_to_num(r_cut), np.nan_to_num(r_full[: t + 1]))


def test_descending_resistance_break_is_long_event():
    # две убывающие вершины, затем рост сквозь линию
    c = np.array([10, 11, 12, 11, 10, 9, 10, 11, 10, 9, 8, 9, 10, 11, 12, 13], float)
    h, l, atr = c + 0.1, c - 0.1, np.full(len(c), 1.0)
    ev, f, rl, _ = trend_breaks(h, l, c, atr, 2)
    t = int(np.flatnonzero(ev == 1)[0])
    assert c[t] > rl[t] and f[t, 0] < 0           # наклон линии отрицательный


def test_triple_barrier_stop_first_and_gap():
    o = np.array([100, 100, 95, 100], float); h = np.array([100, 103, 96, 100], float)
    l = np.array([100, 98, 94, 100], float); c = np.array([100, 100, 95, 100], float)
    atr = np.full(4, 1.0)
    r, b = triple_barrier(np.array([0]), np.array([1]), o, h, l, c, atr, 2.0, 1.0, 10, 0.0)
    assert np.isclose(r[0], 99 / 100 - 1) and b[0] == 1         # бар 1 задевает и тейк 102, и стоп 99 -> стоп
    o2 = np.array([100, 97, 97], float); h2 = np.array([100, 97.5, 98], float); l2 = np.array([100, 96, 96], float)
    r2, _ = triple_barrier(np.array([0]), np.array([1]), o2, h2, l2, o2, np.full(3, 1.0), 2.0, 1.0, 10, 0.0)
    assert np.isclose(r2[0], 97 / 100 - 1)                        # открытие ниже стопа -> по open
