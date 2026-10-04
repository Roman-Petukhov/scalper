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


def test_retest_fill_requires_trade_through_and_pays_maker():
    from research.breakout.lines import retest_barrier
    # пробой вверх на баре 0 линии 100 (горизонтальной); бар 1 касается 100 — не исполнено; бар 2 проходит сквозь
    o = np.array([101, 101, 100.5, 101, 102.5], float); h = np.array([101, 101.5, 101, 101.5, 103], float)
    l = np.array([100.5, 100.0, 99.9, 100.5, 101.5], float); c = np.array([101, 101, 100.8, 101.2, 102.8], float)
    atr = np.full(5, 1.0)
    r, b = retest_barrier(np.array([0]), np.array([1]), np.array([100.0]), np.array([0.0]), o, h, l, c, atr,
                          3, 2.0, 1.0, 10, 2.0, 7.0, 1e-4)
    assert np.isclose(r[0], 102 / 100 - 1 - 2e-4 - 2e-4) and b[0] == 4     # вход 100 (maker), тейк 102 (maker)


def test_luxalgo_method_is_causal_and_matches_definition():
    from research.breakout.lines import luxalgo_breaks
    h, l, c, atr = _walk(4000, seed=3)
    ev, f, up, lo = luxalgo_breaks(h, l, c, atr, 14, 1.0)
    assert (ev == 1).sum() > 20 and (ev == -1).sum() > 20
    for t in (700, 2500, 3999):
        ev2, f2, up2, lo2 = luxalgo_breaks(h[: t + 1], l[: t + 1], c[: t + 1], atr[: t + 1], 14, 1.0)
        assert np.array_equal(ev2, ev[: t + 1]) and np.allclose(np.nan_to_num(up2), np.nan_to_num(up[: t + 1]))
    t = int(np.flatnonzero(ev == 1)[5])
    assert c[t] > up[t] and c[t - 1] <= up[t - 1] + 1e-9 or np.isnan(up[t - 1])
    # линия опускается ровно на ATR/length*mult за бар между вершинами
    k = int(np.flatnonzero(np.isfinite(up))[100])
    if np.isfinite(up[k + 1]) and abs(up[k + 1] - up[k]) < 1:
        assert np.isclose(up[k] - up[k + 1], atr[0] / 14, rtol=1e-6) or up[k + 1] > up[k]
