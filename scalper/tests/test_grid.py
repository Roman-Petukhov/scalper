import numpy as np

from research.grid import grid_machine, rolling_poc


def _run(o, h, lo, c, n_lv=3, anchor=100.0, step=1.0):
    m = len(c)
    return grid_machine(np.asarray(o, float), np.asarray(h, float), np.asarray(lo, float), np.asarray(c, float),
                        np.zeros(m), np.full(m, anchor), np.full(m, step), np.ones(m, np.bool_), n_lv, 0.0, 0.0)


def test_buy_one_step_below_and_take_profit_at_center():
    # бар 1 проходит сквозь 99 (покупка), бар 2 поднимается выше 100 (тейк)
    o, h, lo, c = [100, 100, 99.5], [100.2, 100.0, 100.5], [99.8, 98.8, 99.4], [100, 99.2, 100.3]
    ret, q, fills, tps, stops = _run(o, h, lo, c)
    assert fills == 1 and tps == 1 and stops == 0
    assert np.isclose(ret.sum(), (1 / 3) * (100 / 99 - 1), rtol=1e-3)   # прибыль ступени ~ шаг
    assert q[-1] == 0.0


def test_no_take_profit_in_the_bar_of_the_fill():
    o, h, lo, c = [100, 100], [100.2, 100.5], [99.8, 98.8], [100, 100.4]
    _, q, fills, tps, _ = _run(o, h, lo, c)
    assert fills == 1 and tps == 0 and np.isclose(q[-1], 1 / 3)


def test_stop_beyond_last_level_closes_everything():
    o = [100, 100, 98.5, 96.5]
    h = [100.2, 100.0, 98.6, 96.6]
    lo = [99.8, 98.9, 96.9, 95.0]            # 99, 98, 97 — все три ступени; затем 96 — стоп
    c = [100, 99.0, 97.0, 95.5]
    ret, q, fills, tps, stops = _run(o, h, lo, c)
    assert fills == 3 and stops == 1 and q[-1] == 0.0
    assert ret.sum() < -0.02


def test_poc_is_price_with_most_volume():
    vw = np.array([100.0] * 10 + [105.0] * 2)
    vol = np.ones(12)
    poc = rolling_poc(vw, vol, 12, 10.0, 1)
    assert abs(poc[-1] / 100 - 1) < 0.002
