import numpy as np

from research.limit_engine import limit_machine


def _run(o, h, l, c, side, lim, tp, sl, ttl=3, hold=10, maker=2.0, taker=7.0, through=0.0, funding=None):
    n = len(c)
    f = np.zeros(n) if funding is None else np.asarray(funding, float)
    return limit_machine(np.asarray(side, np.int64), np.asarray(lim, float), np.asarray(tp, float),
                         np.asarray(sl, float), ttl, hold, np.asarray(o, float), np.asarray(h, float),
                         np.asarray(l, float), np.asarray(c, float), f, maker, taker, through)


NAN = np.nan


def test_touch_is_not_fill_but_trade_through_is():
    # лимит 99: бар 1 касается 99 (low == 99) — не исполнено; бар 2 проходит сквозь — исполнено
    o = [100, 100, 100, 100]; h = [100, 100, 100, 101]; l = [100, 99, 98.5, 99.5]; c = [100, 100, 99.5, 100]
    ret, pos, trades, _ = _run(o, h, l, c, [1, 0, 0, 0], [99, NAN, NAN, NAN], [NAN] * 4, [NAN] * 4)
    assert trades == 1
    assert pos[1] == 0 and pos[2] == 1
    assert np.isclose(ret[2], 99.5 / 99 - 1 - 2e-4)


def test_order_expires_after_ttl():
    o = h = c = [100.0] * 5; l = [100, 100, 100, 100, 90]
    _, pos, trades, _ = _run(o, h, l, c, [1, 0, 0, 0, 0], [99] + [NAN] * 4, [NAN] * 5, [NAN] * 5, ttl=2)
    assert trades == 0 and pos.sum() == 0


def test_stop_wins_when_both_levels_inside_bar():
    # вход 99 в баре 1; бар 2 задевает и тейк 101, и стоп 97 — считаем стоп
    o = [100, 100, 99.5, 98]; h = [100, 100, 101.5, 98]; l = [100, 98.9, 96.5, 98]; c = [100, 99.5, 98, 98]
    ret, pos, trades, kind = _run(o, h, l, c, [1, 0, 0, 0], [99, NAN, NAN, NAN], [101] * 4, [97] * 4)
    assert kind[2] == 2 and pos[2] == 0
    total = (1 + ret[1]) * (1 + ret[2]) - 1
    assert np.isclose(total, (99.5 / 99 - 1 - 2e-4 + 1) * (97 / 99.5 - 1 - 7e-4 + 1) - 1)


def test_take_profit_is_maker_and_needs_trade_through():
    o = [100, 100, 99.5, 100.5, 101]; h = [100, 100, 100.5, 101, 102]; l = [100, 98.9, 99, 100, 100.5]
    c = [100, 99.5, 100, 100.8, 101.5]
    ret, pos, trades, kind = _run(o, h, l, c, [1, 0, 0, 0, 0], [99] + [NAN] * 4, [101] * 5, [NAN] * 5)
    assert kind[3] == 0 and kind[4] == 1          # high==101 в баре 3 — касание, не исполнение
    assert np.isclose(ret[4], 101 / 100.8 - 1 - 2e-4)


def test_time_exit_is_taker_and_short_side_symmetric():
    o = [100, 100, 101, 101]; h = [100, 101.5, 101, 101]; l = [100, 100, 100, 99]; c = [100, 101, 100, 99]
    ret, pos, trades, kind = _run(o, h, l, c, [-1, 0, 0, 0], [101] + [NAN] * 3, [NAN] * 4, [NAN] * 4, hold=2)
    assert trades == 1 and kind[3] == 3 and pos[3] == 0
    assert np.isclose(ret[1], -(101 / 101 - 1) - 2e-4)
    assert np.isclose(ret[3], -(99 / 100 - 1) - 7e-4)


def test_funding_charged_on_position_held_into_settlement():
    o = h = l = c = [100.0] * 4
    l = [100, 98, 100, 100]
    ret, pos, _, _ = _run(o, h, l, c, [1, 0, 0, 0], [99] + [NAN] * 3, [NAN] * 4, [NAN] * 4, hold=5,
                          funding=[0, 0, 0.001, 0])
    assert pos[1] == 1 and np.isclose(ret[2], -0.001)
