import numpy as np
import pytest

from research.pyramid import VARIANTS, Rule, simulate
from research.tline import MAKER, TAKER, two_targets


def _path(closes, start=100.0):
    c = np.array([start, *closes], dtype=float)
    o = np.r_[start, c[:-1]]
    return o, np.maximum(o, c), np.minimum(o, c), c, np.zeros(len(c))


def test_control_matches_bot_simulator():
    rng = np.random.default_rng(0)
    for _ in range(50):
        o, h, lo, c, f = _path(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 80))))
        for side, stop in ((1, 95.0), (-1, 105.0)):
            mine = simulate(o, h, lo, c, f, 0, side, 100.0, stop, VARIANTS["контроль (бот)"])
            ref, _ = two_targets(o, h, lo, c, f, 0, side, 100.0, stop, 3.0, 3.0, False, 60, MAKER)
            assert mine == pytest.approx(ref)


def test_add_on_strength_doubles_the_run_and_costs_on_a_round_trip():
    o, h, lo, c, f = _path([102, 105, 108, 112, 116])            # лонг, риск 5: +1R = 105, цель 3R = 115
    full = simulate(o, h, lo, c, f, 0, 1, 100.0, 95.0, Rule(add_at=1.0, add_size=1.0, be_at=1.0))
    base = simulate(o, h, lo, c, f, 0, 1, 100.0, 95.0, Rule())
    assert base == pytest.approx(3.0 - (MAKER + MAKER) * 100 / 5)
    assert full == pytest.approx(base + (115 - 105 - (TAKER + MAKER) * 105) / 5)
    o, h, lo, c, f = _path([102, 105.5, 103, 99, 97])           # дошли до +1R и вернулись: доливка в минус, первая — 0
    r = simulate(o, h, lo, c, f, 0, 1, 100.0, 95.0, Rule(add_at=1.0, add_size=1.0, be_at=1.0))
    assert r == pytest.approx((0 - (MAKER + TAKER) * 100) / 5 + (100 - 105 - (TAKER + TAKER) * 105) / 5)
    assert simulate(o, h, lo, c, f, 0, 1, 100.0, 95.0, Rule(be_at=1.0)) == pytest.approx(-(MAKER + TAKER) * 100 / 5)


def test_time_stop_exits_a_trade_that_does_not_move():
    o, h, lo, c, f = _path([100.5, 100.2, 100.8, 100.1, 100.4, 100.3, 100.6, 99.0])
    r = simulate(o, h, lo, c, f, 0, 1, 100.0, 95.0, Rule(time_bars=6, time_min_r=0.5))
    assert r == pytest.approx((100.3 - 100 - (MAKER + TAKER) * 100) / 5)
