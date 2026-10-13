import numpy as np
import pandas as pd
import pytest

from research import pyramid
from research.dconf import EXITS, daily_breaks, simulate


def _path(closes, start=100.0):
    c = np.array([start, *closes], dtype=float)
    o = np.r_[start, c[:-1]]
    return o, np.maximum(o, c), np.minimum(o, c), c, np.zeros(len(c))


def test_bot_exit_matches_the_pyramid_rule_in_the_bot():
    rng = np.random.default_rng(1)
    for _ in range(50):
        o, h, lo, c, f = _path(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 80))))
        for side, stop in ((1, 95.0), (-1, 105.0)):
            a = simulate(o, h, lo, c, f, 0, side, 100.0, stop, 1.0, EXITS["бот (3R, долив на +1.5R)"])
            b = pyramid.simulate(o, h, lo, c, f, 0, side, 100.0, stop, pyramid.VARIANTS["P2 долив 0.5 на +1.5R"])
            assert a == pytest.approx(b)


def test_ladder_rides_a_long_trend_and_adds_on_the_way():
    up = list(np.linspace(101, 160, 60)) + [150.0]         # лонг, R = 5: ступени 107.5 / 115 / 122.5
    o, h, lo, c, f = _path(up)
    lad = simulate(o, h, lo, c, f, 0, 1, 100.0, 95.0, 2.0, EXITS["лестница"])
    flat = simulate(o, h, lo, c, f, 0, 1, 100.0, 95.0, 2.0, EXITS["лестница без доливок"])
    bot = simulate(o, h, lo, c, f, 0, 1, 100.0, 95.0, 2.0, EXITS["бот (3R, долив на +1.5R)"])
    assert bot < 4 < flat < lad                            # бот взял 3R (+долив), лестница — тренд до отката
    o, h, lo, c, f = _path([104, 108, 112, 107, 101, 99])  # дошли до +2.4R и вернулись: стоп в точке входа
    assert simulate(o, h, lo, c, f, 0, 1, 100.0, 95.0, 2.0, EXITS["лестница без доливок"]) == pytest.approx(
        -(2e-4 + 5.5e-4) * 100 / 5)


def test_daily_breaks_are_known_after_the_day_closes():
    idx = pd.date_range("2022-01-01", periods=400, freq="1D", tz="UTC")
    rng = np.random.default_rng(3)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, len(idx))))
    o = np.r_[c[0], c[:-1]]
    d1 = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.01, "low": np.minimum(o, c) * 0.99, "close": c},
                      index=idx)
    br = daily_breaks(d1)
    assert br and all(k.hour == 0 and k > idx[0] and s in (-1, 1) for k, s in br)
