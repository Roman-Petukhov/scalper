import numpy as np
import pandas as pd
import pytest

from research import placebo as P
from research.tline import MAKER, TAKER


def test_placebo_bars_stay_in_window_and_respect_mask():
    ok = np.zeros(1000, dtype=bool)
    ok[400:700:2] = True
    bars = P.placebo_bars(ok, 500, 500, np.random.default_rng(0))
    assert len(bars) == 500
    assert bars.min() >= 500 - P.WINDOW and bars.max() <= 500 + P.WINDOW
    assert ok[bars].all()
    assert not len(P.placebo_bars(np.zeros(1000, dtype=bool), 500, 10, np.random.default_rng(0)))


def test_placebo_bars_never_pick_the_last_bar():
    ok = np.ones(100, dtype=bool)
    bars = P.placebo_bars(ok, 99, 200, np.random.default_rng(1))
    assert bars.max() <= 98


def test_trade_r_short_hits_3r_target_with_maker_costs():
    c = np.array([100.0, 100.0, 95.0, 80.0])
    o = c.copy()
    hi = np.array([100.0, 100.5, 96.0, 85.0])
    lo = np.array([100.0, 99.0, 94.0, 70.0])
    f = np.zeros(4)
    r = P.trade_r(o, hi, lo, c, f, 1, -1, 0.02)          # стоп 102, цель 94
    risk = 2.0
    assert r == pytest.approx(((100 - 94) - 2 * MAKER * 100) / risk)


def test_trade_r_short_stopped_out():
    c = np.array([100.0, 100.0, 103.0])
    hi = np.array([100.0, 100.0, 104.0])
    lo = np.array([100.0, 99.0, 100.0])
    o = np.array([100.0, 100.0, 100.0])
    r = P.trade_r(o, hi, lo, c, np.zeros(3), 1, -1, 0.02)
    assert r == pytest.approx((-2.0 - (MAKER + TAKER) * 100) / 2.0)


def _frames():
    bot = pd.DataFrame({"tid": ["a", "b"], "R": [1.0, 0.0], "per": ["is", "is"], "side": [-1, -1]})
    k = np.arange(4)
    plc = pd.DataFrame({"tid": np.repeat(["a", "b"], 8), "kind": np.tile(np.repeat(["any", "trend"], 4), 2),
                        "k": np.tile(k, 4),
                        "R": np.r_[[0, 0, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 1, 1]].astype(float)})
    return bot, plc


def test_draw_means_average_over_trades_per_draw():
    bot, plc = _frames()
    np.testing.assert_allclose(P.draw_means(bot, plc, "any"), [0.0, 0.0, 0.5, 0.5])
    np.testing.assert_allclose(P.draw_means(bot, plc, "trend"), [0.5, 0.5, 1.0, 1.0])


def test_compare_reports_gap_and_share_of_draws_not_worse():
    bot, plc = _frames()
    row = P.compare(bot, plc)
    assert row["бот"] == "+0.500 (2)"
    assert row["any"] == "+0.250" and row["бот−any"] == "+0.250" and row["p any"] == "0.500"
    assert row["trend"] == "+0.750" and row["p trend"] == "1.000"


def test_period_of_assigns_periods():
    t = pd.Series(pd.to_datetime(["2020-06-01", "2023-01-01", "2019-01-01"], utc=True))
    assert list(P.period_of(t)) == ["2020", "is", ""]
