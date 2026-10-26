import numpy as np
import pandas as pd
import pytest

from research import crisk as C


def _z(rs, gap_h=8, hold_h=4, be=None):
    t0 = pd.Timestamp("2022-01-03")
    fill = [t0 + pd.Timedelta(hours=gap_h * i) for i in range(len(rs))]
    ex = [f + pd.Timedelta(hours=hold_h) for f in fill]
    return pd.DataFrame({"R": rs, "R_be": rs if be is None else be, "fill": fill, "exit": ex, "exit_be": ex,
                         "be_at": [pd.NaT] * len(rs), "side": -1})


def test_flat_risk_compounds_closed_trades():
    s = C.simulate(_z([1.0, -1.0, 3.0]), C.Policy("f", flat=0.02), 12)
    assert s["eq"] == pytest.approx(1.02 * 0.98 * 1.06)
    assert s["mdd"] == pytest.approx(0.02) and s["n"] == 3


def test_ladder_steps_up_on_takes_and_resets_on_a_loss():
    s = C.simulate(_z([3.0, 3.0, 3.0, -1.0, 3.0]), C.Policy("l", ladder=True), 12)
    eq = 1.0
    for r, risk in zip([3.0, 3.0, 3.0, -1.0, 3.0], [0.02, 0.03, 0.045, 0.045, 0.02]):
        eq *= 1 + risk * r
    assert s["eq"] == pytest.approx(eq)


def test_position_limit_skips_overlapping_signals():
    z = _z([1.0] * 4, gap_h=1, hold_h=10)
    assert C.simulate(z, C.Policy("f", flat=0.01), 2)["n"] == 2
    assert C.simulate(z, C.Policy("f", flat=0.01, limit=False), 2)["n"] == 4


def test_entry_and_exit_in_the_same_bar_closes_the_trade():
    z = _z([-1.0, 1.0], hold_h=0)
    s = C.simulate(z, C.Policy("f", flat=0.01), 1)
    assert s["n"] == 2 and s["eq"] == pytest.approx(0.99 * 1.01)


def test_heat_budget_shrinks_risk_and_be_frees_it():
    z = _z([1.0] * 3, gap_h=1, hold_h=10)
    s = C.simulate(z, C.Policy("h", flat=0.05, heat=True, limit=False), 12)
    assert s["avg_risk"] == pytest.approx(np.mean([0.05, 0.05, 0.015]))
    z.loc[0, "be_at"] = z.fill[0] + pd.Timedelta(minutes=30)
    s = C.simulate(z, C.Policy("h", flat=0.05, heat=True, be=True, limit=False), 12)
    assert s["avg_risk"] == pytest.approx(np.mean([0.05, 0.05, 0.05]))      # риск первой обнулён безубытком


def test_daily_stop_halts_entries_until_next_day():
    z = _z([-1.0] * 6, gap_h=1, hold_h=0)
    s = C.simulate(z, C.Policy("b", flat=0.08, breakers=True, limit=False), 12)
    assert s["n"] == 2                                     # после −15.4% за день новых входов нет


def test_be_path_moves_stop_to_entry_after_the_trigger():
    o = np.array([100.0] * 6)
    hi = np.array([100, 101, 101, 113, 101, 100.0])
    lo = np.array([100, 99, 99, 101, 99.5, 89.0])
    j, rb, k = C.be_path(o, hi, lo, 1, 100.0, 90.0, 0, 5, -1.0, C.TAKER)
    assert j == 3 and k == 4 and rb == pytest.approx(-(2 * C.TAKER) * 100 / 10)
    assert C.be_path(o, hi, lo, 1, 100.0, 80.0, 0, 5, -1.0, C.TAKER)[0] == -1     # +1.2R не достигнуто


def test_autocorr_sees_clustered_outcomes():
    r = np.array([2.0] * 50 + [-1.0] * 50 + [2.0] * 50 + [-1.0] * 50)
    pw, pl, rho, p = C.autocorr(_z(list(r)))
    assert pw > 0.9 and pl < 0.1 and rho > 0.9 and p < 0.01


def test_custom_ladder_steps_are_used():
    s = C.simulate(_z([3.0, 3.0, -1.0]), C.Policy("l", ladder=True, steps=(0.0025, 0.00375, 0.005625)), 3)
    assert s["eq"] == pytest.approx((1 + 0.0075) * (1 + 0.01125) * (1 - 0.005625))


def test_risk_multiplier_scales_each_trade():
    s = C.simulate(_z([1.0, -1.0, 3.0]), C.Policy("f", flat=0.01), 12, mult=np.array([2.0, 1.0, 1.5]))
    assert s["eq"] == pytest.approx(1.02 * 0.99 * 1.045)
    assert s["avg_risk"] == pytest.approx((0.02 + 0.01 + 0.015) / 3)
