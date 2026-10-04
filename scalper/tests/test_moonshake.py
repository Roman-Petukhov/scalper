import numpy as np

from research.moonshake import shakeout


def _run(o, h, lo, c, dip=0.05, rh=6, trail=0.3):
    o, h, lo, c = (np.asarray(x, dtype="float64") for x in (o, h, lo, c))
    return shakeout(o, h, lo, c, np.zeros(len(c)), np.array([0], np.int64), dip, rh, trail, 24, 100, 0.0, 0.0, 0.0)


def test_dip_reclaim_then_pump_exits_on_trail():
    # пролив до 92, возврат к 100 на закрытии, рост до 300, откат на 30% от максимума
    o = [100, 100, 95, 100, 150, 300, 220]
    h = [100, 100, 101, 150, 300, 300, 220]
    lo = [100, 92, 93, 99, 140, 200, 200]
    c = [100, 94, 100, 148, 290, 210, 210]
    r, end, risk, mfe = _run(o, h, lo, c)
    assert end[0] == 5
    assert np.isclose(risk[0], 1 - 92 / 100)
    assert np.isclose(r[0], 300 * 0.7 / 100 - 1)


def test_up_move_first_is_not_a_shakeout_and_no_reclaim_is_nan():
    r, *_ = _run([100, 100, 100], [100, 106, 100], [100, 99, 90], [100, 105, 95])
    assert np.isnan(r[0])
    r, *_ = _run([100] * 4, [100, 96, 96, 96], [100, 94, 90, 90], [100, 95, 92, 93])
    assert np.isnan(r[0])
