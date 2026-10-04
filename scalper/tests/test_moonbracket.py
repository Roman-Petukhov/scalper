import numpy as np

from research.moonbracket import bracket


def _run(o, h, lo, c, band=0.05, stop=0.10, trail=0.30, window=3, hold=100):
    o, h, lo, c = (np.asarray(x, dtype="float64") for x in (o, h, lo, c))
    return bracket(o, h, lo, c, np.zeros(len(c)), np.array([0], np.int64), band, stop, trail, window, hold, 0.0, 0.0)


def test_long_breakout_rides_trend_and_exits_on_trail():
    # +5% пробой во втором часе, рост до 400, откат на 30% от максимума выбивает трейлингом
    o = [100, 100, 106, 200, 400, 300]
    h = [100, 106, 210, 410, 400, 300]
    lo = [100, 99, 104, 195, 280, 250]
    c = [100, 105, 200, 400, 300, 260]
    r, side, end, mfe = _run(o, h, lo, c)
    assert side[0] == 1 and end[0] == 4
    assert np.isclose(r[0], 410 * 0.7 / 105 - 1)             # выход по трейлингу 287
    assert np.isclose(mfe[0], 410 / 105 - 1)


def test_short_breakout_and_initial_stop():
    o = [100, 100, 94, 100]
    h = [100, 101, 96, 106]
    lo = [100, 94, 93, 99]
    c = [100, 95, 95, 105]
    r, side, end, _ = _run(o, h, lo, c, trail=0.5)
    assert side[0] == -1 and end[0] == 3
    assert np.isclose(r[0], -0.10)                           # стоп 104.5 от входа 95


def test_both_levels_in_one_hour_is_worst_case_and_no_trigger_is_nan():
    r, side, _, _ = _run([100, 100], [100, 106], [100, 94], [100, 100])
    assert np.isclose(r[0], -0.10) and side[0] == 0
    r, side, end, _ = _run([100] * 5, [101] * 5, [99] * 5, [100] * 5)
    assert np.isnan(r[0]) and side[0] == 0 and end[0] == 3
