import numpy as np
import pandas as pd

from research.moonshots import _fwd_max, exit_returns


def _run(o, h, lo, c, stop=0.1, tp=0.5, hold=72, fund=None):
    n = len(c)
    f = np.zeros(n) if fund is None else np.array(fund, float)
    return exit_returns(np.array(o, float), np.array(h, float), np.array(lo, float), np.array(c, float), f,
                        np.array([0]), stop, tp, hold, 0.0)[0]


def test_take_profit_from_next_open():
    assert np.isclose(_run([1, 100, 120], [1, 110, 160], [1, 99, 115], [1, 105, 150]), 0.5)


def test_stop_wins_when_both_touched():
    assert np.isclose(_run([1, 100], [1, 200], [1, 50], [1, 100], hold=1), -0.1)


def test_gap_through_stop_exits_at_open():
    assert np.isclose(_run([1, 100, 80], [1, 101, 82], [1, 99, 70], [1, 100, 75]), -0.2)


def test_timeout_and_funding_paid_by_long():
    r = _run([1, 100, 100], [1, 101, 101], [1, 99, 99], [1, 100, 104], hold=2, fund=[0, 0.001, 0.002])
    assert np.isclose(r, 0.04 - 0.003)


def test_incomplete_history_is_nan():
    assert np.isnan(_run([1, 100], [1, 101], [1, 99], [1, 100], hold=5))


def test_fwd_max_excludes_current_bar():
    s = pd.Series([5.0, 1, 2, 3, 0])
    assert list(_fwd_max(s, 2)[:3]) == [2, 3, 3] and _fwd_max(s, 2)[3:].isna().all()
