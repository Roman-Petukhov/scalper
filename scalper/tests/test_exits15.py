import numpy as np
import pandas as pd

from research.dconf import simulate
from research.exits15 import EXITS, month_r


def _bars(path):
    c = np.asarray(path, dtype="float64")
    o = np.r_[c[0], c[:-1]]
    return o, np.maximum(o, c), np.minimum(o, c), c, np.zeros(len(c))


def test_far_target_and_trail_capture_long_short_move():
    o, hi, lo, c, f = _bars([100, 99, 97, 95, 93, 91, 89, 87, 85, 86, 88, 90])     # шорт от 100, стоп 102 (риск 2)
    r3 = simulate(o, hi, lo, c, f, 0, -1, 100.0, 102.0, 2.0, EXITS["R3"], 0.0)
    r6 = simulate(o, hi, lo, c, f, 0, -1, 100.0, 102.0, 2.0, EXITS["R6"], 0.0)
    tr = simulate(o, hi, lo, c, f, 0, -1, 100.0, 102.0, 2.0, EXITS["trail2"], 0.0)
    assert 2.9 < r3 < 3.0 and 5.9 < r6 < 6.0 and tr > r3                          # трейл держит до отката 85 → 89


def test_breakeven_then_trail():
    o, hi, lo, c, f = _bars([100, 98, 99, 101, 103])
    r = simulate(o, hi, lo, c, f, 0, -1, 100.0, 102.0, 2.0, EXITS["be1_trail3"], 0.0)
    assert -0.03 < r < 0.0                                     # на +1R стоп в безубыток, минус только комиссия выхода


def test_month_r_counts_unfilled_as_zero_and_double_cost():
    t = pd.to_datetime(["2023-01-05", "2023-01-06", "2023-01-07"], utc=True)
    df = pd.DataFrame({"t": t, "x": [1.0, np.inf, -1.0], "c": [0.1, 0.1, 0.1]})
    m, m2 = month_r(df, "x", None), month_r(df, "x", "c")
    months = (pd.Timestamp("2024-07-01") - pd.Timestamp("2022-01-01")).days / 30.44
    assert abs(m["is"]) < 1e-12 and abs(m2["is"] - (-0.2 / months)) < 1e-12
