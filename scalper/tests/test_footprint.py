import numpy as np
import pandas as pd

from research.footprint import diff_t, profile


def _sub(close, vol, buy, hi=None, lo=None):
    c = np.asarray(close, dtype="float64")
    return pd.DataFrame({"open": c, "high": c + 0.1 if hi is None else hi, "low": c - 0.1 if lo is None else lo,
                         "close": c, "volume": np.asarray(vol, dtype="float64"),
                         "taker_buy_volume": np.asarray(buy, dtype="float64")})


def test_profile_long_pressure_at_line_then_break():
    # линия 100: 8 подсвечей под линией с давлением покупателей (70%), потом проход и конец свечи — 40% покупок
    close = [99.5] * 8 + [100.5] * 4 + [101.0] * 4
    vol = [10] * 8 + [5] * 4 + [5] * 4
    buy = [7] * 8 + [3] * 4 + [2] * 4
    p = profile(_sub(close, vol, buy), 1, 100.0)
    assert p["F1"] == "< 50%" and p["F2"] == "да" and p["F3"] == "да" and p["F4"] == "нет"


def test_profile_short_mirror_and_strong_close():
    close = [100.5] * 4 + [99.0] * 12
    vol = [1] * 4 + [2] * 11 + [10]
    buy = [0.5] * 4 + [0.8] * 11 + [2]                  # продавцы — 80% в конце
    p = profile(_sub(close, vol, buy), -1, 100.0)
    assert p["F1"] == "≥ 55%" and p["F2"] == "нет" and p["F3"] == "нет" and p["F4"] == "да"


def test_profile_too_few_subbars_and_diff_t():
    assert "F1" not in profile(_sub([1, 2], [1, 1], [1, 1]), 1, 1.5)
    assert diff_t(np.array([1.0, 1.2, 0.8]), np.array([0.0, 0.2, -0.2])) > 2
