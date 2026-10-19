import numpy as np
import pandas as pd

from research import shape as S


def test_features_tight_vs_wide():
    n = 200
    hi, lo = np.full(n, 101.0), np.full(n, 99.0)          # usual candle range 2
    c = np.full(n, 100.0)
    a, t = 120, 150
    tight = S.features(hi.copy(), lo.copy(), c, a, t)
    assert tight["span"] == 29 and abs(tight["bars"] - 1) < 1e-9 and tight["scatter"] < 1e-9
    hi2, lo2 = hi.copy(), lo.copy()
    hi2[a + 1:a + 11], lo2[a + 1:a + 11] = 104, 96            # wide early candles, normal late
    w = S.features(hi2, lo2, c, a, t)
    assert w["width"] > tight["width"] and w["bars"] > 1 and w["narrowing"] < 0.8


def test_features_need_history_and_span():
    hi, lo, c = np.full(300, 1.0), np.zeros(300), np.full(300, 0.5)
    assert "width" not in S.features(hi, lo, c, 50, 80)        # less than BASE bars before anchor
    assert "width" not in S.features(hi, lo, c, 150, 155)      # too short


def test_label_thirds_by_is():
    t = pd.to_datetime(["2023-01-01"] * 6 + ["2025-01-01"] * 3)
    z = pd.DataFrame({"t": t, "width": [1, 2, 3, 4, 5, 6, 1, 9, None], "scatter": 1.0, "bars": 1.0,
                      "narrowing": [0.5] * 9, "span": range(9), "R": 0.0, "side": -1})
    out = S.label(z)
    assert list(out.S1[:8]) == ["да", "да", "нет", "нет", "нет", "нет", "да", "нет"] and pd.isna(out.S1.iloc[8])
    assert (out.S4 == "да").all()
