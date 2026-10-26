import numpy as np
import pandas as pd

from research import formation as F
from research import sharp as S
from tests.test_formation import _rw


def test_sharp_features_count_spikes_and_late_ones():
    n = 12
    d = pd.DataFrame({"high": np.full(n, 101.0), "low": np.full(n, 100.0)})
    d.loc[2, "high"], d.loc[10, "high"] = 103.5, 102.5                      # размах 3.5 и 2.5 ATR
    f = S.sharp_features(d, np.ones(n), {"a": 0}, 12)
    assert f == {"max_bar": 3.5, "spikes": 2, "late": 1}
    d.loc[10, "high"] = 101.0
    assert S.sharp_features(d, np.ones(n), {"a": 0}, 12)["late"] == 0       # вынос только в начале


def test_rows_on_a_random_walk(monkeypatch):
    monkeypatch.setattr(F, "MAX_SLOPE", 1e9)
    rows = pd.DataFrame(F.coin_rows(_rw(1), np.full(30000, 1e9), "RW", "15m", S.sharp_features))
    assert len(rows) > 20 and (rows.max_bar > 0).all() and set(rows.late) <= {0, 1}
    assert (rows.spikes >= rows.late).all() and rows["squeeze"].notna().all()
