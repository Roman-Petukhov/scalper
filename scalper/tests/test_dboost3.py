import numpy as np
import pandas as pd

from research.dboost2 import ensemble
from research.dboost3 import FAST, split_signal


def test_split_signal_uses_full_ensemble_for_top20_only():
    n = 500
    idx = pd.date_range("2022-01-01", periods=n, freq="1D")
    rng = np.random.default_rng(4)
    close = pd.DataFrame({f"C{j}": 10 * np.exp(np.cumsum(rng.normal(0.002, 0.04, n))) for j in range(30)}, index=idx)
    adv = pd.DataFrame({c: (30 - j) * 1e8 for j, c in enumerate(close)}, index=idx)     # C0 — самый ликвидный
    sig = split_signal(close, adv)
    full, fast = ensemble(close), ensemble(close, FAST)
    assert sig["C0"].equals(full["C0"]) and sig["C25"].equals(fast["C25"])
    assert not full["C25"].equals(fast["C25"])
