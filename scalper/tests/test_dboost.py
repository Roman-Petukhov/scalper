import numpy as np
import pandas as pd

from research.dboost import atr_models, judge, pyramid, strongest


def _close(cols: dict[str, np.ndarray]) -> pd.DataFrame:
    n = len(next(iter(cols.values())))
    return pd.DataFrame(cols, index=pd.date_range("2022-01-01", periods=n, freq="1D"))


def test_pyramid_adds_on_new_highs_up_to_three():
    c = _close({"A": np.r_[np.full(25, 10.0), np.linspace(11, 30, 30)]})
    sig = pd.DataFrame({"A": np.r_[np.zeros(25), np.ones(30)]}, index=c.index)
    m = pyramid(c, sig)["A"]
    assert m.iloc[25] == 1.0 and m.iloc[26] == 1.5 and m.iloc[-1] == 2.5 and (m.iloc[:25] == 1).all()


def test_strongest_keeps_k_best_and_concentrates():
    n = 120
    c = _close({s: 10 * np.exp(np.cumsum(np.full(n, g))) * (1 + 0.01 * np.sin(np.arange(n)))
                for s, g in (("A", 0.01), ("B", 0.005), ("C", 0.001))})
    adv = pd.DataFrame(1e9, index=c.index, columns=c.columns)
    sig = pd.DataFrame(1.0, index=c.index, columns=c.columns)
    s = strongest(c, adv, sig, 1)
    last = s.iloc[-1]
    assert last["A"] == 20.0 and last["B"] == 0 and last["C"] == 0


def test_atr_exit_leaves_after_drop_from_peak():
    c = _close({"A": np.r_[np.full(30, 10.0) + 0.1 * np.sin(np.arange(30)), np.linspace(11, 20, 20),
                           np.linspace(19, 10, 20)]})
    st = atr_models(c, 20, 3.0)["A"]
    assert st.iloc[31] == 1 and st.iloc[-1] == 0


def test_judge_needs_every_period():
    idx = pd.date_range("2022-01-01", "2026-09-30", freq="1D")
    rng = np.random.default_rng(1)
    base = pd.Series(rng.normal(0.0005, 0.02, len(idx)), index=idx)
    ok, better, ratio = judge(base + 0.002, base)
    assert ok and all(better) and ratio > 0
    ok, better, _ = judge(base - 0.002, base)
    assert not ok and not any(better)
