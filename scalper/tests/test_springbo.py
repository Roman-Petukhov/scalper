import numpy as np
import pandas as pd

from research.springbo import ENTRIES, coin_rows, sweeps
from research.tline import PIV, last_confirmed, pivots


def test_sweep_needs_a_close_back_above_the_swing_low_within_three_bars():
    lo = np.array([12, 11, 10, 9, 8.5, 8, 9, 10, 11, 12, 13, 14, 7.5, 9, 10, 11, 12, 13], dtype=float)
    c = lo + 0.5
    c[12] = 7.6                                                   # минимум 8 (свеча 5) пробит, закрытие ниже
    sw = last_confirmed(pivots(lo, PIV, False), len(lo))
    sp = sweeps(lo, c, sw)
    assert sp == [(13, 7.5)]                                      # возврат на следующей свече, дно 7.5
    c2 = c.copy()
    c2[13:15] = 7.9                                               # три свечи ниже — не вынос, а пробой
    assert sweeps(lo, c2, sw) == []


def test_rows_have_both_entries_and_labels():
    idx = pd.date_range("2023-01-01", periods=900, freq="4h", tz="UTC")
    rng = np.random.default_rng(3)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 900)))
    o = np.r_[c[0], c[:-1]]
    v = rng.uniform(1e5, 2e5, 900)
    d = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.006, "low": np.minimum(o, c) * 0.994, "close": c,
                      "volume": v, "taker_buy_volume": v * rng.uniform(0.4, 0.6, 900), "funding": 0.0}, index=idx)
    rows = pd.DataFrame(coin_rows(d, np.full(900, 1e9), "X"))
    assert len(rows) and set(ENTRIES) <= set(rows.columns)
    assert rows.spring.dtype == bool and rows["рынок"].notna().all()
