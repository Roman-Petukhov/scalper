import numpy as np
import pandas as pd

from research import latebe as L
from research.tline import MAKER, TAKER


def _rw(seed: int, n: int = 15000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="4h", tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, n)))
    vol = np.full(n, 1000.0)
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": vol,
                         "taker_buy_volume": vol * rng.uniform(0.3, 0.8, n)}, index=idx)


def test_base_matches_bot_r3():
    from research import noline
    for seed in range(2):
        d = _rw(seed)
        rows = pd.DataFrame(L.coin_rows(d, np.full(len(d), 1e9), "RW", "4h"))
        noline.TF = "4h"
        ref = noline.coin_trades(d, np.full(len(d), 1e9), "RW", lines_only=True)
        if len(rows):
            assert np.allclose(rows["base"].to_numpy(), ref["R3"].to_numpy())


def test_late_breakeven_saves_near_miss():
    # long from 100, stop 90 (R = 10): up to 126 (+2.6R), then down through the stop
    c = np.array([100, 110, 126, 118, 100, 85.0])
    o = np.r_[100, c[:-1]]
    hi, lo = np.maximum(o, c) + 0.5, np.minimum(o, c) - 0.5
    f = np.zeros(len(c))
    base, mfe = L.walk(o, hi, lo, c, f, 0, 1, 100.0, 90.0, 3.0, None, 0.0, 100, MAKER)
    be, _ = L.walk(o, hi, lo, c, f, 0, 1, 100.0, 90.0, 3.0, 2.5, 0.0, 100, MAKER)
    assert base < -1 and 2.6 <= mfe < 3
    assert abs(be - (-(MAKER + TAKER) * 100 / 10)) < 1e-9          # out at entry, fees only
    lock, _ = L.walk(o, hi, lo, c, f, 0, 1, 100.0, 90.0, 3.0, 2.5, 1.0, 100, MAKER)
    assert abs(lock - (1 - (MAKER + TAKER) * 100 / 10)) < 1e-9


def test_breakeven_not_applied_in_trigger_bar():
    # bar 1 reaches +2.6R and closes back below entry in the same bar: old stop still applies there
    c = np.array([100, 95, 85.0])
    o = np.array([100, 100, 95.0])
    hi = np.array([100, 126, 96.0])
    lo = np.array([100, 94, 84.0])
    be, _ = L.walk(o, hi, lo, c, np.zeros(3), 0, 1, 100.0, 90.0, 3.0, 2.5, 0.0, 100, MAKER)
    assert abs(be - (-0.5 - (MAKER + TAKER) * 100 / 10)) < 1e-9      # stop moved to entry; bar 2 gaps to 95 → out at open
