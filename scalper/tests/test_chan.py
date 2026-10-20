import numpy as np
import pandas as pd

from research import chan as C


def _frame(c: np.ndarray, noise: float = 0.002, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, noise, len(c))))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, noise, len(c))))
    vol = np.full(len(c), 1000.0)
    idx = pd.date_range("2018-01-01", periods=len(c), freq="4h", tz="UTC")
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": vol,
                         "taker_buy_volume": vol * rng.uniform(0.3, 0.8, len(c))}, index=idx)


def test_rising_channel_detected_and_traded():
    j = np.arange(1200, dtype=float)
    c = 100 + 0.05 * j + 6 * np.sin(j / 12)            # rising channel, width ~12
    d = _frame(c)
    atr = C._atr(d).to_numpy()
    chs = C.channels(c, atr)
    assert chs and all(ch["side"] == 1 for ch in chs)
    assert all(abs(ch["slope"] - 0.05) < 0.02 for ch in chs)
    rows = pd.DataFrame(C.coin_rows(d, np.full(len(c), 1e9), "SYN", "4h"))
    assert len(rows) and (rows.side == 1).all() and (rows.k_bound > 1).all()


def test_falling_channel_is_short():
    j = np.arange(1200, dtype=float)
    c = 200 - 0.05 * j + 6 * np.sin(j / 12)
    chs = C.channels(c, C._atr(_frame(c)).to_numpy())
    assert chs and all(ch["side"] == -1 for ch in chs)


def test_random_walk_has_no_edge():
    out = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, 12000)))
        d = _frame(c, 0.004, seed)
        out += C.coin_rows(d, np.full(len(c), 1e9), "RW", "4h")
    x = pd.DataFrame(out)
    assert len(x) > 100
    for col in ("bound", "r3", "c_bound", "c_r3"):
        assert x[col].mean() < 0.1, (col, x[col].mean(), x[col].notna().sum())
