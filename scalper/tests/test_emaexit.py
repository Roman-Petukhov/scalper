import numpy as np
import pandas as pd
import pytest

from research import emaexit as X
from research.tline import MAKER, TAKER, two_targets


def _walk(n=120, seed=3):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, n)))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.003, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.003, n)))
    return o, hi, lo, c, np.zeros(n)


def test_unreachable_extension_equals_the_bot_exit():
    o, hi, lo, c, f = _walk()
    ema = pd.Series(c).ewm(span=20, adjust=False).mean().to_numpy()
    atr = np.full(len(c), 1.0)
    entry, stop = c[10], c[10] * 0.98
    for side, ent, stp in ((1, entry, stop), (-1, entry, entry * 1.02)):
        want, _ = two_targets(o, hi, lo, c, f, 10, side, ent, stp, 3.0, 3.0, False, 60, MAKER)
        got = X.ext_exit(o, hi, lo, c, f, 10, side, ent, stp, 60, MAKER, ema, atr, 1e9, 1.0)
        assert got == pytest.approx(want)


def _spike():
    """Вход на баре 10 по 100, подъём по 0.5 до 105 (бары 11–20), затем обвал на 94 — стоп 95 бьёт остаток."""
    n = 30
    c = np.r_[np.full(11, 100.0), 100 + np.arange(1, 11) * 0.5, np.full(n - 21, 94.0)]
    return c.copy(), c + 0.05, c - 0.05, c, np.zeros(n), np.full(n, 100.0), np.full(n, 1.0)


def test_full_extension_exit_closes_at_the_close_of_the_far_bar():
    o, hi, lo, c, f, ema, atr = _spike()
    r = X.ext_exit(o, hi, lo, c, f, 10, 1, 100.0, 95.0, 60, TAKER, ema, atr, 2.0, 1.0)
    far = next(i for i in range(11, len(c)) if c[i] - 100.0 >= 2.0)
    assert far == 14 and r == pytest.approx((c[far] - 100.0 - 2 * TAKER * 100.0) / 5.0)


def test_half_extension_keeps_the_remainder_for_the_stop():
    o, hi, lo, c, f, ema, atr = _spike()
    full = X.ext_exit(o, hi, lo, c, f, 10, 1, 100.0, 95.0, 60, TAKER, ema, atr, 2.0, 1.0)
    half = X.ext_exit(o, hi, lo, c, f, 10, 1, 100.0, 95.0, 60, TAKER, ema, atr, 2.0, 0.5)
    assert half == pytest.approx(0.5 * full + 0.5 * (94.0 - 100.0 - 2 * TAKER * 100.0) / 5.0)
    assert half < 0 < full


def test_paired_verdict_needs_every_period_positive_and_t_three():
    t = pd.date_range("2021-01-01", periods=400, freq="1D")
    z = pd.DataFrame({"t": t, "R": -1.0, "v": -1.0 + 0.5})
    ok, txt = X.paired_verdict(z, "v")
    assert "всё время +0.500R" in txt
    bad = z.assign(v=np.where(np.arange(400) % 2 == 0, -1.5, -0.5))
    assert not X.paired_verdict(bad, "v")[0]
