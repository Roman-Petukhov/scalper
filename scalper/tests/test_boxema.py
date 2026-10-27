import numpy as np
import pandas as pd
import pytest

from research import boxema as B
from research.tline import MAKER, TAKER


def _path(c):
    c = np.asarray(c, dtype="float64")
    o = np.r_[c[0], c[:-1]]
    return o, np.maximum(o, c), np.minimum(o, c), c


def test_climax_exit_closes_when_price_runs_k_atr_from_ema20():
    o, hi, lo, c = _path([100, 101, 103, 106, 108])
    n = len(c)
    ema = np.full(n, 100.0)
    r, cost = B.exit_r(o, hi, lo, c, np.zeros(n), 0, 1, 100.0, 98.0, 0, 3.0, 2.0, ema, np.ones(n), 100, TAKER)
    assert r == pytest.approx((3.0 - 2 * TAKER * 100) / 2.0) and cost == pytest.approx(2 * TAKER * 50)


def test_box_height_target_uses_maker():
    o, hi, lo, c = _path([100, 101, 103, 105])
    n = len(c)
    r, _ = B.exit_r(o, hi, lo, c, np.zeros(n), 0, 1, 100.0, 98.0, 2, 1.0, 4.0, np.full(n, 100.0), np.ones(n), 100, TAKER)
    assert r == pytest.approx((4.0 - (TAKER + MAKER) * 100) / 2.0)


def test_signal_needs_box_touches_and_ema_cross():
    n = 120
    c = np.r_[np.linspace(110, 100, 60), 100 + 0.5 * np.sin(np.arange(40) * 1.3), np.full(20, 100.0)]
    c[100] = 101.2                                                     # пробой коробки вверх

    def frame(x):
        o = np.r_[x[0], x[:-1]]
        idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
        return pd.DataFrame({"open": o, "high": np.maximum(o, x) + 0.2, "low": np.minimum(o, x) - 0.2, "close": x},
                            index=idx)

    assert not any(t == 100 for t, *_ in B.signals(frame(c), 32))      # прошлое закрытие уже над EMA10 / EMA20
    c[99] = 99.6                                                       # прошлое — под средними: пробой EMA на 100-й
    assert any(t == 100 and side == 1 for t, side, _, _ in B.signals(frame(c), 32))
