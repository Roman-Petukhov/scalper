import numpy as np
import pandas as pd

from research.candles import engulf_trades, engulfing, retest_kind


def test_engulfing_needs_opposite_colours_and_full_body_cover():
    o = np.array([10.0, 9.0, 9.0, 10.0])
    c = np.array([9.0, 10.5, 8.0, 9.5])
    bull, bear = engulfing(o, c)
    assert list(bull) == [False, True, False, False]      # 9→10.5 накрывает 10→9
    assert not bear.any()                                 # 9→8: открытие 9 ниже закрытия 10.5 — тело не накрыто
    o2, c2 = np.array([9.0, 10.6]), np.array([10.5, 8.8])
    assert list(engulfing(o2, c2)[1]) == [False, True]


def test_retest_kind():
    # лонг, линия 100: закрытие ниже линии — против
    assert retest_kind(101, 102, 98, 99.5, 1, 100.0, False) == "against"
    # длинная нижняя тень к линии, закрытие в верхней половине — пин-бар
    assert retest_kind(101, 102, 96, 101.5, 1, 100.0, False) == "pin"
    # шорт: длинная верхняя тень, закрытие внизу
    assert retest_kind(99, 104, 98, 98.5, -1, 100.0, False) == "pin"
    assert retest_kind(100.5, 103, 100.2, 102.8, 1, 100.0, True) == "engulf"
    assert retest_kind(100.5, 103, 100.2, 102.8, 1, 100.0, False) == "neutral"


def test_engulf_trades_long_hits_target():
    n = 40
    c = np.full(n, 100.0)
    o = c.copy()
    o[20], c[20] = 100.5, 99.5                       # красная
    o[21], c[21] = 99.4, 101.0                       # бычье поглощение
    c[22:] = 120.0
    o[22:] = 101.0
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    d = pd.DataFrame({"open": o, "high": np.maximum(o, c) + 0.2, "low": np.minimum(o, c) - 0.2, "close": c,
                      "funding": 0.0}, index=idx)
    x = engulf_trades(d, np.full(n, 1e9), "X")
    row = x[(x.t == idx[21]) & (x.side == 1)].iloc[0]
    assert row.R3 > 2.9 and row.R2 > 1.9
