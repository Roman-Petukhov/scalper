import numpy as np
import pandas as pd

from research import longexit as L
from research.tline import MAKER, TAKER


def _path(c):
    c = np.asarray(c, dtype="float64")
    o = np.r_[c[0], c[:-1]]
    return o, np.maximum(o, c), np.minimum(o, c), c


def _run(c, kind, atr=1.0, below=None, low10=None, hold=1000):
    o, hi, lo, cl = _path(c)
    n = len(cl)
    return L.long_exit(o, hi, lo, cl, np.zeros(n), 0, 100.0, 90.0, MAKER, kind, hold, np.full(n, atr),
                       np.zeros(n) if below is None else np.asarray(below, float),
                       np.full(n, np.nan) if low10 is None else np.asarray(low10, float))


def test_targets_take_profit_at_their_multiple():
    up = [100, 120, 140, 160, 180, 200]
    assert abs(_run(up, "5R") - (50 - (MAKER + MAKER) * 100) / 10) < 1e-9
    assert abs(_run(up, "8R") - (80 - (MAKER + MAKER) * 100) / 10) < 1e-9


def test_stop_hits_before_anything_else():
    assert abs(_run([100, 95, 85], "5R") - (-10 - (MAKER + TAKER) * 100) / 10) < 1e-9


def test_trail_follows_the_peak():
    r = _run([100, 110, 130, 125], "trail3", atr=2.0)            # пик 130, стоп 124 не задет — выход в конце данных
    assert abs(r - (25 - (MAKER + TAKER) * 100) / 10) < 1e-9
    r = _run([100, 110, 130, 123], "trail3", atr=2.0)            # стоп 124, открытие 130 → выход по 124
    assert abs(r - (24 - (MAKER + TAKER) * 100) / 10) < 1e-9


def test_daily_ema_exit_closes_on_the_flag_bar():
    r = _run([100, 105, 112, 108], "ema20d", below=[0, 0, 1, 0])
    assert abs(r - (12 - (MAKER + TAKER) * 100) / 10) < 1e-9


def test_low10_raises_the_stop():
    r = _run([100, 110, 120, 104], "low10d", low10=[np.nan, 95, 105, 105])
    assert abs(r - (5 - (MAKER + TAKER) * 100) / 10) < 1e-9


def test_daily_levels_are_known_only_after_the_day_closes():
    idx = pd.date_range("2024-01-01", periods=6 * 40, freq="4h", tz="UTC")
    c = np.r_[np.full(6 * 30, 100.0), np.full(6 * 10, 50.0)]
    d = pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c}, index=idx)
    below, low10 = L.daily_levels(d, "4h")
    first = 6 * 30
    assert below[first + 4] == 0.0 and below[first + 5] == 1.0    # дневка 31-го дня известна на её последней свече
    assert low10[first + 5] == 49.0 and low10[first - 1] == 99.0


def test_rows_are_longs_with_every_exit():
    rng = np.random.default_rng(4)
    n = 6000
    idx = pd.date_range("2021-01-01", periods=n, freq="4h", tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, n)))
    v = np.full(n, 1000.0)
    d = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": v,
                      "taker_buy_volume": v * rng.uniform(0.3, 0.8, n), "funding": 0.0}, index=idx)
    rows = pd.DataFrame(L.coin_rows(d, np.full(n, 1e9), "RW", "4h"))
    assert len(rows) >= 3
    assert rows[list(L.EXITS)].notna().all().all() and (rows[list(L.EXITS)] >= -1.2).all().all()
