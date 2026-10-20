import numpy as np
import pandas as pd

from research import exit5 as X
from research import holdlen as H
from research.tline import MAKER, TAKER

FEE = TAKER * 100 / 10          # вход и выход тейкером, вход 100, риск 10


def _rw(seed: int, n: int = 30000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.001, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.001, n)))
    vol = np.full(n, 1000.0)
    idx = pd.date_range("2023-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": vol,
                         "taker_buy_volume": vol * rng.uniform(0.3, 0.8, n)}, index=idx)


def _walk(c, v, line_e=0.0, slope=0.0, buf_abs=None, hi=None, lo=None):
    c = np.asarray(c, dtype="float64")
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) + 0.5 if hi is None else np.asarray(hi, dtype="float64")
    lo = np.minimum(o, c) - 0.5 if lo is None else np.asarray(lo, dtype="float64")
    k, be_at, be_to, part_at, frac, buf, window = X.VARIANTS[v]
    b = (buf_abs if buf_abs is not None else buf) if buf >= 0 else X.NO
    return X.walk(o, hi, lo, c, np.zeros(len(c)), 0, 1, 100.0, 90.0, k, be_at, be_to, part_at, frac, line_e, slope,
                  b, window, 800, TAKER)


def test_base_matches_bot_hold800_on_random_walk():
    d = _rw(5)
    liq = np.full(len(d), 1e9)
    rows = pd.DataFrame(X.coin_rows(d, liq, "RW", "5m"))
    ref = pd.DataFrame(H.coin_rows(d, liq, "RW", "5m"))
    ref = ref[ref.entry == "market"]
    assert len(rows) == len(ref) > 5
    assert np.allclose(rows["base"].to_numpy(), ref["R800"].to_numpy())


def test_near_miss_variants():
    # лонг от 100, стоп 90: до 125 (+2.5R, не 3R), затем в стоп
    c = [100, 110, 124, 118, 100, 85.0]
    base, mfe, why = _walk(c, "base")
    assert why == 0 and 2.0 <= mfe < 3 and base < -1
    tp2, _, why = _walk(c, "tp2")
    assert why == 1 and abs(tp2 - (2 - (TAKER + MAKER) * 100 / 10)) < 1e-9
    be, _, why = _walk(c, "be2")
    assert why == 0 and abs(be + FEE * 2) < 1e-9                       # в ноль, только комиссии
    lock, _, _ = _walk(c, "lock2")
    assert abs(lock - (1 - FEE * 2)) < 1e-9
    half, _, _ = _walk(c, "half2")                                     # ½ по 2R, ½ в ноль
    assert abs(half - (0.5 * 2 - (TAKER + 0.5 * MAKER + 0.5 * TAKER) * 100 / 10)) < 1e-9


def test_half_then_target():
    c = [100, 125, 135.0]
    half, _, why = _walk(c, "half2")
    assert why == 1 and abs(half - (0.5 * 2 + 0.5 * 3 - (TAKER + MAKER) * 100 / 10)) < 1e-9


def test_line_exit_buffer_and_window():
    # лонг, линия на 99 плоская; свеча 1 закрылась на 98 (за линией на 1), затем рост к цели
    c = [100, 98, 105, 135.0]
    lo = [99.5, 97.0, 97.5, 104.0]
    hi = [100.5, 100.5, 105.5, 135.5]
    r, _, why = _walk(c, "back0", line_e=99.0, hi=hi, lo=lo)
    assert why == 3 and abs(r - (-0.2 - FEE * 2)) < 1e-9
    r, _, why = _walk(c, "back05", line_e=99.0, buf_abs=2.0, hi=hi, lo=lo)   # за линией меньше буфера — держим
    assert why == 1
    c2 = [100] + [101.0] * 13 + [95.0, 135.0]
    r, _, why = _walk(c2, "back05h", line_e=99.0, buf_abs=1.0)         # возврат на 14-й свече — вне окна 12
    assert why == 1
