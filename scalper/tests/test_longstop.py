import numpy as np
import pandas as pd

from research import longstop as S


def _bars(c, wick=0.0):
    c = np.asarray(c, dtype="float64")
    o = np.r_[c[0], c[:-1]]
    return o, np.maximum(o, c) + wick, np.minimum(o, c) - wick, c


def test_after_stop_sees_target_and_entry_for_both_sides():
    o, hi, lo, c = _bars([100, 95, 89, 101, 131])
    assert S.after_stop(hi, lo, 2, 1, 100.0, 90.0, 1) == (False, True)
    assert S.after_stop(hi, lo, 2, 1, 100.0, 90.0, 2) == (True, True)
    o, hi, lo, c = _bars([100, 105, 111, 99, 69])
    assert S.after_stop(hi, lo, 2, -1, 100.0, 110.0, 2) == (True, True)


def test_stopped_checks_the_exit_bar_against_the_stop():
    o, hi, lo, c = _bars([100, 95, 89])
    assert S.stopped(hi, lo, 2, 1, 90.0) and not S.stopped(hi, lo, 1, 1, 90.0)


def test_reentry_after_a_sweep_takes_the_new_trade():
    c = [100, 95, 89, 91, 95, 100, 110, 125, 140]
    o, hi, lo, cl = _bars(c)
    atr = np.full(len(c), 2.0)
    r = S.reentry(o, hi, lo, cl, np.zeros(len(c)), atr, 2, 1, 90.0)   # закрытие 91 > 90 → вход, стоп 88.8
    assert r > 2.5


def test_reentry_is_zero_when_price_stays_below_the_stop():
    c = [100, 95, 89, 88, 87, 86, 85]
    o, hi, lo, cl = _bars(c)
    assert S.reentry(o, hi, lo, cl, np.zeros(len(c)), np.full(len(c), 2.0), 2, 1, 90.0) == 0.0


def test_rows_on_random_walk_have_every_variant():
    rng = np.random.default_rng(5)
    n = 12000
    idx = pd.date_range("2021-01-01", periods=n, freq="4h", tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, n)))
    v = np.full(n, 1000.0)
    d = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": v,
                      "taker_buy_volume": v * rng.uniform(0.3, 0.8, n), "funding": 0.0}, index=idx)
    rows = pd.DataFrame(S.coin_rows(d, np.full(n, 1e9), "RW"))
    assert len(rows) >= 3 and rows[list(S.VARIANTS)].notna().all().all()
    assert (rows.loc[~rows.stopped, "reentry"] == rows.loc[~rows.stopped, "R"]).all()
