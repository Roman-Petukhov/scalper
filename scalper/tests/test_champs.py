import numpy as np
import pandas as pd
import pytest

from research import champs as H


def _rw(seed: int, n: int, freq: str, vol: float = 0.006) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq=freq, tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    v = 1000 * np.exp(rng.normal(0, 0.5, n))
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": v, "quote_volume": v * c,
                         "taker_buy_volume": v / 2, "funding": 0.0}, index=idx)


def _liq(d):
    return pd.Series(1e12, index=d.index.floor("D").unique())


def test_walk_bars_stop_target_and_bailout():
    o = np.array([100, 100, 101, 103, 99.0]); hi = o + 1; lo = o - 1; c = o.copy(); f = np.zeros(5)
    r, j = H.walk_bars(o, hi, lo, c, f, 0, 1, 100.0, 97.5, 4, tp=103.5)
    assert j == 3 and r == pytest.approx((3.5 - H.TAKER * 100 - H.MAKER * 103.5) / 2.5)
    bail = np.array([False, False, True, False, False])
    r, j = H.walk_bars(o, hi, lo, c, f, 0, 1, 100.0, 97.5, 4, open_exit=bail)
    assert j == 2 and r > 0                                            # первое открытие в плюсе — выход
    r, j = H.walk_bars(o, hi, lo, c, f, 0, -1, 100.0, 101.5, 4)
    assert j == 2 and r == pytest.approx((-1.5 - H.TAKER * 100 - H.TAKER * 101.5) / 1.5)


def test_williams_levels_stop_at_open_and_no_edge_on_noise():
    h = _rw(1, 24 * 400, "1h", 0.006)
    rows = pd.DataFrame(H.williams_rows(h, _liq(h), "RW"))
    assert len(rows) > 200 and set(rows["exit"]) == {"закрытие дня", "bailout"}
    assert rows.groupby("var").size().index.tolist() == ["k=0.5", "k=0.7"]
    assert abs(rows.R.mean()) < 0.25


def test_kell_and_flag_rows_on_random_walks():
    d = _rw(2, 3000, "1D", 0.04)
    k = pd.DataFrame(H.kell_rows(d, np.full(len(d), 1e12), "RW", "perp", "1d"))
    assert len(k) > 20 and set(k.side) == {1, -1} and k.R.notna().all()
    f = pd.DataFrame(H.flag_rows(_rw(3, 6000, "1D", 0.06), np.full(6000, 1e12), "RW", "spot"))
    assert len(f) > 0 and (f.c2 > 0).all()


def test_flag_found_on_a_built_pole_and_flag():
    n = 120
    c = np.r_[np.full(60, 10.0), np.linspace(10, 20, 20), np.full(10, 18.5), [22.0], np.full(29, 22.0)]
    d = pd.DataFrame({"open": c, "high": c * 1.05, "low": c * 0.95, "close": c,
                      "volume": np.r_[np.full(90, 100.0), [500.0], np.full(29, 100.0)], "funding": 0.0},
                     index=pd.date_range("2023-01-01", periods=n, freq="1D", tz="UTC"))
    rows = H.flag_rows(d, np.full(n, 1e12), "X", "perp")
    assert rows and rows[0]["t"] == d.index[90] and rows[0]["side"] == 1


def test_orb_trades_once_per_session_and_range():
    d = _rw(4, 96 * 120, "15min", 0.003)
    rows = pd.DataFrame(H.orb_rows(d, _liq(d), "RW"))
    per = rows[rows["exit"] == "конец сессии"].groupby([rows.t.dt.floor("D"), "var"]).size()
    assert len(rows) > 100 and per.max() == 1
    us = rows[rows["var"].str.startswith("13:30")]
    assert (us.t.dt.hour * 60 + us.t.dt.minute >= 13 * 60 + 30 + 30).all()
