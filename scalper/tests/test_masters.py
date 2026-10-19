import numpy as np
import pandas as pd
import pytest

from research.masters import adx, coin_rows, leaders, rsi, select, walk
from research.tline import MAKER, TAKER


def _arr(*xs):
    return [np.asarray(x, dtype="float64") for x in xs]


def test_walk_stop_target_half_and_entry_bar():
    o, hi, lo, c = _arr([100] * 5, [101, 102, 104, 111, 101], [99, 99, 100, 103, 99], [100, 101, 103, 110, 100])
    f = np.zeros(5)
    r, j = walk(o, hi, lo, c, f, 0, 1, 100.0, 95.0, 20, tp=110.0)
    assert j == 3 and r == pytest.approx((10 - TAKER * 100 - MAKER * 110) / 5)
    r, j = walk(o, hi, lo, c, f, 0, 1, 100.0, 95.0, 20, half_at=2)            # половина на 103, остаток — стоп в 100
    assert j == 4 and r == pytest.approx((0.5 * (3 - TAKER * 100 - TAKER * 103) + 0.5 * (0 - TAKER * 100 - TAKER * 100)) / 5)
    r, j = walk(o, hi, lo, c, f, 0, 1, 100.0, 99.5, 20, stop_entry_bar=True)   # открытие у входа, минимум ниже стопа
    assert j == 0 and r == pytest.approx((-0.5 - TAKER * 100 - TAKER * 99.5) / 0.5)
    r, j = walk(o, hi, lo, c, f, 0, -1, 100.0, 105.0, 2, close_exit=lambda k: c[k] > 102)
    assert j == 2 and r == pytest.approx((-3 - TAKER * 100 - TAKER * 103) / 5)


def test_indicators():
    up = np.arange(1.0, 60.0)
    assert rsi(up, 2)[-1] == pytest.approx(100.0) and rsi(up[::-1], 2)[-1] == pytest.approx(0.0)
    ax, pdi, mdi = adx(up + 0.5, up - 0.5, up)
    assert ax[-1] > 50 and pdi[-1] > mdi[-1]


def _frame(c, hi=None, lo=None, vol=None):
    c = np.asarray(c, dtype="float64")
    idx = pd.date_range("2022-01-01", periods=len(c), freq="1D", tz="UTC")
    return pd.DataFrame({"open": np.r_[c[0], c[:-1]], "high": c + 1 if hi is None else hi, "low": c - 1 if lo is None else lo,
                         "close": c, "volume": np.ones(len(c)) if vol is None else vol}, index=idx)


def test_turtle_soup_signal():
    n = 300
    c = 100 + np.sin(np.arange(n) / 3.0) * 2                 # боковик: канал 20 дней ~ 97…103
    hi, lo = c + 1, c - 1
    t = 250
    c[t], lo[t], hi[t] = 95.0, 94.0, 96.5                      # новый минимум 20 дней, закрытие под старым минимумом
    old = lo[t - 20:t].min()
    hi[t + 1], lo[t + 1], c[t + 1] = old + 2, 95.5, old + 1.5   # завтра цена вернулась к старому минимуму
    d = _frame(c, hi, lo)
    rows, _ = coin_rows(d, np.full(n, 1e9), "X", "perp")
    ts = [r for r in rows if r["sys"] == "T" and r["t"] == d.index[t]]
    assert {r["exit"] for r in ts} == {"2R", "3R"} and all(r["side"] == 1 for r in ts)


def test_leaders_and_select():
    day = pd.Timestamp("2024-01-01", tz="UTC")
    panel = pd.DataFrame({"symbol": [f"C{i}" for i in range(10)], "src": "perp", "day": day,
                          "ret": np.arange(10) / 10, "liq": 1e9})
    rs = leaders(panel)
    x = pd.DataFrame({"sys": ["Q", "Q", "C"], "symbol": ["C9", "C5", "C1"], "src": "perp", "side": [1, 1, 1],
                      "liq": 1e9, "rs_day": day, "R": [1.0, 1.0, 1.0]})
    out = select(x, rs)
    assert sorted(out.symbol) == ["C1", "C9"]                    # C5 не лидер; RSI(2) без отбора силы


def test_walk_entry_bar_stop_first_when_open_near_stop():
    o, hi, lo, c = _arr([95.2, 100], [101, 104], [94.8, 99], [100, 103])
    r, j = walk(o, hi, lo, c, np.zeros(2), 0, 1, 100.0, 95.0, 1, stop_entry_bar=True)   # стоп раньше входа: сделка идёт
    assert j == 1 and r == pytest.approx((3 - TAKER * 100 - TAKER * 103) / 5)


def test_random_walk_has_no_edge():
    res = []
    for seed in range(4):
        rng = np.random.default_rng(seed)
        n = 2000
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.04, n)))
        op = np.r_[c[0], c[:-1]] * np.exp(rng.normal(0, 0.005, n))
        hi = np.maximum(c, op) * (1 + abs(rng.normal(0, 0.02, n)))
        lo = np.minimum(c, op) * (1 - abs(rng.normal(0, 0.02, n)))
        d = pd.DataFrame({"open": op, "high": hi, "low": lo, "close": c, "volume": rng.lognormal(0, 0.5, n)},
                         index=pd.date_range("2018-01-01", periods=n, freq="1D", tz="UTC"))
        res += coin_rows(d, np.full(n, 1e9), "X", "spot")[0]
    x = pd.DataFrame(res)
    m = x.groupby("sys").R.agg(["mean", "sem"])
    assert (m["mean"] < 2.5 * m["sem"]).all()                  # на случайном блуждании преимущества нет ни у одной
