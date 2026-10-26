import numpy as np
import pandas as pd
import pytest

from research import emasys as S


def _frame(n, freq, seed=5, drift=0.0, vol=0.004):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    v = rng.uniform(80, 120, n)
    idx = pd.date_range("2022-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": v, "quote_volume": v * c,
                         "funding": 0.0}, index=idx)


def test_first_true_and_no_touch_helpers():
    m = np.array([False, True, True, False, True])
    assert S._first_true(m).tolist() == [False, True, False, False, True]
    touch = np.array([False, False, True, False, False, False])
    assert S._no_touch(touch, 2).tolist() == [False, False, True, False, False, True]


def test_mean_reversion_fades_a_far_close_and_targets_the_ema():
    d = _frame(2500, "4h", vol=0.01)
    rows = S.mean_rows(d, np.full(len(d), 5e7), "TEST", "4h")
    assert rows and {r["sys"] for r in rows} == {"M"}
    for r in rows[:50]:
        assert r["side"] in (1, -1) and np.isfinite(r["R"]) and r["exit"] in ("к EMA", "к середине")
    far = [r for r in rows if "k 3 ATR" in r["var"] and "EMA20" in r["var"] and r["exit"] == "к EMA"]
    c = d["close"].to_numpy()
    e = d["close"].ewm(span=20, adjust=False, min_periods=20).mean().to_numpy()
    t = d.index.get_loc(far[0]["t"])
    assert far[0]["side"] == (-1 if c[t] > e[t] else 1)        # против отклонения


def test_pure_ema_rows_cover_cross_and_bounce_on_both_sides():
    d = _frame(3000, "4h", drift=0.0003, vol=0.006)
    rows = S.pure_rows(d, np.full(len(d), 5e7), "TEST", "4h")
    kinds = {r["var"].split(", ")[1].split(" ")[0] for r in rows}
    assert kinds == {"пробой", "отскок"} and {r["side"] for r in rows} == {1, -1}
    assert {r["exit"] for r in rows} >= {"3R", "обратное пересечение", "закрытие за EMA"}
    assert all(np.isfinite(r["R"]) and r["tf"] == "4h" for r in rows)


def _vcp_frame(spike: bool):
    d = _frame(9000, "15min", seed=2, drift=0.0006, vol=0.003)
    v = d["volume"].to_numpy().copy()
    for s_ in range(100, 9000, 30):
        v[s_ - 4:s_] = 25.0                                     # сухие 4 свечи перед каждой 30-й
    if spike:
        v[6000:6600] *= 8                                       # сутки объёма втрое выше обычного
    return d.assign(volume=v)


def test_vcp_rows_need_trend_compression_and_relative_volume():
    d = _vcp_frame(True)
    rows = S.vcp_rows(d, np.full(len(d), 5e7), "TEST")
    assert rows and all(r["sys"] == "G" and r["tf"] == "15m" for r in rows)
    assert {r["exit"] for r in rows} <= {"3R", "закрытие за EMA21"} and all("RVOL" in r["var"] for r in rows)
    assert all(abs(r["c2"]) > 0 for r in rows)
    assert S.vcp_rows(_vcp_frame(False), np.full(len(d), 5e7), "TEST") == []   # без относительного объёма сделок нет


def test_rows_respect_liquidity_filter():
    d = _frame(2500, "4h", vol=0.01)
    assert S.mean_rows(d, np.full(len(d), 1e6), "TEST", "4h") == []
    assert S.pure_rows(d, np.full(len(d), 1e6), "TEST", "4h") == []
