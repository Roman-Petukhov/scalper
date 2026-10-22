import numpy as np
import pandas as pd
import pytest

from research import spec as R


def _rw(seed: int, n: int = 30000, freq: str = "15min") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq=freq, tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.002, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.002, n)))
    vol = np.full(n, 1000.0)
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": vol,
                         "taker_buy_volume": vol * rng.uniform(0.3, 0.8, n), "funding": 0.0}, index=idx)


def test_ker_is_one_on_a_trend_and_small_on_chop():
    assert R.ker(np.arange(1.0, 60.0), 20)[-1] == pytest.approx(1.0)
    assert R.ker(np.tile([1.0, 2.0], 30), 20)[-1] < 0.1


def test_gk_atr_is_causal_and_comparable_to_atr():
    d = _rw(1, 4000)
    g = R.gk_atr(d)
    assert g.iloc[:R.WARMUP].isna().all() and (g.dropna() > 0).all()
    ratio = (g / R._atr(d)).dropna()
    assert 0.7 < ratio.median() < 1.3                     # масштаб выровнен под ATR
    cut = R.gk_atr(d.iloc[:3000])
    assert cut.iloc[-1] == pytest.approx(g.iloc[2999])    # будущее на прошлое не влияет


def test_ts_lines_are_tangents_with_no_vertex_above_and_break_on_the_right_side():
    d = _rw(2)
    lines = [r for r in R.ts_lines(d) if r["t"] > 0]
    assert len(lines) > 5
    c = d["close"].to_numpy()
    for r in lines:
        assert r["side"] * r["slope"] < 0                  # линия против направления пробоя
        assert r["side"] * (c[r["t"]] - r["line_t"]) > 0 and r["side"] * (c[r["t"] - 1] - (r["line_t"] - r["slope"])) <= 1e-9


@pytest.mark.parametrize("variant", R.VARIANTS)
def test_rows_for_every_variant_on_a_random_walk(monkeypatch, variant):
    monkeypatch.setattr(R, "MAX_SLOPE", 1e9)
    rows = pd.DataFrame(R.coin_rows(_rw(3), np.full(30000, 1e9), "RW", "15m", variant))
    assert len(rows) > 10 and rows.R.notna().all() and (rows.variant == variant).all()
    assert rows.ker20.between(0, 1).all()


def test_patches_are_restored():
    before = (R.tline._atr, R.noline._atr, R.noline.zz_lines)
    R.coin_rows(_rw(4, 6000), np.full(6000, 1e9), "RW", "15m", "gk")
    R.coin_rows(_rw(4, 6000), np.full(6000, 1e9), "RW", "15m", "ts")
    assert (R.tline._atr, R.noline._atr, R.noline.zz_lines) == before


def test_cut_verdict_accepts_a_threshold_that_cuts_only_losers():
    rng = np.random.default_rng(0)
    n = 8000
    z = pd.DataFrame({"t": pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 1400, n), unit="D"),
                      "ker20": rng.uniform(0, 1, n)})
    z["R"] = rng.normal(0, 1, n) - np.where(z.ker20 < 0.3, 0.6, 0.0)
    ok = R.cut_verdicts(z, "ker20")
    assert ok[0.2][0] and ok[0.3][0]                       # пороги режут только убыточное
    z["R"] = rng.normal(0.2, 1, n)
    assert not any(v[0] for v in R.cut_verdicts(z, "ker20").values())


def test_variant_verdict_needs_every_period_positive_t3_and_not_worse_than_control():
    rng = np.random.default_rng(1)
    n = 6000
    t = pd.Timestamp("2020-06-01") + pd.to_timedelta(rng.integers(0, 2300, n), unit="D")
    v = pd.DataFrame({"t": t, "R": rng.normal(0.3, 1, n)})
    base = pd.DataFrame({"t": t[:2000], "R": rng.normal(0.0, 1, 2000)})
    assert R.variant_verdict(v, base)[0]
    assert not R.variant_verdict(v.assign(R=v.R - 0.3), base)[0]
