import numpy as np
import pandas as pd

from research import sides15 as S


def _trades(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    side = np.where(rng.random(n) < 0.5, 1, -1)
    base = np.where(side < 0, 0.2, -0.2) + rng.normal(0, 1, n)
    fund = np.where(side < 0, 0.05, -0.05)
    t = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 500, n), unit="D")
    return pd.DataFrame({"t": t, "side": side, "R": base, "R_nofund": base - fund, "risk_pct": 0.01})


def test_welch_and_tstat_have_the_right_sign():
    a, b = np.array([1.0, 2.0, 3.0, 2.5]), np.array([0.0, 0.5, 0.2, 0.1])
    assert S.welch(a, b) > 2
    assert S.tstat(-np.abs(np.random.default_rng(1).normal(1, 0.1, 50))) < -10
    assert np.isnan(S.welch(a[:2], b))


def test_verdict_cuts_longs_only_when_negative_everywhere():
    z = _trades(4000)
    z["t"] = pd.Timestamp("2022-01-01") + (z.t - z.t.min()) * 2          # размазать по IS / VAL / HO
    z.loc[z.side > 0, "R"] -= 0.3
    cut, by_fund, lines = S.verdict(z)
    assert cut and lines
    z2 = z.assign(R=z.R.abs() + 1)                                        # лонги в плюсе — отключать нечего
    assert not S.verdict(z2)[0]


def test_drift_rows_monthly_returns_and_funding():
    idx = pd.date_range("2023-01-01", periods=96 * 90, freq="15min")
    d = pd.DataFrame({"close": np.linspace(100, 90, len(idx)), "funding": 1e-5}, index=idx)
    rows = S.drift_rows(d, "AUSDT", "15m")
    assert rows and all(r["ret"] < 0 for r in rows)
    assert abs(rows[0]["fund_apr"] - 1e-5 * 365 * 96) < 1e-9


def _rw(seed: int, n: int = 30000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="4h")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, n)))
    vol = np.full(n, 1000.0)
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": vol,
                         "taker_buy_volume": vol * rng.uniform(0.3, 0.8, n), "funding": 2e-5}, index=idx)


def test_funding_part_is_charged_to_longs_and_paid_to_shorts():
    d = _rw(1)
    rows = pd.DataFrame(S.coin_rows(d, np.full(len(d), 1e9), "RW", "4h"))
    assert len(rows) > 5 and rows.R.notna().all()
    f = rows.R - rows.R_nofund
    assert f[rows.side > 0].mean() < 0 < f[rows.side < 0].mean()
