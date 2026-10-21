import numpy as np
import pandas as pd

from research import slope as S


def _rw(seed: int, n: int = 30000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.002, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.002, n)))
    vol = np.full(n, 1000.0)
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": vol,
                         "taker_buy_volume": vol * rng.uniform(0.3, 0.8, n), "funding": 0.0}, index=idx)


def test_rows_carry_slope_in_atr_and_both_sides():
    d = _rw(1)
    rows = pd.DataFrame(S.coin_rows(d, np.full(len(d), 1e9), "RW", "15m"))
    assert len(rows) > 20 and rows.slope_atr.ge(0).all() and set(rows.side) <= {-1, 1}
    assert rows.R.notna().all()


def test_verdict_accepts_only_when_the_added_trades_pay_everywhere():
    rng = np.random.default_rng(0)
    n = 6000
    z = pd.DataFrame({"t": pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 1400, n), unit="D"),
                      "slope_atr": rng.choice([0.01, 0.028, 0.035, 0.04, 0.08, 0.15, 0.5], n), "side": -1})
    z["R"] = rng.normal(0, 1, n) + np.where(z.slope_atr > S.BASE, 0.6, 0.0)           # крутые линии платят
    ok = S.verdicts(z)
    assert ok[0.03][0] and ok[0.05][0] and ok[np.inf][0]
    z["R"] = rng.normal(0, 1, n) - np.where(z.slope_atr > S.BASE, 0.4, 0.0)           # крутые линии теряют
    assert not any(v[0] for v in S.verdicts(z).values())
