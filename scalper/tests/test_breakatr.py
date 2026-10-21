import numpy as np
import pandas as pd

from research import breakatr as B


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


def test_rows_have_positive_break_distance_on_the_breakout_side(monkeypatch):
    monkeypatch.setattr(B, "MAX_SLOPE", 1e9)            # на случайном ряде пологих линий почти нет
    rows = pd.DataFrame(B.coin_rows(_rw(1), np.full(30000, 1e9), "RW", "15m"))
    assert len(rows) > 20 and rows.R.notna().all()
    assert (rows.brk_atr > 0).all()                     # пробой — закрытие за линией, значит расстояние положительно


def test_verdict_accepts_a_threshold_that_cuts_only_losers():
    rng = np.random.default_rng(0)
    n = 8000
    z = pd.DataFrame({"t": pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 1400, n), unit="D"),
                      "brk_atr": rng.choice([0.05, 0.15, 0.25, 0.35, 0.6, 1.2], n)})
    z["R"] = rng.normal(0, 1, n) - np.where(z.brk_atr < 0.2, 0.5, 0.0) + np.where(z.brk_atr == 0.6, 1.5, 0.0)
    ok = B.verdicts(z)
    assert ok[0.1][0] and ok[0.2][0] and not ok[1.0][0]                              # 1.0 режет и пробои 0.6, которые платят
    z["R"] = rng.normal(0, 1, n) + 0.2                                               # все прибыльны — резать нечего
    assert not any(v[0] for v in B.verdicts(z).values())
