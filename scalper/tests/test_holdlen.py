import numpy as np
import pandas as pd

from research import holdlen as H
from research import noline


def _rw(seed: int, n: int = 12000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.0015, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.0015, n)))
    vol = np.full(n, 1000.0)
    idx = pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": vol,
                         "taker_buy_volume": vol * rng.uniform(0.3, 0.8, n)}, index=idx)


def test_bot_hold_matches_research_rule_and_longer_hold_times_out_less():
    d = _rw(3)
    liq = np.full(len(d), 1e9)
    rows = pd.DataFrame(H.coin_rows(d, liq, "RW", "15m"))
    noline.TF = "15m"
    ref = noline.coin_trades(d, liq, "RW", lines_only=True)
    assert len(rows) == len(ref) > 10
    assert np.allclose(rows["R200"].to_numpy(), ref["R3"].to_numpy())          # 200 свечей = бэктест бота
    assert rows["to50"].mean() >= rows["to200"].mean() >= rows["to800"].mean()
    assert rows.slope_atr.notna().all()
