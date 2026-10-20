import numpy as np
import pandas as pd

from research import kelly as K


def test_kelly_f_matches_closed_form_for_minus1_plus3():
    p = 0.36
    r = np.r_[np.full(36, 3.0), np.full(64, -1.0)]
    assert abs(K.kelly_f(r) - (p - (1 - p) / 3)) <= 0.005
    assert K.kelly_f(np.r_[np.full(25, 3.0), np.full(80, -1.0)]) == 0.0      # нет края — ноль


def _trades(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.Timestamp("2020-01-01", tz="UTC") + pd.to_timedelta(np.sort(rng.uniform(0, 6 * 365, n)), unit="D")
    r = np.where(rng.random(n) < 0.36, 3.0, -1.0)
    return pd.DataFrame({"t": t, "t_in": t, "t_out": t + pd.Timedelta(days=3), "side": -1, "R3": r,
                         "risk_pct": 0.08, "beta": np.nan, "btc_ret": np.nan})


def test_growth_and_bootstrap_shapes():
    df = _trades()
    res = K.simulate(df, K.cfg(1.0))
    g = K.growth(res.eq, K.E0, pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC"))
    assert np.isfinite(g) and g > 0
    K.BOOT = 5
    gr, dd = K.bootstrap(df, 2.0, np.random.default_rng(0))
    assert len(gr) == len(dd) == 5 and (dd >= 0).all()
