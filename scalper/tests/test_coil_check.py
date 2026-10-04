import numpy as np
import pandas as pd

from research.coil_check import coil_signals, stats


def _bars(n=400, funding=-1e-4):
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    rng = np.random.default_rng(0)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    c[-30:-1] = c[-31]                                   # сжатие: цена стоит
    c[-1] = c[-31] * 1.05                                # пробой вверх
    oi = np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    oi[-30:] = oi[-31] * np.linspace(1, 1.5, 30)          # OI резко набирается
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "funding": funding, "oi": oi}, index=idx)


def test_breakout_up_only_against_crowd():
    up, dn = coil_signals(_bars(funding=-1e-4), 18, 0.2, 0.8, "against")
    assert up[-1] and not dn.any()
    up, _ = coil_signals(_bars(funding=1e-4), 18, 0.2, 0.8, "against")
    assert not up[-1]
    up, _ = coil_signals(_bars(funding=1e-4), 18, 0.2, 0.8, "with")
    assert up[-1]


def test_day_clustered_t_is_lower_for_same_day_trades():
    t = pd.to_datetime(["2024-01-01 01:00"] * 50 + [f"2024-02-{d:02d} 00:00" for d in range(1, 29)], utc=True)
    r = np.r_[np.full(50, 2.0), np.random.default_rng(1).normal(0, 1, 28)]
    s = stats(pd.DataFrame({"t": t, "R": r}))
    assert s["days"] == 29 and s["t_day"] < s["t"]
