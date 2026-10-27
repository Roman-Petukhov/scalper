import numpy as np
import pandas as pd

from research import secorder as O
from research.sectors import panel, sector_map


def _data(seed=1, days=900):
    rng = np.random.default_rng(seed)
    smap = sector_map()
    syms = ["BTCUSDT", "ETHUSDT"] + [s for s in smap][:60] + [f"X{i}USDT" for i in range(180)]
    idx = pd.date_range("2020-01-01", periods=days, freq="D")
    r = rng.normal(0.0, 0.03, (days, len(syms)))
    close = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=syms)
    qv = pd.DataFrame(rng.uniform(6e6, 9e7, (days, len(syms))), index=idx, columns=syms)
    return close, qv


def test_week_and_past_returns():
    s = pd.Series(0.01, index=pd.date_range("2024-01-01", periods=40, freq="D"))
    t = pd.Timestamp("2024-01-15")
    assert abs(O.week_ret(s, t) - (1.01 ** 7 - 1)) < 1e-12
    assert abs(O.past(s, t, 7) - (1.01 ** 7 - 1)) < 1e-12
    assert O.past(s, pd.Timestamp("2024-01-03"), 28) is None


def test_size_baskets_exclude_btc_eth_and_rank():
    close, qv = _data()
    qv["ETHUSDT"] = 1e12
    r = close.pct_change()
    big, small = O.size_baskets(r, qv)
    assert big.notna().sum() > 800 and small.notna().sum() > 800
    assert not np.allclose(big.dropna(), r["ETHUSDT"].reindex(big.dropna().index))


def test_strategies_run_and_pick_weekly_follows_score():
    close, qv = _data()
    r, mkt, idx = panel(close, qv)
    a, b = O.ladder(r, qv, mkt)
    assert len(a) > 20 and len(b) > 20 and {"t", "X"} <= set(a.columns)
    a2, b2 = O.attention(qv, idx, mkt)
    assert len(a2) > 20 and len(b2) <= len(a2)
    assert len(O.breadth(close, r, idx, mkt)) > 20
    days = pd.date_range("2024-01-01", periods=21, freq="D")
    score = pd.DataFrame({"A": 1.0, "B": 2.0, "C": 0.0, "D": 0.5}, index=days)
    ex = pd.DataFrame({"A": 0.0, "B": 0.01, "C": 0.0, "D": 0.0}, index=days)
    z = O.pick_weekly(score, ex)
    assert set(z.sector) == {"B"} and (z.X > 0).all()
    gated = O.pick_weekly(score, ex, gate=pd.DataFrame({"A": 1, "B": -1, "C": 1, "D": -1}, index=days))
    assert set(gated.sector) == {"A"}


def test_cycle_order_without_sector_data(capsys):
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    O.cycle_order({"A": pd.Series(np.nan, index=idx)}, pd.Series(0.0, index=idx))
    assert "секторов с данными нет" in capsys.readouterr().out
