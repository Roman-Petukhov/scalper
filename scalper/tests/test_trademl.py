import numpy as np
import pandas as pd

from research import exit5 as X
from research import trademl as M


def _rw(seed: int, n: int = 30000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.001, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.001, n)))
    vol = np.full(n, 1000.0)
    idx = pd.date_range("2023-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": vol,
                         "taker_buy_volume": vol * rng.uniform(0.3, 0.8, n)}, index=idx)


def test_rows_hold_matches_bot_and_points_are_strided_inside_trade():
    d = _rw(5)
    liq = np.full(len(d), 1e9)
    rows = pd.DataFrame(M.coin_rows(d, liq, "RW", "5m"))
    ref = pd.DataFrame(X.coin_rows(d, liq, "RW", "5m"))
    per_trade = rows.groupby(["t", "side"]).first().reset_index()
    m = per_trade.merge(ref, on=["t", "side"])
    assert len(m) == len(per_trade) > 5
    assert np.allclose(m["hold"].to_numpy(), m["base"].to_numpy())
    pts = rows[rows.pt]
    assert (pts.j % M.SLICE["5m"][3] == 0).all() and (pts.j > 0).all()
    assert pts[M.FEATS].drop(columns=["round_ahead", "swing_ahead"]).notna().all().all()
    assert ((pts.r_now <= pts.mfe + 1e-9) & (pts.giveback >= -1e-9)).all()
    assert (pts.aggr_in.between(0, 1)).all()


def test_policy_exits_at_first_signal_and_keeps_trades_without_points():
    pts = pd.DataFrame({"symbol": ["A"] * 3 + ["B"], "t": [1, 1, 1, 2], "j": [6, 12, 18, 0],
                        "hold": [-1.0, -1.0, -1.0, 3.0], "exit": [0.5, -0.2, 0.9, 3.0], "pt": [True, True, True, False]})
    out = M.policy(pts, np.array([0.1, -0.6, -0.9, -5.0]), 0.25).set_index("symbol")
    assert out.loc["A", "R"] == -0.2 and out.loc["A", "exited"]
    assert out.loc["B", "R"] == 3.0 and not out.loc["B", "exited"]
    never = M.policy(pts, np.zeros(4), 0.25)
    assert (never.R == never.bot).all()
