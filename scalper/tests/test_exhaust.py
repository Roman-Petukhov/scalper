import numpy as np
import pytest

from research.exhaust import exhaust_exit
from research.tline import MAKER, TAKER, two_targets


def bars(closes: list[float], own: list[float], vol: list[float] | None = None):
    c = np.array(closes, dtype="float64")
    o = np.r_[c[0], c[:-1]]
    h, lo = np.maximum(o, c) + 0.01, np.minimum(o, c) - 0.01
    n = len(c)
    return o, h, lo, c, np.zeros(n), np.array(own, dtype="float64"), np.array(vol or [1.0] * n, dtype="float64")


def run(b, mode: int, gate: float = 0.0, side: int = 1, vref: float = 1.0):
    o, h, lo, c, f, own, vol = b
    return exhaust_exit(o, h, lo, c, f, own, vol, 0, side, 100.0, 90.0 if side > 0 else 110.0, 3.0, 60, MAKER,
                        mode, gate, vref)


def test_control_matches_bot_target():
    b = bars([100, 105, 112, 125, 140], [0.6, 0.4, 0.4, 0.4, 0.4])
    o, h, lo, c, f, _, _ = b
    r, j, early = run(b, 0)
    assert early == 0
    assert (r, j) == pytest.approx(two_targets(o, h, lo, c, f, 0, 1, 100.0, 90.0, 3.0, 3.0, False, 60, MAKER))
    assert r == pytest.approx((30 - (MAKER + MAKER) * 100) / 10)


def test_weak_aggressors_two_bars_exit_at_close_in_profit():
    b = bars([100, 105, 108, 109, 140], [0.6, 0.45, 0.48, 0.4, 0.6])
    r, j, early = run(b, 1)
    assert (j, early) == (2, 1)
    assert r == pytest.approx((8 - (MAKER + TAKER) * 100) / 10)


def test_no_early_exit_below_profit_gate():
    b = bars([100, 105, 108, 109, 140], [0.6, 0.45, 0.48, 0.4, 0.6])
    r, j, early = run(b, 1, gate=1.0)        # +0.8R / +0.9R < 1R → держим до тейка
    assert early == 0 and r > 2.9


def test_short_uses_seller_share():
    # шорт: own — доля продавцов; продавцы < 45% при плюсе → выход
    b = bars([100, 97, 95, 60], [0.6, 0.6, 0.4, 0.6])
    r, j, early = run(b, 2, side=-1)
    assert (j, early) == (2, 1)
    assert r == pytest.approx((5 - (MAKER + TAKER) * 100) / 10)


def test_divergence_needs_new_extreme_and_negative_delta():
    b = bars([100, 104, 103, 106, 140], [0.6, 0.6, 0.3, 0.45, 0.6], [1, 1, 3, 1, 1])
    r, j, early = run(b, 4)
    assert (j, early) == (3, 1)       # 106 — новый максимум, дельта за 3 свечи 0.2 − 1.2 − 0.1 < 0


def test_volume_fade():
    b = bars([100, 103, 104, 105, 140], [0.6] * 5, [3, 3, 0.4, 0.4, 3])
    r, j, early = run(b, 3, vref=2.0)
    assert (j, early) == (3, 1)


def test_trade_exits_market_entry_keeps_columns_and_matches_control():
    import pandas as pd

    from research.exhaust import VARIANTS, trade_exits

    idx = pd.date_range("2024-01-01", periods=30, freq="15min", tz="UTC")
    c = np.r_[np.full(25, 100.0), [98, 97, 99, 96, 80]]
    d = pd.DataFrame({"open": np.r_[c[0], c[:-1]], "high": c + 0.5, "low": c - 0.5, "close": c, "funding": 0.0,
                      "volume": 1.0, "taker_buy_volume": 0.3}, index=idx)
    x = pd.DataFrame({"symbol": ["X"], "t": [idx[24]], "side": [-1], "fill_i": [24], "px": [100.0], "stop": [105.0],
                      "risk_pct": [0.05], "R3": [np.nan], "aggr": [0.6]})
    out = trade_exits(d, x, "15m", TAKER, ("aggr",))
    assert out.aggr.iloc[0] == 0.6
    assert out.R_ctl.iloc[0] == pytest.approx((15 - (TAKER + MAKER) * 100) / 5)       # тейк 3R = 85
    assert set(f"R_{v}" for v in VARIANTS) <= set(out.columns)
