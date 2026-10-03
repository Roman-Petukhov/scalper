import numpy as np
import pandas as pd
import pytest

from research import strategies as S
from research.engine import metrics, run, state_machine, stop_target_machine


def make_df(close, funding=None):
    idx = pd.date_range("2023-01-01", periods=len(close), freq="1h", tz="UTC")
    c = np.asarray(close, dtype=float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0, "quote_volume": c,
                         "taker_buy_volume": 0.5,
                         "funding": np.zeros(len(c)) if funding is None else funding}, index=idx)


def test_pnl_uses_next_bar_return_and_charges_costs():
    df = make_df([100, 110, 121])
    res = run(df, np.array([1.0, 1.0, 0.0]), cost_bps=10, bar_min=60)
    # бар0: вход (издержки 10 б.п.); бар1: +10%; бар2: +10% и выход (издержки)
    assert res.pnl.iloc[0] == pytest.approx(-0.001)
    assert res.pnl.iloc[1] == pytest.approx(0.10)
    assert res.pnl.iloc[2] == pytest.approx(0.10 - 0.001)


def test_funding_is_paid_by_longs():
    df = make_df([100, 100, 100], funding=np.array([0, 0.0001, 0]))
    res = run(df, np.array([1.0, 1.0, 1.0]), cost_bps=0, bar_min=60)
    assert res.pnl.iloc[1] == pytest.approx(-0.0001)
    res = run(df, np.array([-1.0, -1.0, -1.0]), cost_bps=0, bar_min=60)
    assert res.pnl.iloc[1] == pytest.approx(0.0001)


def test_state_machine_exit_and_timer():
    el = np.array([1, 0, 0, 0, 0, 0], np.bool_)
    z = np.zeros(6, np.bool_)
    xl = np.array([0, 0, 0, 1, 0, 0], np.bool_)
    assert list(state_machine(el, z, xl, z, 0)) == [1, 1, 1, 0, 0, 0]
    assert list(state_machine(el, z, z, z, 2)) == [1, 1, 0, 0, 0, 0]


def test_stop_target_conservative_when_both_hit():
    close = np.array([100, 100, 100.0])
    high = np.array([100, 103, 100.0])
    low = np.array([100, 97, 100.0])
    el = np.array([1, 0, 0], np.bool_)
    z = np.zeros(3, np.bool_)
    pos, ex = stop_target_machine(el, z, high, low, close, np.full(3, 2.0), np.full(3, 2.0), 0)
    assert pos[0] == 1 and pos[1] == 0
    assert ex[1] == pytest.approx(98.0)       # стоп, а не тейк


def test_random_positions_lose_only_costs():
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 20000)))
    df = make_df(close)
    pos = rng.choice([-1.0, 0.0, 1.0], size=len(close))
    res = run(df, pos, cost_bps=5, bar_min=60)
    turnover = np.abs(np.diff(np.concatenate([[0], pos]))).sum()
    gross = res.pnl.sum() + turnover * 5e-4
    assert abs(gross) < 4 * 0.01 * np.sqrt(len(close) * 2 / 3)   # шум, без систематического плюса


def test_lookahead_canary_would_be_obvious():
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 5000)))
    df = make_df(close)
    future = np.sign(np.concatenate([close[1:] / close[:-1] - 1, [0]]))   # подсматривает в будущее
    m = metrics(run(df, future, 5, 60).pnl)
    assert m["sharpe"] > 10


def test_signals_do_not_change_when_future_is_truncated():
    """Сигнал на баре t не должен зависеть от данных после t."""
    rng = np.random.default_rng(2)
    n = 3000
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = make_df(close)
    df["high"] = df["close"] * 1.003
    df["low"] = df["close"] * 0.997
    df["taker_buy_volume"] = rng.uniform(0.2, 0.8, n)
    cut = 2000
    for fn, kw in [(S.donchian, dict(n=20)), (S.tsmom, dict(lookback=24)),
                   (S.meanrev_z, dict(k=3, thr=2.0, hold=6, volwin=100)),
                   (S.vwap_rev, dict(win=24, thr=2.0, hold=24)),
                   (S.flow, dict(k=3, thr=2.0, hold=6, zwin=500)),
                   (S.absorption, dict(k=3, thr=2.0, move=0.5, hold=6, zwin=500, volwin=100))]:
        full, _ = fn(df, **kw)
        part, _ = fn(df.iloc[:cut], **kw)
        assert np.array_equal(full[:cut], part), fn.__name__
