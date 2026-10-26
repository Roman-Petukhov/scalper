import numpy as np
import pandas as pd
import pytest

from research import emagrid as G
from research.tline import MAKER, TAKER


def _run(c_path, ema=100.0, atr=1.0, entry=0, exit_=0, size="×1", k=1.0, gate=True):
    """5m-свечи без теней: каждая свеча — шаг цены от прошлого закрытия к новому."""
    c = np.asarray(c_path, dtype=np.float64)
    o = np.r_[c[0], c[:-1]]
    h, lo = np.maximum(o, c), np.minimum(o, c)
    n = len(c)
    g = np.full(n, gate)
    pnl, fee = G.simulate(o, h, lo, c, np.zeros(n), np.zeros(n, dtype=np.int64), np.full(n, ema), np.full(n, atr),
                          g, g, 1, entry, exit_, G.size_table(size), k)
    return pnl[0], fee[0]


def test_size_tables_are_normalised_martingale_steps():
    assert G.size_table("×2") == pytest.approx(np.array([1, 2, 4, 8]) / 15)
    assert G.size_table("n+1") == pytest.approx(np.array([1, 2, 3, 4]) / 10)
    assert G.size_table("×1").sum() == pytest.approx(1.0)


def test_limit_grid_buys_the_pullback_and_takes_the_next_line():
    # над EMA 100, шаг 1: линии 100.5 / 101.5 / 102.5 / 103.5; цена 102 → 101.5 (лимитка) → 102.5 (тейк)
    pnl, fee = _run([102.0, 101.5, 102.5], entry=0, exit_=0)
    q = 0.25
    want = q * (1.0 / 101.5) - MAKER * q - MAKER * q * 102.5 / 101.5
    assert pnl == pytest.approx(want) and fee > 0


def test_touching_the_ema_closes_everything_at_a_loss():
    # куплено на 101.5 и 100.5, затем касание EMA 100 — закрыто всё по 100 (taker)
    pnl, _ = _run([102.0, 101.5, 100.5, 99.0], entry=0, exit_=2)
    q = 0.25
    want = (q * (100 - 101.5) / 101.5 + q * (100 - 100.5) / 100.5 - 2 * MAKER * q
            - TAKER * q * 100 / 101.5 - TAKER * q * 100 / 100.5)
    assert pnl == pytest.approx(want)


def test_stop_grid_pyramids_with_the_trend_only():
    # «стоп»: покупки вдогонку на 100.5 / 101.5 / 102.5 при росте; на откате лимиток нет
    pnl, _ = _run([100.2, 103.0, 102.0], entry=1, exit_=2)
    q = 0.25
    unreal = sum(q * (102.0 - x) / x for x in (100.5, 101.5, 102.5))
    assert pnl == pytest.approx(unreal - 3 * TAKER * q)


def test_gate_blocks_new_parts():
    pnl, fee = _run([103.0, 101.5, 102.5], entry=0, exit_=0, gate=False)
    assert pnl == 0 and fee == 0


def test_tf_view_uses_only_closed_bars():
    idx = pd.date_range("2024-01-01", periods=12 * 24 * 30, freq="5min", tz="UTC")
    c = np.linspace(100, 130, len(idx))
    b5 = pd.DataFrame({"open": c, "high": c + 0.1, "low": c - 0.1, "close": c}, index=idx)
    v = G.tf_view(b5, "1h")
    first = v["ema"].first_valid_index()
    assert first == idx[0] + pd.Timedelta(hours=200)               # EMA200 по часам — после закрытия 200-го часа


def test_basket_take_closes_all_above_the_average():
    pnl, _ = _run([102.0, 100.6, 102.5], entry=0, exit_=1)
    q = 0.25
    assert pnl == pytest.approx(q / 101.5 - MAKER * q - MAKER * q * 102.5 / 101.5)


def test_trailing_closes_two_steps_off_the_best_price():
    pnl, _ = _run([100.2, 103.0, 100.9], entry=1, exit_=3)
    q = 0.25
    want = sum(q * (101.0 - x) / x - TAKER * q - TAKER * q * 101.0 / x for x in (100.5, 101.5, 102.5))
    assert pnl == pytest.approx(want)
