import numpy as np
import pandas as pd
import pytest

from research.margsim import E0, MAX_POS, Cfg, max_dd, monthly, simulate, util_stats
from research.tline import TAKER

MARGIN10 = Cfg("m10", True, 10.0, 10, False)
MARGIN10_HEDGE = Cfg("m10h", True, 10.0, 10, True)
MARGIN50 = Cfg("m50", True, 50.0, 10, False)
RISK1 = Cfg("r1", False, 1.0, 5, False)
NAN = np.nan


def trades(rows: list[tuple]) -> pd.DataFrame:
    """rows: (t_in, t_out, side, R3, risk_pct, beta, btc_ret)."""
    df = pd.DataFrame(rows, columns=["t_in", "t_out", "side", "R3", "risk_pct", "beta", "btc_ret"])
    return df.assign(t_in=pd.to_datetime(df.t_in, utc=True), t_out=pd.to_datetime(df.t_out, utc=True))


def test_margin_mode_notional_is_margin_times_leverage():
    # маржа 10% × 10× = номинал в капитал; цель 3R при стопе 2% цены → +6% капитала, стоп стоит 2% капитала
    res = simulate(trades([("2024-01-01", "2024-01-02", -1, 3.0, 0.02, NAN, NAN)]), MARGIN10)
    assert res.eq.iloc[-1] == pytest.approx(E0 * 1.06)
    assert res.loss_at_stop[0] == pytest.approx(2.0)


def test_risk_mode_loss_at_stop_is_fixed():
    res = simulate(trades([("2024-01-01", "2024-01-02", -1, -1.0, 0.04, NAN, NAN)]), RISK1)
    assert res.eq.iloc[-1] == pytest.approx(E0 * 0.99)
    assert res.loss_at_stop[0] == pytest.approx(1.0)


def test_risk_mode_notional_capped_by_leverage():
    # стоп 0.1%: по риску 1% нужен номинал 10× капитала; плечо 5× и запас маржи 5% оставляют 4.75× → стоп стоит 0.475%
    res = simulate(trades([("2024-01-01", "2024-01-02", -1, -1.0, 0.001, NAN, NAN)]), RISK1)
    assert res.loss_at_stop[0] == pytest.approx(0.475)


def test_hedge_pnl_includes_fees_and_sign():
    # шорт, R3 = 0, BTC упал на 5%: лонг BTC на бету 1 × номинал (1000) теряет 50 и платит комиссии на две стороны
    row = [("2024-01-01", "2024-01-02", -1, 0.0, 0.02, 1.0, -0.05)]
    hedged = simulate(trades(row), MARGIN10_HEDGE)
    assert hedged.eq.iloc[-1] == pytest.approx(E0 - 50 - 2 * TAKER * 1000)
    assert simulate(trades(row), MARGIN10).eq.iloc[-1] == pytest.approx(E0)


def test_hedge_margin_cuts_position():
    # маржа 50% × 10× = номинал 5× капитала; хедж с бетой 1 удваивает потребность в марже: размер = 95% × 10 / 2 = 4.75×
    row = [("2024-01-01", "2024-01-02", -1, 1.0, 0.01, 1.0, 0.0)]
    res = simulate(trades(row), Cfg("m50h", True, 50.0, 10, True))
    assert res.loss_at_stop[0] == pytest.approx(4.75)
    assert res.cut == 0                                           # 4.75 ≥ 90% от желаемых 5.0 — это не «урезание»


def test_second_trade_cut_by_free_margin():
    rows = [("2024-01-01 00:00", "2024-01-05", -1, 1.0, 0.01, NAN, NAN),
            ("2024-01-01 04:00", "2024-01-05", -1, 1.0, 0.01, NAN, NAN),
            ("2024-01-01 08:00", "2024-01-05", -1, 1.0, 0.01, NAN, NAN)]
    res = simulate(trades(rows), MARGIN50)
    # первая — 50% маржи; вторая — свободно 50% × 95% × 10 = 4.75× (не урезана); третья — свободно 2.5% → урезана
    assert res.opened == 3 and res.cut == 1 and res.peak_margin <= 1.0


def test_daily_loss_blocks_new_entries_until_next_day():
    rows = [("2024-01-01 00:00", "2024-01-01 04:00", -1, -1.0, 0.05, NAN, NAN),   # −5% капитала
            ("2024-01-01 08:00", "2024-01-02 00:00", -1, 1.0, 0.05, NAN, NAN),    # тот же день → отказ
            ("2024-01-02 04:00", "2024-01-02 08:00", -1, 1.0, 0.05, NAN, NAN)]    # новый день → можно
    res = simulate(trades(rows), MARGIN10)
    assert (res.opened, res.rej_day) == (2, 1)


def test_position_limit():
    rows = [(f"2024-01-01 {h:02d}:00", "2024-01-09", -1, 1.0, 0.01, NAN, NAN) for h in range(MAX_POS + 3)]
    res = simulate(trades(rows), Cfg("tiny", True, 1.0, 10, False))
    assert (res.opened, res.rej_pos, res.peak_open) == (MAX_POS, 3, MAX_POS)


def test_exit_before_entry_frees_slot_and_compounds():
    rows = [("2024-01-01", "2024-01-02", -1, 3.0, 0.02, NAN, NAN),                # +6%
            ("2024-01-02", "2024-01-03", -1, 3.0, 0.02, NAN, NAN)]                # выход ровно в момент входа второй
    res = simulate(trades(rows), MARGIN10)
    assert res.eq.iloc[-1] == pytest.approx(E0 * 1.06 * 1.06)


def test_drawdown_and_monthly():
    eq = pd.Series([1100.0, 880.0, 990.0], index=pd.to_datetime(["2024-01-10", "2024-02-10", "2024-03-10"], utc=True))
    assert max_dd(eq) == pytest.approx(20.0)                                        # 1100 → 880
    assert monthly(eq).round(1).tolist() == [10.0, -20.0, 12.5]


def test_margin_utilization_over_time():
    # сутки под маржой 10% капитала (номинал = капитал, плечо 10×) → среднее 10%, выше 50% не было
    res = simulate(trades([("2024-01-01", "2024-01-02", -1, 0.0, 0.02, NAN, NAN)]), MARGIN10)
    mean, over50, over80 = util_stats(res)
    assert (round(mean, 3), over50, over80) == (0.1, 0.0, 0.0)


def test_hedge_margin_share_and_btc_leverage():
    # хедж с бетой 1 на номинал 1000: при плече BTC 10× — маржа 100 к 100 сделки (загрузка 20%), при 25× — 40 (14%)
    row = [("2024-01-01", "2024-01-02", -1, 0.0, 0.02, 1.0, 0.0)]
    same = simulate(trades(row), MARGIN10_HEDGE)
    wide = simulate(trades(row), Cfg("m10h25", True, 10.0, 10, True, hedge_lev=25))
    assert same.peak_margin == pytest.approx(0.2)
    assert wide.peak_margin == pytest.approx(0.14)


def test_hedge_ratio_scales_hedge_and_max_pos_cfg():
    row = [("2024-01-01", "2024-01-02", -1, 0.0, 0.02, 1.0, -0.05)]
    half = simulate(trades(row), Cfg("m10h50", True, 10.0, 10, True, hedge_ratio=0.5)).eq.iloc[-1]
    assert half == pytest.approx(E0 - 25 - 2 * TAKER * 500)
    rows = [(f"2024-01-01 {h:02d}:00", "2024-01-09", -1, 1.0, 0.01, NAN, NAN) for h in range(5)]
    res = simulate(trades(rows), Cfg("two", True, 1.0, 10, False, max_pos=2))
    assert (res.opened, res.rej_pos) == (2, 3)


def test_hedge15_result_in_r():
    from research.hedge15 import hedge_r
    # шорт, BTC упал на 1%: хедж (лонг BTC, бета 1.5) теряет 1.5%, минус комиссии; стоп 1% → в R делим на 0.01
    r = hedge_r(np.array([-1.0]), np.array([1.5]), np.array([-0.01]), np.array([0.01]))[0]
    assert r == pytest.approx((-0.015 - 2 * TAKER * 1.5) / 0.01)
