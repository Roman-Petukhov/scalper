import numpy as np
import pandas as pd
import pytest

from research import livebook as L


def _z(rows):
    """rows: (tf, час входа, часов в сделке, R, стоп в долях цены)."""
    t0 = pd.Timestamp("2023-01-02")
    fill = [t0 + pd.Timedelta(hours=h) for _, h, _, _, _ in rows]
    return pd.DataFrame({"tf": [r[0] for r in rows], "fill": fill,
                         "exit": [f + pd.Timedelta(hours=r[2]) for f, r in zip(fill, rows)],
                         "R": [r[3] for r in rows], "stop_pct": [r[4] for r in rows], "side": -1,
                         "t": fill, "slope_atr": 0.01, "squeeze": 1.0})


def test_ladder_steps_on_15m_takes_only():
    z = _z([("15m", 0, 1, 3.0, 0.03), ("4h", 2, 1, 3.0, 0.03), ("15m", 4, 1, 3.0, 0.03), ("15m", 6, 1, -1.0, 0.03)])
    s = L.simulate_book(z, L.Book("b", r15=0.01, r4h=0.01, lev15=100, lev4h=100))
    assert s["eq"] == pytest.approx(1.03 * 1.03 * 1.045 * (1 - 0.0225))   # 4h ступень не двигает


def test_margin_rejects_like_bybit_110007():
    z = _z([("15m", 0, 10, 1.0, 0.01), ("15m", 1, 10, 1.0, 0.01), ("15m", 2, 10, 1.0, 0.01)])
    s = L.simulate_book(z, L.Book("b", r15=0.04, ladder_x=1.0, lev15=10))   # номинал 4 капитала, маржа 40% счёта
    assert s["skip"]["маржа"] == 1 and s["eq"] == pytest.approx(1.08)


def test_limits_and_heat_cap():
    z = _z([("15m", h, 10, -1.0, 0.03) for h in range(5)])
    assert L.simulate_book(z, L.Book("b", r15=0.02, ladder_x=1.0, limit15=3))["skip"]["лимит 15m"] == 2
    s = L.simulate_book(z, L.Book("b", r15=0.02, ladder_x=1.0, heat=0.05))
    assert s["skip"]["потолок риска"] == 3 and s["heat_max"] == pytest.approx(0.04)


def test_daily_stop_halts_new_entries():
    z = _z([("15m", h, 0.5, -1.0, 0.03) for h in range(6)])
    s = L.simulate_book(z, L.Book("b", r15=0.06, ladder_x=1.0, lev15=100))
    assert s["skip"]["дневной стоп"] == 3                                    # три стопа по 6% — −17%, дальше стоп


def test_week_blocks_keep_trades_of_a_week_together():
    z = _z([("15m", 24 * d + h, 1, float(d), 0.03) for d in range(21) for h in (0, 1)])
    q = L.week_blocks(z, np.random.default_rng(1))
    assert len(q) == len(z)
    f = q.fill.sort_values().to_numpy()
    pair = (f[1::2] - f[0::2]) / np.timedelta64(1, "h")
    assert (pair == 1.0).all() and (q.sort_values("fill").R.to_numpy()[0::2] == q.sort_values("fill").R.to_numpy()[1::2]).all()
    assert q.fill.min() == z.fill.min()                                      # ряд начинается с той же недели


def test_book_filters_15m_and_boosts_squeezed_4h():
    z = _z([("15m", 0, 1, 3.0, 0.03), ("4h", 2, 1, 3.0, 0.03), ("15m", 4, 1, 3.0, 0.03)])
    z.loc[1, "squeeze"] = 0.4
    z.loc[2, "side"] = 1
    s = L.simulate_book(z, L.Book("b", r15=0.01, ladder_x=1.0, r4h=0.01, squeeze4h=1.5, sides15="short"))
    assert s["eq"] == pytest.approx(1.03 * 1.045)                            # лонг 15m отброшен, 4h ×1.5
    assert L.simulate_book(z, L.Book("b", r15=0.0, r4h=0.01))["eq"] == pytest.approx(1.03)   # только 4h


def test_long_risk_multiplier_only_scales_15m_longs():
    z = _z([("15m", 0, 1, 1.0, 0.03), ("15m", 5, 1, 1.0, 0.03)])
    z["side"] = [1, -1]
    s = L.simulate_book(z, L.Book("b", r15=0.02, ladder_x=1.0, sides15="both", long_x=0.5, lev15=100))
    assert s["eq"] == pytest.approx(1.01 * 1.02)


def test_book_without_4h_drops_4h_trades():
    z = _z([("15m", 0, 1, 1.0, 0.03), ("4h", 5, 1, 1.0, 0.03)])
    s = L.simulate_book(z, L.Book("b", r15=0.02, r4h=0.0, ladder_x=1.0, lev15=100))
    assert s["eq"] == pytest.approx(1.02)


def test_haircut_halves_the_mean_per_timeframe_and_keeps_spread():
    z = _z([("15m", 0, 1, 3.0, 0.03), ("15m", 2, 1, -1.0, 0.03), ("4h", 4, 1, 2.0, 0.03), ("4h", 6, 1, 0.0, 0.03)])
    b = L.Book("b", r15=0.01, r4h=0.01, sides15="short", slope15=0.025)
    R = L.haircut(z, b, 0.5)
    assert R[:2].mean() == pytest.approx(0.5) and R[2:].mean() == pytest.approx(0.5)
    assert R[0] - R[1] == pytest.approx(4.0) and np.array_equal(L.haircut(z, b, 1.0), z.R.to_numpy())


def test_goal_stats_counts_the_hit_inside_the_window():
    t0 = pd.Timestamp("2023-01-01")
    curve = pd.Series([2.0, 0.9, 40.0, 50.0], index=[t0 + pd.Timedelta(days=d) for d in (100, 200, 600, 900)])
    x, hit, years, dd = L.goal_stats(curve, t0, 33.3, 2.0)
    assert x == 40.0 and hit and years == pytest.approx(600 / 365.25) and dd == pytest.approx(0.55)
    assert not L.goal_stats(curve, t0, 45.0, 2.0)[1]


def test_rolling_windows_measure_growth_from_each_month():
    idx = pd.date_range("2022-01-15", periods=40, freq="MS") + pd.Timedelta(days=14)
    curve = pd.Series(2.0 ** (np.arange(40) / 12), index=idx)
    w = L.rolling_windows(curve, 2.0)
    assert len(w) == 16 and np.allclose(w, 4.0)
