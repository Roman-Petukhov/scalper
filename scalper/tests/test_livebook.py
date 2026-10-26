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
