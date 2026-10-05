import numpy as np
import pandas as pd

from research.smc import ladder, order_blocks, swings


def test_swings_confirm_after_three_bars():
    hi = np.array([1, 2, 3, 9, 3, 2, 1, 1, 1.0])
    sh, _ = swings(hi, hi - 0.5)
    assert np.isnan(sh[5]) and sh[6] == 9


def test_ladder_fills_and_hits_target():
    # лонг: лимитки 100 / 99 / 98, стоп 97 → средняя 99, риск 2, тейк 109
    o = np.array([105, 104, 100, 101, 105, 110.0])
    h = np.array([106, 105, 101, 104, 108, 112.0])
    lo = np.array([104, 100.5, 98.5, 100, 104, 108.0])
    c = np.array([105, 101, 100, 103, 107, 111.0])
    r, nf, ex = ladder(o, h, lo, c, np.zeros(6), 0, 1, 100.0, 99.0, 98.0, 97.0, 48, 120)
    assert nf == 2 and ex == 5
    legs = np.array([100.0, 99.0])
    expect = ((109 - legs) - 4e-4 * legs).sum() / 2 / 3
    assert np.isclose(r, expect)


def test_ladder_stop_in_fill_bar_and_missed_setup():
    o = np.array([105, 104.0])
    h = np.array([106, 104.0])
    lo = np.array([104, 96.0])
    c = np.array([105, 97.0])
    r, nf, _ = ladder(o, h, lo, c, np.zeros(2), 0, 1, 100.0, 99.0, 98.0, 97.0, 48, 120)
    assert nf == 3 and r < -1
    r, nf, _ = ladder(np.array([105, 110.0]), np.array([106, 120.0]), np.array([104, 109.0]), np.array([105, 119.0]),
                      np.zeros(2), 0, 1, 100.0, 99.0, 98.0, 97.0, 48, 120)
    assert np.isnan(r) and nf == 0


def test_order_block_with_sweep():
    # свинг-лоу 95, пролив до 93 (сбор стопов), медвежья свеча на дне, затем импульс выше свинг-хая 105
    c = [100, 102, 105, 103, 100, 96, 95, 97, 100, 101, 99, 96, 93.5, 97, 101, 104, 108]
    o = [99, 100, 102, 105, 103, 100, 96, 95, 97, 100, 101, 99, 96, 93.5, 97, 101, 104]
    h = [max(a, b) + 0.5 for a, b in zip(o, c)]
    lo = [min(a, b) - 0.5 for a, b in zip(o, c)]
    lo[12] = 93.0
    d = pd.DataFrame({"open": o, "high": h, "low": lo, "close": c}, dtype="float64")
    obs = order_blocks(d, np.full(len(c), 2.0), with_sweep=True)
    bull = [x for x in obs if x[1] == 1]
    assert bull and bull[-1][4] is True
    assert bull[-1][2] == 93.0                                   # зона — медвежья свеча на дне пролива


def test_fade_and_trap():
    from research.smc import fade, trap
    # бычья зона 98–100, стоп SMC 97.5: цена падает в зону и дальше — шорт-фейд по 100 берёт 1R = 2.5 до 97.5
    # бар касания 103 → 99.5 (максимум 103 был до входа и стопом не считается), дальше падение к 97.5
    o = np.array([105, 103, 99, 97.0])
    h = np.array([106, 103, 100, 98.0])
    lo = np.array([104, 99.5, 97, 96.0])
    c = np.array([105, 100, 97.5, 96.5])
    r = fade(o, h, lo, c, np.zeros(4), 0, 1, 100.0, 97.5, 48, 1.0, 120)
    assert np.isclose(r, (2.5 - (5.5e-4 + 2e-4) * 100) / 2.5)
    # ловушка: пробой 97.5 до 96, возврат закрытием выше 98, затем рост к тейку 3R
    o = np.array([105, 99, 96.5, 98.5, 100, 108.0])
    h = np.array([106, 99.5, 97, 99.0, 104, 110.0])
    lo = np.array([104, 98.5, 96, 97.5, 98.6, 103.0])
    c = np.array([105, 99, 96.5, 98.5, 103, 109.0])
    r = trap(o, h, lo, c, np.zeros(6), 0, 1, 98.0, 97.5, 1.0, 48, 6, 3.0, 120)
    d = 98.5 - (96 - 0.1)
    assert np.isclose(r, (3 * d - (5.5e-4 + 2e-4) * 98.5) / d)


def test_series_number_resets_after_sweep_and_opposite_signal():
    from research.smc import series_number
    t = pd.date_range("2024-01-01", periods=7, freq="1h", tz="UTC")
    df = pd.DataFrame({"symbol": "X", "t": t, "kind": ["FVG", "FVG", "FVG", "OB+SWEEP", "FVG", "FVG", "FVG"],
                       "side": [1, 1, 1, 1, 1, -1, 1]})
    assert list(series_number(df)) == [1, 2, 3, 1, 1, 1, 1]
