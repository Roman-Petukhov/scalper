import numpy as np
import pandas as pd

from research.tline import signals, two_targets


def test_two_targets_ladder_and_breakeven():
    # лонг 100, стоп 98 (R=2): первая цель 106, вторая 110
    o = np.array([100, 101, 105, 104, 101.0])
    h = np.array([100, 103, 107, 105, 102.0])
    lo = np.array([100, 100, 104, 103, 99.5])
    c = np.array([100, 102, 105, 104, 100.0])
    r, ex = two_targets(o, h, lo, c, np.zeros(5), 0, 1, 100.0, 98.0, 3.0, 5.0, True, 100, 0.0)
    # половина на 106 (+1.5R), остаток выбит в безубыток на 100
    assert ex == 4 and np.isclose(r, 0.5 * (6 - 2e-4 * 100) / 2 + 0.5 * (0 - 5.5e-4 * 100) / 2)


def test_signal_on_descending_line_of_closes():
    c = np.array([8, 8, 9, 9, 10, 11, 12, 15, 12, 11, 10, 9, 10, 11, 13, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15, 16.0])
    d = pd.DataFrame({"close": c})
    sig = [s for s in signals(d) if s[2] == 1]
    assert sig, "нет пробоя нисходящей линии"
    t, tc, side, line_b, line_c, i1, i2 = sig[0]
    assert (i1, i2) == (7, 14)
    # линия через закрытия 15 (бар 7) и 13 (бар 14): наклон −2/7; пробой на баре 22, закрепление на 23
    assert (t, tc) == (22, 23)
    assert np.isclose(line_b, 13 - 2 / 7 * 8) and c[t] > line_b and c[t - 1] <= 13 - 2 / 7 * 7


def test_zigzag_confirms_on_k_atr_reversal():
    from research.tline import zigzag
    c = np.array([10, 11, 12, 13, 12.5, 11, 9, 9.5, 10, 12, 13, 14.0])
    hs, ls = zigzag(c, np.ones(len(c)), 2.0)
    # стартовая впадина 10 (бар 0) подтверждена закрытием 12 (бар 2); вершина 13 (бар 3) — закрытием 11 (бар 5);
    # впадина 9 (бар 6) — закрытием 12 (бар 9); откат 12.5 от 13 меньше 2 ATR — не вершина
    assert hs.tolist() == [[3, 5]] and ls.tolist() == [[0, 2], [6, 9]]


def test_zz_line_skips_nearby_minor_touch():
    from research import tline
    from research.tline import zz_lines
    rng = np.random.default_rng(0)
    base = np.concatenate([np.linspace(50, 100, 80), np.linspace(100, 80, 15), np.linspace(80, 95, 10),   # A = 100 (бар 79), рядом мелкий откат до 95
                           np.linspace(95, 70, 30), np.linspace(70, 90, 20),                               # B = 90 (бар 154): далеко от A
                           np.linspace(90, 72, 20), np.linspace(72, 95, 25)])
    c = base + rng.normal(0, 0.01, len(base))
    d = pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c})
    old = tline.ZZ_ANCHOR
    tline.ZZ_ANCHOR = 30
    try:
        recs = [r for r in zz_lines(d) if r["side"] == 1 and r["a"] == 79]
    finally:
        tline.ZZ_ANCHOR = old
    assert recs and recs[0]["b"] >= 79 + tline.ZZ_SPAN and recs[0]["t"] > recs[0]["b"]


def test_fan_line_through_adjacent_lower_highs():
    from research.tline import fan_lines
    seg = [np.linspace(60, 100, 40), np.linspace(100, 80, 20), np.linspace(80, 95, 15), np.linspace(95, 75, 20),
           np.linspace(75, 90, 15), np.linspace(90, 70, 20), np.linspace(70, 100, 30)]
    c = np.concatenate([seg[0]] + [x[1:] for x in seg[1:]]) + np.random.default_rng(1).normal(0, 0.01, 154)
    d = pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c})
    recs = [r for r in fan_lines(d) if r["side"] == 1 and r["k"] == 3.0]
    tops = [39, 72, 105]                                             # вершины 100, 95, 90
    first = [r for r in recs if (r["a"], r["b"]) == (tops[0], tops[1])]
    assert first and first[0]["t"] > tops[1] and c[first[0]["t"]] > first[0]["line_t"]
    # более крутая линия 95 -> 90 пробита на том же баре, что и 100 -> 95, — сигнал не дублируется
    assert len({(r["t"], r["side"]) for r in recs}) == len(recs)


def test_higher_tf_trend_and_structure_use_only_closed_bars():
    from research.tline import htf_structure, htf_trend
    idx = pd.date_range("2024-01-01", periods=24 * 200, freq="1h", tz="UTC")
    c = 100 + np.cumsum(np.random.default_rng(3).normal(0, 0.5, len(idx))) + np.sin(np.arange(len(idx)) / 200) * 20
    d = pd.DataFrame({"open": c, "high": c + 0.3, "low": c - 0.3, "close": c}, index=idx)
    cut = 24 * 150 + 7
    for f, args in ((htf_trend, ("1h", "1D")), (htf_structure, ("1h",)), (htf_trend, ("1h",))):
        full, part = f(d, *args), f(d.iloc[:cut], *args)
        assert np.array_equal(full[:cut], part, equal_nan=True)     # будущее не меняет прошлые значения
