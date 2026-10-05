from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from research.tline import htf_trend, zz_lines
from trader.domain.models import EntryKind, EntryPolicy, Settings, Side, Timeframe, TradePlan
from trader.domain.sizing import position_size
from trader.domain.strategy import detect


def _frame(n: int = 1500, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)) + np.sin(np.arange(n) / 60) * 0.3)
    o = np.r_[c[0], c[:-1]]
    v = rng.uniform(1e3, 2e3, n)
    up = c >= o
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.004, "low": np.minimum(o, c) * 0.996, "close": c,
                         "volume": v, "taker_buy_volume": np.where(up, 0.7, 0.3) * v}, index=idx)


def _expected(d: pd.DataFrame, min_aggr: float, stop_atr: tuple[float, float] = (0.3, 4.0)) -> set[tuple[int, int]]:
    """Правило исследования на полной истории: пробой линии, агрессоры, тренд старшего ТФ, стоп 0.3–4 ATR."""
    from research.smc import _atr
    from research.tline import PIV, last_confirmed, pivots
    trend = htf_trend(d, "4h")
    buy = (d.taker_buy_volume / d.volume).to_numpy()
    a = _atr(d).to_numpy()
    hi, lo, c = d.high.to_numpy(), d.low.to_numpy(), d.close.to_numpy()
    sw_lo, sw_hi = last_confirmed(pivots(lo, PIV, False), len(c)), last_confirmed(pivots(hi, PIV, True), len(c))
    out = set()
    for r in zz_lines(d):
        t, sd = r["t"], r["side"]
        if t < 400 or trend[t] != sd or (buy[t] if sd > 0 else 1 - buy[t]) < min_aggr:
            continue
        sw = sw_lo[t] if sd > 0 else sw_hi[t]
        stop = lo[sw] - 0.1 * a[t] if sd > 0 else hi[sw] + 0.1 * a[t]
        if sw >= 0 and stop_atr[0] <= sd * (c[t] - stop) / a[t] <= stop_atr[1]:
            out.add((t, sd))
    return out


@pytest.mark.parametrize("seed,stop_atr", [(5, (0.3, 4.0)), (11, (0.3, 4.0)), (5, (0.0, 1e9)), (11, (0.0, 1e9))])
def test_detect_matches_research_rule_on_the_last_closed_bar(seed, stop_atr, monkeypatch):
    import trader.domain.strategy as st
    monkeypatch.setattr(st, "STOP_ATR", stop_atr)          # широкий диапазон — сверка всего остального правила
    d = _frame(seed=seed)
    s = Settings(min_aggr=0.55, entry_policy=EntryPolicy.MARKET)
    expected = _expected(d, 0.55, stop_atr)
    candidates = sorted({r["t"] for r in zz_lines(d) if r["t"] >= 400})
    found = set()
    for t in candidates[:40]:
        for x in detect(d.iloc[: t + 1], Timeframe.H4, "X", s):
            found.add((t, int(x.side)))
            assert x.bar_time == d.index[t].to_pydatetime()
            assert x.plan.entry_kind is EntryKind.MARKET and x.plan.entry == pytest.approx(d.close.iloc[t])
            assert int(x.side) * (x.plan.target - x.plan.entry) == pytest.approx(3 * x.plan.risk_per_unit)
    assert found == {e for e in expected if e[0] in set(candidates[:40])}
    if stop_atr[1] > 100:
        assert len(found) >= 2                              # сверка идёт на настоящих сигналах


def test_long_breakout_candle_switches_to_retest_on_the_line():
    d = _frame()
    trend = htf_trend(d, "4h")
    buy = (d.taker_buy_volume / d.volume).to_numpy()
    t, sd, line = next((r["t"], r["side"], r["line_t"]) for r in zz_lines(d) if r["t"] >= 400 and trend[r["t"]] == r["side"]
                       and (buy[r["t"]] if r["side"] > 0 else 1 - buy[r["t"]]) >= 0.55)
    sig = [x for x in detect(d.iloc[: t + 1], Timeframe.H4, "X", Settings(entry_policy=EntryPolicy.HYBRID, hybrid_range_atr=0.5))
           if int(x.side) == sd]
    if sig:                                               # стоп от линии может выйти за 0.3–4 ATR
        assert sig[0].plan.entry_kind is EntryKind.RETEST and sig[0].plan.entry == pytest.approx(line)
        assert sig[0].plan.valid_bars == 12


def test_settings_validation_and_toggle():
    s = Settings()
    assert Timeframe.H1 in s.toggle(Timeframe.H1).timeframes
    assert Timeframe.H4 not in s.toggle(Timeframe.H4).timeframes
    with pytest.raises(ValueError):
        replace(s, risk_pct=10.0)
    assert not Timeframe.M15.has_edge and Timeframe.H4.has_edge


def test_position_size_risk_and_leverage_cap():
    plan = TradePlan(EntryKind.MARKET, entry=100.0, stop=98.0, target=106.0, valid_bars=0)
    z = position_size(1000.0, 1.0, plan)
    assert z.qty == pytest.approx(5.0) and z.risk_usd == pytest.approx(10.0)
    tight = TradePlan(EntryKind.MARKET, entry=100.0, stop=99.99, target=100.03, valid_bars=0)
    z2 = position_size(1000.0, 1.0, tight, max_leverage=5.0)
    assert z2.notional == pytest.approx(5000.0) and z2.risk_usd < 10.0
