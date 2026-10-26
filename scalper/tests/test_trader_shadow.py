import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from research.tline import MAKER, TAKER, zz_lines
from trader.application.shadow import ShadowTracker
from trader.domain import shadow as sh
from trader.domain import strategy
from trader.domain.models import EntryKind, EntryPolicy, Settings, Side, Signal, Timeframe, TradePlan
from trader.domain.strategy import detect
from trader.infrastructure.sqlite_repo import SqliteStore

from test_trader_domain import _frame

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _long_bar(seed=5):
    """(кадр, номер свечи), где при пороге 0.5 есть лонг; порог бота по умолчанию 0.55."""
    d = _frame(seed=seed)
    low = Settings().with_tf(Timeframe.H4, min_aggr=0.5, entry_policy=EntryPolicy.MARKET, min_close_loc=0.0)
    for t in sorted({r["t"] for r in zz_lines(d, log=strategy.LOG_LINES) if r["t"] >= 400}):
        if any(x.side is Side.LONG for x in detect(d.iloc[: t + 1], Timeframe.H4, "X", low)):
            return d, t
    raise AssertionError("в кадре нет лонга")


def _with_buy(d, t, share):
    d = d.copy()
    d.iloc[t, d.columns.get_loc("taker_buy_volume")] = share * d["volume"].iloc[t]
    return d


@pytest.mark.parametrize("seed", [5, 11, 3, 7])
def test_candidates_are_longs_between_the_shadow_and_the_bot_threshold(seed):
    try:
        d, t = _long_bar(seed)
    except AssertionError:
        pytest.skip("нет лонга в этом кадре")
    s = Settings().with_tf(Timeframe.H4, entry_policy=EntryPolicy.MARKET, min_close_loc=0.0)       # порог бота 0.55
    got = sh.candidates(_with_buy(d, t, 0.535).iloc[: t + 1], Timeframe.H4, "X", s)
    assert [k for k, _ in got] == [sh.AGGR525] and got[0][1].side is Side.LONG and 0.525 <= got[0][1].aggr < 0.55
    assert sh.candidates(_with_buy(d, t, 0.60).iloc[: t + 1], Timeframe.H4, "X", s) == []        # бот и так покажет
    assert sh.candidates(_with_buy(d, t, 0.51).iloc[: t + 1], Timeframe.H4, "X", s) == []        # ниже теневого порога
    assert sh.candidates(_with_buy(d, t, 0.535).iloc[: t + 1], Timeframe.M15, "X", s) == []      # только 4h


def _sig(kind=EntryKind.MARKET, entry=100.0, stop=95.0, valid=0):
    plan = TradePlan(kind, entry, stop, entry + 3 * (entry - stop), valid)
    return Signal("X", Timeframe.H4, Side.LONG, T0, entry, entry, ((T0, entry), (T0, entry)), 0.53, 1.0, plan,
                  extra={"spring": True})


def _bars(closes, lows=None, highs=None):
    idx = pd.DatetimeIndex([T0 + timedelta(hours=4 * i) for i in range(len(closes))])
    c = np.asarray(closes, float)
    return pd.DataFrame({"open": np.r_[c[0], c[:-1]], "high": highs if highs is not None else c + 0.1,
                         "low": lows if lows is not None else c - 0.1, "close": c}, index=idx)


S = Settings()
NOW = T0 + timedelta(days=60)


def test_resolve_target_stop_and_open_positions():
    win = sh.resolve(_sig(), _bars([100, 102, 108, 116, 117]), S, NOW)
    assert win.state == "closed" and win.r == pytest.approx((15 - (TAKER + MAKER) * 100) / 5)
    assert win.closed_at == T0 + timedelta(hours=4 * 3 + 4)
    loss = sh.resolve(_sig(), _bars([100, 99, 94, 94.5]), S, NOW)
    assert loss.state == "closed" and loss.r < -1.0 and loss.r == pytest.approx(
        (min(95, 99) - 100 - 100 * (TAKER + TAKER) + 0) / 5, abs=0.3)
    assert sh.resolve(_sig(), _bars([100, 101, 102]), S, NOW) is None           # ни цели, ни стопа, срок не вышел


def test_resolve_retest_needs_a_fill_and_expires():
    sig = _sig(EntryKind.RETEST, entry=100.0, stop=95.0, valid=3)
    up = _bars([102, 103, 104, 105, 106], lows=[101.9, 102.9, 103.9, 104.9, 105.9])
    assert sh.resolve(sig, up, S, T0 + timedelta(hours=8)) is None              # лимитка ещё живёт
    out = sh.resolve(sig, up, S, sig.valid_until() + timedelta(minutes=1))
    assert out.state == "unfilled" and out.r is None
    touch = _bars([102, 100.5, 108, 116, 117], lows=[101.9, 99.9, 107.9, 115.9, 116.9],
                  highs=[102.1, 100.6, 108.1, 116.1, 117.1])
    got = sh.resolve(sig, touch, S, NOW)
    assert got.state == "closed" and got.r > 2.5                                 # вход на 100, цель 115


def test_store_roundtrip_dedup_and_stats(tmp_path):
    st = SqliteStore(tmp_path / "t.db")
    a, b = _sig(), _sig()
    b = Signal(**{**b.__dict__, "symbol": "Y", "extra": {}})
    assert st.add_shadow(sh.AGGR525, a) and not st.add_shadow(sh.AGGR525, a) and st.add_shadow(sh.AGGR525, b)
    opened = st.open_shadow(Timeframe.H4)
    assert [x[2].symbol for x in opened] == ["X", "Y"] and opened[0][2].extra["spring"] and opened[0][2].plan.stop == 95
    st.finish_shadow(opened[0][0], "closed", 2.9, T0 + timedelta(hours=8))
    st.finish_shadow(opened[1][0], "closed", -1.1, T0 + timedelta(hours=12))
    assert st.open_shadow(Timeframe.H4) == []
    rows = st.shadow_rows()
    stats = sh.shadow_stats(rows)[0]
    assert (stats.total, stats.closed, stats.open) == (2, 2, 0) and stats.avg_r == pytest.approx(0.9)
    assert stats.spring_closed == 1 and stats.spring_avg_r == pytest.approx(2.9) and not stats.enough and stats.need == 18


class _Market:
    def __init__(self, bars):
        self.bars = bars

    async def bars_between(self, symbol, tf, start, end):
        return self.bars


def test_tracker_closes_resolved_signals_only(tmp_path):
    st = SqliteStore(tmp_path / "t.db")
    st.add_shadow(sh.AGGR525, _sig())
    long_ago = Signal(**{**_sig().__dict__, "symbol": "Z", "bar_time": T0 + timedelta(days=20)})
    st.add_shadow(sh.AGGR525, long_ago)
    bars = _bars([100, 102, 108, 116, 117])
    n = asyncio.run(ShadowTracker(_Market(bars), st, st).update(Timeframe.H4))
    assert n == 1 and [x[2].symbol for x in st.open_shadow(Timeframe.H4)] == ["Z"]
    row = [r for r in st.shadow_rows() if r.signal.symbol == "X"][0]
    assert row.state == "closed" and row.r > 2.8 and row.closed_at is not None
