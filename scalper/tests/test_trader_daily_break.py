import asyncio
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from research import htftrend
from trader.application.daily_break import DailyBreakPaper
from trader.domain.daily_break import RULES, DailySetup, Status, advance, book, find, funding_r
from trader.infrastructure.sqlite_repo import SqliteStore

RULE = RULES[0]


def _frame(seed: int, n: int = 800, end: str | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = (pd.date_range(end=end, periods=n, freq="1D", tz="UTC") if end
           else pd.date_range("2022-01-01", periods=n, freq="1D", tz="UTC"))
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.04, n)))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.01, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": 1.0, "quote_volume": 3e7,
                         "taker_buy_volume": rng.uniform(0.4, 0.6, n)}, index=idx)


def _research(d: pd.DataFrame) -> list[dict]:
    liq = d["quote_volume"].rolling(30, min_periods=20).mean().shift(1).to_numpy()
    return [r for r in htftrend.coin_rows(d, liq, "X", "perp") if r["t"] >= d.index[400]]


@pytest.mark.parametrize("seed", [0, 6])
def test_setups_and_results_match_research_backtest(seed):
    d = _frame(seed)
    rows = _research(d)
    mine = [s for t in range(400, len(d) - 1) for s in find(d.iloc[:t + 1], "X", RULE)]
    assert {(s.day, s.side) for s in mine} == {(r["t"].date(), r["side"]) for r in rows} and rows
    checked = 0
    for r in rows:
        s = next(x for x in mine if x.day == r["t"].date() and x.side == r["side"])
        assert s.stop == pytest.approx(s.level - s.side * 1.5 * s.atr)
        a = advance(d, s, RULE)
        want = r["retest|1.5 ATR|3R"]
        if a.status is Status.CLOSED:
            assert a.r == pytest.approx(want, abs=1e-12)
            checked += 1
        elif a.status is Status.CANCELLED:
            assert not np.isfinite(want)
    assert checked >= 2


def _setup(**kw) -> DailySetup:
    base = dict(rule=RULE.key, symbol="X", side=1, day=date(2026, 1, 1), level=100.0, stop=97.0, atr=2.0, aggr=0.5,
                slope_atr=0.02)
    return DailySetup(**(base | kw))


def _bars(rows: list[tuple[float, float, float, float]], start: str = "2026-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(rows), freq="1D", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def test_advance_pending_cancel_open_and_close():
    s = _setup()
    flat = [(101.0, 102.0, 100.5, 101.5)] * 5
    assert advance(_bars(flat), s, RULE).status is Status.PENDING                    # ретеста ещё нет
    assert advance(_bars([flat[0]] * 14), s, RULE).status is Status.CANCELLED          # 12 дней без ретеста
    ran = [flat[0], (101.0, 110.0, 100.5, 109.0)]                                    # цена ушла к 3R без нас
    assert advance(_bars(ran), s, RULE).status is Status.CANCELLED
    filled = [flat[0], (101.0, 101.5, 99.5, 100.8), (100.8, 102.0, 100.0, 101.6)]
    a = advance(_bars(filled), s, RULE)
    assert a.status is Status.OPEN and a.fill_day == date(2026, 1, 2) and a.exit_day is None
    assert a.r == pytest.approx((1.6 - (0.0002 + 0.00055) * 100) / 3.0, abs=1e-9)        # maker + taker
    win = filled + [(101.6, 110.0, 101.0, 109.5)]
    w = advance(_bars(win), s, RULE)
    assert w.status is Status.CLOSED and w.exit_day == date(2026, 1, 4) and w.r > 3.0  # 3R + долив на 1.5R
    knocked = [flat[0], (101.0, 101.0, 96.0, 97.5)]                                  # выбит в день ретеста
    k = advance(_bars(knocked), s, RULE)
    assert k.status is Status.CLOSED and k.r < -1.0 and k.fill_day == k.exit_day == date(2026, 1, 2)
    assert advance(_bars(knocked), k, RULE) is k                                     # закрытый не пересчитывается


def test_funding_r_sign_and_book():
    long_, short = _setup(), _setup(side=-1, stop=103.0)
    assert funding_r(long_, 0.001) == pytest.approx(-0.001 * 100 / 3) and funding_r(short, 0.001) > 0
    xs = [replace(_setup(day=date(2026, 1, k)), status=Status.CLOSED, exit_day=date(2026, 2, k), r=r)
          for k, r in ((1, 3.0), (2, -1.0), (3, -1.0), (4, 2.0))]
    xs += [replace(_setup(day=date(2026, 3, 1)), status=Status.OPEN, fill_day=date(2026, 3, 2), r=0.5),
           _setup(day=date(2026, 3, 5)), replace(_setup(day=date(2026, 3, 6)), status=Status.CANCELLED)]
    b = book(xs)
    assert b.closed == 4 and b.total_r == pytest.approx(3.0) and b.mean_r == pytest.approx(0.75)
    assert b.win == 0.5 and b.dd_r == pytest.approx(2.0) and b.open_r == 0.5
    assert len(b.open) == 1 and len(b.pending) == 1 and b.recent[0].r == 2.0 and b.since == date(2026, 1, 1)
    assert book([]).since is None and book([]).mean_r is None


def test_store_roundtrip(tmp_path):
    st = SqliteStore(tmp_path / "t.db")
    s = st.dbreak_add(_setup())
    assert s is not None and s.id is not None and st.dbreak_add(_setup()) is None    # тот же сетап второй раз
    st.dbreak_update(replace(s, status=Status.OPEN, fill_day=date(2026, 1, 2), r=0.4, funding_r=-0.01))
    [a] = st.dbreak_active(RULE.key)
    assert a.status is Status.OPEN and a.fill_day == date(2026, 1, 2) and a.funding_r == -0.01
    st.dbreak_update(replace(a, status=Status.CLOSED, exit_day=date(2026, 1, 9), r=2.9))
    assert st.dbreak_active(RULE.key) == [] and st.dbreak_recent(RULE.key)[0].r == 2.9


class _Market:
    def __init__(self, frames):
        self.frames = frames

    async def universe(self, floor):
        return list(self.frames)

    async def daily_bars(self, symbol, limit):
        return self.frames[symbol].iloc[-limit:]

    async def funding_sum(self, symbol, start, end):
        return 0.0001


class _Note:
    def __init__(self):
        self.msgs = []

    async def text(self, m):
        self.msgs.append(m)


def _breakout_day(d: pd.DataFrame) -> int:
    rows = _research(d)
    return next(int(d.index.get_loc(r["t"])) for r in rows)


def test_paper_finds_breakout_once_per_closed_day_and_tracks_it(tmp_path):
    full = _frame(0)
    t = _breakout_day(full)
    d = full.iloc[:t + 1]                                     # последний закрытый день — день пробоя
    day = d.index[-1].date()
    st, note = SqliteStore(tmp_path / "t.db"), _Note()
    m = _Market({"XUSDT": d})
    now = [datetime.combine(day + timedelta(days=1), datetime.min.time(), timezone.utc) + timedelta(minutes=5)]
    tp = DailyBreakPaper(m, st, note, clock=lambda: now[0])
    assert asyncio.run(tp.maybe_run()) == [] and tp.due_day(now[0]) is None          # день ещё дописывается
    now[0] += timedelta(minutes=5)
    asyncio.run(tp.maybe_run())
    [s] = st.dbreak_active(RULE.key)
    assert s.day == day and s.status is Status.PENDING and "пробой, лимитка" in note.msgs[-1]
    assert tp.due_day(now[0]) is None and asyncio.run(tp.maybe_run()) == []          # этот день уже посчитан
    m.frames["XUSDT"] = full.iloc[:t + 40]                                           # прошло 39 дней
    now[0] += timedelta(days=39)
    asyncio.run(tp.maybe_run())
    [x] = [y for y in st.dbreak_recent(RULE.key) if y.id == s.id]                    # плюс пробои последних 12 дней
    assert x.status in (Status.CLOSED, Status.CANCELLED, Status.OPEN)
    if x.fill_day is not None:
        assert x.funding_r == pytest.approx(funding_r(x, 0.0001)) and "ретест, вошёл" in " ".join(note.msgs)


def test_paper_catches_up_missed_days(tmp_path):
    full = _frame(0)
    t = _breakout_day(full)
    st = SqliteStore(tmp_path / "t.db")
    st.kv_set(f"dbreak_done:{RULE.key}", full.index[t - 3].date().isoformat())       # сервер лежал 3 дня
    d = full.iloc[:t + 2]
    now = datetime.combine(d.index[-1].date() + timedelta(days=1), datetime.min.time(), timezone.utc) \
        + timedelta(minutes=10)
    asyncio.run(DailyBreakPaper(_Market({"XUSDT": d}), st, clock=lambda: now).maybe_run())
    assert [s.day for s in st.dbreak_recent(RULE.key)] == [full.index[t].date()]


def test_paper_refuses_stale_candles(tmp_path):
    d = _frame(1, end="2026-10-10")
    st = SqliteStore(tmp_path / "t.db")
    tp = DailyBreakPaper(_Market({"XUSDT": d}), st, clock=lambda: datetime(2026, 10, 13, 1, 0, tzinfo=timezone.utc))
    assert asyncio.run(tp.maybe_run()) == [] and "ждём 2026-10-12" in tp.last_error
    assert st.kv_get(f"dbreak_done:{RULE.key}") is None
