import asyncio
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from trader.application.trend import TrendPaper
from trader.domain.trend import RULES, TAKER, TrendDay, TrendPosition, TrendTarget, inflow, step, targets, weights
from trader.infrastructure.sqlite_repo import SqliteStore


def _market(n_coins: int = 30, days: int = 900, seed: int = 0, end: str = "2026-10-12"):
    idx = pd.date_range(end=end, periods=days, freq="1D", tz="UTC")
    rng = np.random.default_rng(seed)
    close = pd.DataFrame({f"C{j}USDT": 10 * np.exp(np.cumsum(rng.normal(0.001, 0.04, days))) for j in range(n_coins)},
                         index=idx)
    qv = pd.DataFrame({c: rng.uniform(1e7, 1e9) * np.ones(days) for c in close}, index=idx)
    return close, qv


@pytest.mark.parametrize("rule", RULES, ids=[r.key for r in RULES])
def test_weights_match_research_backtest(rule):
    from research.dboost import ensemble, pyramid, strongest
    from research.dtrend import portfolio
    close, qv = _market()
    c = close.copy()
    c.index = c.index.tz_localize(None)
    adv = qv.rolling(30, min_periods=20).mean().shift(1)
    adv.index = c.index
    base = ensemble(c)
    sig = strongest(c, adv, base * pyramid(c, base), 7)
    research = portfolio(c, pd.DataFrame(0.0, index=c.index, columns=c.columns), adv, sig=sig)
    w = weights(close, qv, rule)
    w.index = c.index
    ret = c.pct_change(fill_method=None).fillna(0.0)
    mine = (w.shift(1).fillna(0.0) * ret).sum(axis=1) - \
        (w - w.shift(1).fillna(0.0)).abs().sum(axis=1).shift(1).fillna(0.0) * TAKER
    assert np.allclose(mine.to_numpy(), research.to_numpy(), atol=1e-12)
    assert (w.iloc[400:] > 0).sum(axis=1).max() <= 7 and w.sum(axis=1).max() <= 2.0 + 1e-9


def test_targets_are_last_day_weights():
    close, qv = _market()
    tg = targets(close, qv, RULES[0])
    w = weights(close, qv, RULES[0]).iloc[-1]
    assert {t.symbol: t.weight for t in tg} == pytest.approx({s: v for s, v in w.items() if v > 0})
    assert all(0 <= t.adds <= 3 and 0 < t.signal <= 1 for t in tg)


def test_step_marks_charges_and_rebalances():
    d0, d1 = date(2026, 10, 1), date(2026, 10, 2)
    first = step(None, "x", d0, {"A": 10.0, "B": 20.0}, {}, [TrendTarget("A", 0.5, 1.0, 0, 1.0)])
    assert first.positions["A"].qty == pytest.approx(0.05) and first.equity == pytest.approx(1 - TAKER * 0.5)
    nxt = step(first, "x", d1, {"A": 12.0, "B": 20.0}, {"A": 0.001}, [TrendTarget("B", 0.25, 0.5, 1, 1.0)])
    gain = 0.05 * 2.0 - 0.05 * 10.0 * 0.001
    eq = first.equity + gain
    assert nxt.equity == pytest.approx(eq - TAKER * (0.05 * 12.0 + 0.25 * eq))
    assert set(nxt.positions) == {"B"} and nxt.positions["B"].entry_day == d1
    assert nxt.pnl == pytest.approx(nxt.equity - first.equity)
    gone = step(nxt, "x", date(2026, 10, 3), {}, {}, [])                       # монета без цены — закрыта по оценке
    assert gone.positions == {} and gone.equity == pytest.approx(nxt.equity - TAKER * 0.25 * eq)


def test_store_roundtrip(tmp_path):
    st = SqliteStore(tmp_path / "t.db")
    d = TrendDay("s7_add", date(2026, 10, 2), 1.05, 0.01, 0.001,
                 {"A": TrendPosition(0.1, 10.0, 0.5, 2, date(2026, 9, 1), 8.0)})
    st.trend_save(d)
    st.trend_save(TrendDay("s7_add", date(2026, 10, 1), 1.04, 0.0, 0.0, {}))
    assert st.trend_last("s7_add") == d and st.trend_last("other") is None
    assert [x.day for x in st.trend_history("s7_add")] == [date(2026, 10, 1), date(2026, 10, 2)]


class _Daily:
    def __init__(self, close, qv):
        self.close, self.qv, self.calls = close, qv, 0

    async def universe(self, floor):
        return list(self.close.columns)

    async def daily_bars(self, symbol, limit):
        self.calls += 1
        return pd.DataFrame({"close": self.close[symbol], "quote_volume": self.qv[symbol]}).iloc[-limit:]

    async def funding_sum(self, symbol, start, end):
        return 0.0001


class _Stables:
    def __init__(self, growing=True, fail=False):
        self.growing, self.fail = growing, fail

    async def supply(self):
        if self.fail:
            raise RuntimeError("нет связи")
        idx = pd.date_range("2025-01-01", "2026-10-12", freq="1D")
        v = np.arange(len(idx), dtype=float) + 1e11
        return pd.Series(v if self.growing else v[::-1], index=idx)


class _Note:
    def __init__(self):
        self.msgs = []

    async def text(self, m):
        self.msgs.append(m)


def test_paper_runs_once_per_closed_day(tmp_path):
    close, qv = _market(end="2026-10-12")
    st = SqliteStore(tmp_path / "t.db")
    note = _Note()
    now = [datetime(2026, 10, 13, 0, 1, tzinfo=timezone.utc)]
    tp = TrendPaper(_Daily(close, qv), st, note, clock=lambda: now[0], stables=_Stables())
    assert asyncio.run(tp.maybe_run()) == []                                  # день ещё дописывается
    now[0] = datetime(2026, 10, 13, 0, 5, tzinfo=timezone.utc)
    out = asyncio.run(tp.maybe_run())
    assert [x.rule for x in out] == [r.key for r in RULES] and all(x.day == date(2026, 10, 12) for x in out)
    assert any("вошёл" in m for m in note.msgs)
    now[0] += timedelta(minutes=30)
    assert asyncio.run(tp.maybe_run()) == [] and tp.due_day(now[0]) is None    # этот день уже посчитан


def test_paper_refuses_stale_candles(tmp_path):
    close, qv = _market(end="2026-10-10")
    st = SqliteStore(tmp_path / "t.db")
    tp = TrendPaper(_Daily(close, qv), st, clock=lambda: datetime(2026, 10, 13, 1, 0, tzinfo=timezone.utc))
    assert asyncio.run(tp.maybe_run()) == [] and "ждём 2026-10-12" in tp.last_error
    assert st.trend_last(RULES[0].key) is None


def test_inflow_matches_research_gate():
    from research.dboost2 import inflow_gate
    idx = pd.date_range("2024-01-01", periods=200, freq="1D")
    rng = np.random.default_rng(2)
    st = pd.Series(1e11 + np.cumsum(rng.normal(0, 1e9, 200)), index=idx).drop(idx[[50, 51, 120]])
    g = inflow_gate(st, idx)
    for d in idx[40:]:
        chg = inflow(st, d.date())
        assert (chg is not None and chg > 0) == bool(g[d])
    assert inflow(st, idx[10].date()) is None                                 # меньше 30 дней истории


def test_gate_off_closes_book_and_falls_back_when_data_missing(tmp_path):
    close, qv = _market(end="2026-10-12")
    st = SqliteStore(tmp_path / "t.db")
    note = _Note()
    now = [datetime(2026, 10, 13, 0, 5, tzinfo=timezone.utc)]
    src = _Stables()
    tp = TrendPaper(_Daily(close, qv), st, note, clock=lambda: now[0], stables=src)
    first = asyncio.run(tp.run(now[0]))[0]
    assert first.gate is True and first.inflow > 0 and first.positions
    close2, qv2 = _market(end="2026-10-13")
    tp.market = _Daily(close2, qv2)
    src.fail = True                                                           # DefiLlama молчит — держим вчерашнее
    now[0] = datetime(2026, 10, 14, 0, 5, tzinfo=timezone.utc)
    second = asyncio.run(tp.run(now[0]))[0]
    assert second.gate is True and second.inflow is None and second.positions
    close3, qv3 = _market(end="2026-10-14")
    tp.market = _Daily(close3, qv3)
    src.fail, src.growing = False, False
    now[0] = datetime(2026, 10, 15, 0, 5, tzinfo=timezone.utc)
    third = asyncio.run(tp.run(now[0]))[0]
    assert third.gate is False and third.positions == {} and third.inflow < 0
    assert any("портфель закрыт" in m for m in note.msgs)
    assert st.trend_last(RULES[0].key).gate is False


def test_first_step_without_data_stays_out(tmp_path):
    close, qv = _market(end="2026-10-12")
    st = SqliteStore(tmp_path / "t.db")
    tp = TrendPaper(_Daily(close, qv), st, stables=_Stables(fail=True))
    d = asyncio.run(tp.run(datetime(2026, 10, 13, 0, 5, tzinfo=timezone.utc)))[0]
    assert d.gate is False and d.positions == {} and d.equity == 1.0


def test_parse_supply_and_old_table_migration(tmp_path):
    import sqlite3
    from trader.infrastructure.defillama import parse_supply
    s = parse_supply([{"date": "1700000000", "totalCirculatingUSD": {"peggedUSD": 1.5e11}},
                      {"date": "x"}, {"date": 1700086400, "totalCirculating": {"peggedUSD": 1.6e11}}, "junk"])
    assert list(s) == [1.5e11, 1.6e11] and s.index[0] == pd.Timestamp("2023-11-14")
    assert parse_supply({"error": 1}).empty
    db = sqlite3.connect(tmp_path / "old.db")
    db.execute("CREATE TABLE trend_days (rule TEXT NOT NULL, day TEXT NOT NULL, equity REAL NOT NULL, pnl REAL NOT NULL, "
               "cost REAL NOT NULL, positions TEXT NOT NULL, PRIMARY KEY (rule, day))")
    db.execute("INSERT INTO trend_days VALUES ('s7_add', '2026-10-01', 1.01, 0.01, 0.0, '{}')")
    db.commit()
    db.close()
    st = SqliteStore(tmp_path / "old.db")
    old = st.trend_last("s7_add")
    assert old.gate is None and old.inflow is None
    st.trend_save(TrendDay("s7_add", date(2026, 10, 2), 1.0, 0.0, 0.0, {}, False, -0.01))
    assert st.trend_last("s7_add").gate is False
