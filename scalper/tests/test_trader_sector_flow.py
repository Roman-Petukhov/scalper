import asyncio
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from trader.application.sector_flow import HIST_KEY, KEY, PAPER_KEY, SectorFlowService
from trader.domain import sector_flow as D
from trader.domain import sector_paper as P

from test_trader_web import env, _login  # noqa: F401  (фикстура панели)

NOW = datetime(2026, 10, 27, 12, tzinfo=timezone.utc)


def _bars(hot="мемы", days=100, seed=7, end="2026-10-26"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=end, periods=days, freq="D", tz="UTC")
    syms = ["BTCUSDT", "1000PEPEUSDT", "DOGEUSDT", "WIFUSDT", "FETUSDT", "TAOUSDT", "WLDUSDT", "AAVEUSDT", "UNIUSDT",
            "CRVUSDT", "XUSDT", "YUSDT"]
    out = {}
    for s in syms:
        r = rng.normal(0, 0.01, days)
        v = np.full(days, 2e7)
        buy = np.full(days, 0.5)
        if D.sector_of(s) == hot:
            r[-7:] += 0.03
            v[-7:] *= 3
            buy[-7:] = 0.6
        out[s] = pd.DataFrame({"close": 100 * np.exp(np.cumsum(r)), "quote_volume": v, "volume": v / 100,
                               "taker_buy_volume": v / 100 * buy}, index=idx)
    return out


def test_base_asset_and_sector():
    assert D.base_asset("1000PEPEUSDT") == "PEPE" and D.base_asset("1MBABYDOGEUSDT") == "BABYDOGE"
    assert D.base_asset("1INCHUSDT") == "1INCH"
    assert D.sector_of("1000PEPEUSDT") == "мемы" and D.sector_of("ONDOUSDT") == "RWA" and D.sector_of("XUSDT") is None


def test_compute_scores_sectors_and_roundtrip():
    f = D.compute(_bars(), NOW, bull=True, btc_ema=1.2)
    assert f.day == "2026-10-26" and f.top is not None and f.top.name == "мемы" and f.bull is True
    assert f.top.ret_7d > 0.1 and f.top.turnover > 1.5 and set(f.top.leaders) <= {"PEPE", "DOGE", "WIF"}
    assert f.top.taker > 0 and f.top.score > 0 and f.top.inflow and len(f.top.members) == 3
    assert [r.top for r in f.rows].count(True) == 1 and f.rows[0].top
    raw = json.dumps(f.to_json(), allow_nan=False)
    assert D.SectorFlow.from_json(json.loads(raw)).to_json() == f.to_json()
    old = json.loads(raw)                                                     # снимок прежней версии — без новых полей
    for r in old["rows"]:
        for k in ("taker", "oi_7d", "score", "members"):
            r.pop(k)
    assert D.SectorFlow.from_json(old).rows[0].name == f.rows[0].name
    assert D.compute({}, NOW).rows == ()


def test_composite_needs_two_features_and_oi_counts():
    sc = D.composite({"a": {"x": 1.0, "y": 0.0, "z": -1.0}, "b": {"x": 2.0, "y": 1.0}})
    assert sc["x"] > sc["y"] and np.isnan(sc["z"])
    idx = pd.date_range(end="2026-10-26", periods=10, freq="D", tz="UTC")
    oi = {s: pd.Series(np.r_[np.full(3, 1e8), np.full(7, 2e8 if D.sector_of(s) == "AI" else 1e8)], index=idx)
          for s in _bars()}
    f = D.compute(_bars(), NOW, oi=oi)
    ai = next(r for r in f.rows if r.name == "AI")
    assert ai.oi_7d > 0.3 and all(r.oi_7d < 0 for r in f.rows if r.name != "AI")


def test_btc_regime_uses_closed_weeks():
    idx = pd.date_range("2024-01-01", "2026-10-25", freq="D", tz="UTC")       # 2026-10-25 — воскресенье
    up = pd.Series(np.linspace(100, 300, len(idx)), index=idx)
    bull, ratio = D.btc_regime(up)
    assert bull is True and ratio > 1
    down = pd.Series(np.linspace(300, 100, len(idx)), index=idx)
    assert D.btc_regime(down)[0] is False
    assert D.btc_regime(up.tail(60)) == (None, None)


def test_history_series_match_snapshot():
    b = _bars(days=220)
    rng = np.random.default_rng(1)
    for x in b.values():                                                      # живые объёмы, как на рынке
        x["quote_volume"] *= np.exp(np.cumsum(rng.normal(0, 0.05, len(x))))
        x["volume"] = x["quote_volume"] / 100
        x["taker_buy_volume"] = x["volume"] * np.clip(0.5 + rng.normal(0, 0.03, len(x)), 0.3, 0.7)
    btc = pd.Series(np.linspace(100, 300, 800), index=pd.date_range(end="2026-10-25", periods=800, freq="D", tz="UTC"))
    h = D.history(b, btc_close=btc)
    f = D.compute(b, NOW)
    assert len(h["days"]) == 90 and set(h["sectors"]) == {r.name for r in f.rows}
    for r in f.rows:
        s = h["sectors"][r.name]
        assert abs(s["turnover"][-1] - r.turnover) < 1e-6 and abs(s["score"][-1] - r.score) < 0.15
        assert all(len(v) == 90 for v in s.values())
    assert h["bull"][-1] is True and json.dumps(h, allow_nan=False)
    assert D.history({}) == {"days": [], "sectors": {}, "bull": []}


def test_paper_pick_settle_and_stats():
    f = D.compute(_bars(end="2026-10-25"), NOW, bull=True)                   # снимок по воскресенье
    assert P.week_start(f) == "2026-10-26" and P.week_start(D.compute(_bars(), NOW)) is None
    new = P.picks(f)
    assert {w.kind for w in new} == {"score", "momentum"} and all(w.sector == "мемы" for w in new)
    weeks, added = P.merge([], new)
    assert len(added) == 2 and P.merge(weeks, P.picks(f))[1] == []
    later = _bars(end="2026-11-01", hot="AI")                                 # неделя прошла
    for s in ("1000PEPEUSDT", "DOGEUSDT", "WIFUSDT"):
        later[s].loc["2026-10-26":, "close"] = later[s].loc["2026-10-25", "close"] * 1.2   # мемы +20% в понедельник
    weeks2, closed = P.settle(weeks, later)
    sc = next(w for w in closed if w.kind == "score")
    assert len(closed) == 2 and sc.sector_ret > 0.15 and sc.excess > 0.05
    st = P.stats(weeks2, "score")
    assert st.weeks == 1 and st.wins == 1 and P.stats(weeks2, "score", bull=False) is None
    text = P.week_text([], closed, f)
    assert "Итог недели с 2026-10-26" in text and "«мемы»" in text
    assert "Сектор недели (бумажно): «мемы»" in P.week_text(added, [], f)
    assert P.PaperWeek.from_json(json.loads(json.dumps(sc.to_json()))) == sc


class _Store:
    def __init__(self):
        self.kv = {}

    def kv_get(self, k):
        return self.kv.get(k)

    def kv_set(self, k, v):
        self.kv[k] = v


class _Market:
    def __init__(self, bars):
        self.bars = bars
        self.oi_calls = 0

    async def universe(self, min_turnover_usd):
        return list(self.bars) + ["BROKENUSDT"]

    async def daily_bars(self, symbol, limit):
        if symbol == "BROKENUSDT":
            raise RuntimeError("451")
        return self.bars[symbol].tail(limit)

    async def oi_daily(self, symbol, days):
        self.oi_calls += 1
        return pd.Series(1e8, index=self.bars[symbol].index[-days:])


class _Notify:
    def __init__(self):
        self.sent = []

    async def text(self, s):
        self.sent.append(s)


def test_service_runs_saves_and_records_weekly_pick():
    bars = {**_bars(end="2026-10-25"), **{f"Z{i}USDT": _bars(end="2026-10-25")["XUSDT"] for i in range(15)}}
    store, note = _Store(), _Notify()
    clock = [NOW]
    m = _Market(bars)
    svc = SectorFlowService(m, store, notifier=note, clock=lambda: clock[0], oi=m)

    async def go():
        await svc.maybe_run()
        await svc._task
    asyncio.run(go())
    assert svc.current().top.name == "мемы" and svc.last_error is None and m.oi_calls > 0
    assert len(svc.paper()) == 2 and len(note.sent) == 1 and "Сектор недели" in note.sent[0]
    assert not svc.due(NOW + timedelta(hours=5)) and svc.due(NOW + timedelta(hours=6))
    clock[0] = NOW + timedelta(hours=7)
    asyncio.run(go())
    assert len(note.sent) == 1 and len(svc.paper()) == 2                # та же неделя — без повтора
    assert PAPER_KEY in store.kv
    h = svc.history()
    assert len(h["days"]) == 90 and h["days"][-1] == "2026-10-25" and "мемы" in h["sectors"]


def test_service_too_few_coins_records_error():
    store = _Store()
    svc = SectorFlowService(_Market(dict(list(_bars().items())[:5])), store, clock=lambda: NOW)

    async def go():
        await svc.maybe_run()
        await svc._task
    asyncio.run(go())
    assert "дневки загрузились" in svc.last_error and KEY not in store.kv
    assert not svc.due(NOW + timedelta(minutes=10))


def test_screener_tab_shows_sectors_and_paper(env):  # noqa: F811
    c, tmp = env
    _login(c)
    from trader.infrastructure.sqlite_repo import SqliteStore
    db = SqliteStore(tmp / "trader.db")
    f = D.compute(_bars(end="2026-10-25"), NOW, bull=False, btc_ema=0.9)
    db.kv_set(KEY, json.dumps(f.to_json()))
    r = c.get("/screener").text
    assert "Куда заходят деньги" in r and "мемы" in r and "Режим не бычий" in r and "первая запись" in r
    weeks, _ = P.merge([], P.picks(f))
    weeks, _ = P.settle(weeks, _bars(end="2026-11-01"))
    db.kv_set(PAPER_KEY, json.dumps([w.to_json() for w in weeks]))
    r = c.get("/screener").text
    assert "Бумажная проверка вперёд" in r and "недель 1" in r and "с 26.10" in r and "data-sector-chart" in r
    assert c.get("/api/sectors/history").json() == {"days": [], "sectors": {}, "bull": []}
    db.kv_set(HIST_KEY, json.dumps(D.history(_bars(days=220))))
    assert len(c.get("/api/sectors/history").json()["days"]) == 90
