import asyncio
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from trader.application.sector_flow import KEY, SectorFlowService
from trader.domain import sector_flow as D

from test_trader_web import env, _login  # noqa: F401  (фикстура панели)

NOW = datetime(2026, 10, 27, 12, tzinfo=timezone.utc)


def _bars(hot="мемы", days=100, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end="2026-10-26", periods=days, freq="D", tz="UTC")
    syms = ["BTCUSDT", "1000PEPEUSDT", "DOGEUSDT", "WIFUSDT", "FETUSDT", "TAOUSDT", "WLDUSDT", "AAVEUSDT", "UNIUSDT",
            "CRVUSDT", "XUSDT", "YUSDT"]
    out = {}
    for s in syms:
        r = rng.normal(0, 0.01, days)
        v = np.full(days, 2e7)
        if D.sector_of(s) == hot:
            r[-7:] += 0.03
            v[-7:] *= 3
        out[s] = pd.DataFrame({"close": 100 * np.exp(np.cumsum(r)), "quote_volume": v}, index=idx)
    return out


def test_base_asset_and_sector():
    assert D.base_asset("1000PEPEUSDT") == "PEPE" and D.base_asset("1MBABYDOGEUSDT") == "BABYDOGE"
    assert D.base_asset("1INCHUSDT") == "1INCH"
    assert D.sector_of("1000PEPEUSDT") == "мемы" and D.sector_of("ONDOUSDT") == "RWA" and D.sector_of("XUSDT") is None


def test_compute_marks_inflow_and_roundtrip():
    f = D.compute(_bars(), NOW)
    assert f.day == "2026-10-26" and f.top is not None and f.top.name == "мемы"
    assert f.top.ret_7d > 0.1 and f.top.turnover > 1.5 and set(f.top.leaders) <= {"PEPE", "DOGE", "WIF"}
    assert [r.top for r in f.rows].count(True) == 1 and not any(r.inflow for r in f.rows if r.name != "мемы")
    assert D.SectorFlow.from_json(json.loads(json.dumps(f.to_json()))) == f
    assert D.compute({}, NOW).rows == ()


def test_flow_changed_text():
    a, b = D.compute(_bars("мемы"), NOW), D.compute(_bars("AI"), NOW)
    assert D.flow_changed(None, a).startswith("Деньги заходят в сектор «мемы»")
    assert D.flow_changed(a, a) is None and "«AI»" in D.flow_changed(a, b)
    quiet = D.SectorFlow(NOW, "", tuple(), 0.0)
    assert D.flow_changed(a, quiet) is None


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

    async def universe(self, min_turnover_usd):
        return list(self.bars) + ["BROKENUSDT"]

    async def daily_bars(self, symbol, limit):
        if symbol == "BROKENUSDT":
            raise RuntimeError("451")
        return self.bars[symbol].tail(limit)


class _Notify:
    def __init__(self):
        self.sent = []

    async def text(self, s):
        self.sent.append(s)


def test_service_runs_saves_and_notifies_once():
    bars = {**_bars(), **{f"Z{i}USDT": _bars()["XUSDT"] for i in range(15)}}
    store, note = _Store(), _Notify()
    clock = [NOW]
    svc = SectorFlowService(_Market(bars), store, notifier=note, clock=lambda: clock[0])

    async def go():
        await svc.maybe_run()
        await svc._task
    asyncio.run(go())
    assert svc.current().top.name == "мемы" and len(note.sent) == 1 and svc.last_error is None
    assert not svc.due(NOW + timedelta(hours=5)) and svc.due(NOW + timedelta(hours=6))
    clock[0] = NOW + timedelta(hours=7)
    asyncio.run(go())
    assert len(note.sent) == 1                                          # тот же сектор — без повтора


def test_service_too_few_coins_records_error():
    store = _Store()
    svc = SectorFlowService(_Market(dict(list(_bars().items())[:5])), store, clock=lambda: NOW)

    async def go():
        await svc.maybe_run()
        await svc._task
    asyncio.run(go())
    assert "дневки загрузились" in svc.last_error and KEY not in store.kv
    assert not svc.due(NOW + timedelta(minutes=10))


def test_screener_tab_shows_sectors(env):  # noqa: F811
    c, tmp = env
    _login(c)
    from trader.infrastructure.sqlite_repo import SqliteStore
    SqliteStore(tmp / "trader.db").kv_set(KEY, json.dumps(D.compute(_bars(), NOW).to_json()))
    r = c.get("/screener").text
    assert "Куда заходят деньги" in r and "мемы" in r and "приток" in r
