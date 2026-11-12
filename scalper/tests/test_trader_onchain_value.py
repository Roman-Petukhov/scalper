import asyncio
import json
from datetime import datetime, timezone

import pytest

from trader.application.onchain_value import ALIAS, KEY, SKIP, OnchainValueService
from trader.domain import onchain_value as O
from trader.infrastructure import coinmetrics as C

from test_trader_web import env, _login  # noqa: F401  (фикстура панели)


def _readings(n=18):
    return [O.Reading(f"c{i:02d}", 1000.0 * (i + 1), 1e9, 30) for i in range(n)]       # c00 — самая дорогая


def test_rank_thirds_filters_and_aliases():
    rows = O.rank(_readings() + [O.Reading("usdt_eth", 5e5, 1e11, 30), O.Reading("usdt", 5e5, 1e11, 30),
                                 O.Reading("thin", 50.0, 1e9, 30), O.Reading("young", 5000.0, 1e9, 5),
                                 O.Reading("avaxc", 4000.0, 8e9, 30), O.Reading("bad", float("nan"), 1e9, 30)],
                  perps={"C00USDT", "AVAXUSDT"}, alias=ALIAS, skip=SKIP)
    syms = [r.symbol for r in rows]
    assert len(rows) == 19 and "USDT" not in syms and "THIN" not in syms and "YOUNG" not in syms and "BAD" not in syms
    assert rows[0].symbol == "AVAX" and rows[0].perp and rows[0].cap_per_address == pytest.approx(2e6)
    assert [r.group for r in rows].count(O.EXPENSIVE) == 6 and [r.group for r in rows].count(O.CHEAP) == 6
    assert rows[1].symbol == "C00" and rows[1].group == O.EXPENSIVE and rows[-1].symbol == "C17" and rows[-1].title == "дешёвая"
    assert O.rank(_readings(10)) == []                                                  # монет мало — трети не считаем
    twice = O.rank(_readings() + [O.Reading("c00", 9e6, 1e9, 30)])                      # дубль — берём большую сеть
    assert sum(r.symbol == "C00" for r in twice) == 1 and twice[-1].symbol == "C00"
    snap = O.OnchainSnapshot(datetime(2026, 11, 12, tzinfo=timezone.utc), tuple(rows), assets=25)
    back = O.OnchainSnapshot.from_json(json.loads(json.dumps(snap.to_json())))
    assert back == snap and back.cheap[0].symbol == "C17" and back.expensive[0].symbol == "AVAX"


def test_coinmetrics_parsers():
    cat = {"data": [{"asset": "btc", "metrics": [{"metric": "AdrActCnt", "frequencies": [{"frequency": "1d"}]}]},
                    {"asset": "xyz", "metrics": [{"metric": "AdrActCnt", "frequencies": [{"frequency": "1b"}]}]}, "мусор"]}
    assert C.parse_catalog(cat) == {"btc"} and C.parse_catalog(None) == set()
    rows = [{"asset": "btc", "time": "2026-11-02T00:00:00Z", "AdrActCnt": "300", "CapMrktCurUSD": "2.1e12"},
            {"asset": "btc", "time": "2026-11-01T00:00:00Z", "AdrActCnt": "100", "CapMrktCurUSD": "2.0e12"},
            {"asset": "eth", "time": "2026-11-01T00:00:00Z", "AdrActCnt": None, "CapMrktCurUSD": "4e11"},
            {"asset": "ltc", "time": "2026-11-01T00:00:00Z", "AdrActCnt": "x", "CapMrktCurUSD": "1"}]
    assert C.parse_readings(rows) == [O.Reading("btc", 200.0, 2.1e12, 2)]


class _Source:
    def __init__(self, readings):
        self.readings_, self.calls = readings, 0

    async def readings(self, days):
        self.calls += 1
        if isinstance(self.readings_, Exception):
            raise self.readings_
        return self.readings_


class _Store:
    def __init__(self):
        self.kv = {}

    def kv_get(self, k):
        return self.kv.get(k)

    def kv_set(self, k, v):
        self.kv[k] = v


def test_service_runs_daily_and_keeps_last_snapshot_on_error():
    async def go():
        now = [datetime(2026, 11, 12, tzinfo=timezone.utc)]
        src, store = _Source(_readings()), _Store()
        svc = OnchainValueService(src, store, clock=lambda: now[0])
        await svc.maybe_run()
        await svc._task
        assert src.calls == 1 and len(svc.current().rows) == 18 and svc.last_error is None
        await svc.maybe_run()
        assert src.calls == 1                                                           # сутки не прошли
        now[0] = datetime(2026, 11, 13, 1, tzinfo=timezone.utc)
        src.readings_ = RuntimeError("нет сети")
        await svc.maybe_run()
        await svc._task
        assert src.calls == 2 and "нет сети" in svc.last_error and len(svc.current().rows) == 18
        src.readings_ = _readings(5)
        now[0] = datetime(2026, 11, 13, 3, tzinfo=timezone.utc)
        await svc.maybe_run()
        await svc._task
        assert "мало" in svc.last_error
    asyncio.run(go())


def test_screener_tab_shows_onchain_card(env):  # noqa: F811
    c, tmp = env
    _login(c)
    from test_trader_screener import _snap
    from trader.application.screener import KEY as SCR_KEY
    from trader.domain import screener as S
    from trader.infrastructure.sqlite_repo import SqliteStore
    store = SqliteStore(tmp / "trader.db")
    store.kv_set(SCR_KEY, json.dumps(_snap([], S.Zone.BULL).to_json()))
    assert "Цена к активным адресам" not in c.get("/screener").text
    snap = O.OnchainSnapshot(datetime(2026, 11, 12, tzinfo=timezone.utc), tuple(O.rank(_readings(), {"C00USDT"})), 25)
    store.kv_set(KEY, json.dumps(snap.to_json()))
    page = c.get("/screener").text
    assert "Цена к активным адресам · дорогих 6 из 18" in page and "Дорогие — осторожно" in page
    assert "$1.0M" in page and "1 000 адр." in page and "перп" in page
