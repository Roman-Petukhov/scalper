import asyncio
import json
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from trader.application.screener import KEY, SEEN_KEY, ScreenerService
from trader.domain import screener as S
from trader.infrastructure import fundamentals as F

from test_trader_web import env, _login  # noqa: F401  (фикстура панели)


def _rows():
    R = S.FeeRow
    return [R("uniswap", "uniswap-v3", "Uniswap V3", "Dexs", S.FEES, 90e6, 900e6),
            R("uniswap", "uniswap-v2", "Uniswap V2", "Dexs", S.FEES, 10e6, 100e6),
            R("uniswap", "uniswap-v3", "Uniswap V3", "Dexs", S.HOLDERS, 2e6, 12e6),
            R("lend", "lend", "Lend", "Lending", S.FEES, 3e6, 24e6),
            R("lend", "lend", "Lend", "Lending", S.REVENUE, 1.5e6, 12e6),
            R("lend", "lend", "Lend", "Lending", S.HOLDERS, 1e6, 6e6),
            R("near", "near", "Near", "Chain", S.HOLDERS, 0.2e6, 2e6),
            R("near-intents", "near-intents", "NEAR Intents", "Bridge", S.REVENUE, 2e6, 10e6),
            R("near-intents", "near-intents", "NEAR Intents", "Bridge", S.HOLDERS, 2e6, 10e6)]


def _q(gid, sym, mcap, fdv, vol=5e6, chg=10.0, price=1.0):
    return S.Quote(gid, sym, sym, price, mcap, fdv, vol, chg, -50.0)


def test_businesses_sum_versions_and_tokens_merge_businesses():
    biz = S.businesses(_rows())
    assert biz["uniswap"].fees_30d == pytest.approx(100e6) and biz["uniswap"].name == "Uniswap V3"
    assert biz["lend"].rev_1y == pytest.approx(12e6)
    tok = S.by_token(biz, {"uniswap": "uniswap", "lend": "lend-token", "near": "near", "near-intents": "near"})
    assert tok["near"].hr_30d == pytest.approx(2.2e6) and tok["near"].name == "NEAR Intents"
    assert set(tok) == {"uniswap", "lend-token", "near"}


def test_evaluate_reasons_and_flags():
    b = S.Business("x", "X", "Chain", hr_30d=10_000, hr_1y=12_000 * 12)
    c = S.evaluate(b, _q("x", "x", 10e6, 40e6, vol=1e5, chg=None), 0.3, S.Rules())
    assert set(c.reasons) == {"нет выручки держателям", "мелкая или нет цены", "неликвидна", "мало в обращении",
                              "разлоки", "моложе года"}
    assert "эмиссия стейкинга не учтена" in c.flags
    ok = S.evaluate(S.Business("y", "Y", "Dexs", hr_30d=1e6, hr_1y=3e6), _q("y", "y", 100e6, 120e6), None, S.Rules())
    assert ok.passed and ok.flags == ("всплеск выручки за месяц",)
    assert ok.fdv_to_hr == pytest.approx(120e6 / (1e6 * 365 / 30))


def test_rank_prefers_cheap_and_growing():
    mk = lambda g, fdv, hr30, hr1y: S.evaluate(S.Business(g, g, "Dexs", hr_30d=hr30, hr_1y=hr1y),
                                               _q(g, g, fdv, fdv), None, S.Rules())
    cheap_fast = mk("a", 50e6, 1e6, 6e6)       # ×4 к выручке, рост ×2
    dear_slow = mk("b", 5e9, 1e6, 24e6)        # дорого, рост ×0.5
    mid = mk("c", 500e6, 1e6, 12e6)
    out = S.rank([dear_slow, mid, cheap_fast])
    assert [c.gecko_id for c in out] == ["a", "c", "b"]
    assert out[0].score > out[1].score > out[2].score


def _weekly(closes, lows=None):
    idx = pd.date_range("2018-01-01", periods=len(closes), freq="W-MON", tz="UTC")
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"close": c, "low": c if lows is None else np.asarray(lows, dtype=float)}, index=idx)


def test_regime_zones():
    assert S.regime(_weekly([100.0] * 50)).zone == S.Zone.UNKNOWN
    up = np.linspace(100, 400, 300)
    assert S.regime(_weekly(up)).zone == S.Zone.BULL
    down = np.r_[np.linspace(100, 400, 250), np.linspace(400, 100, 60)]
    assert S.regime(_weekly(down)).zone == S.Zone.ACCUMULATION
    # после закрытия под EMA200 — резкий возврат над EMA50 на 2 недели подряд → запуск
    w = _weekly(np.r_[down, [600.0, 620.0]])
    r = S.regime(w)
    assert r.zone == S.Zone.LAUNCH and r.weeks_in_zone == 1
    # бычий рынок, минимум недели у EMA50, закрытие над ней → откат
    lows = up.copy()
    e50 = pd.Series(up).ewm(span=50, adjust=False).mean().iloc[-1]
    lows[-1] = e50 * 1.005
    assert S.regime(_weekly(up, lows)).zone == S.Zone.PULLBACK


def test_unlock_share_and_first_seen():
    pts = [(date(2026, 1, 1), 100.0), (date(2026, 6, 1), 110.0), (date(2027, 3, 1), 130.0)]
    assert S.unlock_share(pts, date(2026, 6, 15)) == pytest.approx(20 / 110)
    assert S.unlock_share(pts, date(2025, 1, 1)) is None
    c = S.evaluate(S.Business("y", "Y", "Dexs", hr_30d=1e6, hr_1y=12e6), _q("y", "y", 100e6, 100e6, price=2.0),
                   None, S.Rules())
    out, seen = S.with_first_seen([c], {}, date(2026, 10, 1))
    assert seen == {"y": ("2026-10-01", 2.0)} and out[0].first_seen == date(2026, 10, 1)
    out2, seen2 = S.with_first_seen([S.replace(c, price=3.0)], seen, date(2026, 10, 9))
    assert seen2 == seen and out2[0].since_seen == pytest.approx(0.5)


def _snap(cands, zone=S.Zone.BULL):
    return S.Snapshot(datetime(2026, 10, 27, tzinfo=timezone.utc), S.Regime(date(2026, 10, 19), 100.0, 90.0, 60.0, zone),
                      tuple(cands), (), rules=S.Rules(top=2))


def test_alerts_new_top_entries_and_regime_change_and_json():
    a, b, c = (S.evaluate(S.Business(g, g.upper(), "Dexs", hr_30d=1e6, hr_1y=12e6), _q(g, g, 100e6, 100e6), None,
                          S.Rules()) for g in "abc")
    prev, cur = _snap([a, b]), _snap([c, a, b], S.Zone.PULLBACK)
    assert [x.gecko_id for x in S.new_entries(prev, cur)] == ["c"]
    assert S.new_entries(None, cur) == []
    assert S.regime_changed(prev, cur)
    text = S.alert_text(cur, S.new_entries(prev, cur), True)
    assert "Откат к EMA50" in text and "C (C)" in text
    back = S.Snapshot.from_json(json.loads(json.dumps(cur.to_json())))
    assert back == cur


def test_parsers():
    raw = {"protocols": [{"name": "Uniswap V3", "slug": "uniswap-v3", "parentProtocol": "parent#uniswap",
                          "category": "Dexs", "total30d": 5, "total1y": None}, "мусор"]}
    rows = F.parse_fee_rows(raw, S.FEES)
    assert rows == [S.FeeRow("uniswap", "uniswap-v3", "Uniswap V3", "Dexs", S.FEES, 5.0, 0.0)]
    ids = F.parse_token_ids([{"slug": "aave-v3", "name": "Aave V3", "parentProtocol": "parent#aave", "gecko_id": "aave"},
                             {"slug": "x", "gecko_id": None}], [{"name": "Near", "gecko_id": "near"}])
    assert ids == {"aave": "aave", "aave-v3": "aave", "near": "near"}
    sched = F.parse_schedule({"documentedData": {"data": [
        {"data": [{"timestamp": 1767225600, "unlocked": 10}, {"timestamp": 1767312000, "unlocked": 20}]},
        {"data": [{"timestamp": 1767312000, "unlocked": 5}]}]}})
    assert sched == [(date(2026, 1, 1), 10.0), (date(2026, 1, 2), 25.0)]
    q = F.parse_quotes([{"id": "aave", "symbol": "aave", "name": "Aave", "current_price": 100, "market_cap": 1e9,
                         "fully_diluted_valuation": None, "total_volume": "5", "price_change_percentage_1y_in_currency": None,
                         "ath_change_percentage": -50}])
    assert q["aave"].symbol == "AAVE" and q["aave"].fdv is None and q["aave"].volume == 5.0 and q["aave"].chg_1y is None
    w = F.parse_weekly([[0, "1", "2", "0.5", "1.5", "0", 1000], [2000, "1", "2", "0.5", "1.7", "0", 9_999_999]], 5000)
    assert list(w["close"]) == [1.5] and list(w["low"]) == [0.5]


class _Store:
    def __init__(self):
        self.kv = {}

    def kv_get(self, k):
        return self.kv.get(k)

    def kv_set(self, k, v):
        self.kv[k] = v


class _Source:
    def __init__(self):
        self.asked: list[str] = []

    async def fee_rows(self):
        return _rows()

    async def token_ids(self):
        return {"uniswap": "uniswap", "near": "near"}

    async def token_id(self, key):
        self.asked.append(key)
        return {"lend": "lend-token", "near-intents": "near"}.get(key)

    async def unlock_schedules(self, keys):
        return {"lend": [(date(2026, 1, 1), 100.0), (date(2027, 6, 1), 105.0)]} if "lend" in keys else {}


class _Quotes:
    async def quotes(self, ids):
        return {"uniswap": _q("uniswap", "uni", 4e9, 6e9), "lend-token": _q("lend-token", "lnd", 100e6, 110e6),
                "near": _q("near", "near", 6e9, 6e9)}


class _Weekly:
    async def weekly(self, symbol):
        return _weekly(np.linspace(100, 400, 300))


class _Notify:
    def __init__(self):
        self.texts = []

    async def text(self, m):
        self.texts.append(m)


def test_service_run_snapshot_seen_and_alert():
    store, note, src = _Store(), _Notify(), _Source()
    clock = [datetime(2026, 10, 27, 12, tzinfo=timezone.utc)]
    svc = ScreenerService(src, _Quotes(), _Weekly(), store, notifier=note, clock=lambda: clock[0])

    async def go():
        assert svc.due(clock[0])
        await svc.maybe_run()
        await svc._task
    asyncio.run(go())
    s = svc.current()
    assert svc.last_error is None and s.regime.zone == S.Zone.BULL
    syms = [c.symbol for c in s.candidates]
    assert set(syms) == {"UNI", "LND", "NEAR"} and s.businesses == 4
    lnd = next(c for c in s.candidates if c.symbol == "LND")
    assert lnd.unlock_12m == pytest.approx(0.05, abs=0.02) and lnd.first_seen == date(2026, 10, 27)
    assert json.loads(store.kv[SEEN_KEY])["lend-token"][0] == "2026-10-27"
    assert note.texts == []                                    # первый снимок — без уведомлений
    assert not svc.due(clock[0])
    clock[0] = clock[0].replace(hour=19)
    assert svc.due(clock[0])
    asked = len(src.asked)
    asyncio.run(go())
    assert len(src.asked) == asked                             # id, найденные поштучно, берутся из памяти


def test_screener_tab_renders(env):  # noqa: F811
    c, tmp = env
    _login(c)
    page = c.get("/").text
    assert 'data-go="screener"' in page and 'id="screener-body"' in page
    r = c.get("/screener")
    assert r.status_code == 200 and "Первый расчёт" in r.text
    a = S.evaluate(S.Business("lend-token", "Lend", "Lending", hr_30d=1e6, hr_1y=6e6, rev_30d=2e6),
                   _q("lend-token", "lnd", 100e6, 110e6), None, S.Rules())
    rej = S.evaluate(S.Business("pump", "PumpSwap", "Dexs", hr_30d=1e6, hr_1y=6e6, rev_30d=3e6),
                     _q("pump", "pump", 1.9e9, 3.5e9), 0.15, S.Rules())
    snap = _snap([a], S.Zone.ACCUMULATION)
    snap = S.replace(snap, rejected=(rej,))
    from trader.infrastructure.sqlite_repo import SqliteStore
    SqliteStore(tmp / "trader.db").kv_set(KEY, json.dumps(snap.to_json()))
    r = c.get("/screener")
    assert "Зона накопления" in r.text and "LND" in r.text and "PUMP" in r.text and "разлоки" in r.text
    assert c.post("/screener/run").status_code == 403         # без заголовка панели
