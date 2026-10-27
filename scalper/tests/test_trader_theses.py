import asyncio
import json
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from trader.application.theses import CHECK_KEY, KEY, READ_KEY, TVL_KEY, ThesesService
from trader.domain import screener as S
from trader.domain import theses as T
from trader.infrastructure import fundamentals as F

from test_trader_web import env, _login  # noqa: F401  (фикстура панели)

NOW = datetime(2026, 10, 27, 12, tzinfo=timezone.utc)


def _thesis(**kw):
    base = dict(id="perps", title="Ончейн-деривативы", shift="Объём уходит с CEX", chain="регулирование → DEX",
                beneficiaries=(T.Beneficiary("HYPE", "биржа", "strong"),),
                indicators=(T.IndicatorRef("fees:derivatives", "up", "комиссии растут"),),
                catalysts=("листинг ETF",), kill="доля DEX падает 3 месяца", horizon="1–2 года", confidence="medium",
                priced_in="частично", verdict="new", sources=(T.Source("DefiLlama", "https://defillama.com"),))
    base.update(kw)
    return T.Thesis(**base)


def _model_output(**kw):
    t = {"id": "perps", "title": "Ончейн-деривативы", "shift": "s", "chain": "c",
         "beneficiaries": [{"symbol": " hype ", "role": "биржа", "capture": "strong"},
                           {"symbol": "", "role": "пусто", "capture": "none"}],
         "indicators": [{"id": "fees:derivatives", "direction": "up", "why": "w"},
                        {"id": "made-up", "direction": "up", "why": "w"}],
         "catalysts": ["x"], "kill": "k", "horizon": "1–2 года", "confidence": "medium", "priced_in": "p",
         "verdict": "new", "sources": [{"title": "a", "url": "https://a.example"}], "risk": False}
    t.update(kw)
    return {"macro": "m", "summary": "s", "theses": [t]}


# --- домен ---

def test_indicator_status_ratio_and_level():
    up, down = T.IndicatorRef("x", "up", ""), T.IndicatorRef("x", "down", "")
    ratio = lambda c: T.Reading("x", "x", 1.0, "$", c, "ratio")       # noqa: E731
    level = lambda c: T.Reading("x", "x", 1.0, "$", c, "level")       # noqa: E731
    assert T.indicator_status(up, ratio(1.2)) == T.Status.CONFIRMED
    assert T.indicator_status(down, ratio(1.2)) == T.Status.AGAINST
    assert T.indicator_status(up, ratio(1.01)) == T.Status.FLAT
    assert T.indicator_status(down, ratio(0.9)) == T.Status.CONFIRMED
    assert T.indicator_status(up, level(-0.05)) == T.Status.AGAINST
    assert T.indicator_status(up, level(0.01)) == T.Status.FLAT
    assert T.indicator_status(up, level(None)) == T.Status.NO_DATA
    assert T.indicator_status(up, level(float("nan"))) == T.Status.NO_DATA
    assert T.indicator_status(up, None) == T.Status.NO_DATA


def test_thesis_status_majority():
    refs = tuple(T.IndicatorRef(i, "up", "") for i in "abc")
    t = _thesis(indicators=refs)
    r = lambda i, c: T.Reading(i, i, 1.0, "$", c, "level")             # noqa: E731
    assert T.thesis_status(t, {}) == T.Status.NO_DATA
    assert T.thesis_status(t, {"a": r("a", .1), "b": r("b", .1), "c": r("c", -.1)}) == T.Status.CONFIRMED
    assert T.thesis_status(t, {"a": r("a", -.1), "b": r("b", -.1)}) == T.Status.AGAINST
    assert T.thesis_status(t, {"a": r("a", .1), "b": r("b", -.1)}) == T.Status.FLAT


def test_performance_is_excess_over_btc_and_report_roundtrip():
    rep = T.Report(NOW, "m", "macro", "sum", (_thesis(), _thesis(id="b", beneficiaries=(
        T.Beneficiary("ENA", "r", "weak"), T.Beneficiary("HYPE", "r", "strong")))),
        prices={"BTC": 100.0, "HYPE": 10.0, "ENA": 1.0}, cost_usd=1.5)
    assert rep.symbols == ["HYPE", "ENA"]
    perf = T.performance(rep, {"BTC": 110.0, "HYPE": 15.0})
    assert perf["HYPE"] == pytest.approx(0.4) and perf["ENA"] is None
    assert T.thesis_performance(rep.theses[1], perf) == pytest.approx(0.4)
    assert T.performance(rep, {"HYPE": 15.0})["HYPE"] is None              # нет BTC — сравнивать не с чем
    back = T.Report.from_json(json.loads(json.dumps(rep.to_json())))
    assert back == rep


def test_with_screener_marks():
    rep = T.Report(NOW, "m", "", "", (_thesis(beneficiaries=(T.Beneficiary("HYPE", "r", "strong"),
                                                             T.Beneficiary("PUMP", "r", "strong"),
                                                             T.Beneficiary("ZEC", "r", "none"))),))
    out = T.with_screener(rep, {"HYPE": 3}, {"PUMP": "разлоки"})
    assert [b.screener for b in out.theses[0].beneficiaries] == ["скринер: №3", "скринер отсеял: разлоки",
                                                                "нет в скринере"]


def test_parse_output_drops_unsourced_theses_and_unknown_indicators():
    rep = T.parse_output(_model_output(), NOW, "m", {"fees:derivatives"})
    t = rep.theses[0]
    assert [b.symbol for b in t.beneficiaries] == ["HYPE"]
    assert [i.id for i in t.indicators] == ["fees:derivatives"]
    assert T.parse_output(_model_output(sources=[{"title": "x", "url": "javascript:alert(1)"}]), NOW, "m",
                          set()).theses == ()
    weird = T.parse_output(_model_output(confidence="sure", verdict="??"), NOW, "m", set()).theses[0]
    assert weird.confidence == "low" and weird.verdict == "new"


def test_output_schema_restricts_indicator_ids():
    s = T.output_schema(["fees:dexs", "stablecoins"])
    ind = s["properties"]["theses"]["items"]["properties"]["indicators"]["items"]["properties"]["id"]
    assert ind["enum"] == ["fees:dexs", "stablecoins"]
    assert s["additionalProperties"] is False
    assert T.output_schema([])["properties"]["theses"]["items"]["properties"]["indicators"]["items"][
        "properties"]["id"] == {"type": "string"}


def test_readings_from_fees_tvl_and_stablecoins():
    R = S.FeeRow
    rows = [R("a", "a", "A", "Derivatives", "dailyFees", 30e6, 120e6), R("b", "b", "B", "Derivatives",
                                                                          "dailyFees", 10e6, 120e6),
            R("a", "a", "A", "Derivatives", "dailyHoldersRevenue", 99e6, 99e6),
            R("c", "c", "C", "Tiny", "dailyFees", 0.5e6, 6e6)]
    fr = T.fee_readings(rows)
    assert [r.id for r in fr] == ["fees:derivatives"] and fr[0].change == pytest.approx(40e6 / 20e6)
    day = date(2026, 10, 27)
    tv = T.tvl_readings({"Lending": 50e9, "RWA": 20e9, "Small": 1e6},
                        {"Lending": {day - timedelta(days=31): 40e9}}, day)
    assert [r.id for r in tv] == ["tvl:lending", "tvl:rwa"]
    assert tv[0].change == pytest.approx(0.25) and tv[1].change is None
    series = {day - timedelta(days=d): 300e9 - d * 1e9 for d in range(0, 40)}
    st = T.stablecoin_reading(series, day)
    assert st.value == 300e9 and st.change == pytest.approx(300 / 270 - 1)
    assert T.stablecoin_reading({}, day) is None


def test_alert_and_catalog_text():
    rep = T.Report(NOW, "m", "", "Главное за месяц", (_thesis(), _thesis(id="q", title="Квантовый риск",
                                                                        risk=True, beneficiaries=())))
    txt = T.alert_text(rep, "http://panel")
    assert "Ончейн-деривативы" in txt and "http://panel" in txt
    cat = T.catalog_text([T.Reading("stablecoins", "Стейблкоины", 300e9, "$", 0.03)])
    assert "stablecoins" in cat


# --- инфраструктура ---

def test_parse_prices_and_tvl_by_category():
    raw = [{"symbol": "BTCUSDT", "price": "80000"}, {"symbol": "HYPEUSDT", "price": "40.5"},
           {"symbol": "ZERUSDT", "price": "0"}, {"symbol": "ETHBTC", "price": "0.03"}, {"bad": 1}]
    assert F.parse_prices(raw, ["BTC", "HYPE", "ZER", "ETH"]) == {"BTC": 80000.0, "HYPE": 40.5}
    assert F.parse_prices({"code": -1}, ["BTC"]) == {}
    protos = [{"category": "Lending", "tvl": 10}, {"category": "Lending", "tvl": 5},
              {"category": "CEX", "tvl": 1000}, {"category": "Chain", "tvl": 7}, {"tvl": 3}, "junk"]
    assert F.parse_tvl_by_category(protos) == {"Lending": 15.0}


# --- сервис ---

class _Store:
    def __init__(self):
        self.kv = {}

    def kv_get(self, k):
        return self.kv.get(k)

    def kv_set(self, k, v):
        self.kv[k] = v


class _Fund:
    async def fee_rows(self):
        return [S.FeeRow("a", "a", "A", "Derivatives", "dailyFees", 30e6, 120e6)]

    async def tvl_by_category(self):
        return {"Lending": 50e9}


class _Stables:
    async def supply(self):
        idx = pd.date_range(end=NOW.date(), periods=40, freq="D")
        return pd.Series([270e9 + i * 1e9 for i in range(40)], index=idx)


class _Prices:
    def __init__(self):
        self.asked = []

    async def prices(self, symbols):
        self.asked.append(list(symbols))
        return {"BTC": 80000.0, "HYPE": 40.0}


class _Writer:
    def __init__(self):
        self.contexts, self.schemas = [], []

    async def write(self, context, schema):
        self.contexts.append(context)
        self.schemas.append(schema)
        return _model_output(), T_usage()

    async def check(self, context):
        self.contexts.append(context)
        return {"items": [{"id": "perps", "change": "against", "note": "доля DEX упала", "sources": []},
                          {"id": "ghost", "change": "kill", "note": "нет такого", "sources": []}]}, T_usage()


class T_usage:
    input_tokens, output_tokens, searches, cost_usd = 1000, 200, 3, 0.42


class _Notify:
    def __init__(self):
        self.sent = []

    async def text(self, s):
        self.sent.append(s)


def _snap():
    c = S.evaluate(S.Business("hl", "Hyperliquid", "Derivatives", hr_30d=50e6, hr_1y=500e6),
                   S.Quote("hyperliquid", "HYPE", "Hyperliquid", 40.0, 13e9, 40e9, 300e6, 50.0, -20.0), 0.01,
                   S.Rules())
    return S.Snapshot(NOW, S.Regime(date(2026, 10, 19), 80000.0, 90000.0, 70000.0, S.Zone.BULL), (c,), ())


def _svc(writer=None, clock=None):
    store, note, prices = _Store(), _Notify(), _Prices()
    svc = ThesesService(writer, _Fund(), _Fund(), _Stables(), prices, store, _snap, notifier=note,
                        panel_url="http://panel", model="claude-test", clock=clock or (lambda: NOW))
    return svc, store, note, prices


def test_service_report_check_and_schedule():
    w = _Writer()
    clock = [NOW]
    svc, store, note, prices = _svc(w, lambda: clock[0])
    assert svc._due("readings", NOW) and svc._due("report", NOW) and not svc._due("check", NOW)

    rep = asyncio.run(svc.run_report())
    assert rep.prices == {"BTC": 80000.0, "HYPE": 40.0} and rep.cost_usd == 0.42 and rep.model == "claude-test"
    assert rep.theses[0].beneficiaries[0].screener == "скринер: №1"
    assert "fees:derivatives" in w.contexts[0] and "HYPE" in w.contexts[0] and "Прошлой версии нет" in w.contexts[0]
    enum = w.schemas[0]["properties"]["theses"]["items"]["properties"]["indicators"]["items"]["properties"]["id"]
    assert set(enum["enum"]) == {"fees:derivatives", "tvl:lending", "stablecoins"}
    assert json.loads(store.kv[KEY])["theses"][0]["id"] == "perps" and f"{KEY}:2026-10-27" in store.kv
    assert "Lending" in json.loads(store.kv[TVL_KEY]) and READ_KEY in store.kv
    assert note.sent and "Ончейн-деривативы" in note.sent[0]

    view = svc.view()
    assert view["rows"][0]["status"] == T.Status.CONFIRMED                  # комиссии ×4 к среднему месяцу
    assert view["perf"]["HYPE"] == pytest.approx(0.0) and view["has_ai"]

    assert not svc._due("report", NOW + timedelta(days=29)) and svc._due("report", NOW + timedelta(days=30))
    assert not svc._due("check", NOW + timedelta(days=6)) and svc._due("check", NOW + timedelta(days=7))

    clock[0] = NOW + timedelta(days=7)
    chk = asyncio.run(svc.run_check())
    assert [x["id"] for x in chk["items"]] == ["perps"]                      # чужие id отброшены
    assert json.loads(store.kv[CHECK_KEY])["items"][0]["change"] == "against"
    assert "против" in note.sent[-1]
    assert svc.view()["rows"][0]["check"]["note"] == "доля DEX упала"
    assert not svc._due("check", clock[0] + timedelta(days=1))
    assert "Прошлая версия" in svc._context({}, "")


def test_service_without_key_only_reads_indicators():
    svc, store, _, _ = _svc(None)
    assert not svc._due("report", NOW) and not svc._due("check", NOW) and svc._due("readings", NOW)

    async def go():
        assert not svc.start("report") and svc.start("readings", NOW)
        assert not svc.start("readings", NOW)                               # уже идёт
        await svc._task
    asyncio.run(go())
    v = svc.view()
    assert v["report"] is None and not v["has_ai"] and set(v["readings"]) == {"fees:derivatives", "tvl:lending",
                                                                               "stablecoins"}
    assert v["error"] is None and not svc._due("readings", NOW)
    assert svc._due("readings", NOW + timedelta(hours=25))


def test_service_failure_is_recorded_and_retried_later():
    class Broken(_Writer):
        async def write(self, context, schema):
            raise RuntimeError("overloaded")

    svc, store, _, _ = _svc(Broken())

    async def go():
        await svc.maybe_run()                                               # сначала индикаторы
        await svc._task
        await svc.maybe_run()                                               # затем отчёт — падает
        await svc._task
    asyncio.run(go())
    assert "overloaded" in svc.view()["error"] and KEY not in store.kv
    assert not svc._due("report", NOW + timedelta(minutes=30)) and svc._due("report", NOW + timedelta(hours=1))


# --- вкладка ---

def test_theses_tab_renders(env):  # noqa: F811
    c, tmp = env
    _login(c)
    page = c.get("/").text
    assert 'data-go="theses"' in page and 'id="theses-body"' in page
    r = c.get("/theses")
    assert r.status_code == 200 and "ANTHROPIC_API_KEY" in r.text
    rep = T.Report(NOW, "claude-test", "Ставки высокие", "Деньги идут в ончейн-деривативы",
                   (_thesis(), _thesis(id="q", title="Квантовый риск", risk=True, beneficiaries=())),
                   prices={"BTC": 80000.0, "HYPE": 40.0}, cost_usd=1.23)
    from trader.infrastructure.sqlite_repo import SqliteStore
    db = SqliteStore(tmp / "trader.db")
    db.kv_set(KEY, json.dumps(rep.to_json()))
    db.kv_set(READ_KEY, json.dumps({"at": NOW.isoformat(), "items": [
        {"id": "fees:derivatives", "title": "Комиссии деривативов", "value": 30e6, "unit": "$", "change": 1.4,
         "kind": "ratio", "note": ""}]}))
    db.kv_set(CHECK_KEY, json.dumps({"at": NOW.isoformat(), "items": [
        {"id": "perps", "change": "support", "note": "новый рекорд объёма", "sources": [
            {"title": "x", "url": "https://x.example"}]}], "cost_usd": 0.05}))
    r = c.get("/theses").text
    for s in ("Ончейн-деривативы", "Квантовый риск", "подтверждается", "Комиссии деривативов", "новый рекорд объёма",
              "Токена-получателя нет", "$1.23", "HYPE"):
        assert s in r, s
    assert c.post("/theses/report").status_code == 403                     # без заголовка панели
