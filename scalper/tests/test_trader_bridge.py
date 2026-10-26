import asyncio
import base64
import gzip
import json
from datetime import datetime, timedelta, timezone

import httpx

from trader.application.bridge import (DAY_KEY, REPORT_KEY, REPORT_PATH, STATE_PATH, BridgeService, build_snapshot,
                                       encode, summary)
from trader.domain.execution import Trade, TradeResult, TradeStatus
from trader.domain.models import EntryKind, Settings, Side, Signal, Timeframe, TradePlan
from trader.infrastructure.github_state import GitHubState
from trader.infrastructure.sqlite_repo import SqliteStore

NOW = datetime(2026, 10, 21, 0, 40, tzinfo=timezone.utc)
T0 = NOW - timedelta(days=3)


def _signal(symbol="SOLUSDT", tf=Timeframe.H4, side=Side.LONG, bar=T0) -> Signal:
    return Signal(symbol=symbol, timeframe=tf, side=side, bar_time=bar, close=101.0, line_value=100.0,
                  line_points=((bar - timedelta(days=9), 110.0), (bar - timedelta(days=3), 105.0)), aggr=0.61,
                  range_atr=1.2, plan=TradePlan(EntryKind.MARKET, 101.0, 98.0, 110.0, 0))


def _trade(sig: Signal, pnl: float) -> Trade:
    return Trade(sig.id, sig.symbol, sig.side, EntryKind.MARKET, 12.34567, 101.0, 98.0, 110.0, "oid", "live",
                 status=TradeStatus.CLOSED, created_at=T0, filled_at=T0,
                 result=TradeResult(101.0, 110.0, pnl, T0 + timedelta(hours=8)))


def _store(tmp_path):
    st = SqliteStore(tmp_path / "t.db")
    s1 = st.add(_signal())
    s2 = st.add(_signal(side=Side.SHORT, bar=T0 + timedelta(hours=4)))
    t = st.add_trade(_trade(s1, 777.0))
    st.settle_trade(t.id, TradeStatus.CLOSED, TradeResult(101.0, 110.0, 777.0, T0 + timedelta(hours=8)))
    return st, s1, s2


def test_snapshot_has_r_and_prices_but_no_money(tmp_path):
    st, s1, s2 = _store(tmp_path)
    sigs = st.recent(100)
    snap = build_snapshot(sigs, st.trades_for([s.id for s in sigs]), Settings(), {"day": "2026-10-21"}, NOW)
    text = json.dumps(snap)
    for forbidden in ("qty", "notional_usd", "risk_usd", "pnl_usd", "777.0", "12.34567", "oid"):
        assert forbidden not in text
    rows = {r["signal_id"]: r for r in snap["signals"]}
    assert rows[s1.id]["trade"]["r"] is not None and rows[s1.id]["trade"]["exit"] == 110.0
    assert rows[s2.id]["trade"] is None
    assert snap["weather"] == {"day": "2026-10-21"} and snap["settings"]["tf_params"]["15m"]["long_risk_x"] == 0.5
    assert snap["summary"]["4h"]["long"]["n"] == 1 and "short" not in snap["summary"]["4h"]
    assert json.loads(gzip.decompress(encode(snap)))["signals_n"] == 2


def test_snapshot_drops_signals_older_than_the_window(tmp_path):
    st = SqliteStore(tmp_path / "t.db")
    st.add(_signal())
    sigs = st.recent(10)
    assert build_snapshot(sigs, {}, Settings(), None, sigs[0].created_at + timedelta(days=1))["signals_n"] == 1
    assert build_snapshot(sigs, {}, Settings(), None, sigs[0].created_at + timedelta(days=400))["signals_n"] == 0
    assert summary([{"tf": "4h", "side": -1, "trade": {"r": 2.0}}, {"tf": "4h", "side": -1, "trade": {"r": -1.0}},
                    {"tf": "4h", "side": 1, "trade": None}]) == {"4h": {"short": {"n": 2, "sum_r": 1.0, "avg_r": 0.5}}}


class FakeState:
    def __init__(self, report: str | None = None, fail: bool = False) -> None:
        self.files: dict[str, bytes] = {}
        self.report, self.fail = report, fail

    async def put(self, path, data, message):
        if self.fail:
            raise RuntimeError("нет сети")
        self.files[path] = data

    async def get(self, path):
        return self.report if path == REPORT_PATH else None


class Notes:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def text(self, message: str) -> None:
        self.texts.append(message)


async def _run(svc: BridgeService) -> None:
    await svc.maybe_run()
    if svc._task is not None:
        await svc._task


def test_service_pushes_once_a_day_after_the_close_and_forwards_the_report_once(tmp_path):
    st, *_ = _store(tmp_path)
    pub, notes = FakeState(report="# Неделя\nвсё хорошо"), Notes()
    now = [NOW.replace(hour=0, minute=10)]
    svc = BridgeService(pub, st, st, st, st, notes, clock=lambda: now[0])

    async def go():
        await _run(svc)                                            # 00:10 — дневная свеча ещё не прошла запас
        assert STATE_PATH not in pub.files
        now[0] = NOW
        await _run(svc)
        assert json.loads(gzip.decompress(pub.files[STATE_PATH]))["signals_n"] == 2
        assert st.kv_get(DAY_KEY) == "2026-10-21" and len(notes.texts) == 1 and "всё хорошо" in notes.texts[0]
        pub.files.clear()
        now[0] = NOW + timedelta(hours=3)
        await _run(svc)                                            # в тот же день второй раз не отправляем
        assert not pub.files
        now[0] = NOW + timedelta(days=1)
        await _run(svc)
        assert STATE_PATH in pub.files and len(notes.texts) == 1   # отчёт тот же — в Telegram снова не шлём
        pub.report = "# Следующая неделя"
        now[0] = NOW + timedelta(days=2)
        await _run(svc)
        assert len(notes.texts) == 2 and st.kv_get(REPORT_KEY)

    asyncio.run(go())


def test_service_survives_a_publisher_failure_and_retries_later(tmp_path):
    st, *_ = _store(tmp_path)
    pub = FakeState(fail=True)
    now = [NOW]
    svc = BridgeService(pub, st, st, st, st, None, clock=lambda: now[0])

    async def go():
        await _run(svc)
        assert svc.last_error and st.kv_get(DAY_KEY) is None
        now[0] = NOW + timedelta(minutes=10)
        await _run(svc)
        assert pub.files == {}                                     # пауза 30 минут между попытками
        pub.fail, now[0] = False, NOW + timedelta(minutes=45)
        await _run(svc)
        assert STATE_PATH in pub.files and svc.last_error is None

    asyncio.run(go())


def test_github_state_creates_branch_updates_file_with_sha_and_reads_raw():
    calls: list[tuple[str, str, dict]] = []
    sha = {"x": None}

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content) if req.content else {}
        calls.append((req.method, req.url.path, body))
        assert req.headers["authorization"] == "Bearer tok"
        p = req.url.path
        if p.endswith("/branches/bot-state"):
            return httpx.Response(404 if not any(c[0] == "POST" for c in calls) else 200, json={})
        if p == "/repos/o/r":
            return httpx.Response(200, json={"default_branch": "main"})
        if p.endswith("/git/ref/heads/main"):
            return httpx.Response(200, json={"object": {"sha": "abc"}})
        if p.endswith("/git/refs"):
            return httpx.Response(201, json={})
        if p.endswith("/contents/state/a.bin"):
            if req.method == "GET":
                return httpx.Response(404 if sha["x"] is None else 200, json={"sha": sha["x"]})
            sha["x"] = "s1"
            return httpx.Response(200, json={})
        if p.endswith("/contents/reports/latest.md"):
            return httpx.Response(200, text="# отчёт") if "raw" in req.headers["accept"] else httpx.Response(500)
        return httpx.Response(404)

    gh = GitHubState("tok", "o/r", "bot-state", httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://api.github.com"))

    async def go():
        await gh.put("state/a.bin", b"\x01\x02", "m1")
        await gh.put("state/a.bin", b"\x03", "m2")
        assert await gh.get("reports/latest.md") == "# отчёт" and await gh.get("nope.md") is None

    asyncio.run(go())
    assert ("POST", "/repos/o/r/git/refs", {"ref": "refs/heads/bot-state", "sha": "abc"}) in calls
    puts = [c[2] for c in calls if c[0] == "PUT"]
    assert "sha" not in puts[0] and base64.b64decode(puts[0]["content"]) == b"\x01\x02"
    assert puts[1]["sha"] == "s1" and puts[1]["branch"] == "bot-state"
