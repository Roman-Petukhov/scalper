import asyncio
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from trader.application.services import Scanner, SettingsService, SignalDecisions
from trader.domain.models import EntryKind, EntryPolicy, Mode, Settings, Side, Signal, SignalStatus, Timeframe, TradePlan
from trader.infrastructure.binance_data import BinanceMarketData
from trader.infrastructure.charts import MatplotlibCharts
from trader.infrastructure.sqlite_repo import SqliteStore
from trader.infrastructure.telegram import caption

_NOW = datetime.now(timezone.utc)
T0 = _NOW.replace(hour=_NOW.hour // 4 * 4, minute=0, second=0, microsecond=0) - timedelta(hours=4)   # свеча пробоя — прошлая 4h


def _signal(symbol: str = "SOLUSDT", tf: Timeframe = Timeframe.H4, side: Side = Side.LONG) -> Signal:
    return Signal(symbol=symbol, timeframe=tf, side=side, bar_time=T0, close=101.0, line_value=100.0,
                  line_points=((datetime(2026, 9, 20, tzinfo=timezone.utc), 110.0), (datetime(2026, 9, 26, tzinfo=timezone.utc), 105.0)),
                  aggr=0.61, range_atr=1.2, plan=TradePlan(EntryKind.MARKET, 101.0, 98.0, 110.0, 0))


def test_store_roundtrip_dedup_and_status(tmp_path):
    st = SqliteStore(tmp_path / "t.db")
    s = st.add(_signal())
    assert s is not None and s.id and s.status is SignalStatus.NEW
    assert st.add(_signal()) is None                                    # та же свеча и сторона — дубль
    assert st.add(_signal(side=Side.SHORT)) is not None
    got = st.get(s.id)
    assert got.plan == s.plan and got.line_points == s.line_points and got.bar_time == T0
    st.set_status(s.id, SignalStatus.SKIPPED, "пропущен")
    assert st.get(s.id).status is SignalStatus.SKIPPED
    assert len(st.recent(timeframes={Timeframe.M15})) == 0 and len(st.recent()) == 2


def test_settings_persist_and_service_rules(tmp_path):
    st = SqliteStore(tmp_path / "t.db")
    svc = SettingsService(st)
    assert svc.get() == Settings()
    svc.toggle_timeframe(Timeframe.M15)
    svc.set_mode(Mode.AUTO)
    svc.update({"max_positions": 3}, {Timeframe.M15: {"target_r": 2.0}})
    s = SqliteStore(tmp_path / "t.db").load()
    assert s.mode is Mode.AUTO and s.timeframes == {Timeframe.H4, Timeframe.M15} and s.p(Timeframe.H4).target_r == 3.0
    assert s.max_positions == 3 and s.p(Timeframe.M15).target_r == 2.0
    svc.update({}, {Timeframe.H4: {"risk_pct": 0.5}})
    assert SqliteStore(tmp_path / "t.db").load().p(Timeframe.H4).risk_pct == 0.5
    with pytest.raises(ValueError):
        svc.update({}, {Timeframe.H4: {"risk_pct": 9.0}})
    with pytest.raises(ValueError):
        svc.update({"api_key": 1}, {})
    assert SqliteStore(tmp_path / "t.db").load().p(Timeframe.H4).risk_pct == 0.5      # ошибка — ничего не сохранено


def test_old_settings_format_migrates(tmp_path):
    st = SqliteStore(tmp_path / "t.db")
    old = {"mode": "auto", "timeframes": ["1h", "4h", "15m"], "auto_timeframes": ["1h", "4h"], "risk_pct": 0.8,
           "risk_pct_15m": 0.3, "min_close_loc": 0.6, "min_aggr": 0.6, "target_r": 2.5, "entry_policy": "hybrid",
           "leverage": 7, "max_positions": 4}
    st.db.execute("INSERT INTO settings(id, payload) VALUES (1, ?)", (json.dumps(old),))
    st.db.commit()
    s = st.load()
    assert s.mode is Mode.AUTO and s.leverage == 7 and s.max_positions == 4
    h4, m15 = s.p(Timeframe.H4), s.p(Timeframe.M15)
    assert (h4.risk_pct, h4.min_close_loc, h4.min_aggr, h4.entry_policy) == (0.8, 0.6, 0.6, EntryPolicy.HYBRID)
    assert (m15.risk_pct, m15.min_close_loc, m15.target_r) == (0.3, 0.5, 2.5)
    assert s.timeframes == {Timeframe.H4, Timeframe.M15} and s.auto_timeframes == {Timeframe.H4}   # 1h убран
    st.save(s)
    assert st.load() == s


def test_15m_moves_to_gentle_shorts_once_and_keeps_trader_edits(tmp_path):
    from trader.domain.models import DEFAULT_TF_PARAMS, SideFilter
    st = SqliteStore(tmp_path / "t.db")
    st.db.execute("DELETE FROM kv")                                     # база из версии до новой стратегии 15m
    old = Settings(timeframes=frozenset({Timeframe.H4}), auto_timeframes=frozenset(Timeframe)).with_tf(
        Timeframe.M15, htf_confirm_h=12, entry_policy=EntryPolicy.RETEST, risk_pct=0.4)
    st.save(old)
    st.db.commit()
    s = SqliteStore(tmp_path / "t.db").load()
    assert s.timeframes == set(Timeframe) and s.auto_timeframes == {Timeframe.H4}
    assert s.p(Timeframe.M15) == DEFAULT_TF_PARAMS[Timeframe.M15] and s.p(Timeframe.M15).sides is SideFilter.SHORT
    assert s.p(Timeframe.H4) == old.p(Timeframe.H4)
    st2 = SqliteStore(tmp_path / "t.db")
    st2.save(s.with_tf(Timeframe.M15, risk_pct=0.5).toggle_auto(Timeframe.M15))   # правки трейдера не перетираются
    s2 = SqliteStore(tmp_path / "t.db").load()
    assert s2.p(Timeframe.M15).risk_pct == 0.5 and Timeframe.M15 in s2.auto_timeframes


def test_decisions_only_in_manual_mode_and_once(tmp_path):
    st = SqliteStore(tmp_path / "t.db")
    dec = SignalDecisions(st, st)
    s = st.add(_signal())
    assert asyncio.run(dec.take(s.id)).status is SignalStatus.TAKEN
    with pytest.raises(ValueError):
        dec.skip(s.id)
    s2 = st.add(_signal(side=Side.SHORT))
    st.save(replace(Settings(), mode=Mode.AUTO))
    with pytest.raises(ValueError):
        asyncio.run(dec.take(s2.id))


class _Market:
    def __init__(self, frames):
        self.frames = frames

    async def universe(self, min_turnover_usd):
        return list(self.frames)

    async def closed_bars(self, symbol, tf):
        if symbol == "BAD":
            raise RuntimeError("сеть")
        return self.frames[symbol]


class _Notifier:
    def __init__(self):
        self.sent = []

    async def signal(self, signal, chart_path, panel_url):
        self.sent.append(signal.symbol)

    async def text(self, message):
        pass


def test_scanner_respects_timeframe_chips_and_survives_errors(tmp_path, monkeypatch):
    import trader.application.services as svc_mod
    st = SqliteStore(tmp_path / "t.db")
    idx = pd.date_range("2026-01-01", periods=10, freq="4h", tz="UTC")
    bars = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0, "taker_buy_volume": 0.5},
                        index=idx)
    monkeypatch.setattr(svc_mod, "detect", lambda d, tf, sym, s: [replace(_signal(sym, tf), bar_time=d.index[-1].to_pydatetime())])
    note = _Notifier()

    class _Charts:
        def render(self, signal, b):
            return str(tmp_path / f"{signal.id}.png")

    sc = Scanner(_Market({"AAA": bars, "BAD": bars}), st, st, _Charts(), note, "https://panel")
    rep = asyncio.run(sc.scan(Timeframe.M15))                 # 15m выключен — сканирования нет
    assert rep.symbols == 0 and not rep.signals
    rep = asyncio.run(sc.scan(Timeframe.H4))
    assert rep.errors == 1 and [s.symbol for s in rep.signals] == ["AAA"] and note.sent == ["AAA"]
    assert rep.signals[0].chart_path.endswith(".png")
    rep = asyncio.run(sc.scan(Timeframe.H4))                  # тот же бар — без повторного сигнала
    assert not rep.signals


def test_binance_closed_bars_drop_open_candle_and_merge_tail():
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    hour = 3_600_000
    calls = []

    def kl(start, n):
        return [[start + i * hour, "1", "2", "0.5", "1.5", "10", start + (i + 1) * hour - 1, "15", 3, "6", "9", "0"]
                for i in range(n)]

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(int(req.url.params["limit"]))
        base = now_ms - (now_ms % hour) - 3 * hour
        return httpx.Response(200, json=kl(base, 4))           # последняя свеча ещё не закрыта

    md = BinanceMarketData(httpx.AsyncClient(base_url="https://x", transport=httpx.MockTransport(handler)))
    d = asyncio.run(md.closed_bars("SOLUSDT", Timeframe.M15))
    assert len(d) == 3 and d["taker_buy_volume"].iloc[0] == 6.0 and "close_time" not in d
    d2 = asyncio.run(md.closed_bars("SOLUSDT", Timeframe.M15))
    assert calls == [1500, 6] and len(d2) == 3


def test_binance_universe_filters_turnover_contracts_and_non_crypto():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("exchangeInfo"):
            return httpx.Response(200, json={"symbols": [
                {"symbol": "AUSDT", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT"},
                {"symbol": "BUSDT", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT"},
                {"symbol": "CUSDT", "status": "SETTLING", "contractType": "PERPETUAL", "quoteAsset": "USDT"},
                {"symbol": "XAUTUSDT", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT",
                 "baseAsset": "XAUT", "underlyingType": "COIN"},
                {"symbol": "TSLAUSDT", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT",
                 "baseAsset": "TSLA", "underlyingType": "EQUITY"},
                {"symbol": "DUSDT", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT",
                 "baseAsset": "D", "underlyingType": "COIN", "underlyingSubType": ["TradFi"]}]})
        return httpx.Response(200, json=[{"symbol": "AUSDT", "quoteVolume": "5e7"}, {"symbol": "BUSDT", "quoteVolume": "1e6"},
                                         {"symbol": "CUSDT", "quoteVolume": "9e9"}, {"symbol": "XAUTUSDT", "quoteVolume": "9e9"},
                                         {"symbol": "TSLAUSDT", "quoteVolume": "9e9"}, {"symbol": "DUSDT", "quoteVolume": "9e9"}])

    md = BinanceMarketData(httpx.AsyncClient(base_url="https://x", transport=httpx.MockTransport(handler)))
    assert asyncio.run(md.universe(20e6)) == ["AUSDT"]


def test_chart_and_caption(tmp_path):
    idx = pd.date_range("2026-09-15", periods=200, freq="4h", tz="UTC")
    import numpy as np
    c = 100 + np.cumsum(np.random.default_rng(1).normal(0, 1, 200))
    bars = pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1.0, "taker_buy_volume": 0.5},
                        index=idx)
    sig = replace(_signal(), id=7, line_points=((idx[20].to_pydatetime(), c[20]), (idx[80].to_pydatetime(), c[80])))
    path = MatplotlibCharts(tmp_path).render(sig, bars)
    assert path.endswith(".png") and (tmp_path / path.split("/")[-1]).stat().st_size > 10_000
    text = caption(sig, "https://panel.example")
    assert "SOLUSDT" in text and "#signal-7" in text and "лонг" in text


def test_entry_policy_persists(tmp_path):
    from trader.domain.models import EntryPolicy
    st = SqliteStore(tmp_path / "t.db")
    assert st.load().p(Timeframe.H4).entry_policy is EntryPolicy.RETEST
    SettingsService(st).update({}, {Timeframe.M15: {"entry_policy": EntryPolicy.HYBRID}})
    s = SqliteStore(tmp_path / "t.db").load()
    assert s.p(Timeframe.M15).entry_policy is EntryPolicy.HYBRID and s.p(Timeframe.H4).entry_policy is EntryPolicy.RETEST


def test_png_chart_log_scale(tmp_path):
    idx = pd.date_range("2026-09-01", periods=200, freq="4h", tz="UTC")
    px = 100 * np.exp(np.linspace(0, 0.3, 200))
    bars = pd.DataFrame({"open": px, "high": px * 1.01, "low": px * 0.99, "close": px}, index=idx)
    sig = replace(_signal(), extra={"log_line": True}, id=7,
                  line_points=((idx[20].to_pydatetime(), float(px[20])), (idx[120].to_pydatetime(), float(px[120]))))
    path = MatplotlibCharts(tmp_path).render(sig, bars)
    assert Path(path).stat().st_size > 10_000


def test_stale_signals_expire_and_cannot_be_taken(tmp_path):
    st = SqliteStore(tmp_path / "t.db")
    dec = SignalDecisions(st, st)
    old = st.add(replace(_signal("OLDUSDT"), bar_time=T0 - timedelta(days=2)))
    fresh = st.add(_signal("NEWUSDT"))
    assert old.valid_until() == old.bar_time + timedelta(hours=12)        # свеча пробоя + 2 свечи
    assert dec.expire_stale() == 1
    assert st.get(old.id).status is SignalStatus.EXPIRED and st.get(fresh.id).status is SignalStatus.NEW
    stale = st.add(replace(_signal("ETHUSDT"), bar_time=T0 - timedelta(days=1)))
    with pytest.raises(ValueError, match="устарел"):
        asyncio.run(dec.take(stale.id))
    assert st.get(stale.id).status is SignalStatus.EXPIRED


def test_reset_defaults_keeps_mode(tmp_path):
    st = SqliteStore(tmp_path / "t.db")
    svc = SettingsService(st)
    st.save(replace(Settings(), mode=Mode.AUTO, leverage=8, timeframes=frozenset(Timeframe)).with_tf(
        Timeframe.H4, risk_pct=0.5, entry_policy=EntryPolicy.HYBRID, min_close_loc=0.0, min_aggr=0.6))
    s = svc.reset_defaults()
    h4 = s.p(Timeframe.H4)
    assert s.timeframes == frozenset(Timeframe) and h4.entry_policy is EntryPolicy.RETEST
    assert h4.min_close_loc == 0.5 and h4.min_aggr == 0.55 and h4.target_r == 3.0
    assert (s.mode, h4.risk_pct, s.leverage) == (Mode.AUTO, 1.0, 5)
