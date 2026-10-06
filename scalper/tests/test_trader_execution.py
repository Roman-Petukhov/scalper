import asyncio
from dataclasses import replace
from datetime import timedelta

import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
import pytest

from trader.application.execution import Executor
from trader.application.services import Scanner, SignalDecisions
from trader.domain.execution import (Account, ClosedPnl, ExecutionRefused, Instrument, Position, Trade, TradeStatus,
                                     build_order,
                                     round_down, round_price)
from trader.domain.models import EntryKind, Mode, Settings, Side, SignalStatus, Timeframe, TradePlan
from trader.infrastructure.bybit import BrokerHolder, BybitBroker, BybitCredentials
from trader.infrastructure.sqlite_repo import SqliteStore

from test_trader_app import T0, _Market, _signal

NOW = T0 + timedelta(hours=4, minutes=1)
INST = Instrument("SOLUSDT", qty_step=0.01, min_qty=0.01, tick=0.001, max_leverage=50)
ACC = Account(equity=1000.0, available=1000.0, upnl=0.0)


def _sig(**kw):
    s = _signal()
    return replace(s, id=1, **kw)


# ---------------- домен ----------------
def test_market_order_sized_from_risk_and_rounded():
    o = build_order(_sig(), Settings(), ACC, INST, price=101.0, day_start_equity=1000.0, now=NOW)
    assert o.kind is EntryKind.MARKET and o.price == 101.0 and o.stop == 98.0 and o.target == 110.0
    assert o.qty == 3.33 and o.leverage == 5 and o.risk_usd == pytest.approx(9.99)
    assert o.client_id == "tt-1" and "лонг SOLUSDT" in o.describe()


@pytest.mark.parametrize("acc,price,day,msg", [
    (replace(ACC, positions=(Position("SOLUSDT", Side.LONG, 1, 100, 101, 1),)), 101.0, 1000.0, "уже есть"),
    (replace(ACC, pending_symbols=frozenset({f"X{i}USDT" for i in range(12)})), 101.0, 1000.0, "открыто 12 из 12"),
    (replace(ACC, equity=950.0), 101.0, 1000.0, "дневной стоп"),
    (ACC, 97.0, 1000.0, "за стопом"),
    (ACC, 111.0, 1000.0, "до цели"),
    (ACC, 106.0, 1000.0, "меньше 1.5 стопа"),
    (replace(ACC, available=0.0), 101.0, 1000.0, "свободной маржи"),
])
def test_guards_refuse_with_reason(acc, price, day, msg):
    with pytest.raises(ExecutionRefused, match=msg):
        build_order(_sig(), Settings(), acc, INST, price, day, NOW)


def test_leverage_caps_notional_and_is_sent_to_exchange():
    tight = _sig(plan=TradePlan(EntryKind.MARKET, 101.0, 100.5, 102.5, 0))      # стоп 0.5%: на риск 1% нужно 2×
    o = build_order(tight, Settings(leverage=1), ACC, INST, 101.0, None, NOW)
    assert o.leverage == 1 and o.qty * 101.0 <= 1000.0 and o.risk_usd < 10.0    # плечо 1× урезало риск
    o = build_order(tight, Settings(leverage=10), ACC, INST, 101.0, None, NOW)
    assert o.leverage == 10 and o.risk_usd == pytest.approx(10.0, abs=0.01)
    o = build_order(tight, Settings(leverage=10), ACC, replace(INST, max_leverage=3), 101.0, None, NOW)
    assert o.leverage == 3                                                      # не больше, чем позволяет монета
    for bad in (0, 11):
        with pytest.raises(ValueError, match="плечо"):
            Settings(leverage=bad)


def test_retest_limit_has_expiry_and_refuses_when_late():
    plan = TradePlan(EntryKind.RETEST, 100.0, 98.0, 106.0, 12)
    sig = _sig(plan=plan)
    o = build_order(sig, Settings(), ACC, INST, 101.5, None, NOW)
    assert o.kind is EntryKind.RETEST and o.price == 100.0 and o.expires_at == T0 + timedelta(hours=4 * 13)
    with pytest.raises(ExecutionRefused, match="ретест"):
        build_order(sig, Settings(), ACC, INST, 101.5, None, T0 + timedelta(hours=4 * 13))


def test_too_small_account_refused():
    inst = replace(INST, min_qty=10.0, qty_step=1.0)
    with pytest.raises(ExecutionRefused, match="меньше минимального"):
        build_order(_sig(), Settings().with_tf(Timeframe.H4, risk_pct=0.1), ACC, inst, 101.0, None, NOW)


def test_rounding_helpers():
    assert round_down(3.339, 0.01) == 3.33 and round_down(7.9, 1) == 7 and round_price(101.2345, 0.01) == 101.23


# ---------------- сервис ----------------
class FakeBroker:
    network = "demo"

    def __init__(self, acc=ACC, price=101.0, fail=None):
        self.acc, self.px, self.fail = acc, price, fail
        self.placed, self.cancelled, self.open, self.closed = [], [], set(), []
        self.pnl = []                                          # (монета, ClosedPnl)
        self.adjusted = []                                     # (монета, объём со знаком) — ордера хеджа

    async def account(self):
        return self.acc

    async def instrument(self, symbol):
        return replace(INST, symbol=symbol) if symbol != "NOPEUSDT" else None

    async def price(self, symbol):
        return self.px

    async def place(self, order):
        if self.fail:
            raise self.fail
        self.placed.append(order)
        oid = f"o{len(self.placed)}"
        self.open.add(oid)
        return oid

    async def open_order_ids(self):
        return set(self.open)

    async def cancel(self, symbol, order_id):
        self.cancelled.append(order_id)
        self.open.discard(order_id)

    async def close_position(self, symbol):
        self.closed.append(symbol)

    async def adjust(self, symbol, qty, leverage):
        self.adjusted.append((symbol, qty))

    async def closed_pnl(self, symbol, since, until):
        return [r for sym, r in self.pnl if sym == symbol and since <= r.closed_at < until]

    async def close(self):
        pass


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def _exec(tmp_path, broker, clock=None):
    st = SqliteStore(tmp_path / "t.db")
    return st, Executor(lambda: broker, st, st, st, None, clock or Clock(NOW))


def test_take_places_order_and_marks_signal(tmp_path):
    b = FakeBroker()
    st, ex = _exec(tmp_path, b)
    s = st.add(_signal())
    dec = SignalDecisions(st, st, ex)
    out = asyncio.run(dec.take(s.id))
    assert out.status is SignalStatus.TAKEN and out.note.startswith("Bybit демо: лонг SOLUSDT")
    assert len(b.placed) == 1 and st.trades_for([s.id])[s.id].status is TradeStatus.FILLED
    with pytest.raises(ValueError):
        asyncio.run(dec.take(s.id))                       # второй раз нельзя
    assert len(b.placed) == 1


def test_refusal_and_exchange_error_keep_signal_new(tmp_path):
    b = FakeBroker(price=97.0)
    st, ex = _exec(tmp_path, b)
    s = st.add(_signal())
    with pytest.raises(ValueError, match="Ордер не отправлен: цена уже за стопом"):
        asyncio.run(SignalDecisions(st, st, ex).take(s.id))
    assert st.get(s.id).status is SignalStatus.NEW
    b.px, b.fail = 101.0, ccxt.InsufficientFunds("bybit 110007 ab not enough for new order")
    with pytest.raises(ValueError, match="биржа отклонила: bybit 110007"):
        asyncio.run(SignalDecisions(st, st, ex).take(s.id))
    assert st.get(s.id).status is SignalStatus.NEW and not st.trades_for([s.id])


def test_without_exchange_take_only_marks(tmp_path):
    st = SqliteStore(tmp_path / "t.db")
    ex = Executor(lambda: None, st, st, st)
    s = st.add(_signal())
    out = asyncio.run(SignalDecisions(st, st, ex).take(s.id))
    assert out.status is SignalStatus.TAKEN and "биржа не подключена" in out.note


def test_housekeep_cancels_expired_retest_and_marks_fills(tmp_path):
    b = FakeBroker(price=101.5)
    clock = Clock(NOW)
    st, ex = _exec(tmp_path, b, clock)
    a = st.add(_signal(symbol="AAAUSDT"))
    a = replace(a, plan=TradePlan(EntryKind.RETEST, 100.0, 98.0, 106.0, 2))
    st.db.execute("UPDATE signals SET payload = ? WHERE id = ?", (st._payload(a), a.id))
    st.db.commit()
    c = st.add(_signal(symbol="CCCUSDT"))
    c = replace(c, plan=TradePlan(EntryKind.RETEST, 100.0, 98.0, 106.0, 12))
    st.db.execute("UPDATE signals SET payload = ? WHERE id = ?", (st._payload(c), c.id))
    st.db.commit()
    asyncio.run(ex.execute(st.get(a.id)))
    asyncio.run(ex.execute(st.get(c.id)))
    assert [t.status for t in st.pending_trades()] == [TradeStatus.PLACED, TradeStatus.PLACED]
    b.open.discard("o2")                                   # лимитка C исполнилась
    b.acc = replace(ACC, positions=(Position("CCCUSDT", Side.LONG, 1, 100, 101, 1),))
    clock.t = T0 + timedelta(hours=4 * 3)                 # у A вышло время (2 свечи)
    asyncio.run(ex.housekeep())
    tr = st.trades_for([a.id, c.id])
    assert tr[a.id].status is TradeStatus.EXPIRED and b.cancelled == ["o1"]
    assert tr[c.id].status is TradeStatus.FILLED and st.get(a.id).status is SignalStatus.EXPIRED


def test_wallet_day_pnl_from_first_reading(tmp_path):
    b = FakeBroker()
    st, ex = _exec(tmp_path, b)
    w = asyncio.run(ex.wallet())
    assert w.day_start == 1000.0 and w.day_pnl == 0
    b.acc = replace(ACC, equity=1012.5)
    w = asyncio.run(ex.wallet())
    assert w.day_pnl == 12.5 and w.day_pnl_pct == pytest.approx(1.25)


def test_auto_mode_scanner_executes_and_skips_with_reason(tmp_path, monkeypatch):
    import trader.application.services as svc_mod
    b = FakeBroker()
    st, ex = _exec(tmp_path, b)
    st.save(replace(Settings(), mode=Mode.AUTO))
    idx = pd.date_range("2026-01-01", periods=10, freq="4h", tz="UTC")
    bars = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
                         "taker_buy_volume": 0.5}, index=idx)
    monkeypatch.setattr(svc_mod, "detect", lambda d, tf, sym, s: [replace(_signal(sym, tf), bar_time=T0)])

    class _Charts:
        def render(self, signal, bb):
            return "x.png"

    sc = Scanner(_Market({"SOLUSDT": bars, "NOPEUSDT": bars}), st, st, _Charts(), None, "", executor=ex)
    rep = asyncio.run(sc.scan(Timeframe.H4))
    by = {s.symbol: s for s in rep.signals}
    assert by["SOLUSDT"].status is SignalStatus.TAKEN and len(b.placed) == 1
    assert by["NOPEUSDT"].status is SignalStatus.SKIPPED and "не торгуется на Bybit" in by["NOPEUSDT"].note


# ---------------- адаптер Bybit ----------------
class FakeCcxt:
    def __init__(self):
        self.calls = []

    async def load_markets(self):
        return {"SOL/USDT:USDT": {"id": "SOLUSDT", "symbol": "SOL/USDT:USDT", "swap": True, "linear": True,
                                  "settle": "USDT", "active": True, "precision": {"amount": 0.1, "price": 0.001},
                                  "limits": {"amount": {"min": 0.1}, "leverage": {"max": 75}, "cost": {"min": None}}},
                "SOL/USDT": {"id": "SOLUSDT", "symbol": "SOL/USDT", "spot": True, "settle": None}}

    async def set_leverage(self, lev, sym):
        self.calls.append(("lev", lev, sym))
        raise ccxt.BadRequest('bybit {"retCode":110043,"retMsg":"leverage not modified"}')

    async def create_order(self, sym, typ, side, qty, price, params):
        self.calls.append(("order", sym, typ, side, qty, price, params))
        return {"id": "123"}

    async def fetch_balance(self):
        return {"info": {"result": {"list": [{"totalEquity": "1500.5", "totalAvailableBalance": "1200"}]}}}

    async def fetch_positions(self, symbols, params):
        return [{"contracts": 2.0, "side": "short", "entryPrice": 150, "markPrice": 145, "unrealizedPnl": 10,
                 "stopLossPrice": 160, "takeProfitPrice": None, "info": {"symbol": "SOLUSDT"}},
                {"contracts": 0, "side": "long", "info": {"symbol": "ETHUSDT"}}]

    async def fetch_open_orders(self, symbol, since, limit, params):
        return [{"id": "9", "reduceOnly": False, "info": {"symbol": "ADAUSDT"}},
                {"id": "10", "reduceOnly": True, "info": {"symbol": "SOLUSDT", "reduceOnly": True}}]

    async def fetch_positions_history(self, symbols, since, limit, params):
        self.calls.append(("pnl", symbols, since, params["until"]))
        return [{"info": {"symbol": "SOLUSDT", "side": "Sell", "closedSize": "2", "avgEntryPrice": "100.5",
                          "avgExitPrice": "110", "closedPnl": "18.7", "updatedTime": "1791000000000"}}]

    async def close(self):
        pass


CREDS = BybitCredentials("k" * 18, "s" * 36, "demo")


def test_bybit_adapter_places_with_attached_stop_and_target():
    fx = FakeCcxt()
    br = BybitBroker(CREDS, exchange=fx)
    inst = asyncio.run(br.instrument("SOLUSDT"))
    assert inst.qty_step == 0.1 and inst.min_qty == 0.1 and inst.max_leverage == 75 and inst.min_notional == 5.0
    assert asyncio.run(br.instrument("XYZUSDT")) is None
    o = build_order(_sig(), Settings(), ACC, inst, 101.0, None, NOW)
    assert asyncio.run(br.place(o)) == "123"
    (_, sym, typ, side, qty, price, params) = fx.calls[-1]
    assert (sym, typ, side, price) == ("SOL/USDT:USDT", "market", "buy", None) and qty == o.qty
    assert params["stopLoss"] == {"triggerPrice": 98.0} and params["takeProfit"] == {"triggerPrice": 110.0}
    assert params["clientOrderId"] == "tt-1" and params["positionIdx"] == 0
    r = build_order(_sig(plan=TradePlan(EntryKind.RETEST, 100.0, 98.0, 106.0, 12)), Settings(), ACC, inst, 101.0,
                    None, NOW)
    asyncio.run(br.place(r))
    assert fx.calls[-1][2] == "limit" and fx.calls[-1][5] == 100.0 and fx.calls[-1][6]["timeInForce"] == "GTC"


def test_bybit_adapter_account_parsing():
    br = BybitBroker(CREDS, exchange=FakeCcxt())
    a = asyncio.run(br.account())
    assert a.equity == 1500.5 and a.available == 1200 and a.upnl == 10
    assert [(p.symbol, p.side, p.qty, p.stop, p.target) for p in a.positions] == [("SOLUSDT", Side.SHORT, 2.0, 160, None)]
    assert a.pending_symbols == frozenset({"ADAUSDT"}) and a.busy_symbols == {"SOLUSDT", "ADAUSDT"}
    assert asyncio.run(br.open_order_ids()) == {"9", "10"}


def test_credentials_and_holder(tmp_path):
    with pytest.raises(ValueError):
        BybitCredentials("short", "s" * 36, "demo")
    with pytest.raises(ValueError):
        BybitCredentials("k" * 18, "s" * 36, "mainnet")
    st = SqliteStore(tmp_path / "t.db")
    h = BrokerHolder(st)
    assert h() is None
    h.save(CREDS)
    assert h.credentials == CREDS and CREDS.masked_key == "kkkk…kkkk"
    b1 = h()
    assert b1 is not None and b1.network == "demo" and h() is b1
    assert "api-demo" in str(b1.ex.urls["api"])
    h.forget()
    assert h() is None


def test_order_uses_timeframe_risk():
    o4 = build_order(_sig(), Settings(), ACC, INST, 101.0, None, NOW)
    o1 = build_order(_sig(timeframe=Timeframe.M15, bar_time=NOW - timedelta(minutes=16)), Settings(), ACC, INST,
                     101.0, None, NOW)
    assert o4.risk_usd == pytest.approx(9.99, abs=0.02) and o1.risk_usd == pytest.approx(2.49, abs=0.02)


def _fresh_15m():
    """Последняя закрытая свеча 15m по реальным часам: ручной вход проверяет срок сигнала по текущему времени."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    return now.replace(minute=now.minute - now.minute % 15, second=0, microsecond=0) - timedelta(minutes=15)


def test_auto_mode_trades_only_selected_timeframes(tmp_path, monkeypatch):
    import trader.application.services as svc_mod
    b = FakeBroker()
    st, ex = _exec(tmp_path, b)
    st.save(replace(Settings(), mode=Mode.AUTO, timeframes=frozenset({Timeframe.H4, Timeframe.M15}),
                    auto_timeframes=frozenset({Timeframe.H4})).with_tf(Timeframe.M15, htf_confirm_h=0))
    idx = pd.date_range("2026-01-01", periods=10, freq="1h", tz="UTC")
    bars = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
                         "taker_buy_volume": 0.5}, index=idx)
    monkeypatch.setattr(svc_mod, "detect", lambda d, tf, sym, s: [replace(_signal(sym, tf), bar_time=_fresh_15m())])

    class _Charts:
        def render(self, signal, bb):
            return "x.png"

    sc = Scanner(_Market({"SOLUSDT": bars}), st, st, _Charts(), None, "", executor=ex)
    rep = asyncio.run(sc.scan(Timeframe.M15))
    assert rep.signals[0].status is SignalStatus.NEW and not b.placed      # 15m не отмечен для автобота
    out = asyncio.run(SignalDecisions(st, st, ex).take(rep.signals[0].id))  # но вручную войти можно
    assert out.status is SignalStatus.TAKEN and len(b.placed) == 1
    assert Settings().toggle_auto(Timeframe.M15).auto_timeframes == {Timeframe.H4, Timeframe.M15}


def test_15m_signal_only_after_fresh_4h_breakout(tmp_path, monkeypatch):
    import trader.application.services as svc_mod
    from trader.domain.models import Side
    b = FakeBroker()
    st, ex = _exec(tmp_path, b)
    st.save(replace(Settings(), timeframes=frozenset({Timeframe.H4, Timeframe.M15})).with_tf(Timeframe.M15, htf_confirm_h=12))
    idx = pd.date_range("2026-01-01", periods=10, freq="15min", tz="UTC")
    bars = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
                         "taker_buy_volume": 0.5}, index=idx)
    bar_time = _fresh_15m()
    monkeypatch.setattr(svc_mod, "detect", lambda d, tf, sym, s: [replace(_signal(sym, tf), bar_time=bar_time)])

    class _Charts:
        def render(self, signal, bb):
            return "x.png"

    sc = Scanner(_Market({"SOLUSDT": bars}), st, st, _Charts(), None, "", executor=ex)
    monkeypatch.setattr(svc_mod, "htf_breakouts", lambda d, tf: [])
    assert asyncio.run(sc.scan(Timeframe.M15)).signals == []                 # пробоя 4h не было — сигнала нет
    seen = bar_time + timedelta(minutes=15) - timedelta(hours=3)
    monkeypatch.setattr(svc_mod, "htf_breakouts", lambda d, tf: [(seen, Side.LONG)])
    rep = asyncio.run(sc.scan(Timeframe.M15))
    assert len(rep.signals) == 1 and rep.signals[0].extra["htf_age_h"] == 3.0 and rep.signals[0].extra["htf"] == "4h"


def test_housekeep_cancels_retest_when_price_reached_target_first(tmp_path):
    b = FakeBroker(price=101.5)
    st, ex = _exec(tmp_path, b)
    a = st.add(_signal(symbol="AAAUSDT"))
    a = replace(a, plan=TradePlan(EntryKind.RETEST, 100.0, 98.0, 106.0, 12))
    st.db.execute("UPDATE signals SET payload = ? WHERE id = ?", (st._payload(a), a.id))
    st.db.commit()
    asyncio.run(ex.execute(st.get(a.id)))
    asyncio.run(ex.housekeep())
    assert st.trades_for([a.id])[a.id].status is TradeStatus.PLACED and b.cancelled == []
    b.px = 106.2                                           # цена ушла к цели, ретеста не было
    asyncio.run(ex.housekeep())
    assert st.trades_for([a.id])[a.id].status is TradeStatus.EXPIRED and b.cancelled == ["o1"]
    assert "до цели без ретеста" in st.get(a.id).note


def test_housekeep_closes_position_after_max_hold_only_for_latest_trade(tmp_path):
    b = FakeBroker()
    clock = Clock(NOW)
    st, ex = _exec(tmp_path, b, clock)
    old = st.add(_signal())                                # рыночный вход: сделка сразу «на бирже»
    asyncio.run(ex.execute(st.get(old.id)))
    b.acc = replace(ACC, positions=(Position("SOLUSDT", Side.LONG, 1, 100, 101, 1),))
    clock.t = NOW + timedelta(hours=4 * 59)
    asyncio.run(ex.housekeep())
    assert b.closed == [] and st.trades_for([old.id])[old.id].status is TradeStatus.FILLED
    clock.t = NOW + timedelta(hours=4 * 60)                # 60 свечей 4h — срок сделки
    asyncio.run(ex.housekeep())
    assert b.closed == ["SOLUSDT"] and st.trades_for([old.id])[old.id].status is TradeStatus.TIMED_OUT
    b.acc = replace(ACC, positions=(Position("SOLUSDT", Side.SHORT, 1, 100, 99, 1),))   # уже другая позиция
    asyncio.run(ex.housekeep())
    assert b.closed == ["SOLUSDT"]


def test_scanner_takes_only_top_n_liquid_symbols(tmp_path, monkeypatch):
    import trader.application.services as svc_mod
    st = SqliteStore(tmp_path / "t.db")
    st.save(replace(Settings(), timeframes=frozenset(Timeframe)).with_tf(Timeframe.M15, top_n=2))
    idx = pd.date_range("2026-01-01", periods=10, freq="15min", tz="UTC")
    bars = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
                         "taker_buy_volume": 0.5}, index=idx)
    seen = []
    monkeypatch.setattr(svc_mod, "detect", lambda d, tf, sym, s: seen.append((tf, sym)) or [])

    class _Charts:
        def render(self, signal, bb):
            return "x.png"

    m = _Market({"AUSDT": bars, "BUSDT": bars, "CUSDT": bars})
    sc = Scanner(m, st, st, _Charts(), None, "")
    asyncio.run(sc.scan(Timeframe.M15))
    asyncio.run(sc.scan(Timeframe.H4))
    assert sorted(s for tf, s in seen if tf is Timeframe.M15) == list(m.frames)[:2]
    assert len([s for tf, s in seen if tf is Timeframe.H4]) == 3


def test_timeframe_position_limit_inside_common_limit():
    s15 = replace(_sig(), timeframe=Timeframe.M15)
    with pytest.raises(ExecutionRefused, match="по 15m открыто 3 из 3"):
        build_order(s15, Settings(), ACC, INST, 101.0, 1000.0, NOW, tf_open=3)
    assert build_order(s15, Settings(), ACC, INST, 101.0, 1000.0, NOW, tf_open=2).qty > 0
    assert build_order(_sig(), Settings(), ACC, INST, 101.0, 1000.0, NOW, tf_open=11).qty > 0   # 4h: только общий


def test_executor_counts_open_positions_per_timeframe(tmp_path):
    b = FakeBroker()
    st, ex = _exec(tmp_path, b)
    syms = ["AUSDT", "BUSDT", "CUSDT"]
    for sym in syms:
        asyncio.run(ex.execute(st.add(_signal(sym, Timeframe.M15))))
    b.acc = replace(ACC, positions=tuple(Position(x, Side.LONG, 1, 100, 101, 1) for x in syms))
    with pytest.raises(ExecutionRefused, match="по 15m открыто 3 из 3"):
        asyncio.run(ex.execute(st.add(_signal("DUSDT", Timeframe.M15))))
    asyncio.run(ex.execute(st.add(_signal("EUSDT", Timeframe.H4))))           # 4h места 15m не занимает
    b.acc = replace(ACC, positions=tuple(Position(x, Side.LONG, 1, 100, 101, 1) for x in syms[:2]))
    asyncio.run(ex.execute(st.get(st.add(_signal("FUSDT", Timeframe.M15)).id)))   # одна закрылась — место есть
    assert [o.symbol for o in b.placed] == syms + ["EUSDT", "FUSDT"]



def test_bybit_closed_pnl_parsing():
    fx = FakeCcxt()
    br = BybitBroker(CREDS, exchange=fx)
    since, until = NOW - timedelta(days=1), NOW
    (r,) = asyncio.run(br.closed_pnl("SOLUSDT", since, until))
    assert r.side is Side.LONG and r.qty == 2 and r.entry == 100.5 and r.exit == 110 and r.pnl == 18.7
    assert fx.calls[-1] == ("pnl", ["SOL/USDT:USDT"], int(since.timestamp() * 1000), int(until.timestamp() * 1000))


def test_journal_records_result_in_r_slippage_and_reason(tmp_path):
    from trader.domain.journal import tf_stats
    b = FakeBroker()
    clock = Clock(NOW)
    st, ex = _exec(tmp_path, b, clock)
    s = st.add(replace(_signal(), extra={"level": True}))  # лонг по рынку: план вход 101, стоп 98, цель 110
    asyncio.run(ex.execute(st.get(s.id)))
    b.acc = replace(ACC, positions=(Position("SOLUSDT", Side.LONG, 3.33, 101.3, 105, 10),))
    clock.t = NOW + timedelta(hours=1)
    asyncio.run(ex.housekeep())                            # позиция ещё открыта — итога нет
    assert st.unsettled_trades()[0].result is None
    b.acc = ACC
    b.pnl = [("SOLUSDT", ClosedPnl(Side.LONG, 3.33, 101.3, 110.0, 28.5, NOW + timedelta(hours=2))),
             ("SOLUSDT", ClosedPnl(Side.SHORT, 1, 1, 1, -5.0, NOW + timedelta(hours=2))),       # чужая сторона
             ("SOLUSDT", ClosedPnl(Side.LONG, 1, 1, 1, -5.0, NOW - timedelta(hours=1)))]        # до входа
    clock.t = NOW + timedelta(hours=3)
    asyncio.run(ex.housekeep())
    t = st.trades_for([s.id])[s.id]
    assert t.status is TradeStatus.CLOSED and t.result.pnl_usd == 28.5 and t.exit_reason == "цель"
    assert t.r_multiple == pytest.approx(28.5 / (3.33 * 3)) and t.slippage_r == pytest.approx(0.1)
    assert not st.unsettled_trades()
    h4 = next(x for x in tf_stats(st.journal()) if x.timeframe is Timeframe.H4)
    assert h4.closed == 1 and h4.avg_r == pytest.approx(t.r_multiple) and h4.need == 29 and h4.win_share == 1.0
    assert st.journal()[0].at_level and h4.level_closed == 1 and h4.level_avg_r == pytest.approx(t.r_multiple)
    assert h4.plain_avg_r is None
    assert st.get(s.id).status is SignalStatus.CLOSED and "Bybit" in st.get(s.id).note   # из ленты — в архив


def test_journal_gives_up_without_exchange_record_and_hold_counts_from_fill(tmp_path):
    b = FakeBroker()
    clock = Clock(NOW)
    st, ex = _exec(tmp_path, b, clock)
    s = st.add(_signal())
    asyncio.run(ex.execute(st.get(s.id)))
    clock.t = NOW + timedelta(days=2)
    asyncio.run(ex.housekeep())
    assert st.trades_for([s.id])[s.id].status is TradeStatus.FILLED      # итог ещё ищем
    clock.t = NOW + timedelta(days=9)
    asyncio.run(ex.housekeep())
    t = st.trades_for([s.id])[s.id]
    assert t.status is TradeStatus.CLOSED and t.result is None and t.exit_reason == "нет данных"
    assert st.get(s.id).status is SignalStatus.CLOSED


def test_taken_signals_with_finished_trades_are_closed_on_start(tmp_path):
    st = SqliteStore(tmp_path / "t.db")
    done, live = st.add(_signal()), st.add(_signal(symbol="ETHUSDT"))
    for s in (done, live):
        st.set_status(s.id, SignalStatus.TAKEN, "ордер")
    t = st.add_trade(Trade(done.id, "SOLUSDT", Side.LONG, EntryKind.MARKET, 1, 101, 98, 110, "o", "demo", TradeStatus.FILLED))
    st.add_trade(Trade(live.id, "ETHUSDT", Side.LONG, EntryKind.MARKET, 1, 101, 98, 110, "o2", "demo", TradeStatus.FILLED))
    st.settle_trade(t.id, TradeStatus.CLOSED, None)        # закрылась до обновления: сигнал так и остался «в работе»
    st2 = SqliteStore(tmp_path / "t.db")
    assert st2.get(done.id).status is SignalStatus.CLOSED and st2.get(live.id).status is SignalStatus.TAKEN


def test_retest_filled_and_closed_within_a_minute_is_journaled_not_cancelled(tmp_path):
    b = FakeBroker()
    st, ex = _exec(tmp_path, b, Clock(NOW))
    s = st.add(replace(_signal(), plan=TradePlan(EntryKind.RETEST, 100.0, 98.0, 106.0, 12)))
    asyncio.run(ex.execute(st.get(s.id)))
    b.open.clear()                                          # лимитка исчезла, позиции уже нет
    b.pnl = [("SOLUSDT", ClosedPnl(Side.LONG, 5, 100.0, 97.9, -10.5, NOW - timedelta(seconds=30)))]
    asyncio.run(ex.housekeep())
    t = st.trades_for([s.id])[s.id]
    assert t.status is TradeStatus.CLOSED and t.filled_at == NOW and t.exit_reason == "стоп"
    assert t.r_multiple == pytest.approx(-10.5 / (t.qty * 2))


class _PnlBroker(FakeBroker):
    def __init__(self, recs):
        super().__init__()
        self.recs, self.calls = recs, []

    async def closed_pnl_all(self, since, until):
        self.calls.append((since, until))
        return [r for r in self.recs if since <= r.closed_at < until]


def test_pnl_periods_backfill_history_and_count_today(tmp_path):
    from trader.application.pnl import PnlHistory
    now = NOW.replace(hour=12, minute=0, second=0, microsecond=0)
    recs = [ClosedPnl(Side.LONG, 1, 1, 1, 10.0, now - timedelta(hours=1)),          # сегодня
            ClosedPnl(Side.SHORT, 1, 1, 1, -4.0, now - timedelta(days=3)),          # в неделе
            ClosedPnl(Side.LONG, 1, 1, 1, 20.0, now - timedelta(days=20)),          # в 30 днях
            ClosedPnl(Side.LONG, 1, 1, 1, 100.0, now - timedelta(days=100))]        # в полугоде
    st = SqliteStore(tmp_path / "t.db")
    h = PnlHistory(st, Clock(now))
    b = _PnlBroker(recs)
    asyncio.run(h.refresh(b))
    assert all(until - since <= timedelta(days=7) for since, until in b.calls)
    p = {x.label: x for x in h.periods("demo", 1000.0)}
    assert p["Неделя"].usd == 6.0 and p["30 дней"].usd is None and p["Полгода"].usd is None   # догружается
    asyncio.run(h.refresh(b))
    assert h.periods("demo", 1000.0)[1].usd == 26.0
    for _ in range(10):
        asyncio.run(h.refresh(b))
    p = {x.label: x for x in h.periods("demo", 1000.0)}
    assert p["Полгода"].usd == 126.0 and p["Неделя"].pct == pytest.approx(6.0 / 994.0 * 100)
    n = len(b.calls)
    asyncio.run(h.refresh(b))
    assert len(b.calls) == n + 1                                # история есть — только сегодняшний день
    assert h.periods("live", 1000.0)[0].usd is None             # другая сеть — своя история


def test_health_check_flags_a_strategy_that_stopped_working():
    from trader.domain.journal import Health, health
    assert health([1.0] * 5, 0.45, 3.0).level is Health.EARLY
    rng = np.random.default_rng(1)
    normal = list(np.where(rng.random(60) < 0.31, 3.0, -1.0))          # около +0.22R — половина бэктеста 4h
    assert health(normal, 0.45, 3.0).level is Health.OK
    broken = [-1.0] * 9 + [3.0] + [-1.0] * 10 + [3.0] + [-1.0] * 9     # 2 цели из 30: ≈ −0.73R
    h = health(broken, 0.45, 3.0)
    assert h.level is Health.STOP and "выключить авто" in h.text and h.drawdown_r > h.drawdown_limit_r


# ---------------- хедж BTC ----------------
def _hedge_market(k=1.5):
    """4h-свечи монеты с бетой k к BTC (плюс шум) и самого BTC."""
    from test_trader_app import _Market as _M
    rng = np.random.default_rng(3)
    idx = pd.date_range("2026-06-01", periods=400, freq="4h", tz="UTC")
    rb = rng.normal(0, 0.01, len(idx))
    btc = pd.DataFrame({"close": 60000 * np.exp(np.cumsum(rb))}, index=idx)
    coin = pd.DataFrame({"close": 100 * np.exp(np.cumsum(k * rb + rng.normal(0, 0.002, len(idx))))}, index=idx)
    return _M({"SOLUSDT": coin, "BTCUSDT": btc})


def test_beta_rebalance_and_hedge_r_domain():
    from trader.domain.execution import HedgeLeg
    from trader.domain.hedge import beta, rebalance, target_qty, without_hedge
    m = _hedge_market(1.5)
    assert beta(m.frames["SOLUSDT"]["close"], m.frames["BTCUSDT"]["close"]) == pytest.approx(1.5, abs=0.05)
    assert beta(m.frames["SOLUSDT"]["close"].iloc[:50], m.frames["BTCUSDT"]["close"]) is None   # мало свечей
    # шорт на $1000 с бетой 1.5 и лонг на $500 с бетой 1 → BTC лонг $1500 − $500 = $1000
    assert target_qty([(Side.SHORT, 1000, 1.5), (Side.LONG, 500, 1.0)], 50000) == pytest.approx(0.02)
    btc = replace(INST, symbol="BTCUSDT", qty_step=0.001, min_qty=0.001)
    assert rebalance(0.020, 0.0, btc, 50000) == pytest.approx(0.020)
    assert rebalance(0.020, 0.019, btc, 50000) == 0.0                     # разница < 10% — не трогаем
    assert rebalance(0.0, 0.019, btc, 50000) == pytest.approx(-0.019)     # сделок нет — закрыть всё
    acc = replace(ACC, positions=(Position("BTCUSDT", Side.LONG, 0.02, 50000, 50000, 0),
                                  Position("SOLUSDT", Side.SHORT, 10, 100, 100, 0)))
    assert [p.symbol for p in without_hedge(acc).positions] == ["SOLUSDT"]
    from trader.domain.execution import Trade
    t = Trade(1, "SOLUSDT", Side.SHORT, EntryKind.MARKET, 10, 100, 102, 94, "o", "demo",
              hedge=HedgeLeg(1.5, 50000, 51000))                          # BTC +2%, хедж-лонг на $1500
    assert t.hedge_r == pytest.approx((1500 * 0.02 - 2 * 5.5e-4 * 1500) / 20)


def test_hedged_trade_opens_and_closes_btc_position(tmp_path):
    b = FakeBroker()
    clock = Clock(NOW)
    st = SqliteStore(tmp_path / "t.db")
    st.save(Settings().with_tf(Timeframe.H4, hedge_btc=True))
    ex = Executor(lambda: b, st, st, st, None, clock, market=_hedge_market(1.5))
    s = st.add(_signal())                                   # лонг SOL по рынку
    tr = asyncio.run(ex.execute(st.get(s.id)))
    assert tr.hedge is not None and tr.hedge.beta == pytest.approx(1.5, abs=0.05)
    assert "хедж BTC β" in st.get(s.id).note
    b.px = 20000.0                                          # цена BTC (FakeBroker отдаёт одну цену на всё)
    b.acc = replace(ACC, positions=(Position("SOLUSDT", Side.LONG, tr.qty, 101, 101, 0),))
    asyncio.run(ex.housekeep())
    (sym, qty), = b.adjusted                                # лонг SOL → шорт BTC на бету × номинал
    assert sym == "BTCUSDT" and qty == pytest.approx(-tr.hedge.beta * tr.qty * 101 / 20000, abs=0.01)
    assert st.trades_for([s.id])[s.id].hedge.btc_in == 20000.0
    # BTC-позиция хеджа не занимает место сделки: при лимите в 1 позицию новая монета всё равно открывается
    b.acc = replace(ACC, positions=(Position("SOLUSDT", Side.LONG, tr.qty, 101, 101, 0),
                                    Position("BTCUSDT", Side.SHORT, -qty, 50000, 50000, 0)))
    st.save(replace(st.load(), max_positions=2))
    b.px = 101.0
    s2 = st.add(_signal(symbol="ETHUSDT"))
    asyncio.run(ex.execute(st.get(s2.id)))
    btc_sig = st.add(_signal(symbol="BTCUSDT"))
    with pytest.raises(ExecutionRefused, match="занят хеджем"):
        asyncio.run(ex.execute(st.get(btc_sig.id)))
    # SOL закрылась по цели: итог в журнал, цена BTC при снятии хеджа, позиция BTC закрывается
    b.acc = replace(ACC, positions=(Position("BTCUSDT", Side.SHORT, -qty, 50000, 50000, 0),))
    st.settle_trade(st.trades_for([s2.id])[s2.id].id, TradeStatus.CANCELLED, None)
    b.pnl = [("SOLUSDT", ClosedPnl(Side.LONG, tr.qty, 101.0, 110.0, 20.0, NOW + timedelta(hours=2)))]
    b.px = 19000.0
    clock.t = NOW + timedelta(hours=3)
    asyncio.run(ex.housekeep())
    t = st.trades_for([s.id])[s.id]
    assert t.status is TradeStatus.CLOSED and t.hedge.btc_out == 19000.0 and t.hedge_r > 0   # BTC упал — шорт BTC в плюсе
    assert b.closed == ["BTCUSDT"] and st.kv_get("hedge:held") == "0"


def test_hedge_off_leaves_manual_btc_position_alone(tmp_path):
    b = FakeBroker(acc=replace(ACC, positions=(Position("BTCUSDT", Side.LONG, 0.01, 50000, 50000, 0),)))
    st, ex = _exec(tmp_path, b)
    s = st.add(_signal())
    asyncio.run(ex.execute(st.get(s.id)))
    asyncio.run(ex.housekeep())
    assert not b.adjusted and not b.closed and st.trades_for([s.id])[s.id].hedge is None


def test_order_says_why_risk_was_cut_and_reserves_margin_for_hedge():
    from trader.domain.sizing import exposure, risk_notes
    s = Settings().with_tf(Timeframe.H4, risk_pct=5.0)
    poor = replace(ACC, available=100.0)                    # маржи хватает на 500$ номинала при плече 5
    o = build_order(_sig(), s, poor, INST, price=101.0, day_start_equity=1000.0, now=NOW)
    assert o.cut.startswith("вместо $50.00: не хватило свободной маржи") and "вместо $50.00" in o.describe()
    h = build_order(_sig(), s, poor, INST, price=101.0, day_start_equity=1000.0, now=NOW, hedge_beta=1.0)
    assert h.qty == pytest.approx(o.qty / 2, abs=0.02) and "с учётом хеджа BTC" in h.cut
    assert build_order(_sig(), Settings(), ACC, INST, price=101.0, day_start_equity=1000.0, now=NOW).cut == ""
    e = exposure(False, 5.0, 10.0, 1.7, 10, hedge=True)     # как в панели трейдера: риск 5%, плечо 10×, хедж
    assert e.notional_x == pytest.approx(5 / 1.7) and e.fits == 1
    notes = risk_notes(5.0, 4.0, e, 13.3)
    assert any("дневного лимита" in n for n in notes) and any("−66%" in n for n in notes)
    assert any("одну такую сделку" in n for n in notes)
    assert risk_notes(1.0, 4.0, exposure(False, 1.0, 10.0, 3.0, 5, hedge=False), 13.3) == []
    m = exposure(True, 1.0, 10.0, 3.0, 10, hedge=False)    # как на Bybit: маржа 10% × 10 = позиция 1× капитала
    assert m.notional_x == pytest.approx(1.0) and m.loss_pct == pytest.approx(3.0) and m.fits == 9


def test_margin_sizing_like_bybit():
    from trader.domain.models import Sizing
    s = replace(Settings(sizing=Sizing.MARGIN, leverage=10).with_tf(Timeframe.H4, margin_pct=10.0))
    o = build_order(_sig(), s, ACC, INST, price=101.0, day_start_equity=1000.0, now=NOW)
    # маржа $100 × 10 = позиция $1000 → 9.9 монеты по 101; риск = объём × (101 − 98)
    assert o.qty == pytest.approx(9.9) and o.risk_usd == pytest.approx(9.9 * 3) and o.leverage == 10
    assert "маржа $99.99 × 10" in o.describe() and o.cut == ""
    poor = build_order(_sig(), s, replace(ACC, available=50.0), INST, price=101.0, day_start_equity=1000.0, now=NOW)
    assert poor.cut.startswith("маржа $") and "не хватило свободной маржи" in poor.cut
