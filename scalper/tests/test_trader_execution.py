import asyncio
from dataclasses import replace
from datetime import timedelta

import ccxt.async_support as ccxt
import pandas as pd
import pytest

from trader.application.execution import Executor
from trader.application.services import Scanner, SignalDecisions
from trader.domain.execution import (Account, ExecutionRefused, Instrument, Position, TradeStatus, build_order,
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
    (replace(ACC, pending_symbols=frozenset({f"X{i}USDT" for i in range(5)})), 101.0, 1000.0, "открыто 5 из 5"),
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
        self.placed, self.cancelled, self.open = [], [], set()

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
                    auto_timeframes=frozenset({Timeframe.H4})))
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
