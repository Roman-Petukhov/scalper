import pytest

from core.models import OrderCmd, OrderKind, OrderStatus, OrderUpdate, Side, Signal, Trade
from core.risk import Instrument, RiskManager
from exchange.broker_paper import PaperBroker

INST = Instrument("BTCUSDT", 0.1, 0.001, 100.0)


def sig(stop=50.0):
    return Signal(ts=0, side=Side.LONG, setup="t", price=100000, stop_dist=stop, tp_dist=80)


def test_size_by_risk(cfg):
    cfg.risk.max_leverage_used = 10
    r = RiskManager(cfg, 1000.0)
    qty, _ = r.size(sig(50.0), 100000.0, INST)       # 2.5 USDT риска / 50 = 0.05
    assert qty == pytest.approx(0.05)


def test_size_capped_by_leverage(cfg):
    cfg.risk.max_leverage_used = 3
    r = RiskManager(cfg, 1000.0)
    qty, _ = r.size(sig(10.0), 100000.0, INST)       # по риску 0.25, но номинал ≤ 3000
    assert qty == pytest.approx(0.03)


def test_below_min_notional_is_skipped_not_inflated(cfg):
    fine = Instrument("BTCUSDT", 0.1, 0.0001, 100.0)
    r = RiskManager(cfg, 100.0)
    qty, why = r.size(sig(500.0), 100000.0, fine)    # 0.0005 BTC = 50 USDT < 100
    assert qty == 0 and why.startswith("below_min_notional")


def test_daily_loss_halt(cfg):
    halts = []
    r = RiskManager(cfg, 1000.0, on_halt=halts.append)
    ts = 1_700_000_000
    assert r.can_trade(ts)[0]
    r.on_trade_closed(-25.0, ts)                     # -2.5% > лимита 2%
    assert not r.can_trade(ts + 10)[0]
    assert halts
    assert r.can_trade(ts + 86400)[0]                # новые сутки


def test_loss_streak_pause(cfg):
    r = RiskManager(cfg, 10000.0)
    ts = 1_700_000_000
    for i in range(cfg.risk.max_consecutive_losses):
        r.on_trade_closed(-1.0, ts + i)
    ok, why = r.can_trade(ts + 10)
    assert not ok and why == "loss_streak_pause"


# ---------- paper-брокер ----------

class Collector:
    def __init__(self):
        self.updates = []

    def __call__(self, m):
        if isinstance(m, OrderUpdate):
            self.updates.append(m)

    def last(self, cid):
        return [u for u in self.updates if u.cid == cid][-1]


def make_broker(cfg):
    b = PaperBroker(cfg, 0.1, 0.001, 1000.0)
    c = Collector()
    b.set_sink(c)
    b.on_event(Trade(1.0, 100.0, 1, False))           # аск=100.0, бид=99.9
    return b, c


def test_post_only_crossing_is_rejected(cfg):
    b, c = make_broker(cfg)
    b.submit(OrderCmd("place", "o1", OrderKind.LIMIT, "buy", 0.01, price=100.0, post_only=True))
    b.on_event(Trade(1.1, 100.0, 1, False))
    assert c.last("o1").status == OrderStatus.REJECTED


def test_limit_fills_only_when_traded_through(cfg):
    b, c = make_broker(cfg)
    b.submit(OrderCmd("place", "o1", OrderKind.LIMIT, "buy", 0.01, price=99.9, post_only=True))
    b.on_event(Trade(1.1, 99.9, 5, True))             # касание: не исполняем
    assert c.last("o1").status == OrderStatus.NEW
    b.on_event(Trade(1.2, 99.8, 5, True))             # прошли сквозь
    u = c.last("o1")
    assert u.status == OrderStatus.FILLED and u.last_price == 99.9
    assert u.fee == pytest.approx(0.01 * 99.9 * cfg.fees.maker)
    assert b.pos == pytest.approx(0.01)


def test_stop_triggers_as_taker_and_reduce_only_caps(cfg):
    b, c = make_broker(cfg)
    b.submit(OrderCmd("place", "m1", OrderKind.MARKET, "buy", 0.01))
    b.on_event(Trade(1.1, 100.0, 1, False))
    assert b.pos == pytest.approx(0.01)
    b.submit(OrderCmd("place", "s1", OrderKind.STOP, "sell", 0.05, stop_price=99.5, reduce_only=True))
    b.on_event(Trade(1.2, 99.9, 1, True))
    assert c.last("s1").status == OrderStatus.NEW
    b.on_event(Trade(1.3, 99.4, 1, True))
    u = c.last("s1")
    assert u.last_qty == pytest.approx(0.01)          # reduceOnly не перевернул позицию
    assert u.fee == pytest.approx(0.01 * 99.4 * cfg.fees.taker)
    assert b.pos == 0
    assert b.realized == pytest.approx((99.4 - 100.0) * 0.01)


def test_stop_that_would_trigger_immediately_is_rejected(cfg):
    b, c = make_broker(cfg)
    b.submit(OrderCmd("place", "m1", OrderKind.MARKET, "buy", 0.01))
    b.on_event(Trade(1.1, 100.0, 1, False))
    b.submit(OrderCmd("place", "s1", OrderKind.STOP, "sell", 0.01, stop_price=100.5, reduce_only=True))
    b.on_event(Trade(1.2, 100.0, 1, False))
    assert c.last("s1").status == OrderStatus.REJECTED


def test_latency_delays_activation(cfg):
    cfg.paper.latency_ms = 500
    b, c = make_broker(cfg)
    b.submit(OrderCmd("place", "m1", OrderKind.MARKET, "buy", 0.01))
    b.on_event(Trade(1.2, 100.0, 1, False))
    assert not c.updates
    b.on_event(Trade(1.4, 101.0, 1, False))           # рынок ушёл, пока ордер "летел"
    assert not c.updates
    b.on_event(Trade(1.6, 101.0, 1, False))
    assert c.last("m1").last_price == pytest.approx(101.0)
