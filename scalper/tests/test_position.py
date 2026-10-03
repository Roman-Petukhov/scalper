import pytest

from core.features import Features
from core.models import OrderKind, OrderUpdate, Side, Signal, Trade
from core.position import FLAT, OPEN, PositionManager
from core.risk import Instrument, RiskManager
from core.strategy import FlowScalper
from exchange.broker_paper import PaperBroker

INST = Instrument("BTCUSDT", 0.1, 0.001, 100.0)


class Rig:
    def __init__(self, cfg):
        cfg.exits.flow_reversal_z = 1e9
        self.cfg = cfg
        self.broker = PaperBroker(cfg, 0.1, 0.001, 10000.0)
        self.features = Features(cfg, 0.1)
        self.strategy = FlowScalper(cfg)
        self.risk = RiskManager(cfg, 10000.0)
        self.events = []
        self.pm = PositionManager(cfg, INST, self.broker, self.risk, self.strategy,
                                  notify=lambda k, p: self.events.append((k, p)))
        self.broker.set_sink(lambda m: self.pm.on_order_update(m) if isinstance(m, OrderUpdate)
                             else self.pm.on_position_sync(m))
        self.snap = None

    def feed(self, ts, price, buyer_maker=True):
        t = Trade(ts, price, 1.0, buyer_maker)
        self.broker.on_event(t)
        self.features.on_trade(t)
        self.snap = self.features.snapshot(ts)
        self.pm.on_market(self.snap)
        return self.snap

    def enter_long(self, ts=10.0, stop=50.0, tp=80.0):
        self.feed(ts, 100000.0, True)                 # бид = 100000.0
        s = Signal(ts=ts, side=Side.LONG, setup="test", price=100000.0, stop_dist=stop, tp_dist=tp)
        self.pm.on_signal(s, self.snap)
        self.feed(ts + 0.1, 100000.0, True)           # активация ордера
        self.feed(ts + 0.2, 99999.9, True)            # прошли сквозь лимит: заливка


def open_orders(rig, kind):
    return [o.cmd for o in rig.broker.open.values() if o.cmd.kind == kind]


def test_take_profit_cycle(cfg):
    r = Rig(cfg)
    r.enter_long()
    assert r.pm.state == OPEN
    qty = r.pm.qty
    assert qty == pytest.approx(0.3)                  # риск 25 USDT/50 = 0.5, потолок 3x -> 0.3
    stops, tps = open_orders(r, OrderKind.STOP), open_orders(r, OrderKind.LIMIT)
    r.feed(10.3, 100000.0, False)                     # активация стопа и TP
    stops, tps = open_orders(r, OrderKind.STOP), open_orders(r, OrderKind.LIMIT)
    assert stops[0].stop_price == pytest.approx(99950.0) and stops[0].reduce_only
    assert tps[0].price == pytest.approx(100080.0) and tps[0].post_only
    r.feed(11.0, 100080.1, False)                     # цена прошла сквозь TP
    assert r.pm.state == FLAT
    t = r.pm.trades[-1]
    assert t.exit_reason == "tp"
    assert t.gross_pnl == pytest.approx(80.0 * qty)
    fees = qty * 100000.0 * cfg.fees.maker + qty * 100080.0 * cfg.fees.maker
    assert t.fees == pytest.approx(fees)
    assert t.net_pnl == pytest.approx(80.0 * qty - fees)
    r.feed(11.2, 100080.0, False)                     # отмена стопа дошла
    assert not r.broker.open and r.broker.pos == 0


def test_stop_loss_cycle(cfg):
    r = Rig(cfg)
    r.enter_long()
    r.feed(10.3, 100000.0, False)
    r.feed(11.0, 99949.0, True)
    assert r.pm.state == FLAT
    t = r.pm.trades[-1]
    assert t.exit_reason == "stop"
    assert t.gross_pnl == pytest.approx((99949.0 - 100000.0) * 0.3)
    assert r.risk.equity == pytest.approx(10000.0 + t.net_pnl)


def test_breakeven_moves_stop(cfg):
    r = Rig(cfg)
    r.enter_long(tp=300.0)
    r.feed(10.3, 100000.0, False)
    r.feed(10.5, 100045.0, False)                     # +45 >= 0.8*50, но вход+комиссии = 100070: рано
    r.feed(10.6, 100045.0, False)
    assert open_orders(r, OrderKind.STOP)[0].stop_price == pytest.approx(99950.0)
    r.feed(10.7, 100100.0, False)                     # выше уровня вход+комиссии
    r.feed(10.8, 100100.0, False)
    r.feed(10.9, 100100.0, False)
    stops = open_orders(r, OrderKind.STOP)
    assert len(stops) == 1
    assert stops[0].stop_price == pytest.approx(100070.0)
    r.feed(11.0, 100060.0, True)                      # откат: выбивает в безубыток
    t = r.pm.trades[-1]
    assert t.exit_reason == "breakeven"


def test_time_stop(cfg):
    cfg.exits.time_stop_s = 5
    r = Rig(cfg)
    r.enter_long()
    for k in range(80):
        r.feed(10.3 + k * 0.1, 99999.9, True)
    assert r.pm.state == FLAT
    assert r.pm.trades[-1].exit_reason == "time_stop"


def test_entry_timeout_aborts_without_trade(cfg):
    r = Rig(cfg)
    r.feed(10.0, 100000.0, True)
    s = Signal(ts=10.0, side=Side.LONG, setup="test", price=100000.0, stop_dist=50, tp_dist=80)
    r.pm.on_signal(s, r.snap)
    for k in range(60):
        r.feed(10.1 + k * 0.1, 100000.0, True)        # касания без прохода
    assert r.pm.state == FLAT
    assert not r.pm.trades
    assert r.broker.pos == 0 and not r.broker.open


def test_reprice_follows_market(cfg):
    r = Rig(cfg)
    r.feed(10.0, 100000.0, True)
    s = Signal(ts=10.0, side=Side.LONG, setup="test", price=100000.0, stop_dist=50, tp_dist=80)
    r.pm.on_signal(s, r.snap)
    r.feed(10.1, 100000.0, True)
    r.feed(10.2, 100000.3, False)                     # бид ушёл на 2+ тика вверх
    r.feed(10.3, 100000.3, False)
    r.feed(10.4, 100000.3, False)
    entries = open_orders(r, OrderKind.LIMIT)
    assert len(entries) == 1 and entries[0].price == pytest.approx(100000.2)
    assert r.pm.reprices == 1


def test_taker_entry_mode_uses_market_order(cfg):
    cfg.strategy.momentum.entry = "taker"
    r = Rig(cfg)
    r.feed(10.0, 100000.0, False)                     # аск = 100000.0
    s = Signal(ts=10.0, side=Side.LONG, setup="momentum", price=100000.0, stop_dist=50, tp_dist=80)
    r.pm.on_signal(s, r.snap)
    r.feed(10.1, 100000.0, False)
    assert r.pm.state == OPEN
    assert r.pm.entry_fees == pytest.approx(r.pm.qty * 100000.0 * cfg.fees.taker)
