"""Связка признаки -> стратегия -> позиция -> брокер. Одна и та же для всех режимов."""
from __future__ import annotations

from .features import Features
from .models import OrderUpdate, PositionSync
from .position import PositionManager
from .risk import Instrument, RiskManager
from .strategy import FlowScalper


class TradingSession:
    def __init__(self, cfg, inst: Instrument, broker, equity: float, notify=None):
        self.cfg = cfg
        self.notify = notify or (lambda kind, payload: None)
        self.features = Features(cfg, inst.tick_size)
        self.strategy = FlowScalper(cfg)
        self.risk = RiskManager(cfg, equity, on_halt=lambda msg: self.notify("alert", msg))
        self.broker = broker
        self.manager = PositionManager(cfg, inst, broker, self.risk, self.strategy, notify=self.notify)
        self.interval = min(float(cfg.strategy.eval_interval_s), 0.25)
        self.last_snap_ts = 0.0
        self.last_snap = None
        self.simulated = hasattr(broker, "on_event")   # paper-брокер сам матчит ордера по рынку
        self.trading_enabled = True
        broker.set_sink(self._on_broker)

    def _on_broker(self, msg) -> None:
        if isinstance(msg, OrderUpdate):
            self.manager.on_order_update(msg)
        elif isinstance(msg, PositionSync):
            self.manager.on_position_sync(msg)

    def on_event(self, ev) -> None:
        if self.simulated:
            self.broker.on_event(ev)
        self.features.on_event(ev)
        ts = ev.ts
        if ts - self.last_snap_ts < self.interval:
            return
        self.last_snap_ts = ts
        snap = self.features.snapshot(ts)
        self.last_snap = snap
        self.manager.on_market(snap)
        if not self.trading_enabled:
            return
        sig = self.strategy.evaluate(snap)
        if sig is not None:
            self.manager.on_signal(sig, snap)
