"""
Стейт-машина позиции. Одинакова для бэктеста, paper и live.

FLAT -> ENTERING (post-only лимит) -> OPEN (биржевой стоп + TP-лимит) -> EXITING -> FLAT

Менеджер не ждёт ответов брокера: отправляет команды с собственными clientOrderId
и реагирует на OrderUpdate, которые брокер присылает позже.
"""
from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass

from .features import Snapshot
from .models import (OrderCmd, OrderKind, OrderStatus, OrderUpdate, PositionSync,
                     Side, Signal, TradeRecord)
from .risk import Instrument, RiskManager

log = logging.getLogger("position")

FLAT, ENTERING, OPEN, EXITING = "FLAT", "ENTERING", "OPEN", "EXITING"


@dataclass
class _Order:
    role: str            # entry | stop | tp | exit
    filled: float = 0.0
    final: bool = False


class PositionManager:
    def __init__(self, cfg, inst: Instrument, broker, risk: RiskManager, strategy, notify=None):
        self.cfg = cfg
        self.x = cfg.exits
        self.ex = cfg.execution
        self.fees = cfg.fees
        self.inst = inst
        self.broker = broker
        self.risk = risk
        self.strategy = strategy
        self.notify = notify or (lambda kind, payload: None)
        self._ids = itertools.count(1)
        self._run = format(int(time.time()) % 10**8, "x")
        self.trades: list[TradeRecord] = []
        self.signals_seen = 0
        self.skips: dict[str, int] = {}
        self.last_price = 0.0
        self.last_flat_ts = 0.0
        self._reset()

    # ---------- служебное ----------

    def _reset(self) -> None:
        self.state = FLAT
        self.side: Side | None = None
        self.sig: Signal | None = None
        self.orders: dict[str, _Order] = {}
        self.entry_cid = self.stop_cid = self.tp_cid = self.exit_cid = None
        self.entry_px = 0.0
        self.entry_placed_ts = 0.0
        self.reprices = 0
        self.pending_reprice = False
        self.qty = 0.0
        self.entry_value = 0.0
        self.entry_fees = 0.0
        self.exit_qty = 0.0
        self.exit_value = 0.0
        self.exit_fees = 0.0
        self.stop_px = 0.0
        self.stop_qty = 0.0
        self.open_ts = 0.0
        self.mfe = self.mae = 0.0
        self.be_done = False
        self.exit_reason = ""
        self.exit_sent_ts = 0.0
        self.target_qty = 0.0
        self.next_entry_px = 0.0
        self.retry_entry = False

    def _cid(self, role: str) -> str:
        return f"fs{self._run}{role[0]}{next(self._ids)}"

    def _skip(self, why: str) -> None:
        self.skips[why] = self.skips.get(why, 0) + 1

    def _send(self, role: str, **kw) -> str:
        cid = self._cid(role)
        self.orders[cid] = _Order(role)
        self.broker.submit(OrderCmd(action="place", cid=cid, **kw))
        return cid

    def _cancel(self, cid: str | None) -> None:
        if cid and cid in self.orders and not self.orders[cid].final:
            kind = OrderKind.STOP if self.orders[cid].role == "stop" else OrderKind.LIMIT
            self.broker.submit(OrderCmd(action="cancel", cid=cid, kind=kind))

    @property
    def avg_entry(self) -> float:
        return self.entry_value / self.qty if self.qty else 0.0

    @property
    def remaining(self) -> float:
        return max(self.qty - self.exit_qty, 0.0)

    @property
    def is_flat(self) -> bool:
        return self.state == FLAT

    # ---------- вход ----------

    def on_signal(self, sig: Signal, snap: Snapshot) -> None:
        self.signals_seen += 1
        if self.state != FLAT:
            return self._skip("busy")
        ok, why = self.risk.can_trade(sig.ts)
        if not ok:
            return self._skip(why)
        px = self._entry_price(sig.side, snap)
        if getattr(getattr(self.cfg.strategy, sig.setup, None), "entry", "maker") == "taker":
            px = snap.ask if sig.side is Side.LONG else snap.bid
        qty, why = self.risk.size(sig, px, self.inst)
        if qty <= 0:
            return self._skip(why.split("(")[0])
        self.side, self.sig = sig.side, sig
        self.state = ENTERING
        self.target_qty = qty
        self._place_entry(px, sig.ts)
        self.strategy.notify_entry(sig.ts)
        log.info("SIGNAL %s %s @%.2f qty=%s stop=%.2f tp=%.2f | %s", sig.setup, sig.side.name,
                 px, qty, sig.stop_dist, sig.tp_dist, sig.reason)

    def _entry_price(self, side: Side, snap: Snapshot) -> float:
        off = self.ex.entry_offset_ticks * self.inst.tick_size
        if side is Side.LONG:
            return self.inst.round_price(snap.bid - off, "down")
        return self.inst.round_price(snap.ask + off, "up")

    def _entry_mode(self) -> str:
        return getattr(getattr(self.cfg.strategy, self.sig.setup, None), "entry", "maker")

    def _place_entry(self, px: float, ts: float) -> None:
        self.entry_px = px
        self.entry_placed_ts = ts
        qty = self.inst.round_qty(self.target_qty - self.qty)
        if self._entry_mode() == "taker":
            self.entry_cid = self._send("entry", kind=OrderKind.MARKET, side=self.side.order_side, qty=qty)
            return
        self.entry_cid = self._send("entry", kind=OrderKind.LIMIT, side=self.side.order_side,
                                    qty=qty, price=px, post_only=True)

    # ---------- обработка ордеров ----------

    def on_order_update(self, u: OrderUpdate) -> None:
        o = self.orders.get(u.cid)
        if o is None:
            return
        if o.final and not u.last_qty:
            return
        if u.last_qty > 0:
            o.filled += u.last_qty
            if o.role == "entry":
                self._on_entry_fill(u)
            else:
                self._on_exit_fill(u, o.role)
        if u.status.is_final:
            o.final = True
            self._on_final(u, o)

    def _on_entry_fill(self, u: OrderUpdate) -> None:
        if self.state not in (ENTERING, OPEN):
            return
        if self.qty == 0:
            self.open_ts = u.ts
        self.qty += u.last_qty
        self.entry_value += u.last_qty * u.last_price
        self.entry_fees += u.fee
        self._sync_stop(u.ts)
        if self.state == ENTERING and self.qty >= self.target_qty - self.inst.step_size / 2:
            self._go_open(u.ts)

    def _sync_stop(self, ts: float) -> None:
        """Стоп всегда покрывает весь набранный объём. Новый ставим до отмены старого."""
        if self.remaining <= 0 or abs(self.stop_qty - self.remaining) < self.inst.step_size / 2:
            return
        if not self.stop_px:
            d = self.side.value
            self.stop_px = self.inst.round_price(self.avg_entry - d * self.sig.stop_dist,
                                                 "down" if d > 0 else "up")
        old = self.stop_cid
        self.stop_qty = self.inst.round_qty(self.remaining)
        self.stop_cid = self._send("stop", kind=OrderKind.STOP, side=self.side.close_side,
                                   qty=self.stop_qty, stop_price=self.stop_px, reduce_only=True)
        self._cancel(old)

    def _go_open(self, ts: float) -> None:
        if self.state == OPEN:
            return
        self.state = OPEN
        d = self.side.value
        # стоп и TP пересчитываем от фактической средней цены входа
        self.stop_px = self.inst.round_price(self.avg_entry - d * self.sig.stop_dist,
                                             "down" if d > 0 else "up")
        self.stop_qty = 0.0
        self._sync_stop(ts)
        tp_px = self.inst.round_price(self.avg_entry + d * self.sig.tp_dist, "up" if d > 0 else "down")
        self.tp_cid = self._send("tp", kind=OrderKind.LIMIT, side=self.side.close_side,
                                 qty=self.inst.round_qty(self.remaining), price=tp_px,
                                 post_only=self.ex.tp_post_only, reduce_only=True)
        self.notify("open", {"side": self.side.name, "setup": self.sig.setup, "qty": self.qty,
                             "price": self.avg_entry, "stop": self.stop_px, "tp": tp_px,
                             "reason": self.sig.reason})

    def _on_exit_fill(self, u: OrderUpdate, role: str) -> None:
        self.exit_qty += u.last_qty
        self.exit_value += u.last_qty * u.last_price
        self.exit_fees += u.fee
        if not self.exit_reason:
            self.exit_reason = {"stop": "breakeven" if self.be_done else "stop",
                                "tp": "tp"}.get(role, "exit")
        if self.remaining < self.inst.step_size / 2:
            self._finalize(u.ts)
        elif role in ("tp", "exit"):
            self._sync_stop(u.ts)     # частичный выход: уменьшаем стоп

    def _on_final(self, u: OrderUpdate, o: _Order) -> None:
        if o.role == "entry" and u.cid == self.entry_cid:
            if self.state != ENTERING:
                return
            if self.pending_reprice and self.qty < self.target_qty:
                self.pending_reprice = False
                self.reprices += 1
                self._place_entry(self.next_entry_px, u.ts)
            elif self.qty > 0:
                self._go_open(u.ts)
            elif u.status == OrderStatus.REJECTED and self.reprices < self.ex.max_reprices:
                # GTX отклонён: цена ушла через наш уровень. Повторяем по новой лучшей цене.
                self.reprices += 1
                self.entry_cid = None
                self.retry_entry = True      # переставим на следующем тике по свежей цене
            else:
                self._abort_entry("entry_" + u.status.value.lower())
        elif o.role == "tp" and u.status == OrderStatus.REJECTED and self.state == OPEN:
            # post-only TP отклонён: цена уже за уровнем TP. Забираем прибыль рынком.
            self._market_exit(u.ts, "tp_taker")
        elif o.role == "exit" and u.status in (OrderStatus.REJECTED, OrderStatus.EXPIRED, OrderStatus.CANCELED):
            if self.state == EXITING and self.remaining > 0:
                self.exit_cid = None   # повторим на следующем тике
        elif o.role == "stop" and u.status == OrderStatus.REJECTED and u.cid == self.stop_cid:
            # стоп отклонён (цена уже за ним): немедленно закрываемся
            if self.remaining > 0 and self.state in (ENTERING, OPEN):
                log.warning("stop rejected (%s): market exit", u.reason)
                self._market_exit(u.ts, "stop_rejected")

    def _abort_entry(self, why: str) -> None:
        self._skip(why)
        for cid, o in self.orders.items():
            if not o.final:
                self._cancel(cid)
        self.last_flat_ts = self.entry_placed_ts if self.entry_placed_ts != float("inf") else self.last_flat_ts
        self._reset()

    # ---------- выход ----------

    def _market_exit(self, ts: float, reason: str) -> None:
        if self.remaining <= 0:
            return
        if self.state == ENTERING:
            self._cancel(self.entry_cid)
        self._cancel(self.tp_cid)
        self.state = EXITING
        if not self.exit_reason:
            self.exit_reason = reason
        self.exit_sent_ts = ts
        self.exit_cid = self._send("exit", kind=OrderKind.MARKET, side=self.side.close_side,
                                   qty=self.inst.round_qty(self.remaining), reduce_only=True)

    def _move_stop(self, new_px: float, ts: float) -> None:
        self.stop_px = new_px
        self.stop_qty = 0.0
        self._sync_stop(ts)

    def _finalize(self, ts: float) -> None:
        for cid, o in self.orders.items():
            if not o.final:
                self._cancel(cid)
        d = self.side.value
        exit_px = self.exit_value / self.exit_qty if self.exit_qty else self.last_price
        gross = d * (exit_px - self.avg_entry) * self.qty
        fees = self.entry_fees + self.exit_fees
        rec = TradeRecord(symbol=self.inst.symbol, side=self.side.name, setup=self.sig.setup,
                          entry_ts=self.open_ts, exit_ts=ts, qty=self.qty,
                          entry_price=self.avg_entry, exit_price=exit_px, gross_pnl=gross,
                          fees=fees, net_pnl=gross - fees, exit_reason=self.exit_reason or "exit",
                          mfe=self.mfe, mae=self.mae, signal_reason=self.sig.reason)
        self.trades.append(rec)
        self.risk.on_trade_closed(rec.net_pnl, ts)
        log.info("CLOSE %s %s %s net=%.4f (%s) hold=%.0fs", rec.setup, rec.side, rec.exit_reason,
                 rec.net_pnl, f"{exit_px:.2f}", ts - self.open_ts)
        self.notify("close", rec)
        self.last_flat_ts = ts
        self._reset()

    # ---------- тик рынка ----------

    def on_market(self, snap: Snapshot) -> None:
        ts, p = snap.ts, snap.price
        self.last_price = p
        if self.retry_entry and self.state == ENTERING:
            self.retry_entry = False
            self._place_entry(self._entry_price(self.side, snap), ts)
            return
        if self.state == ENTERING:
            self._manage_entry(snap)
        elif self.state == OPEN:
            self._manage_open(snap)
        elif self.state == EXITING:
            if self.exit_cid is None and self.remaining > 0:
                self._market_exit(ts, self.exit_reason)
            elif ts - self.exit_sent_ts > 10:
                log.warning("exit not confirmed for 10s, resending")
                self.exit_cid = None
                self.exit_sent_ts = ts

    def _manage_entry(self, snap: Snapshot) -> None:
        if self.pending_reprice or self.entry_cid is None:
            return
        ts = snap.ts
        if ts - self.entry_placed_ts > self.ex.entry_timeout_s:
            self.pending_reprice = False
            self._cancel(self.entry_cid)        # финал придёт в _on_final -> open или abort
            self.entry_placed_ts = float("inf")
            return
        best = self._entry_price(self.side, snap)
        away = round((best - self.entry_px) * self.side.value / self.inst.tick_size, 6)
        if away >= self.ex.reprice_threshold_ticks and self.reprices < self.ex.max_reprices:
            self.pending_reprice = True
            self.next_entry_px = best
            self._cancel(self.entry_cid)

    def _manage_open(self, snap: Snapshot) -> None:
        ts, p = snap.ts, snap.price
        d = self.side.value
        entry = self.avg_entry
        move = d * (p - entry) / entry
        self.mfe = max(self.mfe, move)
        self.mae = min(self.mae, move)
        held = ts - self.open_ts

        # безубыток: стоп за вход + комиссии круга
        if not self.be_done and d * (p - entry) >= self.x.breakeven_trigger * self.sig.stop_dist:
            fee_cover = entry * (self.fees.maker + self.fees.taker)
            be = self.inst.round_price(entry + d * fee_cover, "up" if d > 0 else "down")
            if d * (p - be) > 2 * self.inst.tick_size:
                self.be_done = True
                self._move_stop(be, ts)

        if held >= self.x.max_hold_s:
            return self._market_exit(ts, "max_hold")
        if held >= self.x.time_stop_s and move <= 0:
            return self._market_exit(ts, "time_stop")
        zf = snap.z[min(snap.z)]
        if zf * d <= -self.x.flow_reversal_z:
            return self._market_exit(ts, "flow_reversal")

    # ---------- сверка и аварийные действия ----------

    def on_position_sync(self, ps: PositionSync) -> None:
        """Live: биржа — источник истины по размеру позиции."""
        tol = self.inst.step_size / 2
        if ps.ts < max(self.open_ts, self.last_flat_ts) + 5:   # снимок биржи мог устареть
            return
        if self.state in (OPEN, EXITING) and abs(ps.qty) < tol and self.remaining > tol:
            log.warning("exchange reports flat, local remaining=%s: closing record", self.remaining)
            self.exit_qty += self.remaining
            self.exit_value += self.remaining * self.last_price
            self.exit_reason = self.exit_reason or "exchange_flat"
            self._finalize(ps.ts)
        elif self.state == FLAT and abs(ps.qty) >= tol:
            self.notify("alert", f"Обнаружена чужая/осиротевшая позиция {ps.qty} @ {ps.entry_price}. Закрываю.")
            side = "sell" if ps.qty > 0 else "buy"
            self.broker.submit(OrderCmd(action="place", cid=self._cid("orphan"), kind=OrderKind.MARKET,
                                        side=side, qty=self.inst.round_qty(abs(ps.qty)), reduce_only=True))

    def flatten(self, ts: float, reason: str = "shutdown") -> None:
        if self.state == ENTERING and self.qty == 0:
            self._abort_entry(reason)
        elif self.state in (ENTERING, OPEN):
            self._market_exit(ts, reason)
