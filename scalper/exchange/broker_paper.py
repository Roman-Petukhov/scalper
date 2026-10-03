"""
Симулятор биржи для бэктеста и paper-режима.

Модель исполнения консервативная:
- команды применяются с задержкой latency_ms;
- post-only лимит, который пересёк бы спред, отклоняется (как GTX на Binance);
- лимит считается исполненным, только когда сделка прошла СКВОЗЬ его цену
  (уровень полностью съеден), либо при касании, если fill_on_touch=true;
- рыночные и стоп-ордера исполняются как тейкер со слиппеджем;
- reduceOnly никогда не увеличивает позицию.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from core.models import (Book, OrderCmd, OrderKind, OrderStatus, OrderUpdate,
                         PositionSync, Trade)
from .broker_base import Broker

log = logging.getLogger("paper")


@dataclass
class _PaperOrder:
    cmd: OrderCmd
    filled: float = 0.0
    value: float = 0.0

    @property
    def remaining(self) -> float:
        return self.cmd.qty - self.filled


class PaperBroker(Broker):
    def __init__(self, cfg, tick_size: float, step_size: float, equity: float):
        super().__init__()
        p = cfg.paper
        self.latency = p.latency_ms / 1000.0
        self.slip = p.slippage_bps / 1e4
        self.touch = bool(p.fill_on_touch)
        self.maker_fee = cfg.fees.maker
        self.taker_fee = cfg.fees.taker
        self.tick = tick_size
        self.step = step_size
        self.pending: list[tuple[float, OrderCmd]] = []
        self.open: dict[str, _PaperOrder] = {}
        self.now = 0.0
        self.last_price = 0.0
        self.last_buyer_maker = False
        self.book: Book | None = None
        # позиция и счёт
        self.pos = 0.0
        self.pos_value = 0.0       # Σ qty*price открытой части (для средней)
        self.realized = 0.0
        self.fees_paid = 0.0
        self.start_equity = equity
        self.last_sync = 0.0

    @property
    def equity(self) -> float:
        return self.start_equity + self.realized - self.fees_paid

    # ---------- команды ----------

    def submit(self, cmd: OrderCmd) -> None:
        self.pending.append((self.now + self.latency, cmd))

    def _activate(self, ts: float) -> None:
        if not self.pending:
            return
        due = [c for c in self.pending if c[0] <= ts]
        if not due:
            return
        self.pending = [c for c in self.pending if c[0] > ts]
        for _, cmd in due:
            if cmd.action == "cancel":
                o = self.open.pop(cmd.cid, None)
                if o:
                    self._update(o, OrderStatus.CANCELED, ts)
            else:
                self._place(cmd, ts)

    def _bid_ask(self) -> tuple[float, float]:
        bk = self.book
        if bk and bk.bids and bk.asks and self.now - bk.ts < 1.0:
            bid, ask = bk.bids[0][0], bk.asks[0][0]
            p = self.last_price
            if p >= ask:
                ask, bid = p, max(bid, p - self.tick)
            elif p <= bid:
                bid, ask = p, min(ask, p + self.tick)
            return bid, ask
        p = self.last_price
        return (p, p + self.tick) if self.last_buyer_maker else (p - self.tick, p)

    def _place(self, cmd: OrderCmd, ts: float) -> None:
        o = _PaperOrder(cmd)
        bid, ask = self._bid_ask()
        if cmd.reduce_only and self._reducible(cmd.side) <= 0:
            return self._update(o, OrderStatus.EXPIRED, ts, reason="reduce_only_no_position")
        if cmd.kind == OrderKind.MARKET:
            px = ask * (1 + self.slip) if cmd.side == "buy" else bid * (1 - self.slip)
            return self._fill(o, o.remaining, px, ts, taker=True)
        if cmd.kind == OrderKind.STOP:
            # стоп, который сработал бы сразу, биржа отклоняет (-2021)
            if (cmd.side == "sell" and self.last_price <= cmd.stop_price) or \
               (cmd.side == "buy" and self.last_price >= cmd.stop_price):
                return self._update(o, OrderStatus.REJECTED, ts, reason="would_immediately_trigger")
            self.open[cmd.cid] = o
            return self._update(o, OrderStatus.NEW, ts)
        crosses = cmd.price >= ask if cmd.side == "buy" else cmd.price <= bid
        if crosses:
            if cmd.post_only:
                return self._update(o, OrderStatus.REJECTED, ts, reason="post_only_would_take")
            px = ask if cmd.side == "buy" else bid
            return self._fill(o, o.remaining, px * (1 + self.slip if cmd.side == "buy" else 1 - self.slip),
                              ts, taker=True)
        self.open[cmd.cid] = o
        self._update(o, OrderStatus.NEW, ts)

    # ---------- рынок ----------

    def on_event(self, ev) -> None:
        ts = ev.ts
        self.now = max(self.now, ts)
        self._activate(ts)
        if isinstance(ev, Book):
            self.book = ev
        elif isinstance(ev, Trade):
            self.last_price = ev.price
            self.last_buyer_maker = ev.buyer_maker
            if self.open:
                self._match(ev)
        if ts - self.last_sync >= 1.0:
            self.last_sync = ts
            avg = self.pos_value / self.pos if self.pos else 0.0
            self.emit(PositionSync(ts=ts, qty=self.pos, entry_price=avg))

    def _match(self, t: Trade) -> None:
        for cid in list(self.open):
            o = self.open.get(cid)
            if o is None:
                continue
            c = o.cmd
            if c.kind == OrderKind.STOP:
                hit = t.price <= c.stop_price if c.side == "sell" else t.price >= c.stop_price
                if hit:
                    px = t.price * (1 - self.slip) if c.side == "sell" else t.price * (1 + self.slip)
                    self._fill(o, o.remaining, px, t.ts, taker=True)
            else:
                if c.side == "buy":
                    hit = t.price < c.price or (self.touch and t.price <= c.price)
                else:
                    hit = t.price > c.price or (self.touch and t.price >= c.price)
                if hit:
                    self._fill(o, o.remaining, c.price, t.ts, taker=False)

    # ---------- исполнение и учёт ----------

    def _reducible(self, side: str) -> float:
        if side == "sell":
            return max(self.pos, 0.0)
        return max(-self.pos, 0.0)

    def _fill(self, o: _PaperOrder, qty: float, px: float, ts: float, taker: bool) -> None:
        c = o.cmd
        if c.reduce_only:
            qty = min(qty, self._reducible(c.side))
            if qty < self.step / 2:
                self.open.pop(c.cid, None)
                return self._update(o, OrderStatus.EXPIRED, ts, reason="reduce_only_no_position")
        fee = qty * px * (self.taker_fee if taker else self.maker_fee)
        self._apply_position(c.side, qty, px)
        self.fees_paid += fee
        o.filled += qty
        o.value += qty * px
        done = o.remaining < self.step / 2 or c.reduce_only  # reduceOnly с остатком: остаток экспирится
        status = OrderStatus.FILLED if o.remaining < self.step / 2 else (
            OrderStatus.EXPIRED if c.reduce_only else OrderStatus.PARTIAL)
        if done:
            self.open.pop(c.cid, None)
        self._update(o, status, ts, last_qty=qty, last_price=px, fee=fee)

    def _apply_position(self, side: str, qty: float, px: float) -> None:
        signed = qty if side == "buy" else -qty
        if self.pos == 0 or (self.pos > 0) == (signed > 0):
            self.pos += signed
            self.pos_value += abs(signed) * px
            return
        avg = self.pos_value / abs(self.pos)
        closing = min(abs(signed), abs(self.pos))
        direction = 1 if self.pos > 0 else -1
        self.realized += direction * (px - avg) * closing
        self.pos_value -= avg * closing
        self.pos += signed
        if abs(self.pos) < self.step / 2:
            self.pos, self.pos_value = 0.0, 0.0
        elif (self.pos > 0) != (direction > 0):   # переворот позиции
            self.pos_value = abs(self.pos) * px

    def _update(self, o: _PaperOrder, status: OrderStatus, ts: float, last_qty: float = 0.0,
                last_price: float = 0.0, fee: float = 0.0, reason: str = "") -> None:
        avg = o.value / o.filled if o.filled else 0.0
        self.emit(OrderUpdate(cid=o.cmd.cid, status=status, filled=o.filled, avg_price=avg,
                              last_qty=last_qty, last_price=last_price, fee=fee, ts=ts, reason=reason))
