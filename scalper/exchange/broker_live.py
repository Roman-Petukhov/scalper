"""
Боевой брокер: Binance USDⓈ-M через ccxt (async + pro для потока ордеров).

- Входы и TP: лимиты с timeInForce=GTX (post-only).
- Стоп: STOP_MARKET reduceOnly, на Binance это algo-ордер. ccxt>=4.5 маршрутизирует сам.
- Статусы ордеров берутся из user-data потока (watch_orders) и дублируются опросом.
- Позиция периодически сверяется с биржей (PositionSync).
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import ccxt.pro as ccxtpro
from ccxt.base.errors import (InvalidOrder, OrderImmediatelyFillable, OrderNotFillable,
                              OrderNotFound)

from core.models import OrderCmd, OrderKind, OrderStatus, OrderUpdate, PositionSync
from core.risk import Instrument
from .broker_base import Broker

log = logging.getLogger("live")

STATUS_MAP = {
    "open": OrderStatus.NEW,
    "closed": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELED,
    "cancelled": OrderStatus.CANCELED,
    "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
}


@dataclass
class _Tracked:
    cmd: OrderCmd
    filled: float = 0.0
    value: float = 0.0
    final: bool = False
    last_seen: float = 0.0
    new_sent: bool = False


class LiveBroker(Broker):
    def __init__(self, cfg, api_key: str, secret: str, on_alert=None):
        super().__init__()
        self.cfg = cfg
        self.env = cfg.exchange.env
        self.ccxt_symbol = _ccxt_symbol(cfg.exchange.symbol)
        self.ex = ccxtpro.binanceusdm({
            "apiKey": api_key, "secret": secret, "enableRateLimit": True,
            "options": {"defaultType": "future", "adjustForTimeDifference": True},
        })
        if self.env == "demo":
            self.ex.enable_demo_trading(True)
        self.queue: asyncio.Queue[OrderCmd] = asyncio.Queue()
        self.tracked: dict[str, _Tracked] = {}
        self.on_alert = on_alert or (lambda msg: None)
        self.maker_fee = cfg.fees.maker
        self.taker_fee = cfg.fees.taker
        self.tasks: list[asyncio.Task] = []
        self.instrument: Instrument | None = None
        self._stopping = False

    # ---------- запуск ----------

    async def start(self) -> Instrument:
        await self.ex.load_markets()
        m = self.ex.market(self.ccxt_symbol)
        self.instrument = Instrument(
            symbol=self.cfg.exchange.symbol,
            tick_size=float(m["precision"]["price"]),
            step_size=float(m["precision"]["amount"]),
            min_notional=float((m["limits"].get("cost") or {}).get("min") or self.cfg.instrument.min_notional),
        )
        try:
            await self.ex.set_position_mode(False, self.ccxt_symbol)   # one-way, не hedge
        except Exception as e:
            log.info("set_position_mode: %s", e)
        try:
            await self.ex.set_margin_mode(self.cfg.exchange.margin_mode, self.ccxt_symbol)
        except Exception as e:   # "No need to change margin type" и т.п.
            log.info("set_margin_mode: %s", e)
        try:
            await self.ex.set_leverage(int(self.cfg.exchange.leverage), self.ccxt_symbol)
        except Exception as e:
            log.warning("set_leverage: %s", e)
        await self.cancel_all()
        self.tasks = [
            asyncio.create_task(self._worker(), name="orders-worker"),
            asyncio.create_task(self._watch_orders(), name="watch-orders"),
            asyncio.create_task(self._poll_orders(), name="poll-orders"),
            asyncio.create_task(self._reconcile(), name="reconcile"),
        ]
        return self.instrument

    async def fetch_equity(self) -> float:
        bal = await self.ex.fetch_balance()
        return float(bal.get("USDT", {}).get("total") or 0.0)

    async def cancel_all(self) -> None:
        for params in ({}, {"trigger": True}):
            try:
                await self.ex.cancel_all_orders(self.ccxt_symbol, params)
            except Exception as e:
                log.info("cancel_all_orders%s: %s", params, e)

    async def close(self) -> None:
        self._stopping = True
        for t in self.tasks:
            t.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.ex.close()

    # ---------- команды ----------

    def submit(self, cmd: OrderCmd) -> None:
        self.queue.put_nowait(cmd)

    async def _worker(self) -> None:
        while True:
            cmd = await self.queue.get()
            try:
                if cmd.action == "cancel":
                    await self._cancel(cmd)
                else:
                    await self._place(cmd)
            except Exception as e:      # не роняем воркер из-за одной команды
                log.exception("order command failed: %s", cmd)
                self.on_alert(f"Ошибка ордера {cmd.cid}: {e}")

    async def _place(self, cmd: OrderCmd) -> None:
        t = self.tracked[cmd.cid] = _Tracked(cmd, last_seen=time.time())
        params: dict = {"reduceOnly": True} if cmd.reduce_only else {}
        try:
            if cmd.kind == OrderKind.LIMIT:
                params["newClientOrderId"] = cmd.cid
                if cmd.post_only:
                    params["timeInForce"] = "GTX"
                o = await self.ex.create_order(self.ccxt_symbol, "limit", cmd.side, cmd.qty, cmd.price, params)
            elif cmd.kind == OrderKind.MARKET:
                params["newClientOrderId"] = cmd.cid
                o = await self.ex.create_order(self.ccxt_symbol, "market", cmd.side, cmd.qty, None, params)
            else:
                params.update({"clientOrderId": cmd.cid, "stopPrice": cmd.stop_price,
                               "workingType": "CONTRACT_PRICE"})
                o = await self.ex.create_order(self.ccxt_symbol, "STOP_MARKET", cmd.side, cmd.qty, None, params)
        except (OrderNotFillable, OrderImmediatelyFillable) as e:
            # GTX пересёк бы спред (-5022) или стоп сработал бы сразу (-2021)
            return self._final(t, OrderStatus.REJECTED, reason=type(e).__name__)
        except InvalidOrder as e:
            self.on_alert(f"Ордер отклонён биржей ({cmd.kind.value} {cmd.side} {cmd.qty}): {e}")
            return self._final(t, OrderStatus.REJECTED, reason=str(e)[:120])
        self._on_ccxt_order(o)

    async def _cancel(self, cmd: OrderCmd) -> None:
        t = self.tracked.get(cmd.cid)
        if t is None or t.final:
            return
        params = {"clientOrderId": cmd.cid}
        if t.cmd.kind == OrderKind.STOP:
            params["trigger"] = True
        try:
            o = await self.ex.cancel_order(None, self.ccxt_symbol, params)
            self._on_ccxt_order(o)
        except OrderNotFound:
            await self._refresh(t)     # уже исполнен или отменён: узнаём итоговый статус

    # ---------- статусы ----------

    def _on_ccxt_order(self, o: dict) -> None:
        cid = o.get("clientOrderId")
        t = self.tracked.get(cid)
        if t is None or t.final:
            return
        t.last_seen = time.time()
        status = STATUS_MAP.get(o.get("status") or "open", OrderStatus.NEW)
        filled = float(o.get("filled") or 0.0)
        avg = float(o.get("average") or o.get("price") or 0.0)
        last_qty = filled - t.filled
        last_px, fee = 0.0, 0.0
        if last_qty > 1e-12:
            new_value = filled * avg
            last_px = (new_value - t.value) / last_qty if new_value > t.value else avg
            maker = t.cmd.kind == OrderKind.LIMIT and t.cmd.post_only
            fee = last_qty * last_px * (self.maker_fee if maker else self.taker_fee)
            t.filled, t.value = filled, new_value
        else:
            last_qty = 0.0
        if status == OrderStatus.NEW and filled > 0:
            status = OrderStatus.PARTIAL
        if status == OrderStatus.EXPIRED and t.cmd.post_only and filled == 0:
            status = OrderStatus.REJECTED   # GTX, принятый и сразу снятый биржей
        if status.is_final:
            t.final = True
        elif last_qty == 0 and status == OrderStatus.NEW and t.new_sent:
            return
        t.new_sent = True
        self.emit(OrderUpdate(cid=cid, status=status, filled=t.filled,
                              avg_price=t.value / t.filled if t.filled else 0.0,
                              last_qty=last_qty, last_price=last_px, fee=fee, ts=self.clock()))

    def _final(self, t: _Tracked, status: OrderStatus, reason: str = "") -> None:
        t.final = True
        self.emit(OrderUpdate(cid=t.cmd.cid, status=status, filled=t.filled,
                              avg_price=t.value / t.filled if t.filled else 0.0,
                              ts=self.clock(), reason=reason))

    async def _refresh(self, t: _Tracked) -> None:
        params = {"clientOrderId": t.cmd.cid}
        if t.cmd.kind == OrderKind.STOP:
            params["trigger"] = True
        try:
            o = await self.ex.fetch_order(None, self.ccxt_symbol, params)
            self._on_ccxt_order(o)
        except OrderNotFound:
            if time.time() - t.last_seen > 30:
                self._final(t, OrderStatus.CANCELED, reason="not_found")

    async def _watch_orders(self) -> None:
        while not self._stopping:
            try:
                orders = await self.ex.watch_orders(self.ccxt_symbol)
                for o in orders:
                    self._on_ccxt_order(o)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("watch_orders: %s; reconnecting", e)
                await asyncio.sleep(2)

    async def _poll_orders(self) -> None:
        """Подстраховка на случай потери сообщений из user-data потока."""
        while not self._stopping:
            await asyncio.sleep(1.0)
            now = time.time()
            for t in list(self.tracked.values()):
                if not t.final and now - t.last_seen > 2.0:
                    try:
                        await self._refresh(t)
                    except Exception as e:
                        log.debug("refresh %s: %s", t.cmd.cid, e)
            # чистим старые финальные
            if len(self.tracked) > 500:
                for cid in [c for c, t in self.tracked.items() if t.final][:250]:
                    self.tracked.pop(cid, None)

    async def _reconcile(self) -> None:
        interval = float(self.cfg.exchange.reconcile_interval_s)
        while not self._stopping:
            await asyncio.sleep(interval)
            started = self.clock()
            try:
                positions = await self.ex.fetch_positions([self.ccxt_symbol])
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("fetch_positions: %s", e)
                continue
            qty, entry = 0.0, 0.0
            for p in positions:
                c = float(p.get("contracts") or 0.0)
                if c:
                    qty = c if p.get("side") == "long" else -c
                    entry = float(p.get("entryPrice") or 0.0)
            self.emit(PositionSync(ts=started, qty=qty, entry_price=entry))


def _ccxt_symbol(symbol: str) -> str:
    """BTCUSDT -> BTC/USDT:USDT"""
    base = symbol[:-4] if symbol.endswith("USDT") else symbol
    return f"{base}/USDT:USDT"
