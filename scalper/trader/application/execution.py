"""Исполнение на бирже: ордер по сигналу (кнопка «Вхожу» или авто-режим), снятие просроченных лимиток,
состояние кошелька для панели."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from datetime import timedelta

from ..domain.execution import Account, ExecutionRefused, Trade, TradeStatus, build_order, target_reached
from ..domain.models import Signal, SignalStatus
from .ports import Broker, Notifier, SettingsRepository, SignalRepository, TradeRepository

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Wallet:
    network: str
    account: Account
    day_start: float
    updated: datetime

    @property
    def day_pnl(self) -> float:
        """Изменение капитала с 00:00 UTC (учитывает и закрытые сделки, и открытые позиции)."""
        return self.account.equity - self.day_start

    @property
    def day_pnl_pct(self) -> float:
        return self.day_pnl / self.day_start * 100 if self.day_start > 0 else 0.0


class Executor:
    def __init__(self, broker: Callable[[], Broker | None], signals: SignalRepository,
                 settings: SettingsRepository, trades: TradeRepository, notifier: Notifier | None = None,
                 clock: Callable[[], datetime] = _utcnow) -> None:
        self.broker, self.signals, self.settings, self.trades = broker, signals, settings, trades
        self.notifier, self.clock = notifier, clock
        self.lock = asyncio.Lock()          # ручной и авто-вход не должны открыть одну монету дважды

    @property
    def connected(self) -> bool:
        return self.broker() is not None

    def _day_start(self, equity: float) -> float:
        """Капитал на начало UTC-дня: первый замер за день запоминается в базе."""
        key = f"day_equity:{self.clock().date().isoformat()}"
        v = self.trades.kv_get(key)
        if v is None:
            self.trades.kv_set(key, repr(equity))
            return equity
        return float(v)

    async def wallet(self) -> Wallet | None:
        b = self.broker()
        if b is None:
            return None
        acc = await b.account()
        return Wallet(b.network, acc, self._day_start(acc.equity), self.clock())

    async def execute(self, signal: Signal) -> Trade:
        """Отправить ордер по сигналу. ExecutionRefused — не отправлен (причина в тексте); сигнал тогда не меняется."""
        b = self.broker()
        if b is None:
            raise ExecutionRefused("биржа не подключена")
        async with self.lock:
            fresh = self.signals.get(signal.id) if signal.id is not None else None
            if fresh is None or fresh.status is not SignalStatus.NEW:
                raise ExecutionRefused("сигнал уже обработан")
            try:
                inst = await b.instrument(signal.symbol)
                if inst is None:
                    raise ExecutionRefused(f"{signal.symbol} не торгуется на Bybit")
                acc, price = await asyncio.gather(b.account(), b.price(signal.symbol))
                req = build_order(signal, self.settings.load(), acc, inst, price, self._day_start(acc.equity),
                                  self.clock())
                order_id = await b.place(req)
            except ExecutionRefused:
                raise
            except Exception as e:                                   # сеть, ключи, отказ биржи
                log.exception("ордер %s", signal.symbol)
                raise ExecutionRefused(f"биржа отклонила: {_short(e)}") from e
            trade = self.trades.add_trade(Trade.from_order(req, order_id, b.network, self.clock()))
            label = "Bybit демо" if b.network == "demo" else "Bybit"
            self.signals.set_status(signal.id, SignalStatus.TAKEN, f"{label}: {req.describe()}")
        log.info("ордер %s: %s", order_id, req.describe())
        return trade

    async def housekeep(self) -> None:
        """Раз в минуту, как в бэктесте: лимитка ретеста исполнилась — помечаем; истекло время или цена дошла до цели
        без ретеста — снимаем; позиция дольше срока сделки (max_hold_bars свечей с отправки ордера) — закрываем."""
        b = self.broker()
        pending, filled = self.trades.pending_trades(), self.trades.filled_trades()
        if b is None or not (pending or filled):
            return
        open_ids, acc = await asyncio.gather(b.open_order_ids(), b.account())
        held = {p.symbol: p for p in acc.positions}
        now = self.clock()
        for t in pending:
            if t.order_id not in open_ids:                         # исчезла из открытых: исполнилась или снята
                self.trades.set_trade_status(t.id, TradeStatus.FILLED if t.symbol in held else TradeStatus.CANCELLED)
                continue
            why = None
            if t.expires_at is not None and now >= t.expires_at:
                why = "ретеста не было — лимитка снята"
            else:
                try:
                    if target_reached(t.side, t.target, await b.price(t.symbol)):
                        why = "цена дошла до цели без ретеста — лимитка снята"
                except Exception:
                    log.exception("цена %s", t.symbol)
            if why is None:
                continue
            try:
                await b.cancel(t.symbol, t.order_id)
            except Exception:
                log.exception("снятие лимитки %s", t.symbol)
                continue
            self.trades.set_trade_status(t.id, TradeStatus.EXPIRED)
            self.signals.set_status(t.signal_id, SignalStatus.EXPIRED, why)
            await self._say(f"{t.symbol}: {why}")
        settings, seen = self.settings.load(), set()
        for t in filled:                                           # новые первыми: по монете — только последняя сделка
            if t.symbol in seen:
                continue
            seen.add(t.symbol)
            pos, sig = held.get(t.symbol), self.signals.get(t.signal_id)
            if pos is None or pos.side is not t.side or sig is None or t.created_at is None:
                continue
            bars = settings.p(sig.timeframe).max_hold_bars
            if now < t.created_at + timedelta(minutes=sig.timeframe.minutes * bars):
                continue
            try:
                await b.close_position(t.symbol)
            except Exception:
                log.exception("закрытие по времени %s", t.symbol)
                continue
            self.trades.set_trade_status(t.id, TradeStatus.TIMED_OUT)
            await self._say(f"{t.symbol}: позиция закрыта по рынку — прошло {bars} свечей {sig.timeframe.value}")

    async def _say(self, text: str) -> None:
        if self.notifier is not None:
            try:
                await self.notifier.text(text)
            except Exception:
                log.exception("уведомление")


def _short(e: Exception) -> str:
    msg = str(e) or type(e).__name__
    return msg if len(msg) <= 160 else msg[:157] + "…"
