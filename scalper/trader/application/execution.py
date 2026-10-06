"""Исполнение на бирже: ордер по сигналу (кнопка «Вхожу» или авто-режим), снятие просроченных лимиток,
состояние кошелька для панели."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from datetime import timedelta

from ..domain.execution import (Account, ExecutionRefused, Trade, TradeResult, TradeStatus, build_order, settle,
                                target_reached)
from ..domain.journal import Health, tf_stats
from ..domain.models import Signal, SignalStatus, Timeframe
from .pnl import PnlHistory, PnlPeriod
from .ports import Broker, Notifier, SettingsRepository, SignalRepository, TradeRepository

log = logging.getLogger(__name__)

PNL_WINDOW = timedelta(days=6, hours=23)    # биржа отдаёт закрытия окнами не длиннее 7 дней
GIVE_UP = timedelta(days=8)                 # закрытия так и не нашлось — итог «нет данных», больше не ищем


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Wallet:
    network: str
    account: Account
    day_start: float
    updated: datetime
    periods: tuple[PnlPeriod, ...] = ()

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
        self.pnl = PnlHistory(trades, clock)

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
        return Wallet(b.network, acc, self._day_start(acc.equity), self.clock(),
                      tuple(self.pnl.periods(b.network, acc.equity)))

    async def refresh_pnl(self) -> None:
        """Раз в минуту: реализованный PnL за сегодня и догрузка истории для недели / 30 дней / полугода."""
        b = self.broker()
        if b is not None:
            await self.pnl.refresh(b)

    def _tf_open(self, acc: Account, tf: Timeframe) -> int:
        """Сколько занятых монет (позиция или лимитка) заняты сделками таймфрейма tf: по монете — последняя сделка."""
        latest: dict[str, Trade] = {}
        for t in sorted(self.trades.pending_trades() + self.trades.filled_trades(),
                        key=lambda t: t.id or 0, reverse=True):
            latest.setdefault(t.symbol, t)
        n = 0
        for sym in acc.busy_symbols:
            t = latest.get(sym)
            sig = self.signals.get(t.signal_id) if t is not None else None
            n += sig is not None and sig.timeframe is tf
        return n

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
                                  self.clock(), self._tf_open(acc, signal.timeframe))
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
        без ретеста — снимаем; позиция дольше срока сделки (max_hold_bars свечей с исполнения) — закрываем;
        закрытая позиция — итог с биржи в журнал."""
        b = self.broker()
        pending, filled = self.trades.pending_trades(), self.trades.filled_trades()
        unsettled = self.trades.unsettled_trades()
        if b is None or not (pending or filled or unsettled):
            return
        open_ids, acc = await asyncio.gather(b.open_order_ids(), b.account())
        held = {p.symbol: p for p in acc.positions}
        now = self.clock()
        for t in pending:
            if t.order_id not in open_ids:                         # исчезла из открытых: исполнилась или снята
                if t.symbol in held:
                    self.trades.set_trade_filled(t.id, now)
                    continue
                res = await self._result(b, t, None, now)          # исполнилась и уже закрылась за эту минуту?
                if res is None:
                    self.trades.set_trade_status(t.id, TradeStatus.CANCELLED)
                    self._close_signal(t.signal_id)
                else:
                    self.trades.set_trade_filled(t.id, now)
                    await self._settle(replace(t, status=TradeStatus.FILLED, filled_at=now), TradeStatus.CLOSED, res)
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
            if pos is None or pos.side is not t.side or sig is None or t.opened_at is None:
                continue
            bars = settings.p(sig.timeframe).max_hold_bars
            if now < t.opened_at + timedelta(minutes=sig.timeframe.minutes * bars):
                continue
            try:
                await b.close_position(t.symbol)
            except Exception:
                log.exception("закрытие по времени %s", t.symbol)
                continue
            self.trades.set_trade_status(t.id, TradeStatus.TIMED_OUT)
            await self._say(f"{t.symbol}: позиция закрыта по рынку — прошло {bars} свечей {sig.timeframe.value}")
        await self._settle_closed(b, held, now)

    async def _settle_closed(self, b: Broker, held: dict, now: datetime) -> None:
        """Сделки, позиции которых уже нет, получают итог с биржи. Окно поиска — от исполнения до следующей сделки
        по той же монете: старая сделка не заберёт закрытие новой."""
        later: dict[int, datetime] = {}
        last_at: dict[str, datetime] = {}
        for e in self.trades.journal():                            # новые первыми
            t = e.trade
            if t.id is not None and t.symbol in last_at:
                later[t.id] = last_at[t.symbol]
            if t.created_at is not None:
                last_at[t.symbol] = t.created_at
        for t in self.trades.unsettled_trades():
            pos = held.get(t.symbol)
            if t.id not in later and t.status is TradeStatus.FILLED and pos is not None and pos.side is t.side:
                continue                                           # ещё открыта
            until = later.get(t.id, now)
            res = await self._result(b, t, until, now)
            status = TradeStatus.TIMED_OUT if t.status is TradeStatus.TIMED_OUT else TradeStatus.CLOSED
            if res is not None:
                await self._settle(t, status, res)
            elif t.opened_at is not None and now - t.opened_at > GIVE_UP:
                self.trades.settle_trade(t.id, status, None)
                self._close_signal(t.signal_id)

    async def _result(self, b: Broker, t: Trade, until: datetime | None, now: datetime) -> TradeResult | None:
        since = (t.opened_at or now) - timedelta(minutes=1)
        end = until or now
        try:
            recs = await b.closed_pnl(t.symbol, max(since, end - PNL_WINDOW), end)
        except Exception:
            log.exception("итог сделки %s", t.symbol)
            return None
        return settle(recs, t.side, since, end)

    async def _settle(self, t: Trade, status: TradeStatus, res: TradeResult) -> None:
        self.trades.settle_trade(t.id, status, res)
        self._close_signal(t.signal_id)
        done = replace(t, status=status, result=res)
        sig = self.signals.get(t.signal_id)
        tf = f" {sig.timeframe.value}" if sig is not None else ""
        r = done.r_multiple
        await self._say(f"{t.symbol}{tf} {t.side.label} закрыта: "
                        f"{f'{r:+.2f}R' if r is not None else f'{res.pnl_usd:+.2f} $'} ({done.exit_reason})")
        if sig is not None:
            await self._check_health(sig.timeframe)

    def _close_signal(self, signal_id: int) -> None:
        """Сделка завершена — сигнал уходит из ленты в архив; описание ордера остаётся."""
        sig = self.signals.get(signal_id)
        if sig is not None and sig.status is SignalStatus.TAKEN:
            self.signals.set_status(signal_id, SignalStatus.CLOSED, sig.note)

    async def _check_health(self, tf: Timeframe) -> None:
        """Детектор «стратегия перестала работать»: при переходе в «присмотреться» или «остановить» — уведомление."""
        st = next(x for x in tf_stats(self.trades.journal(), {tf: self.settings.load().p(tf).target_r})
                  if x.timeframe is tf)
        if st.health is None:
            return
        key = f"health:{tf.value}"
        prev = self.trades.kv_get(key)
        self.trades.kv_set(key, st.health.level.value)
        if st.health.level in (Health.WATCH, Health.STOP) and prev != st.health.level.value:
            icon = "🛑" if st.health.level is Health.STOP else "⚠️"
            await self._say(f"{icon} {tf.value}: {st.closed} сделок, в среднем {st.health.live_r:+.2f}R. {st.health.text}")

    async def _say(self, text: str) -> None:
        if self.notifier is not None:
            try:
                await self.notifier.text(text)
            except Exception:
                log.exception("уведомление")


def _short(e: Exception) -> str:
    msg = str(e) or type(e).__name__
    return msg if len(msg) <= 160 else msg[:157] + "…"
