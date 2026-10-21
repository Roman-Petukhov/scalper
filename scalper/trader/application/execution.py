"""Исполнение на бирже: ордер по сигналу (кнопка «Вхожу» или авто-режим), снятие просроченных лимиток,
состояние кошелька для панели."""
from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from datetime import timedelta

from ..domain.execution import (PYRAMID_AT_R, Account, ExecutionRefused, HedgeLeg, Position, Trade, TradeResult,
                                TradeStatus, build_order, pyramid_due, pyramid_qty, settle, target_reached)
from ..domain.hedge import HEDGE_SYMBOL, beta, held_qty, hedge_leverage, rebalance, target_qty, without_hedge
from ..domain.journal import Health, tf_stats
from ..domain.models import Signal, SignalStatus, Timeframe
from .pnl import PnlHistory, PnlPeriod
from .ports import Broker, MarketData, Notifier, SettingsRepository, SignalRepository, TradeRepository

log = logging.getLogger(__name__)

PNL_WINDOW = timedelta(days=6, hours=23)    # биржа отдаёт закрытия окнами не длиннее 7 дней
NOTICE_AFTER = timedelta(minutes=30)            # позиции нет, записи о закрытии нет — один раз сказать, что искали
NOTICE_KEY = "noresult:{}"
GIVE_UP = timedelta(days=2)                 # позиции нет уже столько, а закрытия на бирже нет — итог «нет данных»
GONE_KEY = "gone:{}"                        # когда бот впервые не увидел позицию сделки (id) на бирже
LOST_REOPEN_KEY = "lost_reopen_2026_10"     # один раз: вернуть в поиск итоги, потерянные старым правилом «8 дней с входа»
LOST_LOOKBACK = timedelta(days=30)
HEDGE_ERR_KEY = "hedge:error"               # последняя ошибка хеджа: уведомление — один раз, а не каждую минуту
HEDGE_KEY = "hedge:held"                    # "1" — позицию BTCUSDT ведёт хедж (не трогать ручную, пока хеджа не было)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Wallet:
    network: str
    account: Account
    day_start: float
    updated: datetime
    periods: tuple[PnlPeriod, ...] = ()
    hedge_symbol: str | None = None         # эта позиция — хедж BTC, а не сделка

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
                 clock: Callable[[], datetime] = _utcnow, market: MarketData | None = None) -> None:
        self.broker, self.signals, self.settings, self.trades = broker, signals, settings, trades
        self.notifier, self.clock, self.market = notifier, clock, market
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
                      tuple(self.pnl.periods(b.network, acc.equity)),
                      HEDGE_SYMBOL if self._hedge_managed(self.settings.load()) else None)

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
            settings = self.settings.load()
            hedged = self._hedge_managed(settings)
            if hedged and signal.symbol == HEDGE_SYMBOL:
                raise ExecutionRefused(f"{HEDGE_SYMBOL} занят хеджем — по нему бот не входит")
            try:
                inst = await b.instrument(signal.symbol)
                if inst is None:
                    raise ExecutionRefused(f"{signal.symbol} не торгуется на Bybit")
                acc, price = await asyncio.gather(b.account(), b.price(signal.symbol))
                if hedged:
                    acc = without_hedge(acc)                         # позиция хеджа не занимает место сделки
                leg = await self._hedge_leg(signal) if settings.p(signal.timeframe).hedge_btc else None
                req = build_order(signal, settings, acc, inst, price, self._day_start(acc.equity),
                                  self.clock(), self._tf_open(acc, signal.timeframe), leg.beta if leg else 0.0,
                                  settings.hedge_lev)
                order_id = await b.place(req)
            except ExecutionRefused:
                raise
            except Exception as e:                                   # сеть, ключи, отказ биржи
                log.exception("ордер %s", signal.symbol)
                raise ExecutionRefused(f"биржа отклонила: {_short(e)}") from e
            trade = self.trades.add_trade(replace(Trade.from_order(req, order_id, b.network, self.clock()), hedge=leg))
            label = "Bybit демо" if b.network == "demo" else "Bybit"
            note = ""
            if settings.p(signal.timeframe).hedge_btc:
                note = f" · хедж BTC β {leg.beta:.2f}" if leg else " · без хеджа: бету посчитать не удалось"
            self.signals.set_status(signal.id, SignalStatus.TAKEN, f"{label}: {req.describe()}{note}")
        log.info("ордер %s: %s", order_id, req.describe())
        return trade

    async def housekeep(self) -> None:
        """Раз в минуту, как в бэктесте: лимитка ретеста исполнилась — помечаем; истекло время или цена дошла до цели
        без ретеста — снимаем; позиция дольше срока сделки (max_hold_bars свечей с исполнения) — закрываем;
        закрытая позиция — итог с биржи в журнал."""
        b = self.broker()
        if b is not None and self.trades.kv_get(LOST_REOPEN_KEY) is None:
            self._reopen_lost(self.clock())
        pending, filled = self.trades.pending_trades(), self.trades.filled_trades()
        unsettled = self.trades.unsettled_trades()
        if b is None or not (pending or filled or unsettled or self.trades.kv_get(HEDGE_KEY) == "1"):
            return
        open_ids, acc = await asyncio.gather(b.open_order_ids(), b.account())
        held = {p.symbol: p for p in acc.positions}
        now = self.clock()
        try:
            await self._watch_orders(b, pending, filled, held, open_ids, now, acc)
        except Exception:                                       # сбой тут не должен лишать закрытые сделки итога
            log.exception("сверка ордеров и сроков")
        await self._settle_closed(b, held, now)
        try:
            await self._rebalance_hedge(b, acc)
        except Exception:
            log.exception("хедж BTC")

    async def _watch_orders(self, b: Broker, pending: list[Trade], filled: list[Trade], held: dict, open_ids: set,
                            now: datetime, acc: Account) -> None:
        """Лимитки: исполнилась / истекла / цена ушла к цели; позиции: срок сделки и доливка."""
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
                if settings.p(sig.timeframe).pyramid and pyramid_due(t, pos.mark):
                    await self._pyramid(b, t, pos, sig, settings, acc)
                continue
            try:
                await b.close_position(t.symbol)
            except Exception:
                log.exception("закрытие по времени %s", t.symbol)
                continue
            self.trades.set_trade_status(t.id, TradeStatus.TIMED_OUT)
            await self._say(f"{t.symbol}: позиция закрыта по рынку — прошло {bars} свечей {sig.timeframe.value}")

    async def _pyramid(self, b: Broker, t: Trade, pos: Position, sig: Signal, settings, acc: Account) -> None:
        """Цена прошла +1.5R: стоп всей позиции — в точку входа, затем долить 0.5 исходного объёма (стоп и цель
        биржа держит на весь объём). Стоп переставляется первым: если доливка не пройдёт, риск всё равно ушёл."""
        tf = sig.timeframe.value
        try:
            await b.protect(t.symbol, t.price, t.target)
        except Exception as e:
            log.exception("стоп в безубыток %s", t.symbol)
            key = f"pyramid:error:{t.id}"
            if self.trades.kv_get(key) != _short(e):
                self.trades.kv_set(key, _short(e))
                await self._say(f"⚠️ {t.symbol} {tf}: +{PYRAMID_AT_R:g}R, но стоп в безубыток не переставлен: "
                                f"{_short(e)}. Повторю через минуту.")
            return
        qty, why = 0.0, ""
        inst = await b.instrument(t.symbol)
        if inst is None:
            why = "монета не торгуется"
        else:
            lev = int(max(1, min(settings.lev(sig.timeframe), inst.max_leverage)))
            qty, why = pyramid_qty(t, inst, pos.mark, acc.available, lev)
            if qty:
                try:
                    await b.adjust(t.symbol, int(t.side) * qty, lev)
                    await b.protect(t.symbol, t.price, t.target)   # на случай, если биржа не растянула их на долив
                except Exception as e:
                    log.exception("доливка %s", t.symbol)
                    qty, why = 0.0, _short(e)
        self.trades.set_trade_pyramid(t.id, qty, pos.mark)
        if qty:
            await self._say(f"{t.symbol} {tf}: +{PYRAMID_AT_R:g}R — долил {qty:g} по ~{pos.mark:g}, стоп всей позиции "
                            f"в точку входа {t.price:g}, цель {t.target:g}")
        else:
            await self._say(f"{t.symbol} {tf}: +{PYRAMID_AT_R:g}R — стоп в точку входа {t.price:g}, без доливки ({why})")

    def _hedge_managed(self, settings) -> bool:
        """Позицию BTCUSDT ведёт хедж: он включён в настройках или ещё держится по открытым сделкам."""
        return settings.hedging or self.trades.kv_get(HEDGE_KEY) == "1"

    async def _hedge_leg(self, signal: Signal) -> HedgeLeg | None:
        """Бета монеты к BTC по 4h-свечам (60 дней); None — данных нет, сделка пойдёт без хеджа."""
        if self.market is None:
            return None
        try:
            coin, btc = await asyncio.gather(self.market.closed_bars(signal.symbol, Timeframe.H4),
                                             self.market.closed_bars(HEDGE_SYMBOL, Timeframe.H4))
        except Exception:
            log.exception("бета %s", signal.symbol)
            return None
        b = beta(coin["close"], btc["close"])
        return HedgeLeg(b) if b is not None else None

    async def _rebalance_hedge(self, b: Broker, acc: Account) -> None:
        """Позиция BTCUSDT = сумма хеджей открытых сделок (против стороны сделки на бету × номинал позиции).
        Хедж сделки добавляется, когда её позиция появилась на бирже, и снимается, когда позиция закрылась."""
        marked = self.trades.kv_get(HEDGE_KEY) == "1"
        latest: dict[str, Trade] = {}
        for t in self.trades.filled_trades():                    # новые первыми: по монете — последняя сделка
            latest.setdefault(t.symbol, t)
        held = {p.symbol: p for p in acc.positions}
        open_ = [t for t in latest.values() if t.hedge is not None and t.symbol in held
                 and held[t.symbol].side is t.side]
        if not open_ and not marked:
            return
        px = await b.price(HEDGE_SYMBOL)
        cur = held_qty(acc)
        if not open_:
            if cur:
                await b.close_position(HEDGE_SYMBOL)
                await self._say(f"Хедж BTC снят: открытых сделок с хеджем нет (было {cur:+.6g} BTC)")
            self.trades.kv_set(HEDGE_KEY, "0")
            return
        for t in open_:
            if t.hedge.btc_in is None and t.id is not None:
                self.trades.set_trade_hedge(t.id, replace(t.hedge, btc_in=px))
        target = target_qty(((t.side, held[t.symbol].notional, t.hedge.beta) for t in open_), px)
        inst = await b.instrument(HEDGE_SYMBOL)
        if inst is None:
            return
        self.trades.kv_set(HEDGE_KEY, "1")
        delta = rebalance(target, cur, inst, px)
        if delta:
            try:
                await b.adjust(HEDGE_SYMBOL, delta, hedge_leverage(self.settings.load(), inst))
            except Exception as e:                               # обычно не хватает маржи — сказать один раз
                why = _short(e)
                if self.trades.kv_get(HEDGE_ERR_KEY) != why:
                    self.trades.kv_set(HEDGE_ERR_KEY, why)
                    await self._say(f"⚠️ Хедж BTC не подогнан ({delta:+.6g} BTC): {why}. Повторю через минуту.")
                raise
            self.trades.kv_set(HEDGE_ERR_KEY, "")
            side = "лонг" if cur + delta > 0 else "шорт"
            await self._say(f"Хедж BTC: {side} {abs(cur + delta):.6g} BTC (≈${abs(cur + delta) * px:,.0f}) "
                            f"под {len(open_)} сделок")

    async def _close_hedge_leg(self, b: Broker, t: Trade) -> None:
        """Сделка закрылась: цена BTC в её хедж — для оценки итога хеджа в журнале."""
        if t.hedge is None or t.hedge.btc_in is None or t.hedge.btc_out is not None or t.id is None:
            return
        try:
            self.trades.set_trade_hedge(t.id, replace(t.hedge, btc_out=await b.price(HEDGE_SYMBOL)))
        except Exception:
            log.exception("цена BTC для хеджа %s", t.symbol)

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
            try:
                await self._settle_one(b, t, held.get(t.symbol), later.get(t.id), now)
            except Exception:                                      # одна сделка не должна держать остальные
                log.exception("итог сделки %s", t.symbol)

    async def _settle_one(self, b: Broker, t: Trade, pos: Position | None, until: datetime | None,
                          now: datetime) -> None:
        if until is None and t.status is TradeStatus.FILLED and pos is not None and pos.side is t.side:
            return                                                 # ещё открыта
        res = await self._result(b, t, until or now, now)
        status = TradeStatus.TIMED_OUT if t.status is TradeStatus.TIMED_OUT else TradeStatus.CLOSED
        if res is not None:
            await self._settle(t, status, res)
            return
        gone = self._gone_since(t, now)
        if now - gone > GIVE_UP:
            self.trades.settle_trade(t.id, status, None)
            self._close_signal(t.signal_id)
            await self._close_hedge_leg(b, t)
        elif now - gone > NOTICE_AFTER and not self.trades.kv_get(NOTICE_KEY.format(t.id)):
            self.trades.kv_set(NOTICE_KEY.format(t.id), now.isoformat())
            await self._say(f"⚠️ {t.symbol}: позиции на бирже нет уже {int((now - gone).total_seconds() // 60)} мин, а записи "
                            f"о закрытии не нашёл ({await self._diagnose(b, t, until or now, now)}). "
                            f"Продолжу искать до {GIVE_UP.days} суток.")

    async def _diagnose(self, b: Broker, t: Trade, end: datetime, now: datetime) -> str:
        """Что отдаёт биржа по монете в окне поиска: для разбора, почему итог не находится."""
        since = (t.opened_at or now) - timedelta(minutes=1)
        start = max(since, end - PNL_WINDOW)
        parts = [f"окно {start:%d.%m %H:%M}–{end:%d.%m %H:%M}"]
        for name, call in (("по монете", lambda: b.closed_pnl(t.symbol, start, end)),
                           ("по всем", lambda: b.closed_pnl_all(start, end))):
            try:
                recs = await call()
            except Exception as e:
                parts.append(f"{name}: ошибка {_short(e)}")
                continue
            own = [r for r in recs if not r.symbol or r.symbol == t.symbol]
            last = max(own, key=lambda r: r.closed_at, default=None)
            parts.append(f"{name}: {len(own)} зап." + (f", последняя {last.side.label} {last.closed_at:%d.%m %H:%M}"
                                                       f" {last.pnl:+.2f}$" if last else ""))
        return "; ".join(parts)

    def _gone_since(self, t: Trade, now: datetime) -> datetime:
        """Когда бот впервые не увидел позицию: срок поиска итога считается от этого момента, а не от входа —
        сделка 4h живёт до 10 суток, и закрытие на 9-й нельзя бросать с первой же пустой выдачи биржи."""
        key = GONE_KEY.format(t.id)
        v = self.trades.kv_get(key)
        if v:
            return datetime.fromisoformat(v)
        self.trades.kv_set(key, now.isoformat())
        return now

    def _reopen_lost(self, now: datetime) -> None:
        """Сделки, которым старое правило (8 суток с входа) записало «нет данных», хотя позиция могла закрыться
        только что или ещё открыта: снова «на бирже» — итог ищется заново, хедж учитывает их как открытые."""
        for t in self.trades.lost_trades(now - LOST_LOOKBACK):
            self.trades.reopen_trade(t.id)
            if t.hedge is not None:
                self.trades.set_trade_hedge(t.id, replace(t.hedge, btc_out=None))
            sig = self.signals.get(t.signal_id)
            if sig is not None and sig.status is SignalStatus.CLOSED:
                self.signals.set_status(t.signal_id, SignalStatus.TAKEN, sig.note)
            log.info("сделка %s %s: итог снова ищем", t.id, t.symbol)
        self.trades.kv_set(LOST_REOPEN_KEY, now.isoformat())

    async def _result(self, b: Broker, t: Trade, until: datetime | None, now: datetime) -> TradeResult | None:
        since = (t.opened_at or now) - timedelta(minutes=1)
        end = until or now
        start = max(since, end - PNL_WINDOW)
        try:
            res = settle(await b.closed_pnl(t.symbol, start, end), t.side, since, end)
        except Exception:
            log.exception("итог сделки %s", t.symbol)
            res = None
        if res is not None:
            return res
        try:                                                       # запрос по монете пуст или упал — общая выдача по всем
            recs = [r for r in await b.closed_pnl_all(start, end) if r.symbol == t.symbol]
        except Exception:
            log.exception("итог сделки %s (общая выдача)", t.symbol)
            return None
        return settle(recs, t.side, since, end)

    async def _settle(self, t: Trade, status: TradeStatus, res: TradeResult) -> None:
        self.trades.settle_trade(t.id, status, res)
        self._close_signal(t.signal_id)
        b = self.broker()
        if b is not None:
            await self._close_hedge_leg(b, t)
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
            await self._say(f"{icon} {tf.value}: {st.closed} сделок, в среднем {st.avg_r:+.2f}R. {st.health.text}")

    async def _say(self, text: str) -> None:
        if self.notifier is not None:
            try:
                await self.notifier.text(text)
            except Exception:
                log.exception("уведомление")


_EXCHANGE_ERR = re.compile(r'"retCode"\s*:\s*(\d+)\s*,\s*"retMsg"\s*:\s*"([^"]*)"')


def _short(e: Exception) -> str:
    """Причина отказа одной строкой. Из ответа Bybit — только код и текст: в нём есть меняющееся поле time, и без
    этого одна и та же ошибка выглядела бы новой (уведомление каждую минуту)."""
    msg = str(e) or type(e).__name__
    if m := _EXCHANGE_ERR.search(msg):
        code, text = m.groups()
        hint = " — не хватает свободной маржи" if ("not enough" in text.lower() or "insufficient" in text.lower()) else ""
        msg = f"Bybit {code}: {text}{hint}"
    return msg if len(msg) <= 160 else msg[:157] + "…"
