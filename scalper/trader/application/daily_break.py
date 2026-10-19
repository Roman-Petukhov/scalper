"""Пробой линии на дневках, бумажный счёт: раз в сутки после закрытия дня UTC найти новые пробои и пересчитать
исход открытых сетапов по свежим свечам.

Ордеров нет — это проверка стратегии перед реальными деньгами. Сетап хранит только линию, стоп и ATR; где он сейчас
(ждёт ретеста, в позиции, закрыт), каждый раз заново считается по истории свечей (domain.daily_break.advance) — так
исход совпадает с бэктестом. Пропущенные дни (сервер был выключен) догоняются: пробои ищутся и на них, но не дальше
окна ретеста — более старый сигнал уже не исполнить. Размер сделки задаётся один раз, на дне входа, по капиталу и
свободной марже того дня (сделки обходятся по дням входа); хедж BTC — по закрытиям BTCUSDT."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone

import pandas as pd

from ..domain.daily_break import (HISTORY_DAYS, RULES, DailyBreakRule, DailySetup, Status, add_fits, advance,
                                  beta_before, equity_on, find, funding_r, hedge_pnl, margin_on, size)
from ..domain.hedge import HEDGE_SYMBOL
from .ports import DailyBreakRepository, MarketData, Notifier
from .trend import RETRY

log = logging.getLogger(__name__)
AFTER_CLOSE = timedelta(minutes=8)      # тренд на дневках качает те же ~200 дневок в 00:03 — лимит веса Binance
#                                         (2400 в минуту) на двоих не делим; скан 15m в 00:15 ещё впереди


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _key(rule: DailyBreakRule) -> str:
    return f"dbreak_done:{rule.key}"


def _midnight(d: date) -> datetime:
    return datetime.combine(d, time(), timezone.utc)


def _close_on(close: pd.Series | None, d: date | None) -> float | None:
    """Закрытие дня d (последнее известное до него включительно)."""
    if close is None or d is None:
        return None
    x = close[close.index < _midnight(d + timedelta(days=1))]
    return float(x.iloc[-1]) if len(x) else None


class DailyBreakPaper:
    def __init__(self, market: MarketData, repo: DailyBreakRepository, notifier: Notifier | None = None,
                 rules: tuple[DailyBreakRule, ...] = RULES, concurrency: int = 4,
                 clock: Callable[[], datetime] = _utcnow) -> None:
        self.market, self.repo, self.notifier, self.rules = market, repo, notifier, rules
        self.sem = asyncio.Semaphore(concurrency)
        self.clock = clock
        self._tried: datetime | None = None
        self.last_error: str | None = None

    def _done(self, rule: DailyBreakRule) -> date | None:
        raw = self.repo.kv_get(_key(rule))
        return date.fromisoformat(raw) if raw else None

    def due_day(self, now: datetime) -> date | None:
        """День, закрытие которого пора обработать; None — всё обработано или день ещё не дописан."""
        if now - _midnight(now.date()) < AFTER_CLOSE:
            return None
        day = now.date() - timedelta(days=1)
        if all((x := self._done(r)) is not None and x >= day for r in self.rules):
            return None
        return day

    async def maybe_run(self) -> list[DailySetup]:
        """Для фоновой проверки раз в минуту; после сбоя — не чаще раза в 15 минут. Возвращает изменившиеся сетапы."""
        now = self.clock()
        if self.due_day(now) is None or (self._tried is not None and now - self._tried < RETRY):
            return []
        self._tried = now
        try:
            out = await self.run(now)
            self.last_error = None
            return out
        except Exception as e:
            self.last_error = str(e)[:200]
            log.exception("пробой на дневках: шаг бумажного счёта")
            return []

    async def _bars(self, symbol: str) -> pd.DataFrame | None:
        async with self.sem:
            try:
                d = await self.market.daily_bars(symbol, HISTORY_DAYS)
            except Exception as e:
                log.warning("пробой на дневках: свечи %s — %s", symbol, e)
                return None
        if d is None or not len(d):
            return None
        d = d.sort_index()
        d.index = d.index.tz_convert("UTC") if d.index.tz is not None else d.index.tz_localize("UTC")
        return d

    async def _funding(self, symbol: str, start: datetime, end: datetime) -> float:
        async with self.sem:
            try:
                return await self.market.funding_sum(symbol, start, end)
            except Exception as e:
                log.warning("пробой на дневках: funding %s — %s", symbol, e)
                return 0.0

    async def _with_funding(self, d: pd.DataFrame, s: DailySetup, rule: DailyBreakRule, now: datetime) -> DailySetup:
        new = advance(d, s, rule)
        if new.fill_day is None:
            return new
        end = _midnight(new.exit_day + timedelta(days=1)) if new.exit_day is not None else _midnight(now.date())
        base = await self._funding(s.symbol, _midnight(new.fill_day), end)
        extra = await self._funding(s.symbol, _midnight(new.add_day), end) \
            if new.add_day is not None and new.add_ok is not False else 0.0
        fr = funding_r(new, base, extra, rule.add_size)
        return advance(d, replace(s, funding_r=fr), rule)

    @staticmethod
    def _account(s: DailySetup, book: dict[int | None, DailySetup], bars: dict[str, pd.DataFrame],
                 btc: pd.Series | None, rule: DailyBreakRule) -> DailySetup:
        """Деньги сделки: размер на дне входа (один раз), хедж — по закрытию BTC дня входа и выхода (или последнему)."""
        if s.fill_day is None or s.status is Status.SKIPPED:
            return s
        if not s.sized:
            others = [x for k, x in book.items() if k != s.id]
            b = None
            if rule.hedge and btc is not None and s.symbol != HEDGE_SYMBOL:
                b = beta_before(bars[s.symbol]["close"], btc, s.fill_day, rule)
            s = size(s, equity_on(others, s.fill_day), margin_on(others, s.fill_day, rule.leverage), rule, b)
            if s.status is Status.SKIPPED:
                return s
            s = replace(s, btc_in=_close_on(btc, s.fill_day) if s.beta else None)
        if not s.beta or btc is None:
            return s
        if s.status is Status.CLOSED:
            out = _close_on(btc, s.exit_day)
            return replace(s, btc_out=out, hedge_pnl=hedge_pnl(s, out, rule.add_size))
        return replace(s, hedge_pnl=hedge_pnl(s, float(btc.iloc[-1]), rule.add_size))

    async def run(self, now: datetime) -> list[DailySetup]:
        day = now.date() - timedelta(days=1)
        active = {r.key: self.repo.dbreak_active(r.key) for r in self.rules}
        held = {s.symbol for xs in active.values() for s in xs}
        floor = min(r.min_adv for r in self.rules) / 2     # список — по обороту за сутки, правило — за 30 дней
        symbols = list(dict.fromkeys([*await self.market.universe(floor), *sorted(held), HEDGE_SYMBOL]))
        frames = await asyncio.gather(*(self._bars(s) for s in symbols))
        bars = {s: f for s, f in zip(symbols, frames) if f is not None}
        if not bars:
            raise RuntimeError("нет дневных свечей ни по одной монете")
        latest = max(f.index[-1].date() for f in bars.values())
        if latest != day:
            raise RuntimeError(f"последняя дневная свеча {latest}, ждём {day}")
        out: list[DailySetup] = []
        for rule in self.rules:
            done = self._done(rule)
            if done is not None and done >= day:
                continue
            events: list[tuple[str, DailySetup]] = []
            first = day if done is None else max(done + timedelta(days=1), day - timedelta(days=rule.retest_bars))
            scan = [first + timedelta(days=k) for k in range((day - first).days + 1)]
            for sym, d in bars.items():
                for x in scan:
                    part = d[d.index < _midnight(x + timedelta(days=1))]
                    if not len(part) or part.index[-1].date() != x:
                        continue
                    for s in await asyncio.to_thread(find, part, sym, rule):
                        saved = self.repo.dbreak_add(s)
                        if saved is not None:
                            active[rule.key].append(saved)
                            events.append(("new", saved))
            moved: list[tuple[DailySetup, DailySetup]] = []
            for s in active[rule.key]:
                d = bars.get(s.symbol)
                if d is not None:
                    moved.append((s, await self._with_funding(d, s, rule, now)))
            btc = bars[HEDGE_SYMBOL]["close"] if HEDGE_SYMBOL in bars else None
            book = {x.id: x for x in self.repo.dbreak_recent(rule.key, 100_000)}
            done_: dict[int | None, DailySetup] = {}
            for s, new in sorted(moved, key=lambda p: (p[1].fill_day or date.max, p[1].day)):
                done_[new.id] = book[new.id] = self._account(new, book, bars, btc, rule)
            for s, _ in sorted(moved, key=lambda p: (done_[p[1].id].add_day or date.max, p[1].day)):
                x = done_[s.id]
                if x.add_day is None or x.add_ok is not None or not x.sized or x.status is Status.SKIPPED:
                    continue
                others = [y for k, y in book.items() if k != x.id]
                used = margin_on(others, x.add_day, rule.leverage, rule.add_size) + x.margin(rule.leverage)
                ok = add_fits(x, equity_on(others, x.add_day), used, rule)
                x = replace(x, add_ok=ok, btc_add=_close_on(btc, x.add_day) if ok and x.beta else None)
                if not ok:                                         # без доливки — пересчитать исход и funding
                    x = await self._with_funding(bars[x.symbol], replace(x, status=Status.OPEN), rule, now)
                done_[x.id] = book[x.id] = self._account(x, book, bars, btc, rule)
            for s, _ in moved:
                new = done_[s.id]
                if new == s:
                    continue
                self.repo.dbreak_update(new)
                out.append(new)
                if new.status is Status.SKIPPED:
                    events.append(("skip", new))
                    continue
                if new.fill_day is not None and s.fill_day is None:
                    events.append(("fill", new))
                if new.add_ok is not None and s.add_ok is None:
                    events.append(("add", new))
                if new.status is Status.CLOSED:
                    events.append(("close", new))
                elif new.status is Status.CANCELLED:
                    events.append(("cancel", new))
            self.repo.kv_set(_key(rule), day.isoformat())
            await self._notify(rule, events)
        return out

    async def _notify(self, rule: DailyBreakRule, events: list[tuple[str, DailySetup]]) -> None:
        if self.notifier is None or not any(k != "cancel" for k, _ in events):
            return
        side = {1: "лонг", -1: "шорт"}
        parts = []
        new = [s for k, s in events if k == "new"]
        if new:
            parts.append("пробой, лимитка на линии: " + ", ".join(f"{s.symbol} {side[s.side]}" for s in new))
        fills = [s for k, s in events if k == "fill"]
        if fills:
            parts.append("ретест, вошёл: " + ", ".join(f"{s.symbol} {side[s.side]}" for s in fills))
        added = [s for k, s in events if k == "add"]
        if added:
            parts.append("долив: " + ", ".join(s.symbol + ("" if s.add_ok else " — пропущен, нет свободной маржи")
                                              for s in added))
        closed = [s for k, s in events if k == "close"]
        if closed:
            parts.append("закрыл: " + ", ".join(f"{s.symbol} {s.r:+.2f}R ({(s.pnl or 0) * 100:+.1f}% счёта)"
                                                for s in closed))
        skipped = [s for k, s in events if k == "skip"]
        if skipped:
            parts.append("пропустил, нет свободной маржи: " + ", ".join(s.symbol for s in skipped))
        try:
            await self.notifier.text(f"Дневки, бумага · {rule.title}: " + "; ".join(parts) + ".")
        except Exception:
            log.exception("пробой на дневках: уведомление")
