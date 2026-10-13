"""Тренд на дневках, бумажный счёт: раз в сутки после закрытия дня UTC пересчитать портфель и оценить счёт.

Ордера не отправляются — это проверка стратегии перед реальными деньгами. Капитал каждой стратегии начинается с 1.0
(100%), в панели — доходность в процентах. Пропущенные дни (сервер был выключен) догоняются одним шагом: позиции
переоцениваются по свежим закрытиям, funding берётся за весь перерыв."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import date, datetime, time, timedelta, timezone

import pandas as pd

from ..domain.trend import HISTORY_DAYS, RULES, TrendDay, TrendRule, step, targets
from .ports import MarketData, Notifier, TrendRepository

log = logging.getLogger(__name__)
AFTER_CLOSE = timedelta(minutes=3)      # Binance дописывает дневную свечу и funding 00:00, а сканы 15m / 4h в 00:00
#                                         уже отработали — лимит запросов Binance не делим с ними
RETRY = timedelta(minutes=15)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TrendPaper:
    def __init__(self, market: MarketData, repo: TrendRepository, notifier: Notifier | None = None,
                 rules: tuple[TrendRule, ...] = RULES, concurrency: int = 4,
                 clock: Callable[[], datetime] = _utcnow) -> None:
        self.market, self.repo, self.notifier, self.rules = market, repo, notifier, rules
        self.sem = asyncio.Semaphore(concurrency)
        self.clock = clock
        self._tried: datetime | None = None
        self.last_error: str | None = None

    def due_day(self, now: datetime) -> date | None:
        """День, закрытие которого пора обработать; None — всё обработано или день ещё не дописан."""
        if now - datetime.combine(now.date(), time(), timezone.utc) < AFTER_CLOSE:
            return None
        day = now.date() - timedelta(days=1)
        if all((x := self.repo.trend_last(r.key)) is not None and x.day >= day for r in self.rules):
            return None
        return day

    async def maybe_run(self) -> list[TrendDay]:
        """Для фоновой проверки раз в минуту: считает, только когда закрылся новый день; после сбоя — не чаще
        раза в 15 минут."""
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
            log.exception("тренд: шаг бумажного счёта")
            return []

    async def _bars(self, symbol: str) -> pd.DataFrame | None:
        async with self.sem:
            try:
                return await self.market.daily_bars(symbol, HISTORY_DAYS)
            except Exception as e:
                log.warning("тренд: свечи %s — %s", symbol, e)
                return None

    async def _funding(self, symbol: str, start: datetime, end: datetime) -> float:
        async with self.sem:
            try:
                return await self.market.funding_sum(symbol, start, end)
            except Exception as e:
                log.warning("тренд: funding %s — %s", symbol, e)
                return 0.0

    async def run(self, now: datetime) -> list[TrendDay]:
        day = now.date() - timedelta(days=1)
        last = {r.key: self.repo.trend_last(r.key) for r in self.rules}
        held = {s for x in last.values() if x is not None for s in x.positions}
        floor = min(r.min_adv for r in self.rules)
        symbols = list(dict.fromkeys([*await self.market.universe(floor), *sorted(held)]))
        frames = await asyncio.gather(*(self._bars(s) for s in symbols))
        bars = {s: f for s, f in zip(symbols, frames) if f is not None and len(f)}
        if not bars:
            raise RuntimeError("нет дневных свечей ни по одной монете")
        close = pd.DataFrame({s: f["close"] for s, f in bars.items()}).sort_index()
        qv = pd.DataFrame({s: f["quote_volume"] for s, f in bars.items()}).reindex(close.index)
        close.index = close.index.tz_convert("UTC") if close.index.tz is not None else close.index.tz_localize("UTC")
        qv.index = close.index
        if close.index[-1].date() != day:
            raise RuntimeError(f"последняя дневная свеча {close.index[-1].date()}, ждём {day}")
        prices = {s: float(v) for s, v in close.iloc[-1].items() if pd.notna(v) and v > 0}
        close_at = datetime.combine(now.date(), time(), timezone.utc)
        out = []
        for rule in self.rules:
            prev = last[rule.key]
            if prev is not None and prev.day >= day:
                continue
            tgt = await asyncio.to_thread(targets, close, qv, rule)       # секунда-две счёта — не в цикле событий
            fund: dict[str, float] = {}
            if prev is not None and prev.positions:
                since = datetime.combine(prev.day + timedelta(days=1), time(), timezone.utc)
                names = list(prev.positions)
                vals = await asyncio.gather(*(self._funding(s, since, close_at) for s in names))
                fund = dict(zip(names, vals))
            new = step(prev, rule.key, day, prices, fund, tgt)
            self.repo.trend_save(new)
            out.append(new)
            await self._notify(rule, prev, new)
        return out

    async def _notify(self, rule: TrendRule, prev: TrendDay | None, new: TrendDay) -> None:
        if self.notifier is None:
            return
        old = set(prev.positions) if prev is not None else set()
        cur = set(new.positions)
        more = [s for s in cur & old if new.positions[s].adds > prev.positions[s].adds] if prev is not None else []
        if old == cur and not more:
            return
        parts = [f"Тренд, бумага · {rule.title}: капитал {(new.equity - 1) * 100:+.1f}% с начала"]
        if cur - old:
            parts.append("вошёл: " + ", ".join(sorted(cur - old)))
        if old - cur:
            parts.append("вышел: " + ", ".join(sorted(old - cur)))
        if more:
            parts.append("долил: " + ", ".join(sorted(more)))
        try:
            await self.notifier.text("; ".join(parts) + ".")
        except Exception:
            log.exception("тренд: уведомление")
