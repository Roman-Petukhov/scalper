"""Сектора альтов (domain/sector_flow.py) в фоне: дневки ликвидных перпетуалов Binance и открытый интерес монет секторов
раз в EVERY, режим BTC, снимок в kv. Бумажная проверка (domain/sector_paper.py): по понедельникам — выбор недели,
через неделю — итог; уведомление раз в неделю."""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

import pandas as pd

from ..domain.sector_flow import BARS_FOR_HISTORY, SectorFlow, btc_regime, compute, history, sector_of
from ..domain.sector_paper import PaperWeek, merge, picks, settle, week_text
from .ports import KeyValueStore, MarketData, Notifier, OpenInterestHistory

log = logging.getLogger(__name__)
KEY = "sectorflow"
PAPER_KEY = "sectorpaper"
HIST_KEY = "sectorhist"
OI_DAYS = 30                        # Binance отдаёт историю открытого интереса только за 30 дней
EVERY = timedelta(hours=6)
RETRY = timedelta(minutes=30)
UNIVERSE_MIN = 2e6                  # оборот за 24 ч: ниже — монета не влияет ни на сектор, ни на рынок
PARALLEL = 6
BTC_DAYS = 1000                     # недельная EMA50 — нужно >= 30 закрытых недель, берём с запасом


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SectorFlowService:
    def __init__(self, market: MarketData, store: KeyValueStore, notifier: Notifier | None = None,
                 clock: Callable[[], datetime] = _utcnow, every: timedelta = EVERY,
                 oi: OpenInterestHistory | None = None) -> None:
        self.market, self.store, self.notifier, self.clock, self.every = market, store, notifier, clock, every
        self.oi = oi
        self._task: asyncio.Task | None = None
        self._tried: datetime | None = None
        self.last_error: str | None = None

    def current(self) -> SectorFlow | None:
        raw = self.store.kv_get(KEY)
        if not raw:
            return None
        try:
            return SectorFlow.from_json(json.loads(raw))
        except (ValueError, KeyError, TypeError):
            log.exception("сектора: снимок не читается")
            return None

    def history(self) -> dict | None:
        """Ряды для графиков секторов (domain.sector_flow.history) из последнего расчёта."""
        raw = self.store.kv_get(HIST_KEY)
        try:
            return json.loads(raw) if raw else None
        except ValueError:
            log.exception("сектора: история не читается")
            return None

    def paper(self) -> list[PaperWeek]:
        raw = self.store.kv_get(PAPER_KEY)
        if not raw:
            return []
        try:
            return [PaperWeek.from_json(d) for d in json.loads(raw)]
        except (ValueError, KeyError, TypeError):
            log.exception("сектора: журнал бумажной проверки не читается")
            return []

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def due(self, now: datetime) -> bool:
        if self.running or (self._tried is not None and now - self._tried < RETRY):
            return False
        s = self.current()
        return s is None or now - s.computed_at >= self.every

    async def maybe_run(self) -> None:
        now = self.clock()
        if self.due(now):
            self._tried = now
            self._task = asyncio.create_task(self._guarded())

    async def _guarded(self) -> None:
        try:
            flow, bars, hist = await self.run()
            self.store.kv_set(KEY, json.dumps(flow.to_json()))
            self.store.kv_set(HIST_KEY, json.dumps(hist))
            self.last_error = None
            weeks, closed = settle(self.paper(), bars)
            weeks, added = merge(weeks, picks(flow))
            if added or closed:
                self.store.kv_set(PAPER_KEY, json.dumps([w.to_json() for w in weeks]))
            text = week_text(added, closed, flow)
            if text and self.notifier is not None:
                try:
                    await self.notifier.text(text)
                except Exception:
                    log.exception("сектора: уведомление")
        except Exception as e:
            self.last_error = str(e)[:200]
            log.exception("сектора")

    async def _gather(self, coro: Callable, keys: list[str]) -> dict:
        sem = asyncio.Semaphore(PARALLEL)

        async def one(s: str):
            async with sem:
                try:
                    return s, await coro(s)
                except Exception as e:
                    log.warning("сектора: %s — %s", s, e)
                    return s, None

        return {s: v for s, v in await asyncio.gather(*(one(s) for s in keys)) if v is not None and len(v)}

    async def run(self) -> tuple[SectorFlow, dict[str, pd.DataFrame], dict]:
        syms = await self.market.universe(UNIVERSE_MIN)
        bars = await self._gather(lambda s: self.market.daily_bars(s, BARS_FOR_HISTORY), syms)
        if len(bars) < 20:
            raise RuntimeError(f"дневки загрузились у {len(bars)} монет")
        oi: dict[str, pd.Series] = {}
        if self.oi is not None:
            oi = await self._gather(lambda s: self.oi.oi_daily(s, OI_DAYS),          # type: ignore[union-attr]
                                    [s for s in bars if sector_of(s) is not None and s != "BTCUSDT"])
        bull, ratio, btc_close = None, None, None
        try:
            btc_close = (await self.market.daily_bars("BTCUSDT", BTC_DAYS))["close"]
            bull, ratio = btc_regime(btc_close)
        except Exception as e:
            log.warning("сектора: режим BTC — %s", e)
        flow = compute(bars, self.clock(), oi=oi, bull=bull, btc_ema=ratio)
        if not flow.rows:
            raise RuntimeError("ни в одном секторе нет трёх ликвидных монет")
        return flow, bars, history(bars, oi=oi, btc_close=btc_close)
