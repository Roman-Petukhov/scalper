"""Куда заходят деньги по секторам (domain/sector_flow.py) в фоне: дневки ликвидных перпетуалов Binance раз в EVERY,
снимок в kv, уведомление при смене сектора, куда заходят деньги."""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

import pandas as pd

from ..domain.sector_flow import LONG, SectorFlow, compute, flow_changed
from .ports import KeyValueStore, MarketData, Notifier

log = logging.getLogger(__name__)
KEY = "sectorflow"
EVERY = timedelta(hours=6)
RETRY = timedelta(minutes=30)
UNIVERSE_MIN = 2e6                  # оборот за 24 ч: ниже — монета не влияет ни на сектор, ни на рынок
PARALLEL = 6


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SectorFlowService:
    def __init__(self, market: MarketData, store: KeyValueStore, notifier: Notifier | None = None,
                 clock: Callable[[], datetime] = _utcnow, every: timedelta = EVERY) -> None:
        self.market, self.store, self.notifier, self.clock, self.every = market, store, notifier, clock, every
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
            prev = self.current()
            flow = await self.run()
            self.store.kv_set(KEY, json.dumps(flow.to_json()))
            self.last_error = None
            text = flow_changed(prev, flow)
            if text and self.notifier is not None:
                try:
                    await self.notifier.text(text)
                except Exception:
                    log.exception("сектора: уведомление")
        except Exception as e:
            self.last_error = str(e)[:200]
            log.exception("сектора")

    async def run(self) -> SectorFlow:
        syms = await self.market.universe(UNIVERSE_MIN)
        sem = asyncio.Semaphore(PARALLEL)

        async def one(s: str) -> tuple[str, pd.DataFrame | None]:
            async with sem:
                try:
                    return s, await self.market.daily_bars(s, LONG + 10)
                except Exception as e:
                    log.warning("сектора: %s — %s", s, e)
                    return s, None

        got = await asyncio.gather(*(one(s) for s in syms))
        bars = {s: b[["close", "quote_volume"]] for s, b in got if b is not None and len(b)}
        if len(bars) < 20:
            raise RuntimeError(f"дневки загрузились у {len(bars)} монет")
        flow = compute(bars, self.clock())
        if not flow.rows:
            raise RuntimeError("ни в одном секторе нет трёх ликвидных монет")
        return flow
