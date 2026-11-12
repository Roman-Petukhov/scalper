"""Цена к активным адресам (domain/onchain_value.py) в фоне: раз в сутки — данные CoinMetrics и снимок в kv."""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from ..domain.onchain_value import OnchainSnapshot, rank
from .ports import KeyValueStore, MarketData, OnchainMetrics

log = logging.getLogger(__name__)
KEY = "onchainvalue"
EVERY = timedelta(hours=24)         # метрики сети — дневные
RETRY = timedelta(hours=1)
WINDOW_DAYS = 30
ALIAS = {"avaxc": "AVAX", "matic_eth": "MATIC", "pol_eth": "POL"}
SKIP = frozenset({"USDT", "USDC", "DAI", "BUSD", "TUSD", "PAX", "USDP", "GUSD", "HUSD", "PYUSD", "FDUSD", "WBTC",
                  "WETH", "STETH", "PAXG", "XAUT"})          # стейблкоины, обёртки и золото — не сети


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class OnchainValueService:
    def __init__(self, source: OnchainMetrics, store: KeyValueStore, market: MarketData | None = None,
                 clock: Callable[[], datetime] = _utcnow, every: timedelta = EVERY) -> None:
        self.source, self.store, self.market, self.clock, self.every = source, store, market, clock, every
        self._task: asyncio.Task | None = None
        self._tried: datetime | None = None
        self.last_error: str | None = None

    def current(self) -> OnchainSnapshot | None:
        raw = self.store.kv_get(KEY)
        if not raw:
            return None
        try:
            return OnchainSnapshot.from_json(json.loads(raw))
        except (ValueError, KeyError, TypeError):
            log.exception("цена к адресам: снимок не читается")
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
            snap = await self.run()
            self.store.kv_set(KEY, json.dumps(snap.to_json()))
            self.last_error = None
        except Exception as e:
            self.last_error = str(e)[:200]
            log.exception("цена к адресам")

    async def run(self) -> OnchainSnapshot:
        readings = await self.source.readings(WINDOW_DAYS)
        perps: set[str] = set()
        if self.market is not None:
            try:
                perps = set(await self.market.universe(0.0))
            except Exception as e:
                log.warning("цена к адресам: список перпетуалов — %s", e)
        rows = rank(readings, perps, ALIAS, SKIP)
        if not rows:
            raise RuntimeError(f"монет с адресами и капитализацией — {len(readings)}, мало для сравнения")
        return OnchainSnapshot(self.clock(), tuple(rows), assets=len(readings))
