"""Счёт ликвидаций за сутки для «Погоды рынка»: поток слушается, пока работает бот; корзины сохраняются в kv раз
в минуту, чтобы перезапуск не обнулял сутки."""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone

from ..domain.liquidations import LiqBook, LiqTotals, parse_force_order
from .ports import TradeRepository

log = logging.getLogger(__name__)
KEY = "liquidations"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class LiquidationWatcher:
    def __init__(self, repo: TradeRepository, clock: Callable[[], datetime] = _utcnow) -> None:
        self.repo, self.clock = repo, clock
        raw = repo.kv_get(KEY)
        try:
            self.book = LiqBook.from_json(json.loads(raw)) if raw else LiqBook()
        except (ValueError, TypeError):
            self.book = LiqBook()
        now = clock()
        self.book.trim(now)
        if self.book.started is None or not self.book.buckets:
            self.book.started = now                    # данных за окно нет — считаем заново с этого момента

    def on_message(self, msg: dict) -> None:
        ev = parse_force_order(msg)
        if ev is not None:
            self.book.add(*ev)

    def save(self) -> None:
        self.book.trim(self.clock())
        self.repo.kv_set(KEY, json.dumps(self.book.to_json()))

    def totals(self) -> LiqTotals | None:
        return self.book.totals(self.clock())
