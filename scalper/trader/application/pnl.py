"""Реализованный PnL счёта за неделю, 30 дней и полгода — по закрытиям позиций на бирже (все монеты, с комиссиями).

Биржа отдаёт закрытия окнами не длиннее 7 дней, поэтому прошедшие дни считаются один раз и хранятся в базе
(по сети: демо и реальный счёт отдельно), а сегодняшний день пересчитывается раз в минуту. Полгода догружаются
постепенно, несколько окон за проход, чтобы не упираться в лимиты биржи."""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Protocol

from .ports import Broker

log = logging.getLogger(__name__)

PERIODS: tuple[tuple[str, int], ...] = (("Неделя", 7), ("30 дней", 30), ("Полгода", 182))
HISTORY_DAYS = max(d for _, d in PERIODS)
WINDOW_DAYS = 7
WINDOWS_PER_PASS = 4


class KeyValue(Protocol):
    def kv_get(self, key: str) -> str | None: ...

    def kv_set(self, key: str, value: str) -> None: ...


@dataclass(frozen=True)
class PnlPeriod:
    label: str
    days: int
    usd: float | None           # None — история ещё догружается
    pct: float | None           # к капиталу на начало периода (капитал сейчас минус PnL периода)


def _midnight(d: date) -> datetime:
    return datetime.combine(d, time(0), tzinfo=timezone.utc)


class PnlHistory:
    def __init__(self, kv: KeyValue, clock: Callable[[], datetime]) -> None:
        self.kv, self.clock = kv, clock

    @staticmethod
    def _day_key(network: str, d: date) -> str:
        return f"pnl_day:{network}:{d.isoformat()}"

    async def refresh(self, b: Broker) -> None:
        """Сегодняшний PnL и до WINDOWS_PER_PASS недостающих недель истории."""
        now = self.clock()
        today = now.date()
        recs = await b.closed_pnl_all(_midnight(today), now)
        self.kv.kv_set(f"pnl_today:{b.network}", json.dumps({"day": today.isoformat(), "usd": sum(r.pnl for r in recs)}))
        oldest = today - timedelta(days=HISTORY_DAYS - 1)
        for _ in range(WINDOWS_PER_PASS):
            missing = [today - timedelta(days=k) for k in range(1, HISTORY_DAYS)
                       if self.kv.kv_get(self._day_key(b.network, today - timedelta(days=k))) is None]
            if not missing:
                return
            end = missing[0]
            start = max(end - timedelta(days=WINDOW_DAYS - 1), oldest)
            recs = await b.closed_pnl_all(_midnight(start), _midnight(end + timedelta(days=1)))
            sums: dict[date, float] = {}
            for r in recs:
                sums[r.closed_at.date()] = sums.get(r.closed_at.date(), 0.0) + r.pnl
            d = start
            while d <= end:
                self.kv.kv_set(self._day_key(b.network, d), repr(sums.get(d, 0.0)))
                d += timedelta(days=1)

    def periods(self, network: str, equity: float) -> list[PnlPeriod]:
        today = self.clock().date()
        raw = self.kv.kv_get(f"pnl_today:{network}")
        cur = json.loads(raw) if raw else None
        today_usd = cur["usd"] if cur and cur["day"] == today.isoformat() else None
        out = []
        for label, days in PERIODS:
            vals = [self.kv.kv_get(self._day_key(network, today - timedelta(days=k))) for k in range(1, days)]
            if today_usd is None or any(v is None for v in vals):
                out.append(PnlPeriod(label, days, None, None))
                continue
            usd = today_usd + sum(float(v) for v in vals if v is not None)
            base = equity - usd
            out.append(PnlPeriod(label, days, usd, usd / base * 100 if base > 0 else None))
        return out
