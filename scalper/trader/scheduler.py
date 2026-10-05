"""Запуск сканера сразу после закрытия свечи каждого таймфрейма (UTC-сетка Binance)."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from .application.services import Scanner
from .domain.models import Timeframe

log = logging.getLogger(__name__)


def next_close(now: datetime, tf: Timeframe) -> datetime:
    """Ближайшее закрытие свечи tf строго после now (свечи выровнены по полуночи UTC)."""
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    step = timedelta(minutes=tf.minutes)
    k = (now - day) // step + 1
    return day + k * step


async def run(scanner: Scanner, delay_s: float, stop: asyncio.Event) -> None:
    """Бесконечный цикл: ждём ближайшее закрытие любой свечи, сканируем закрывшиеся ТФ. Выключенные кнопками ТФ
    сканер пропускает сам, поэтому переключение действует со следующей свечи без перезапуска."""
    while not stop.is_set():
        now = datetime.now(timezone.utc)
        closes = {tf: next_close(now, tf) for tf in Timeframe}
        t = min(closes.values())
        due = [tf for tf, c in closes.items() if c == t]
        wait = (t - now).total_seconds() + delay_s
        try:
            await asyncio.wait_for(stop.wait(), timeout=max(wait, 0.0))
            return
        except asyncio.TimeoutError:
            pass
        for tf in sorted(due, key=lambda x: -x.minutes):
            try:
                await scanner.scan(tf)
            except Exception:
                log.exception("скан %s", tf.value)
