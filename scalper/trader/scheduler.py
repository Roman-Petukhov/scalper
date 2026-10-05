"""Запуск сканера сразу после закрытия свечи каждого таймфрейма (UTC-сетка Binance).

Цикл сверяется с настенными часами короткими шагами, а не спит до закрытия одним таймером: после сна компьютера
таймер может сработать с опозданием или не на ту свечу. Проснувшись, цикл видит пропущенные закрытия и сразу
сканирует эти таймфреймы — последняя закрытая свеча ещё та самая, если сон был короче одной свечи."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from .application.services import Scanner
from .domain.models import Timeframe

log = logging.getLogger(__name__)


def _day(now: datetime) -> datetime:
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def last_close(now: datetime, tf: Timeframe) -> datetime:
    """Последнее закрытие свечи tf не позже now (свечи выровнены по полуночи UTC)."""
    step = timedelta(minutes=tf.minutes)
    return _day(now) + ((now - _day(now)) // step) * step


def next_close(now: datetime, tf: Timeframe) -> datetime:
    """Ближайшее закрытие свечи tf строго после now."""
    return last_close(now, tf) + timedelta(minutes=tf.minutes)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def run(scanner: Scanner, delay_s: float, stop: asyncio.Event, poll_s: float = 20.0,
              clock: Callable[[], datetime] = _utcnow) -> None:
    """Бесконечный цикл: как только прошло delay_s после закрытия свечи, сканируем этот ТФ (старшие первыми).
    Выключенные кнопками ТФ сканер пропускает сам, поэтому переключение действует со следующей свечи."""
    lag = timedelta(seconds=delay_s)
    start = clock() - lag
    done = {tf: last_close(start, tf) for tf in Timeframe}           # при старте уже закрытые свечи не трогаем
    while not stop.is_set():
        now = clock()
        ref = now - lag
        due = [tf for tf in Timeframe if last_close(ref, tf) > done[tf]]
        for tf in sorted(due, key=lambda x: -x.minutes):
            missed = (last_close(ref, tf) - done[tf]) // timedelta(minutes=tf.minutes) - 1
            if missed > 0:
                log.warning("%s: пропущено закрытий %d (компьютер спал?) — сканирую последнее", tf.value, missed)
            done[tf] = last_close(ref, tf)
            try:
                await scanner.scan(tf)
            except Exception:
                log.exception("скан %s", tf.value)
        until = min(next_close(ref, tf) for tf in Timeframe) - ref
        try:
            await asyncio.wait_for(stop.wait(), timeout=max(min(poll_s, until.total_seconds()), 0.01))
            return
        except asyncio.TimeoutError:
            pass
