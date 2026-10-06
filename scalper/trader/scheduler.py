"""Запуск сканера сразу после закрытия свечи каждого таймфрейма (UTC-сетка Binance).

Цикл сверяется с настенными часами короткими шагами, а не спит до закрытия одним таймером: после сна компьютера
таймер может сработать с опозданием или не на ту свечу. Проснувшись, цикл видит пропущенные закрытия и сразу
сканирует эти таймфреймы вместе с пропущенными свечами (до CATCHUP_BARS назад): сигнал с них ещё рабочий, пока не
вышло время на вход. Последняя проверенная свеча каждого ТФ хранится в базе — после перезапуска сервера пропущенное
тоже догоняется."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import Protocol

from .application.services import CATCHUP_BARS, Scanner
from .domain.models import Timeframe

log = logging.getLogger(__name__)


class ScanState(Protocol):
    """Где хранить последнюю проверенную свечу каждого ТФ (kv базы)."""
    def kv_get(self, key: str) -> str | None: ...

    def kv_set(self, key: str, value: str) -> None: ...


def _key(tf: Timeframe) -> str:
    return f"scan_done:{tf.value}"


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
              clock: Callable[[], datetime] = _utcnow, tick: Callable[[], Awaitable[None]] | None = None,
              tick_s: float = 60.0, state: ScanState | None = None) -> None:
    """Бесконечный цикл: как только прошло delay_s после закрытия свечи, сканируем этот ТФ (старшие первыми).
    Выключенные кнопками ТФ сканер пропускает сам, поэтому переключение действует со следующей свечи.
    tick — фоновая работа раз в tick_s секунд (снятие просроченных лимиток на бирже).
    state — последняя проверенная свеча по ТФ: при старте пропущенные с тех пор свечи сканируются сразу."""
    lag = timedelta(seconds=delay_s)
    start = clock() - lag
    done = {tf: last_close(start, tf) for tf in Timeframe}           # без сохранённого состояния — с текущей свечи
    for tf in Timeframe:
        raw = state.kv_get(_key(tf)) if state is not None else None
        saved = datetime.fromisoformat(raw) if raw else None
        if saved is not None and saved < done[tf]:
            done[tf] = saved
    last_tick: datetime | None = None
    while not stop.is_set():
        now = clock()
        if tick is not None and (last_tick is None or (now - last_tick).total_seconds() >= tick_s):
            last_tick = now
            try:
                await tick()
            except Exception:
                log.exception("фоновая проверка ордеров")
        ref = now - lag
        due = [tf for tf in Timeframe if last_close(ref, tf) > done[tf]]
        for tf in sorted(due, key=lambda x: -x.minutes):
            missed = (last_close(ref, tf) - done[tf]) // timedelta(minutes=tf.minutes) - 1
            back = min(missed, CATCHUP_BARS)
            if missed > 0:
                log.warning("%s: пропущено закрытий %d — проверяю и %d прошлых свечей", tf.value, missed, back)
            done[tf] = last_close(ref, tf)
            try:
                await (scanner.scan(tf, back=back) if back > 0 else scanner.scan(tf))
            except Exception:
                log.exception("скан %s", tf.value)
                continue
            if state is not None:
                state.kv_set(_key(tf), done[tf].isoformat())
        until = min(next_close(ref, tf) for tf in Timeframe) - ref
        try:
            await asyncio.wait_for(stop.wait(), timeout=max(min(poll_s, until.total_seconds()), 0.01))
            return
        except asyncio.TimeoutError:
            pass
