"""Общий интерфейс брокера: команды уходят через submit, ответы приходят через sink."""
from __future__ import annotations

import time
from typing import Callable

from core.models import OrderCmd, OrderUpdate, PositionSync

Sink = Callable[[OrderUpdate | PositionSync], None]


class Broker:
    def __init__(self):
        self._sink: Sink = lambda msg: None
        # часы в домене биржевого времени; live-раннер подменяет на синхронизированные
        self.clock: Callable[[], float] = time.time

    def set_sink(self, sink: Sink) -> None:
        self._sink = sink

    def emit(self, msg: OrderUpdate | PositionSync) -> None:
        self._sink(msg)

    def submit(self, cmd: OrderCmd) -> None:  # pragma: no cover - интерфейс
        raise NotImplementedError
