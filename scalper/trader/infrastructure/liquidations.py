"""Живой поток ликвидаций Binance USDⓈ-M по всем монетам (без ключей), с переподключением."""
from __future__ import annotations

import asyncio
import json
import logging
import random
from collections.abc import Callable

import websockets

log = logging.getLogger(__name__)
URL = "wss://fstream.binance.com/ws/!forceOrder@arr"
SILENCE_S = 600                 # ликвидации идут постоянно; 10 минут тишины — соединение повисло


async def stream(on_message: Callable[[dict], None], stop: asyncio.Event, url: str = URL) -> None:
    backoff = 1.0
    while not stop.is_set():
        try:
            async with websockets.connect(url, open_timeout=10, ping_interval=20, ping_timeout=20) as ws:
                log.info("ликвидации: поток подключён")
                backoff = 1.0
                while not stop.is_set():
                    msg = await asyncio.wait_for(ws.recv(), SILENCE_S)
                    data = json.loads(msg)
                    for m in data if isinstance(data, list) else [data]:
                        on_message(m)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            log.warning("ликвидации: тишина %s с, переподключаюсь", SILENCE_S)
        except Exception as e:
            log.warning("ликвидации: %s: %s; переподключение через %.0f с", type(e).__name__, e, backoff)
        if stop.is_set():
            break
        try:
            await asyncio.wait_for(stop.wait(), backoff + random.random())
        except asyncio.TimeoutError:
            pass
        backoff = min(backoff * 2, 60.0)
