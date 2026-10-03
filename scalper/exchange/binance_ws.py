"""
Рыночные потоки Binance USDⓈ-M.

На mainnet потоки разделены по эндпоинтам: стакан идёт через /public, а сделки,
mark price и ликвидации через /market. Подключаемся к каждому отдельно,
с переподключением и watchdog по тишине.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
from collections import defaultdict
from typing import Callable

import websockets

from core.models import Book, Liquidation, MarkPrice, Trade

log = logging.getLogger("ws")

SILENCE_TIMEOUT_S = 30


def parse(stream: str, d: dict):
    e = d.get("e")
    if e == "aggTrade":
        return Trade(ts=d["T"] / 1000, price=float(d["p"]), qty=float(d["q"]), buyer_maker=bool(d["m"]))
    if e == "depthUpdate":
        return Book(ts=d.get("T", d["E"]) / 1000,
                    bids=[(float(p), float(q)) for p, q in d["b"]],
                    asks=[(float(p), float(q)) for p, q in d["a"]])
    if e == "markPriceUpdate":
        return MarkPrice(ts=d["E"] / 1000, mark=float(d["p"]), index=float(d.get("i") or 0),
                         funding=float(d.get("r") or 0))
    if e == "forceOrder":
        o = d["o"]
        px = float(o.get("ap") or o["p"])
        qty = float(o.get("z") or o["q"])
        return Liquidation(ts=o["T"] / 1000, side=o["S"], price=px, qty=qty)
    return None


class MarketStream:
    def __init__(self, cfg, env: str, on_event: Callable, raw_sink: Callable | None = None):
        sym = cfg.exchange.symbol.lower()
        urls = getattr(cfg.exchange.ws, env)
        routes = defaultdict(list)
        routes[urls.public].append(f"{sym}@depth20@100ms")
        routes[urls.market] += [f"{sym}@aggTrade", f"{sym}@markPrice@1s", f"{sym}@forceOrder"]
        self.routes = dict(routes)
        self.on_event = on_event
        self.raw_sink = raw_sink
        self.connected: dict[str, bool] = {u: False for u in self.routes}
        self._stop = False

    async def run(self) -> None:
        await asyncio.gather(*(self._conn(url, streams) for url, streams in self.routes.items()))

    def stop(self) -> None:
        self._stop = True

    async def _conn(self, base: str, streams: list[str]) -> None:
        url = f"{base}?streams={'/'.join(streams)}"
        backoff = 1.0
        while not self._stop:
            try:
                async with websockets.connect(url, open_timeout=10, ping_interval=20,
                                              ping_timeout=20, max_queue=4096) as ws:
                    log.info("connected %s (%s)", base, ", ".join(streams))
                    self.connected[base] = True
                    backoff = 1.0
                    while not self._stop:
                        msg = await asyncio.wait_for(ws.recv(), SILENCE_TIMEOUT_S)
                        m = json.loads(msg)
                        stream, data = m.get("stream", ""), m.get("data", m)
                        if self.raw_sink:
                            self.raw_sink(stream, data)
                        ev = parse(stream, data)
                        if ev is not None:
                            self.on_event(ev)
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                log.warning("%s: нет данных %ss, переподключаюсь", base, SILENCE_TIMEOUT_S)
            except Exception as e:
                log.warning("%s: %s: %s; переподключение через %.0fs", base, type(e).__name__, e, backoff)
            self.connected[base] = False
            if self._stop:
                break
            await asyncio.sleep(backoff + random.random())
            backoff = min(backoff * 2, 30.0)
