"""Публичные данные фьючерсов Binance USDⓈ-M: в свечах есть объём рыночных покупок (taker buy), на котором стоит
фильтр агрессоров. Кеш свечей в памяти: первый запрос — 1500 свечей, дальше — только хвост."""
from __future__ import annotations

import asyncio
import logging
import time

import httpx
import pandas as pd

from ..domain.models import Timeframe

log = logging.getLogger(__name__)

BASE = "https://fapi.binance.com"
HISTORY = 1500                    # 4h: 250 дней — хватает на EMA50 дневок и жизнь линий (300 свечей)
TAIL = 6


class BinanceMarketData:
    def __init__(self, client: httpx.AsyncClient | None = None, universe_ttl_s: float = 3600.0) -> None:
        self.client = client or httpx.AsyncClient(base_url=BASE, timeout=20.0)
        self.cache: dict[tuple[str, Timeframe], pd.DataFrame] = {}
        self._universe: tuple[float, float, list[str]] | None = None
        self.universe_ttl_s = universe_ttl_s

    async def _get(self, path: str, params: dict | None = None) -> list | dict:
        for attempt in range(5):
            r = await self.client.get(path, params=params)
            if r.status_code in (418, 429):
                wait = float(r.headers.get("Retry-After", 2 ** attempt))
                log.warning("Binance %s: лимит запросов, ждём %.0f с", path, wait)
                await asyncio.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"Binance {path}: лимит запросов не отпустил")

    async def universe(self, min_turnover_usd: float) -> list[str]:
        now = time.monotonic()
        if self._universe and self._universe[1] == min_turnover_usd and now - self._universe[0] < self.universe_ttl_s:
            return self._universe[2]
        info, tick = await asyncio.gather(self._get("/fapi/v1/exchangeInfo"), self._get("/fapi/v1/ticker/24hr"))
        live = {s["symbol"] for s in info["symbols"] if s.get("status") == "TRADING"
                and s.get("contractType") == "PERPETUAL" and s.get("quoteAsset") == "USDT"}
        out = sorted(t["symbol"] for t in tick if t["symbol"] in live and float(t["quoteVolume"]) >= min_turnover_usd)
        self._universe = (now, min_turnover_usd, out)
        return out

    @staticmethod
    def _frame(rows: list) -> pd.DataFrame:
        df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume", "close_time",
                                         "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume",
                                         "ignore"])
        df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        cols = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume"]
        out = df[cols].astype("float64")
        out["close_time"] = df["close_time"].astype("int64")
        return out

    async def closed_bars(self, symbol: str, tf: Timeframe) -> pd.DataFrame:
        key = (symbol, tf)
        old = self.cache.get(key)
        limit = HISTORY if old is None else TAIL
        rows = await self._get("/fapi/v1/klines", {"symbol": symbol, "interval": tf.value, "limit": limit})
        new = self._frame(rows)
        df = new if old is None else pd.concat([old, new])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        now_ms = int(time.time() * 1000)
        df = df[df["close_time"] < now_ms]                  # только закрытые свечи
        df = df.iloc[-HISTORY:]
        self.cache[key] = df
        return df.drop(columns="close_time")

    async def aclose(self) -> None:
        await self.client.aclose()
