"""Предложение долларовых стейблкоинов (DefiLlama, без ключа): включатель тренда на дневках «приток за 30 дней»."""
from __future__ import annotations

import httpx
import pandas as pd

URL = "https://stablecoins.llama.fi/stablecoincharts/all"


def parse_supply(raw: object) -> pd.Series:
    """Дневной ряд суммарного предложения, $; индекс — день UTC без пояса. Непонятные строки пропускаются."""
    out: dict[pd.Timestamp, float] = {}
    for row in raw if isinstance(raw, list) else []:
        try:
            day = pd.Timestamp(int(row["date"]), unit="s").normalize()
            v = (row.get("totalCirculatingUSD") or row.get("totalCirculating") or {}).get("peggedUSD")
            if v is not None:
                out[day] = float(v)
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    return pd.Series(out, dtype=float).sort_index()


class DefiLlamaStablecoins:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self.client = client or httpx.AsyncClient(timeout=30.0, headers={"User-Agent": "trendline-trader/1.0"})

    async def supply(self) -> pd.Series:
        r = await self.client.get(URL)
        r.raise_for_status()
        s = parse_supply(r.json())
        if not len(s):
            raise RuntimeError("DefiLlama: пустой или непонятный ответ")
        return s

    async def aclose(self) -> None:
        await self.client.aclose()
