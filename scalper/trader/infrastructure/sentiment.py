"""Настроение толпы для карточки «Погода рынка», всё без ключей: индекс страха и жадности (alternative.me); funding,
базис к споту, открытый интерес и лонги / шорты всех счетов и крупных трейдеров на BTC (публичные данные фьючерсов
Binance); доминация BTC (CoinGecko); индекс волатильности опционов BTC DVOL (Deribit)."""
from __future__ import annotations

import time

import httpx

FNG_URL = "https://api.alternative.me/fng/"
BINANCE = "https://fapi.binance.com"
COINGECKO_GLOBAL = "https://api.coingecko.com/api/v3/global"
DERIBIT_DVOL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
FNG_RU = {"extreme fear": "сильный страх", "fear": "страх", "neutral": "нейтрально", "greed": "жадность",
          "extreme greed": "сильная жадность"}


def parse_fng(raw: object) -> list[tuple[int, str]]:
    """(значение, подпись по-русски), новые первыми; непонятные строки пропускаются."""
    rows = raw.get("data") if isinstance(raw, dict) else None
    out = []
    for r in rows if isinstance(rows, list) else []:
        try:
            label = str(r.get("value_classification", "")).strip()
            out.append((int(r["value"]), FNG_RU.get(label.lower(), label)))
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    return out


def parse_premium(raw: object) -> dict[str, tuple[float, float]]:
    """По монетам USDT-перпетуалов: (последняя ставка funding, базис = mark / index − 1)."""
    out = {}
    for r in raw if isinstance(raw, list) else []:
        try:
            sym = str(r["symbol"])
            index = float(r["indexPrice"])
            if sym.endswith("USDT") and index > 0:
                out[sym] = (float(r["lastFundingRate"]), float(r["markPrice"]) / index - 1)
        except (KeyError, TypeError, ValueError):
            continue
    return out


def parse_series(raw: object, key: str) -> list[float]:
    """Ряд значения `key` из статистики фьючерсов Binance, старые первыми."""
    rows = [r for r in raw if isinstance(r, dict)] if isinstance(raw, list) else []
    out = []
    for r in sorted(rows, key=lambda x: int(x.get("timestamp", 0))):
        try:
            out.append(float(r[key]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def parse_long_short(raw: object) -> list[float]:
    """Соотношение лонгов к шортам, старые первыми."""
    return parse_series(raw, "longShortRatio")


def parse_dominance(raw: object) -> float | None:
    """Доля BTC в капитализации крипторынка, %."""
    try:
        return float(raw["data"]["market_cap_percentage"]["btc"])          # type: ignore[index]
    except (KeyError, TypeError, ValueError):
        return None


def parse_dvol(raw: object) -> list[float]:
    """Закрытия DVOL по дням, старые первыми."""
    try:
        rows = raw["result"]["data"]                                       # type: ignore[index]
    except (KeyError, TypeError):
        return []
    out = []
    for r in sorted((x for x in rows if isinstance(x, list) and len(x) >= 5), key=lambda x: x[0]):
        try:
            out.append(float(r[4]))
        except (TypeError, ValueError):
            continue
    return out


class MarketMood:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self.client = client or httpx.AsyncClient(timeout=20.0, headers={"User-Agent": "trendline-trader/1.0"})

    async def fear_greed(self, days: int = 8) -> list[tuple[int, str]]:
        r = await self.client.get(FNG_URL, params={"limit": days})
        r.raise_for_status()
        return parse_fng(r.json())

    async def premium_now(self) -> dict[str, tuple[float, float]]:
        r = await self.client.get(f"{BINANCE}/fapi/v1/premiumIndex")
        r.raise_for_status()
        return parse_premium(r.json())

    async def _btc_stat(self, path: str, key: str, days: int) -> list[float]:
        r = await self.client.get(f"{BINANCE}/futures/data/{path}",
                                  params={"symbol": "BTCUSDT", "period": "1d", "limit": days})
        r.raise_for_status()
        return parse_series(r.json(), key)

    async def btc_long_short(self, days: int = 8) -> list[float]:
        return await self._btc_stat("globalLongShortAccountRatio", "longShortRatio", days)

    async def btc_top_long_short(self, days: int = 8) -> list[float]:
        return await self._btc_stat("topLongShortPositionRatio", "longShortRatio", days)

    async def btc_open_interest(self, days: int = 8) -> list[float]:
        return await self._btc_stat("openInterestHist", "sumOpenInterestValue", days)

    async def btc_dominance(self) -> float | None:
        r = await self.client.get(COINGECKO_GLOBAL)
        r.raise_for_status()
        return parse_dominance(r.json())

    async def btc_dvol(self, days: int = 8) -> list[float]:
        end = int(time.time() * 1000)
        r = await self.client.get(DERIBIT_DVOL, params={"currency": "BTC", "resolution": "1D",
                                                        "start_timestamp": end - (days + 1) * 86_400_000,
                                                        "end_timestamp": end})
        r.raise_for_status()
        return parse_dvol(r.json())

    async def aclose(self) -> None:
        await self.client.aclose()
