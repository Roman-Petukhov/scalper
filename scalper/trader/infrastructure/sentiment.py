"""Настроение толпы для карточки «Погода рынка»: индекс страха и жадности (alternative.me, без ключа), funding всех
перпетуалов и соотношение лонгов / шортов по счетам на BTC (публичные данные фьючерсов Binance)."""
from __future__ import annotations

import httpx

FNG_URL = "https://api.alternative.me/fng/"
BINANCE = "https://fapi.binance.com"
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


def parse_funding(raw: object) -> dict[str, float]:
    """Последняя ставка funding по монетам USDT-перпетуалов."""
    out = {}
    for r in raw if isinstance(raw, list) else []:
        try:
            sym = str(r["symbol"])
            if sym.endswith("USDT"):
                out[sym] = float(r["lastFundingRate"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def parse_long_short(raw: object) -> list[float]:
    """Соотношение лонгов к шортам по счетам, старые первыми."""
    rows = [r for r in raw if isinstance(r, dict)] if isinstance(raw, list) else []
    out = []
    for r in sorted(rows, key=lambda x: int(x.get("timestamp", 0))):
        try:
            out.append(float(r["longShortRatio"]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


class MarketMood:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self.client = client or httpx.AsyncClient(timeout=20.0, headers={"User-Agent": "trendline-trader/1.0"})

    async def fear_greed(self, days: int = 8) -> list[tuple[int, str]]:
        r = await self.client.get(FNG_URL, params={"limit": days})
        r.raise_for_status()
        return parse_fng(r.json())

    async def funding_now(self) -> dict[str, float]:
        r = await self.client.get(f"{BINANCE}/fapi/v1/premiumIndex")
        r.raise_for_status()
        return parse_funding(r.json())

    async def btc_long_short(self, days: int = 8) -> list[float]:
        r = await self.client.get(f"{BINANCE}/futures/data/globalLongShortAccountRatio",
                                  params={"symbol": "BTCUSDT", "period": "1d", "limit": days})
        r.raise_for_status()
        return parse_long_short(r.json())

    async def aclose(self) -> None:
        await self.client.aclose()
