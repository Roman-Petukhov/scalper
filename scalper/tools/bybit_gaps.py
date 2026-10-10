"""Какие ликвидные USDT-перпетуалы Binance бот не может открыть на Bybit (нет монеты или другое имя).

Запуск на сервере, из контейнера бота (из облака API Bybit закрыт):
    docker compose -f docker-compose.tailscale.yml exec -T trader python -m tools.bybit_gaps
Ликвидность — оборот Binance за 24 ч (у бота порог — средний за 30 дней, так что список приблизительный).
"""
from __future__ import annotations

import asyncio
import sys

import ccxt.async_support as ccxt

from trader.infrastructure.bybit import binance_aliases

MIN_TURNOVER = 20e6


def _perps(markets: dict) -> set[str]:
    return {m["id"] for m in markets.values()
            if m.get("swap") and m.get("linear") and m.get("settle") == "USDT" and m.get("active", True)}


async def main(min_turnover: float) -> None:
    bn, by = ccxt.binanceusdm(), ccxt.bybit({"options": {"defaultType": "swap"}})
    try:
        b_ids, y_ids = _perps(await bn.load_markets()), _perps(await by.load_markets())
        tickers = await bn.fetch_tickers()
    finally:
        await asyncio.gather(bn.close(), by.close())
    vol = {(t.get("info") or {}).get("symbol"): float(t.get("quoteVolume") or 0) for t in tickers.values()}
    aliases = binance_aliases(y_ids)
    liquid = sorted((i for i in b_ids if vol.get(i, 0) >= min_turnover), key=lambda i: -vol[i])
    mapped = [i for i in liquid if i not in y_ids and i in aliases]
    missing = [i for i in liquid if i not in y_ids and i not in aliases]
    print(f"Binance: {len(b_ids)} перпетуалов, с оборотом от ${min_turnover / 1e6:.0f}M за 24 ч — {len(liquid)}")
    print(f"Bybit: {len(y_ids)} перпетуалов")
    print(f"под другим именем, бот их теперь находит ({len(mapped)}): "
          + ", ".join(f"{i}→{aliases[i]}" for i in mapped))
    print(f"бот пропустит — нет на Bybit под таким именем ({len(missing)}, {len(missing) / max(len(liquid), 1):.0%} "
          f"ликвидных): " + ", ".join(f"{i} (${vol[i] / 1e6:.0f}M)" for i in missing))


if __name__ == "__main__":
    asyncio.run(main(float(sys.argv[1]) * 1e6 if len(sys.argv) > 1 else MIN_TURNOVER))
