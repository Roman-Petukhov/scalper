"""Данные скринера без ключей: DefiLlama (комиссии, выручка, токены протоколов, разлоки), CoinGecko (цена, FDV,
оборот), спот Binance (недельные свечи BTC с 2017 — недельной EMA200 нужно 200 недель)."""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, datetime, timezone

import httpx
import pandas as pd

from ..domain.screener import DATA_TYPES, FeeRow, Quote, group_key

log = logging.getLogger(__name__)

LLAMA = "https://api.llama.fi"
FEES_PATH = "/overview/fees"
EMISSIONS_LIST = "https://defillama-datasets.llama.fi/emissionsProtocolsList"
EMISSIONS = "https://defillama-datasets.llama.fi/emissions/{key}"
COINGECKO = "https://api.coingecko.com/api/v3/coins/markets"
BINANCE_SPOT = "https://api.binance.com/api/v3/klines"
UA = {"User-Agent": "trendline-trader/1.0", "Accept": "application/json"}
CG_BATCH = 200
CG_PAUSE_S = 6.0                 # бесплатный CoinGecko — около 10 запросов в минуту


def _num(x: object) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
        return v if v == v else 0.0
    except (TypeError, ValueError):
        return 0.0


def parse_fee_rows(raw: object, data_type: str) -> list[FeeRow]:
    """Строки обзора /overview/fees: версия протокола × вид данных; непонятные строки пропускаются."""
    rows = []
    for p in (raw or {}).get("protocols", []) if isinstance(raw, dict) else []:
        if not isinstance(p, dict):
            continue
        slug = p.get("slug") or p.get("module") or p.get("name")
        rows.append(FeeRow(group=group_key(slug, p.get("name"), p.get("parentProtocol")), slug=str(slug or ""),
                           name=str(p.get("displayName") or p.get("name") or slug or "?"),
                           category=str(p.get("category") or "—"), data_type=data_type,
                           d30=_num(p.get("total30d")), y1=_num(p.get("total1y"))))
    return rows


def parse_token_ids(protocols: object, chains: object) -> dict[str, str]:
    """id CoinGecko по ключу бизнеса и slug из /protocols и /v2/chains (сети — по имени)."""
    out: dict[str, str] = {}
    for p in protocols if isinstance(protocols, list) else []:
        g = p.get("gecko_id") if isinstance(p, dict) else None
        if not g:
            continue
        out.setdefault(group_key(p.get("slug"), p.get("name"), p.get("parentProtocol")), g)
        if p.get("slug"):
            out.setdefault(str(p["slug"]).lower(), g)
    for c in chains if isinstance(chains, list) else []:
        if isinstance(c, dict) and c.get("gecko_id"):
            out.setdefault(group_key(None, c.get("name"), None), c["gecko_id"])
    return out


def parse_schedule(raw: object) -> list[tuple[date, float]]:
    """Накопленный разлок по дням: сумма категорий (команда, фонды, экосистема…) с переносом последнего значения."""
    if not isinstance(raw, dict):
        return []
    cats = ((raw.get("documentedData") or {}).get("data")) or ((raw.get("data") or {}).get("data")) or []
    series = []
    for c in cats:
        pts = c.get("data") if isinstance(c, dict) else None
        if not pts:
            continue
        s = {}
        for x in pts:
            try:
                s[pd.Timestamp(int(x["timestamp"]), unit="s").normalize()] = float(x.get("unlocked") or 0.0)
            except (KeyError, TypeError, ValueError):
                continue
        if s:
            series.append(pd.Series(s))
    if not series:
        return []
    cum = pd.concat(series, axis=1).sort_index().ffill().fillna(0.0).sum(axis=1)
    cum = cum[~cum.index.duplicated()]
    return [(t.date(), float(v)) for t, v in cum.items()]


def parse_quotes(raw: object) -> dict[str, Quote]:
    out = {}
    for m in raw if isinstance(raw, list) else []:
        if not isinstance(m, dict) or not m.get("id"):
            continue

        def f(k: str) -> float | None:
            v = m.get(k)
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        out[m["id"]] = Quote(gecko_id=m["id"], symbol=str(m.get("symbol") or "").upper(), name=str(m.get("name") or ""),
                             price=f("current_price"), mcap=f("market_cap"), fdv=f("fully_diluted_valuation"),
                             volume=f("total_volume"), chg_1y=f("price_change_percentage_1y_in_currency"),
                             ath_chg=f("ath_change_percentage"))
    return out


def parse_weekly(rows: object, now_ms: int) -> pd.DataFrame:
    """Только закрытые недели (close_time в прошлом); индекс — понедельник открытия недели, UTC."""
    data = [(int(r[0]), float(r[3]), float(r[4])) for r in rows if isinstance(r, list) and int(r[6]) < now_ms] \
        if isinstance(rows, list) else []
    df = pd.DataFrame(data, columns=["open_time", "low", "close"])
    df.index = pd.to_datetime(df.pop("open_time"), unit="ms", utc=True)
    return df


class DefiLlamaFundamentals:
    def __init__(self, client: httpx.AsyncClient | None = None, list_ttl_s: float = 6 * 3600.0) -> None:
        self.client = client or httpx.AsyncClient(timeout=90.0, headers=UA, follow_redirects=True)
        self.list_ttl_s = list_ttl_s
        self._emis_list: tuple[float, set[str]] | None = None

    async def _get(self, url: str, params: dict | None = None, quiet: tuple[int, ...] = ()) -> object | None:
        for attempt in range(4):
            try:
                r = await self.client.get(url, params=params)
            except httpx.HTTPError as e:
                log.warning("DefiLlama %s: %s", url, e)
                await asyncio.sleep(2 ** attempt)
                continue
            if r.status_code in quiet:
                return None
            if r.status_code == 429 or r.status_code >= 500:
                await asyncio.sleep(2 ** (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"DefiLlama не ответил: {url}")

    async def fee_rows(self) -> list[FeeRow]:
        rows: list[FeeRow] = []
        for dt in DATA_TYPES:
            raw = await self._get(LLAMA + FEES_PATH, {"excludeTotalDataChart": "true",
                                                      "excludeTotalDataChartBreakdown": "true", "dataType": dt})
            rows += parse_fee_rows(raw, dt)
        if not rows:
            raise RuntimeError("DefiLlama: пустой обзор комиссий")
        return rows

    async def token_ids(self) -> dict[str, str]:
        return parse_token_ids(await self._get(LLAMA + "/protocols"), await self._get(LLAMA + "/v2/chains"))

    async def token_id(self, key: str) -> str | None:
        for path in (f"/summary/fees/{key}", f"/protocol/{key}"):
            raw = await self._get(LLAMA + path, {"dataType": "dailyFees"} if "summary" in path else None,
                                  quiet=(400, 404))
            if isinstance(raw, dict) and raw.get("gecko_id"):
                return str(raw["gecko_id"])
        return None

    async def _emission_keys(self) -> set[str]:
        now = time.monotonic()
        if self._emis_list is None or now - self._emis_list[0] > self.list_ttl_s:
            raw = await self._get(EMISSIONS_LIST, quiet=(403, 404))
            self._emis_list = (now, {str(x).lower() for x in raw} if isinstance(raw, list) else set())
        return self._emis_list[1]

    async def unlock_schedules(self, keys: set[str]) -> dict[str, list[tuple[date, float]]]:
        have = await self._emission_keys()
        out = {}
        for k in sorted(keys & have):
            raw = await self._get(EMISSIONS.format(key=k), quiet=(403, 404))
            pts = parse_schedule(raw)
            if pts:
                out[k] = pts
        return out

    async def aclose(self) -> None:
        await self.client.aclose()


class CoinGeckoQuotes:
    def __init__(self, client: httpx.AsyncClient | None = None, api_key: str | None = None) -> None:
        headers = dict(UA)
        if api_key:
            headers["x-cg-demo-api-key"] = api_key
        self.client = client or httpx.AsyncClient(timeout=60.0, headers=headers)
        self.pause_s = CG_PAUSE_S

    async def quotes(self, ids: list[str]) -> dict[str, Quote]:
        out: dict[str, Quote] = {}
        uniq = sorted(set(ids))
        for i in range(0, len(uniq), CG_BATCH):
            params = {"vs_currency": "usd", "ids": ",".join(uniq[i:i + CG_BATCH]), "per_page": 250, "page": 1,
                      "price_change_percentage": "1y"}
            for attempt in range(5):
                r = await self.client.get(COINGECKO, params=params)
                if r.status_code == 429:
                    wait = float(r.headers.get("Retry-After") or 15 * (attempt + 1))
                    log.warning("CoinGecko: лимит запросов, ждём %.0f с", wait)
                    await asyncio.sleep(wait)
                    continue
                r.raise_for_status()
                out.update(parse_quotes(r.json()))
                break
            else:
                raise RuntimeError("CoinGecko: лимит запросов не отпустил")
            if i + CG_BATCH < len(uniq):
                await asyncio.sleep(self.pause_s)
        return out

    async def aclose(self) -> None:
        await self.client.aclose()


class BinanceSpotWeekly:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self.client = client or httpx.AsyncClient(timeout=30.0, headers=UA)

    async def weekly(self, symbol: str) -> pd.DataFrame:
        r = await self.client.get(BINANCE_SPOT, params={"symbol": symbol, "interval": "1w", "limit": 1000})
        r.raise_for_status()
        return parse_weekly(r.json(), int(datetime.now(timezone.utc).timestamp() * 1000))

    async def aclose(self) -> None:
        await self.client.aclose()
