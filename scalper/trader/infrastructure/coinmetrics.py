"""CoinMetrics community API (без ключа): активные адреса и капитализация монет по дням — для карточки «цена к
активным адресам» (domain/onchain_value.py)."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import httpx

from ..domain.onchain_value import Reading

log = logging.getLogger(__name__)

BASE = "https://community-api.coinmetrics.io/v4"
METRICS = ("AdrActCnt", "CapMrktCurUSD")
UA = {"User-Agent": "trendline-trader/1.0", "Accept": "application/json"}
BATCH = 20
MAX_PAGES = 20


def parse_catalog(raw: object) -> set[str]:
    """Монеты, у которых метрика есть на дневках."""
    out = set()
    for row in (raw or {}).get("data", []) if isinstance(raw, dict) else []:
        if not isinstance(row, dict) or not row.get("asset"):
            continue
        if any(isinstance(f, dict) and f.get("frequency") == "1d"
               for m in row.get("metrics", []) if isinstance(m, dict) for f in m.get("frequencies", [])):
            out.add(str(row["asset"]))
    return out


def parse_readings(rows: list[dict]) -> list[Reading]:
    """Строки timeseries (asset, time, AdrActCnt, CapMrktCurUSD) → среднее адресов за окно и последняя капитализация."""
    acc: dict[str, dict] = {}
    for d in sorted((r for r in rows if isinstance(r, dict) and r.get("asset")), key=lambda r: str(r.get("time", ""))):
        a = acc.setdefault(str(d["asset"]), {"adr": [], "cap": None})
        try:
            if d.get("AdrActCnt") is not None:
                a["adr"].append(float(d["AdrActCnt"]))
            if d.get("CapMrktCurUSD") is not None:
                a["cap"] = float(d["CapMrktCurUSD"])
        except (TypeError, ValueError):
            continue
    return [Reading(k, sum(v["adr"]) / len(v["adr"]), v["cap"], len(v["adr"]))
            for k, v in acc.items() if v["adr"] and v["cap"] is not None]


class CoinMetricsOnchain:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self.client = client or httpx.AsyncClient(timeout=60.0, headers=UA, follow_redirects=True)

    async def _get(self, url: str, params: dict | None = None) -> dict:
        for attempt in range(4):
            try:
                r = await self.client.get(url, params=params)
            except httpx.HTTPError as e:
                log.warning("CoinMetrics %s: %s", url, e)
                await asyncio.sleep(2 ** attempt)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                await asyncio.sleep(2 ** (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            return data if isinstance(data, dict) else {}
        raise RuntimeError(f"CoinMetrics не ответил: {url}")

    async def assets(self) -> list[str]:
        have: set[str] | None = None
        for m in METRICS:
            got = parse_catalog(await self._get(f"{BASE}/catalog-v2/asset-metrics", {"metrics": m, "page_size": 10000}))
            have = got if have is None else have & got
        return sorted(have or set())

    async def readings(self, days: int) -> list[Reading]:
        """Средние активные адреса за days дней и последняя капитализация по всем монетам источника."""
        assets = await self.assets()
        if not assets:
            raise RuntimeError("CoinMetrics: пустой каталог")
        start = (datetime.now(timezone.utc) - timedelta(days=days + 2)).strftime("%Y-%m-%d")
        rows: list[dict] = []
        for i in range(0, len(assets), BATCH):
            url: str | None = f"{BASE}/timeseries/asset-metrics"
            params: dict | None = {"assets": ",".join(assets[i:i + BATCH]), "metrics": ",".join(METRICS),
                                   "frequency": "1d", "start_time": start, "page_size": 10000,
                                   "ignore_unsupported_errors": "true"}
            for _ in range(MAX_PAGES):
                data = await self._get(url, params)
                rows += [d for d in data.get("data", []) if isinstance(d, dict)]
                url, params = data.get("next_page_url"), None
                if not url:
                    break
        return parse_readings(rows)

    async def aclose(self) -> None:
        await self.client.aclose()
