"""Этап 1 проверки «недооценённые альты»: есть ли бесплатная история для мер стоимости из литературы.
    адреса к капитализации   Liu, Tsyvinski, Wu («Accounting for Cryptocurrency Value»); Babiak, Bianchi —
                             CoinMetrics community API (без ключа): AdrActCnt, CapMrktCurUSD, SplyCur, FeeTotUSD
    капитализация к комиссиям DeFi — DefiLlama: комиссии по протоколам с историей, связь с монетой (gecko_id)
    история капитализации    CoinGecko без ключа (market_chart days=max), DefiLlama coins (только цена)
Печатает, что отвечает, сколько монет и с какой даты, — по этому решается схема этапа 3. Сеть есть только на машине
сборки, поэтому всё — в report.

    python -m research.onchainprobe report
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

CM = "https://community-api.coinmetrics.io/v4"
CM_METRICS = ("AdrActCnt", "AdrNewCnt", "CapMrktCurUSD", "SplyCur", "FeeTotUSD", "PriceUSD")
LLAMA_FEES = ("https://api.llama.fi/overview/fees?excludeTotalDataChart=true&excludeTotalDataChartBreakdown=true"
              "&dataType=dailyFees")
LLAMA_ONE = "https://api.llama.fi/summary/fees/{slug}?dataType=dailyFees"
LLAMA_PROTO = "https://api.llama.fi/protocol/{slug}"
GECKO = "https://api.coingecko.com/api/v3/coins/{id}/market_chart?vs_currency=usd&days=max"


def get(url: str) -> tuple[int, object | str | None]:
    """(HTTP-код, JSON или начало текста); 0 — сеть не ответила."""
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
                try:
                    return r.status, json.loads(raw)
                except ValueError:
                    return r.status, raw[:300].decode(errors="replace")
        except urllib.error.HTTPError as e:
            body = e.read()[:300].decode(errors="replace")
            if e.code in (400, 401, 403, 404):
                return e.code, body
        except Exception as e:
            body = str(e)
        time.sleep(2 ** attempt)
    return 0, body


def coinmetrics() -> None:
    print("\n  --- CoinMetrics community: какие монеты и с какой даты (дневки) ---")
    for metric in CM_METRICS:
        code, x = get(f"{CM}/catalog-v2/asset-metrics?metrics={metric}&page_size=10000")
        if code != 200 or not isinstance(x, dict):
            code2, x2 = get(f"{CM}/catalog-all-v2/asset-metrics?metrics={metric}&page_size=10000")
            if code2 == 200 and isinstance(x2, dict):
                code, x = code2, x2
        if code != 200 or not isinstance(x, dict):
            print(f"    {metric}: HTTP {code} {str(x)[:160]}")
            continue
        assets = []
        for row in x.get("data", []):
            for m in row.get("metrics", []):
                for f in m.get("frequencies", []):
                    if f.get("frequency") == "1d":
                        assets.append((row.get("asset"), str(f.get("min_time", ""))[:10], f.get("community")))
        early = sum(1 for _, t, _ in assets if t and t < "2021-01-01")
        print(f"    {metric}: монет {len(assets)}, с историей до 2021 — {early}; пример "
              + ", ".join(f"{a} {t}" for a, t, _ in assets[:12]))
    code, x = get(f"{CM}/timeseries/asset-metrics?assets=btc,eth,uni,link,aave&metrics=AdrActCnt,CapMrktCurUSD"
                  f"&frequency=1d&start_time=2024-01-01&page_size=10")
    print(f"    проба рядов (btc, eth, uni, link, aave): HTTP {code}; "
          + (json.dumps(x.get("data", [])[:3], ensure_ascii=False)[:400] if isinstance(x, dict) else str(x)[:300]))


def defillama() -> None:
    print("\n  --- DefiLlama: комиссии протоколов ---")
    code, x = get(LLAMA_FEES)
    if code != 200 or not isinstance(x, dict):
        print(f"    обзор комиссий: HTTP {code} {str(x)[:160]}")
        return
    ps = x.get("protocols", [])
    with_gecko = [p for p in ps if p.get("gecko_id")]
    big = sorted(with_gecko, key=lambda p: -(p.get("total30d") or 0))[:40]
    print(f"    протоколов {len(ps)}, со ссылкой на монету (gecko_id) {len(with_gecko)}; ключи: "
          f"{sorted(ps[0].keys())[:30] if ps else []}")
    print("    крупнейшие по комиссиям за 30 дней: " + ", ".join(f"{p.get('slug')}→{p.get('gecko_id')}" for p in big[:20]))
    for slug in ("uniswap", "aave", "lido", "pancakeswap"):
        code, y = get(LLAMA_ONE.format(slug=slug))
        if code == 200 and isinstance(y, dict):
            ch = y.get("totalDataChart") or []
            first = time.strftime("%Y-%m-%d", time.gmtime(ch[0][0])) if ch else "—"
            print(f"    {slug}: дней комиссий {len(ch)}, с {first}; gecko_id {y.get('gecko_id')}")
        else:
            print(f"    {slug}: HTTP {code} {str(y)[:120]}")
    code, y = get(LLAMA_PROTO.format(slug="uniswap"))
    if code == 200 and isinstance(y, dict):
        print(f"    protocol/uniswap: ключи {sorted(y.keys())[:40]}; mcap {y.get('mcap')}")
    else:
        print(f"    protocol/uniswap: HTTP {code}")


def coingecko() -> None:
    print("\n  --- CoinGecko без ключа: история капитализации ---")
    for cid in ("uniswap", "chainlink"):
        code, x = get(GECKO.format(id=cid))
        if code == 200 and isinstance(x, dict):
            mc = x.get("market_caps") or []
            first = time.strftime("%Y-%m-%d", time.gmtime(mc[0][0] / 1000)) if mc else "—"
            print(f"    {cid}: точек капитализации {len(mc)}, с {first}")
        else:
            print(f"    {cid}: HTTP {code} {str(x)[:160]}")


def report() -> None:
    print("\n===== ONCHAINPROBE: бесплатная история для мер стоимости (адреса, комиссии, капитализация) =====")
    coinmetrics()
    defillama()
    coingecko()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
