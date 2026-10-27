"""Этап 3 проверки «недооценённые альты»: ончейн-стоимость против удержания BTC (меры из литературы, обзор 2026-10).
    адреса / капитализация   активные адреса за 30 дней к капитализации на конце месяца — «network-to-market»
                             (Babiak, Bianchi; Liu, Tsyvinski, Wu «Accounting for Cryptocurrency Value»: дешёвые по
                             цене на адрес растут сильнее). CoinMetrics community: AdrActCnt, CapMrktCurUSD (этап 1:
                             ~135 монет, 113 с историей до 2021)
    комиссии / капитализация комиссии протокола за 30 дней × 12 к капитализации (DefiLlama — комиссии по протоколам,
                             капитализация — CoinMetrics; только монеты, у которых есть обе)
Цены, отбор и выход — как в research/spotrev.py (спот Binance с делистнутыми, оборот >= $1M, возраст >= 180 дней, вход по
закрытию первого дня следующего месяца, умершая монета — последнее закрытие минус 2%, издержки 0.8% за круг).
Монет с данными мало, поэтому корзина — треть самых дешёвых (не 10%), отобранных >= 15. Справка — треть самых дорогих и
«дешёвые минус дорогие» (схема статей). Варианты: 2 меры × H 1 / 3 / 6 × режим всегда / BTC > SMA200 = 12.
Принятие: как у spotrev — сверх BTC > 0 в каждом периоде (когорт >= 6), t Ньюи–Уэста >= 3, плюс при издержках ×2.

    python -m research.spotrev collect && python -m research.onchainval report   (сеть — на машине сборки)
"""
from __future__ import annotations

import itertools
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings

import numpy as np
import pandas as pd

from . import spotrev as S

CM = "https://community-api.coinmetrics.io/v4"
LLAMA_LIST = ("https://api.llama.fi/overview/fees?excludeTotalDataChart=true&excludeTotalDataChartBreakdown=true"
              "&dataType=dailyFees")
LLAMA_FEES = "https://api.llama.fi/summary/fees/{slug}?dataType=dailyFees"
LLAMA_PROTO = "https://api.llama.fi/protocol/{slug}"
START = "2019-01-01"
TOP_PROTOCOLS = 150
TERCILE = 1 / 3
MIN_ELIG = 15
CM_ALIAS = {"avaxc": "AVAX", "matic_eth": "MATIC", "pol_eth": "POL"}
MEASURES = ("адреса / капитализация", "комиссии / капитализация")


def get(url: str) -> object | None:
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 403, 404):
                print(f"  onchainval: {url[:90]} → HTTP {e.code}", flush=True)
                return None
        except Exception as e:
            print(f"  onchainval: {url[:90]} → {e}", flush=True)
        time.sleep(2 ** attempt)
    return None


def cm_assets() -> list[str]:
    """Монеты CoinMetrics community, у которых есть и AdrActCnt, и CapMrktCurUSD на дневках."""
    have: dict[str, set[str]] = {}
    for metric in ("AdrActCnt", "CapMrktCurUSD"):
        x = get(f"{CM}/catalog-v2/asset-metrics?metrics={metric}&page_size=10000")
        for row in (x or {}).get("data", []) if isinstance(x, dict) else []:
            if any(f.get("frequency") == "1d" for m in row.get("metrics", []) for f in m.get("frequencies", [])):
                have.setdefault(row["asset"], set()).add(metric)
    return sorted(a for a, ms in have.items() if len(ms) == 2)


def binance_symbol(asset: str) -> str | None:
    if asset in CM_ALIAS:
        return CM_ALIAS[asset] + "USDT"
    return None if "_" in asset else asset.upper() + "USDT"


def cm_series(assets: list[str]) -> pd.DataFrame:
    """Дневные AdrActCnt и CapMrktCurUSD: столбцы asset, day, adr, cap."""
    rows = []
    for i in range(0, len(assets), 10):
        q = urllib.parse.urlencode({"assets": ",".join(assets[i:i + 10]), "metrics": "AdrActCnt,CapMrktCurUSD",
                                    "frequency": "1d", "start_time": START, "page_size": 10000,
                                    "ignore_unsupported_errors": "true"})
        url = f"{CM}/timeseries/asset-metrics?{q}"
        for _ in range(50):
            x = get(url)
            if not isinstance(x, dict):
                break
            for d in x.get("data", []):
                rows.append((d.get("asset"), d.get("time", "")[:10], d.get("AdrActCnt"), d.get("CapMrktCurUSD")))
            url = x.get("next_page_url")
            if not url:
                break
    df = pd.DataFrame(rows, columns=["asset", "day", "adr", "cap"])
    df["day"] = pd.to_datetime(df["day"])
    for k in ("adr", "cap"):
        df[k] = pd.to_numeric(df[k], errors="coerce")
    return df


def llama_fees() -> pd.DataFrame:
    """Дневные комиссии крупнейших протоколов DefiLlama, сложенные по символу монеты: столбцы symbol, day, fees."""
    x = get(LLAMA_LIST)
    ps = (x or {}).get("protocols", []) if isinstance(x, dict) else []
    ps = sorted((p for p in ps if p.get("slug")), key=lambda p: -(p.get("total1y") or 0))[:TOP_PROTOCOLS]
    rows = []
    for p in ps:
        proto = get(LLAMA_PROTO.format(slug=p["slug"]))
        sym = (proto or {}).get("symbol") if isinstance(proto, dict) else None
        if not sym or sym == "-":
            continue
        y = get(LLAMA_FEES.format(slug=p["slug"]))
        for ts, v in ((y or {}).get("totalDataChart") or []) if isinstance(y, dict) else []:
            rows.append((sym.upper() + "USDT", pd.Timestamp(int(ts), unit="s"), float(v or 0.0)))
    df = pd.DataFrame(rows, columns=["symbol", "day", "fees"])
    return df.groupby(["symbol", "day"], as_index=False)["fees"].sum()


def month_end(daily: pd.DataFrame, col: str, how: str) -> pd.DataFrame:
    """Помесячно (метка — начало месяца, как в spotrev): last — значение на последний день, sum30 / mean30 — за 30 дней."""
    w = daily.pivot_table(index="day", columns="symbol", values=col).sort_index()
    w = w.asfreq("D")
    if how == "sum30":
        w = w.rolling(30, min_periods=25).sum()
    elif how == "mean30":
        w = w.rolling(30, min_periods=20).mean()
    out = w.groupby(w.index.to_period("M")).last()
    out.index = out.index.to_timestamp()
    return out


def cohorts(c_end: pd.DataFrame, c_d1: pd.DataFrame, adv: pd.DataFrame, meta: pd.DataFrame, regime: pd.Series,
            sig: pd.DataFrame, H: int, btc_sym: str = "BTCUSDT") -> pd.DataFrame:
    """Когорты по мере «дешевизны» sig (больше — дешевле): дешёвая и дорогая трети, BTC, режим."""
    months = c_end.index
    first = meta["first"].reindex(c_end.columns)
    out = []
    for i in range(len(months) - 1):
        m = months[i]
        entry_m, exit_m = m + pd.offsets.MonthBegin(1), m + pd.offsets.MonthBegin(1 + H)
        if entry_m not in c_d1.index or exit_m > months[-1] or m not in sig.index:
            continue
        s = sig.loc[m].reindex(c_end.columns)
        p0 = c_d1.loc[entry_m]
        age_ok = ((first + S.MIN_AGE) <= (m + pd.offsets.MonthEnd(0))).reindex(s.index, fill_value=False)
        ok = s.notna() & np.isfinite(s) & (s > 0) & p0.notna() & (adv.loc[m] >= S.BANDS["от $1M"]) & age_ok
        ok[btc_sym] = False
        if ok.sum() < MIN_ELIG:
            continue
        p1 = S.exit_prices(c_d1, c_end, meta, exit_m)
        r = (p1 / p0 - 1)[ok].dropna()
        rk = s[r.index].rank(pct=True)
        b0, b1 = c_d1[btc_sym].get(entry_m, np.nan), p1.get(btc_sym, np.nan)
        cheap, dear = r[rk > 1 - TERCILE].mean(), r[rk <= TERCILE].mean()
        out.append({"t": entry_m, "дешёвые": cheap, "дорогие": dear, "btc": b1 / b0 - 1,
                    "on": bool(regime.get(m, False)), "n": len(r), "ls": cheap - dear})
    return pd.DataFrame(out)


def report() -> None:
    print("\n===== ONCHAINVAL: ончейн-стоимость — адреса и комиссии к капитализации — против удержания BTC (спот Binance, "
          "треть самых дешёвых) =====")
    got = S.load()
    if got is None:
        print("  onchainval: нет цен spotrev")
        return
    c_end, c_d1, adv, meta, regime, _ = got
    assets = cm_assets()
    print(f"  CoinMetrics: монет с адресами и капитализацией {len(assets)}", flush=True)
    cm = cm_series(assets)
    cm["symbol"] = cm["asset"].map(binance_symbol)
    cm = cm.dropna(subset=["symbol"])
    cm = cm[cm["symbol"].isin(c_end.columns)]
    print(f"  CoinMetrics: строк {len(cm)}, монет с парой на споте Binance {cm.symbol.nunique()}", flush=True)
    if cm.empty:
        print("  onchainval: нет данных CoinMetrics")
        return
    adr, cap = month_end(cm, "adr", "mean30"), month_end(cm, "cap", "last")
    signals = {"адреса / капитализация": (adr / cap.replace(0, np.nan))}
    fees = llama_fees()
    fees = fees[fees["symbol"].isin(cap.columns)]
    print(f"  DefiLlama: монет с комиссиями и капитализацией {fees.symbol.nunique()}", flush=True)
    if not fees.empty:
        f30 = month_end(fees, "fees", "sum30").reindex(columns=cap.columns)
        signals["комиссии / капитализация"] = (f30 * 12 / cap.reindex_like(f30).replace(0, np.nan))
    passed = []
    for name, H in itertools.product(MEASURES, S.HS):
        if name not in signals:
            print(f"\n    {name}: нет данных")
            continue
        z = cohorts(c_end, c_d1, adv, meta, regime, signals[name], H)
        if z.empty:
            print(f"\n    {name}, H {H}: когорт нет (отобранных < {MIN_ELIG})")
            continue
        print(f"\n    {name}, H {H} мес.: когорт {len(z)} ({z.t.min():%Y-%m} … {z.t.max():%Y-%m}), отобранных в медиане "
              f"{z.n.median():.0f}; дешёвые минус дорогие {z.ls.mean():+.1%} (t {S.nw_t(z.ls.to_numpy(), H):.1f})")
        for basket, reg in itertools.product(("дешёвые", "дорогие"), S.REGIMES):
            good, txt = S.verdict(z, basket, reg, H)
            tag = ("принято" if good else "нет") if basket == "дешёвые" else "справка"
            print(f"      {basket}, {reg}: {tag}\n        {txt}")
            if H == 1 and basket == "дешёвые":
                print(f"        счёт: {S.compound(z, basket, reg)}")
            if good and basket == "дешёвые":
                passed.append(f"{name} H {H} {reg}")
    print("\n  ИТОГ: " + ("прошли — " + "; ".join(passed) if passed else
                         "ончейн-«дешёвые» монеты не обгоняют удержание BTC"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
