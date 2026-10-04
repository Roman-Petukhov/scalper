"""
Событийное исследование: публичные объявления Binance (листинги, запуск фьючерсов, Launchpool/HODLer, мониторинг,
делистинги) и реакция цены той же монеты на Bybit по тиковым сделкам.

Это законный новостной трейдинг: только опубликованная информация, вопрос лишь в скорости реакции.
Время события — releaseDate статьи Binance (мс). Вход: через L секунд после публикации по цене первой сделки
агрессора в нужную сторону (лонг — первая покупка, шорт — первая продажа). Выход через H: по последней сделке
до момента выхода. Комиссии: taker 5.5 б.п. на сторону + 5 б.п. проскальзывания на сторону.

    python -m research.news.events --out <dir>
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

CMS = "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query?type=1&catalogId={c}&pageNo={p}&pageSize=50"
CATALOGS = {48: "listing", 161: "delisting"}
BYBIT_TRADES = "https://public.bybit.com/trading/{s}/{s}{d}.csv.gz"
LAT_S = (1, 3, 10, 30, 60)
HOR_S = (60, 300, 900, 3600, 14400)
COST_BPS = 2 * (5.5 + 5.0)
UA = {"User-Agent": "Mozilla/5.0 (research)"}


def _get_json(url: str):
    for k in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(2 ** k)
    return None


def fetch_announcements() -> pd.DataFrame:
    rows = []
    for cid, cat in CATALOGS.items():
        p = 1
        while True:
            js = _get_json(CMS.format(c=cid, p=p))
            arts = (((js or {}).get("data") or {}).get("catalogs") or [{}])[0].get("articles") or []
            if not arts:
                break
            for a in arts:
                rows.append({"id": a["id"], "catalog": cat, "title": a["title"], "ts_ms": int(a["releaseDate"])})
            p += 1
            time.sleep(0.4)
    df = pd.DataFrame(rows).drop_duplicates("id").sort_values("ts_ms")
    df["time"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    return df


TICK = r"[A-Z0-9]{2,15}"


def classify(title: str, catalog: str) -> tuple[str, list[str], int]:
    """Тип события, тикеры, ожидаемое направление (+1 рост, -1 падение)."""
    t = title.replace("USDⓈ-M", "USDS-M")
    if catalog == "delisting" or re.search(r"\bWill Delist\b|\bDelisting\b", t):
        if "Futures" in t:
            ticks = re.findall(rf"\b({TICK})USDT\b", t)
            return "futures_delist", ticks, -1
        m = re.search(r"Will Delist (.+?)(?: on \d|$)", t)
        ticks = re.findall(rf"\b({TICK})\b", m.group(1)) if m else []
        return "spot_delist", [x for x in ticks if x not in ("AND", "ON")], -1
    if re.search(r"Monitoring Tag", t):
        m = re.search(r"Add (.+?) to Monitoring", t)
        ticks = re.findall(rf"\b({TICK})\b", m.group(1)) if m else []
        return "monitoring_tag", [x for x in ticks if x not in ("AND",)], -1
    if re.search(r"Futures Will Launch", t):
        return "futures_launch", re.findall(rf"\b({TICK})USDT\b", t), 1
    if re.search(r"Launchpool|HODLer Airdrop|Megadrop", t):
        return "launchpool_airdrop", re.findall(rf"\(({TICK})\)", t), 1
    if re.search(r"Will List\b", t):
        return "spot_listing", re.findall(rf"\(({TICK})\)", t), 1
    if re.search(r"Will Add", t):
        return "product_add", re.findall(rf"\(({TICK})\)", t), 1
    return "other", [], 0


def _trades(sym: str, day: str) -> pd.DataFrame | None:
    url = BYBIT_TRADES.format(s=sym, d=day)
    for k in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                raw = r.read()
            t = pd.read_csv(io.BytesIO(gzip.decompress(raw)), usecols=["timestamp", "side", "price"])
            t["ts"] = (t["timestamp"] * 1000).astype("int64")
            return t.sort_values("ts", kind="stable")
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
        except Exception:
            time.sleep(2 ** k)
    return None


def event_paths(ev: dict) -> dict | None:
    """Цены вокруг события по сделкам Bybit (перпетуал SYMUSDT)."""
    sym = f"{ev['ticker']}USDT"
    tau = ev["ts_ms"]
    days = sorted({pd.Timestamp(x, unit="ms", tz="UTC").strftime("%Y-%m-%d")
                   for x in (tau - 3600_000, tau, tau + max(HOR_S) * 1000 + max(LAT_S) * 1000)})
    parts = [p for p in (_trades(sym, d) for d in days) if p is not None]
    if not parts:
        return None
    t = pd.concat(parts).drop_duplicates().sort_values("ts", kind="stable")
    ts, px = t["ts"].to_numpy(), t["price"].to_numpy(dtype="float64")
    buy = (t["side"].to_numpy() == "Buy")
    if len(ts) < 50 or ts[0] > tau - 600_000:        # монета должна торговаться до объявления
        return None

    def last_before(q):
        i = np.searchsorted(ts, q, side="right") - 1
        return px[i] if i >= 0 else np.nan

    def first_aggr_after(q, want_buy):
        sel_ts, sel_px = (ts[buy], px[buy]) if want_buy else (ts[~buy], px[~buy])
        i = np.searchsorted(sel_ts, q, side="left")
        return sel_px[i] if i < len(sel_ts) and sel_ts[i] - q <= 60_000 else np.nan

    out = {"symbol": sym, "p_pre1h": last_before(tau - 3600_000), "p_pre5m": last_before(tau - 300_000),
           "p0": last_before(tau)}
    d = ev["direction"]
    for L in LAT_S:
        out[f"entry_{L}"] = first_aggr_after(tau + L * 1000, want_buy=(d > 0))
    for H in HOR_S:
        out[f"exit_{H}"] = last_before(tau + H * 1000)
    return out


def study(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    with ThreadPoolExecutor(8) as ex:
        for ev, res in zip(events.to_dict("records"), ex.map(event_paths, events.to_dict("records"))):
            if res:
                rows.append({**ev, **res})
    return pd.DataFrame(rows)


def report(r: pd.DataFrame) -> None:
    d = r["direction"].to_numpy()
    r = r.copy()
    r["pre1h_bps"] = (r["p0"] / r["p_pre1h"] - 1) * 1e4 * d
    r["pre5m_bps"] = (r["p0"] / r["p_pre5m"] - 1) * 1e4 * d
    for L in LAT_S:
        r[f"slip_{L}_bps"] = (r[f"entry_{L}"] / r["p0"] - 1) * 1e4 * d       # сколько «уехало» до нашего входа
        for H in HOR_S:
            r[f"net_{L}_{H}"] = (r[f"exit_{H}"] / r[f"entry_{L}"] - 1) * 1e4 * d - COST_BPS
    r["year"] = pd.to_datetime(r["ts_ms"], unit="ms", utc=True).dt.year
    print(f"\nсобытий с данными Bybit: {len(r)}")
    print(r.groupby("type").size().to_string())
    print("\nДвижение ДО объявления (в ожидаемую сторону, б.п.; медиана) и «уехало» к моменту входа:")
    g = r.groupby("type")
    print(pd.DataFrame({"n": g.size(), "pre1h": g["pre1h_bps"].median(), "pre5m": g["pre5m_bps"].median(),
                        "к 1с": g["slip_1_bps"].median(), "к 10с": g["slip_10_bps"].median(),
                        "к 60с": g["slip_60_bps"].median()}).round(1).to_string())
    for stat in ("mean", "median"):
        print(f"\nЧистый результат сделки, б.п. ({'среднее' if stat == 'mean' else 'медиана'}), "
              f"строки — задержка входа, столбцы — удержание:")
        for typ, gg in r.groupby("type"):
            if len(gg) < 15:
                continue
            tab = pd.DataFrame({f"{H // 60}м": [getattr(gg[f"net_{L}_{H}"], stat)() for L in LAT_S] for H in HOR_S},
                               index=[f"{L}с" for L in LAT_S])
            print(f"\n[{typ}] n={len(gg)}")
            print(tab.round(1).to_string())
    print("\nПо годам (вход через 3 с, удержание 5 мин; среднее / медиана / доля плюсовых):")
    k = "net_3_300"
    print(r.groupby(["type", "year"])[k].agg(["count", "mean", "median", lambda x: (x > 0).mean()])
          .round(2).to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ann = fetch_announcements()
    ann.to_csv(out / "binance_announcements.csv", index=False)
    print(f"объявлений: {len(ann)} ({ann['time'].min():%Y-%m-%d} … {ann['time'].max():%Y-%m-%d})")
    ev = []
    for row in ann.itertuples():
        typ, ticks, d = classify(row.title, row.catalog)
        if d == 0 or row.time < pd.Timestamp("2022-01-01", tz="UTC"):
            continue
        for tk in dict.fromkeys(ticks):
            ev.append({"id": row.id, "type": typ, "ticker": tk, "direction": d, "ts_ms": row.ts_ms, "title": row.title})
    ev = pd.DataFrame(ev)
    print(f"событий (тикер × объявление) с 2022 года: {len(ev)}")
    print(ev.groupby("type").size().to_string())
    res = study(ev)
    res.to_csv(out / "news_events.csv", index=False)
    report(res)
