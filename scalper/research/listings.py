"""
Событийный тест: анонсы Binance (листинги в споте, делистинги, Monitoring Tag) и реакция перпетуала той же монеты.

Анонсы — публичный CMS Binance (каталоги 48 «New Cryptocurrency Listing», 49 «Latest Binance News», 161 «Delisting»),
время публикации releaseDate с точностью до мс. Монета из заголовка берётся в тест, только если её USDT-перпетуал
торговался на Binance минимум сутки ДО анонса: только по таким монетам бот может войти в момент новости.

Классы (по заголовку, правила заданы до запуска):
    spot_list       листинг в споте: Will List / HODLer Airdrops / Launchpool / Megadrop / Seed Tag / Innovation
    delist          делистинг токена из спота
    futures_delist  делистинг самого перпетуала
    monitoring      Monitoring Tag (биржа предупреждает о риске делистинга)
    other           остальное с найденной монетой (маржа, Earn, пары и т.п.) — для сравнения
Цены — минутные свечи перпетуала (дневные архивы data.binance.vision).
    «мгновенно»     от close последней минуты до анонса до +1 мин / +5 мин / +15 мин / +1 ч / +24 ч
    run-up          от −24 ч до анонса (утечки)
    торговля        вход по open первой минуты после задержки 1 или 5 мин; выход через 1 / 4 / 24 / 72 ч по close;
                    сторона — лонг для spot_list, шорт для delist / futures_delist / monitoring; издержки 12 б.п.
t — кластеризованный по дням анонса; разбивка по годам.

    python -m research.listings --root <binance data> --symbols <все перпетуалы>
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import threading
import time
import urllib.request
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D

CMS = "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query?type=1&catalogId={c}&pageNo={p}&pageSize=50"
CATALOGS = (48, 49, 161)
SINCE = pd.Timestamp("2022-01-01", tz="UTC")
UNTIL = pd.Timestamp("2026-10-01", tz="UTC")
COST = 12e-4
NOT_TICKERS = {"USDT", "USDC", "FDUSD", "BUSD", "TUSD", "TRY", "EUR", "BRL", "USD", "UAH", "JPY", "BNB", "BTC", "ETH",
               "UTC", "API", "NFT", "VIP", "APR", "APY", "KYC", "FAQ", "AMA", "CEO", "USDⓈ", "P2P", "OTC", "ID"}
SIDE = {"spot_list": 1, "delist": -1, "futures_delist": -1, "monitoring": -1, "other": 1}
DELAYS = (1, 5)
HOLDS = {"1h": 60, "4h": 240, "24h": 1440, "72h": 4320}
INSTANT = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "24h": 1440}


# ---------- анонсы ----------

def _json(url: str) -> dict:
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"не удалось получить {url}")


def fetch_announcements() -> pd.DataFrame:
    rows = []
    for c in CATALOGS:
        p = 1
        while True:
            cats = (_json(CMS.format(c=c, p=p)).get("data") or {}).get("catalogs") or []
            arts = cats[0]["articles"] if cats else []
            if not arts:
                print(f"  каталог {c}: страниц {p - 1}", flush=True)
                break
            rows += [{"catalog": c, "id": a["id"], "title": a["title"], "ts": a["releaseDate"]} for a in arts]
            if pd.Timestamp(min(a["releaseDate"] for a in arts), unit="ms", tz="UTC") < SINCE:
                break
            p += 1
            time.sleep(0.3)
    df = pd.DataFrame(rows).drop_duplicates("id")
    df["t"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df[(df["t"] >= SINCE) & (df["t"] < UNTIL)].sort_values("t").reset_index(drop=True)


def classify(title: str, catalog: int) -> str:
    t = title.lower()
    if "monitoring tag" in t:
        return "monitoring"
    if catalog == 161 or "delist" in t:
        if "futures" in t or "perpetual" in t:
            return "futures_delist"
        if re.search(r"will delist|delisting of|to delist", t) and not re.search(r"margin|trading pairs|loans|earn", t):
            return "delist"
        return "other"
    if re.search(r"margin|convert|earn|collateral|futures|options|trading pairs|loans|copy|bots|bstocks|stock|"
                 r"vip|pre-market|alpha points", t):
        return "other"
    if re.search(r"will list|hodler airdrops|launchpool|megadrop|seed tag|innovation zone|will add .* on binance spot|"
                 r"available on binance spot|spot listing", t):
        return "spot_list"
    return "other"


def tickers(title: str) -> list[str]:
    cand = set(re.findall(r"\(([A-Z0-9]{2,15})\)", title)) | set(re.findall(r"\b([A-Z0-9]{2,15})\b", title))
    return sorted(c for c in cand if c not in NOT_TICKERS and not c.isdigit())


def events(ann: pd.DataFrame, perps: dict[str, pd.Timestamp]) -> pd.DataFrame:
    rows = []
    for a in ann.itertuples():
        cls = classify(a.title, a.catalog)
        for tk in tickers(a.title):
            for sym in (f"{tk}USDT", f"1000{tk}USDT", tk if tk.endswith("USDT") else ""):
                if sym in perps and perps[sym] <= a.t - pd.Timedelta(days=1):
                    rows.append({"t": a.t, "cls": cls, "symbol": sym, "title": a.title})
                    break
    ev = pd.DataFrame(rows)
    return ev.drop_duplicates(["symbol", "cls", "t"]).reset_index(drop=True)


# ---------- минутные цены ----------

def _day_1m(sym: str, day: str, cache: Path) -> pd.DataFrame | None:
    out = cache / f"{sym}-1m-{day}.parquet"
    miss = out.with_suffix(".missing")
    if out.exists():
        try:
            return pd.read_parquet(out)
        except Exception:
            out.unlink(missing_ok=True)                       # битый файл (например, прерванная запись)
    if miss.exists():
        return None
    try:
        blob = D._get(f"{D.DAILY}/klines/{sym}/1m/{sym}-1m-{day}.zip", strict=True)
    except D.NotFound:
        miss.touch()
        return None
    if blob is None:
        return None
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0]).decode()
    first = raw.split("\n", 1)[0]
    df = pd.read_csv(io.StringIO(raw), header=0 if first.startswith("open_time") else None, usecols=range(6))
    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
    df = df.astype({"open_time": "int64", "open": "float64", "high": "float64", "low": "float64", "close": "float64"})
    tmp = out.with_suffix(f".{threading.get_ident()}.tmp")    # атомарная запись: события делят файлы дней
    df.to_parquet(tmp, index=False)
    tmp.replace(out)
    return df


def minutes(sym: str, t0: pd.Timestamp, cache: Path) -> pd.DataFrame | None:
    days = pd.date_range((t0 - pd.Timedelta(days=1)).floor("D"), (t0 + pd.Timedelta(days=3, hours=2)).floor("D"))
    parts = [p for d in days if (p := _day_1m(sym, d.strftime("%Y-%m-%d"), cache)) is not None]
    if not parts:
        return None
    m = pd.concat(parts).drop_duplicates("open_time").sort_values("open_time")
    m.index = pd.to_datetime(m["open_time"], unit="ms", utc=True)
    return m


def measure(m: pd.DataFrame, t0: pd.Timestamp, side: int) -> dict | None:
    start = m.index
    before = m[start + pd.Timedelta(minutes=1) <= t0]                 # минуты, закрывшиеся до анонса
    if not len(before) or before.index[-1] < t0 - pd.Timedelta(minutes=10):
        return None
    p0 = before["close"].iloc[-1]
    row: dict = {}
    for k, mins in INSTANT.items():
        x = m[start + pd.Timedelta(minutes=1) <= t0 + pd.Timedelta(minutes=mins)]
        row[f"i_{k}"] = (x["close"].iloc[-1] / p0 - 1) * 1e4 if len(x) else np.nan
    pre = m[start + pd.Timedelta(minutes=1) <= t0 - pd.Timedelta(hours=24)]
    row["runup24"] = (p0 / pre["close"].iloc[-1] - 1) * 1e4 if len(pre) else np.nan
    for dl in DELAYS:
        after = m[start >= t0 + pd.Timedelta(minutes=dl)]
        if not len(after):
            continue
        e_t, e_px = after.index[0], after["open"].iloc[0]
        for hk, hm in HOLDS.items():
            x = m[(start + pd.Timedelta(minutes=1) <= e_t + pd.Timedelta(minutes=hm))]
            if x.index[-1] + pd.Timedelta(minutes=1) < e_t + pd.Timedelta(minutes=hm) - pd.Timedelta(minutes=5):
                continue                                               # нет данных до конца удержания
            row[f"d{dl}_{hk}"] = (side * (x["close"].iloc[-1] / e_px - 1) - COST) * 1e4
    return row


# ---------- отчёт ----------

def _t_day(r: pd.Series, t: pd.Series) -> float:
    ok = r.notna()
    r, d = r[ok].to_numpy(dtype="float64"), t[ok].dt.floor("D").to_numpy()
    if len(r) < 5:
        return np.nan
    g = pd.Series(r - r.mean()).groupby(d).sum()
    return r.sum() / np.sqrt((g ** 2).sum()) if (g ** 2).sum() > 0 else np.nan


def report(res: pd.DataFrame) -> None:
    cols_i = [f"i_{k}" for k in INSTANT] + ["runup24"]
    cols_d = [f"d{dl}_{hk}" for dl in DELAYS for hk in HOLDS]
    print("\n=== 1. «Мгновенная» реакция от цены до анонса, б.п. (среднее / медиана), и run-up за сутки до ===")
    rows = []
    for cls, g in res.groupby("cls"):
        rows.append({"класс": cls, "n": len(g), "монет": g.symbol.nunique(),
                     **{c: f"{g[c].mean():+.0f} / {g[c].median():+.0f}" for c in cols_i}})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n=== 2. Торговля после задержки (сторона по классу, после 12 б.п.): средний результат, б.п. (t по дням, win) ===")
    rows = []
    for cls, g in res.groupby("cls"):
        r = {"класс": cls, "n": len(g)}
        for c in cols_d:
            x = g[c].dropna()
            r[c] = f"{x.mean():+.0f} ({_t_day(g[c], g.t):.1f}, {np.mean(x > 0):.0%})" if len(x) >= 5 else "—"
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n=== 3. По годам: вход через 1 мин, выход через 4 ч / 24 ч (средний результат, б.п., n) ===")
    rows = []
    for (cls, y), g in res.groupby(["cls", res.t.dt.year]):
        rows.append({"класс": cls, "год": y, "n": len(g),
                     "4h": f"{g['d1_4h'].mean():+.0f} ({_t_day(g['d1_4h'], g.t):.1f})",
                     "24h": f"{g['d1_24h'].mean():+.0f} ({_t_day(g['d1_24h'], g.t):.1f})",
                     "мгновенно 5m": f"{g['i_5m'].mean():+.0f}"})
    print(pd.DataFrame(rows).to_string(index=False))
    sl = res[res.cls == "spot_list"].copy()
    if len(sl):
        sl["sub"] = np.select([sl.title.str.contains("HODLer", case=False), sl.title.str.contains("Seed Tag", case=False),
                               sl.title.str.contains("Launchpool|Megadrop", case=False)],
                              ["hodler", "seed_tag", "launchpool"], "will_list")
        print("\n=== 4. Листинги в споте по подтипам (вход через 1 мин) ===")
        rows = []
        for s, g in sl.groupby("sub"):
            rows.append({"подтип": s, "n": len(g), "мгновенно 5m": f"{g['i_5m'].mean():+.0f}",
                         **{hk: f"{g[f'd1_{hk}'].mean():+.0f} ({_t_day(g[f'd1_{hk}'], g.t):.1f})" for hk in HOLDS}})
        print(pd.DataFrame(rows).to_string(index=False))
        print("\n  примеры (лучшие и худшие по d1_24h):")
        show = sl.sort_values("d1_24h").dropna(subset=["d1_24h"])
        print(pd.concat([show.head(5), show.tail(5)])[["t", "symbol", "i_5m", "d1_4h", "d1_24h", "title"]]
              .assign(title=lambda d: d.title.str.slice(0, 70)).round(0).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 40)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    root = Path(a.root)
    D.set_host("cdn")
    ann = fetch_announcements()
    print(f"анонсов с 2022 года: {len(ann)} (" + ", ".join(f"каталог {c}: {n}" for c, n in ann.catalog.value_counts().items()) + ")")
    perps = {}
    for s in a.symbols.split(","):
        p = root / f"{s}-1h.parquet"
        if p.exists():
            perps[s] = pd.Timestamp(pd.read_parquet(p, columns=["open_time"])["open_time"].min(), unit="ms", tz="UTC")
    ev = events(ann, perps)
    print(f"событий с монетой, у которой уже был перпетуал: {len(ev)}; по классам: " +
          ", ".join(f"{k} {v}" for k, v in ev.cls.value_counts().items()), flush=True)
    cache = root / "cache" / "1m"
    cache.mkdir(parents=True, exist_ok=True)

    def one(r) -> dict | None:
        m = minutes(r.symbol, r.t, cache)
        if m is None:
            return None
        x = measure(m, r.t, SIDE[r.cls])
        return None if x is None else {"t": r.t, "cls": r.cls, "symbol": r.symbol, "title": r.title, **x}

    with ThreadPoolExecutor(a.workers) as ex:
        res = [x for x in ex.map(one, ev.itertuples()) if x is not None]
    res = pd.DataFrame(res)
    print(f"===== LISTINGS: событий с минутными данными {len(res)} =====")
    report(res)
