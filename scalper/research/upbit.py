"""
Листинги на Upbit (KRW-рынок) и перпетуал той же монеты на Binance: что делает цена до и после открытия торгов.

Страница анонсов Upbit с серверов GitHub закрыта (Cloudflare), поэтому событие — момент ОТКРЫТИЯ торгов на Upbit:
первая минутная свеча KRW-рынка (публичный API api.upbit.com). Бот в реальном времени видит это так же — новый рынок
в /v1/market/all и первые сделки. Анонс обычно раньше открытия, поэтому основной всплеск может прийтись на время
до t_open — он виден как run-up. Рынки, уже снятые с Upbit, в список не попадают (лёгкая ошибка выжившего).
Монета — USDT-перпетуал Binance, торговавшийся >= 1 дня до t_open.
    run-up:   цена перпа от t_open - 24 ч и от t_open - 6 ч до t_open
    сделки:   лонг и шорт перпа, вход через 1 / 5 / 60 / 240 мин после t_open, выход через 4 / 24 / 72 ч
              (research.listings2.trades: издержки 12 б.п., funding Binance; цены Binance 1m)

    python -m research.upbit collect --symbols <все перпетуалы>   (по частям)
    python -m research.upbit report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .listings import minutes
from .listings2 import DELAYS, HOLDS, _cell, funding, trades
from .shard import all_parts, mine, part_path

API = "https://api.upbit.com/v1"
SINCE = pd.Timestamp("2022-01-01", tz="UTC")
UNTIL = pd.Timestamp("2026-09-25", tz="UTC")


def _get(url: str, cache: Path | None = None) -> list | dict | None:
    f = None
    if cache is not None:
        f = cache / (url.split("v1/", 1)[1].replace("/", "_").replace("?", "_").replace("&", "_").replace(":", "")[:200] + ".json")
        if f.exists():
            return json.loads(f.read_text())
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
            time.sleep(0.15)                                    # лимит Upbit ~10 запросов/с
            if f is not None:
                f.write_text(json.dumps(data))
            return data
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(2 + 2 * attempt)
                continue
            if 400 <= e.code < 500:
                return None
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return None


def first_day(market: str, cache: Path) -> pd.Timestamp | None:
    to = None
    earliest = None
    for _ in range(20):
        url = f"{API}/candles/days?market={market}&count=200" + (f"&to={to}" if to else "")
        rows = _get(url, cache if to else None)
        if not rows:
            break
        earliest = min(pd.Timestamp(r["candle_date_time_utc"], tz="UTC") for r in rows)
        if len(rows) < 200:
            break
        to = earliest.strftime("%Y-%m-%dT%H:%M:%SZ")
    return earliest


def first_minute(market: str, day: pd.Timestamp, cache: Path) -> pd.Timestamp | None:
    to = (day + pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    earliest = None
    for _ in range(10):
        rows = _get(f"{API}/candles/minutes/1?market={market}&count=200&to={to}", cache)
        if not rows:
            break
        earliest = min(pd.Timestamp(r["candle_date_time_utc"], tz="UTC") for r in rows)
        if len(rows) < 200 or earliest < day:
            break
        to = earliest.strftime("%Y-%m-%dT%H:%M:%SZ")
    if earliest is not None and earliest.normalize() < day:
        return None                                            # рынок существовал раньше — это не листинг
    return earliest


def collect(root: Path, syms: list[str]) -> None:
    cache = root / "cache" / "upbit"
    cache.mkdir(parents=True, exist_ok=True)
    markets = [m["market"] for m in (_get(f"{API}/market/all") or []) if m["market"].startswith("KRW-")]
    have = set(syms)

    def to_sym(m: str) -> str | None:
        tk = m.split("-", 1)[1]
        for s in (f"{tk}USDT", f"1000{tk}USDT"):
            if s in have:
                return s
        return None

    pairs = [(m, s) for m in markets if (s := to_sym(m))]
    pairs = [p for p in pairs if p[1] in set(mine(sorted({s for _, s in pairs})))]
    print(f"KRW-рынков {len(markets)}, с перпетуалом Binance в части: {len(pairs)}", flush=True)
    c1 = root / "cache" / "1m"
    c1.mkdir(parents=True, exist_ok=True)
    rows = []
    for m, s in pairs:
        try:
            d0 = first_day(m, cache)
            if d0 is None or not (SINCE <= d0 < UNTIL):
                continue
            t0 = first_minute(m, d0, cache)
            p = root / f"{s}-1h.parquet"
            if t0 is None or not p.exists():
                continue
            perp_start = pd.Timestamp(pd.read_parquet(p, columns=["open_time"])["open_time"].min(), unit="ms", tz="UTC")
            if perp_start > t0 - pd.Timedelta(days=1):
                continue
            mm = minutes(s, t0, c1)
            if mm is None:
                continue
            fund = funding(root, s)
            row = {"market": m, "symbol": s, "t": t0}
            before = mm[mm.index + pd.Timedelta(minutes=1) <= t0]
            for k, h in (("runup24", 24), ("runup6", 6)):
                x = mm[mm.index + pd.Timedelta(minutes=1) <= t0 - pd.Timedelta(hours=h)]
                row[k] = (before["close"].iloc[-1] / x["close"].iloc[-1] - 1) * 1e4 if len(x) and len(before) else np.nan
            for side, pref in ((1, "L_"), (-1, "S_")):
                row.update(trades(mm, t0, side, fund, pref))
            rows.append(row)
        except Exception as e:
            print(f"  {m}: пропуск ({e})", flush=True)
    print(f"  событий с ценами: {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("upbit"), index=False)


def report() -> None:
    parts = all_parts("upbit")
    if not parts:
        print("частей нет")
        return
    res = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    res["t"] = pd.to_datetime(res["t"], utc=True)
    print(f"===== UPBIT: листингов KRW с перпетуалом Binance {len(res)}, частей {len(parts)} =====")
    print(f"run-up до открытия торгов, б.п.: 24 ч {res['runup24'].mean():+.0f} / медиана {res['runup24'].median():+.0f}; "
          f"6 ч {res['runup6'].mean():+.0f} / медиана {res['runup6'].median():+.0f}")
    print("ячейка: среднее, б.п. (t по дням, доля прибыльных, n); с funding")
    rows = []
    for dl in DELAYS:
        for hk in HOLDS:
            rows.append({"вход через": f"{dl} мин", "держим": hk,
                         "лонг": _cell(res.get(f"L_d{dl}_{hk}", pd.Series(dtype=float)), res["t"]),
                         "шорт": _cell(res.get(f"S_d{dl}_{hk}", pd.Series(dtype=float)), res["t"])})
    print(pd.DataFrame(rows).to_string(index=False))
    for col in ("S_d60_24h", "S_d60_72h"):
        if col in res:
            y = res.groupby(res["t"].dt.year)[col].agg(["size", "mean"])
            print(f"  по годам ({col}): " + ", ".join(f"{k}: {v['mean']:+.0f} (n={int(v['size'])})" for k, v in y.iterrows()))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 320)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x])
    else:
        report()
