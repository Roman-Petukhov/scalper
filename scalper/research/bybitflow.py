"""Правдивы ли агрессоры Binance: та же доля рыночных покупок на свечах сигналов бота 4h по сделкам Bybit.

Пробои линий 4h со всеми фильтрами бота, кроме агрессоров (research/topcrowd.py coin_rows: оборот, дневной тренд,
закрытие в своей половине, стоп за свингом, ретест; сделка как у бота), с 2023 года. Для каждого — доля рыночных
покупок в свече пробоя (в сторону сделки) на Binance (свечи, taker buy volume) и на Bybit (архив сделок
public.bybit.com/trading: сторона каждой сделки — сторона агрессора). Независимая биржа: если поток Binance накручен,
он разойдётся с Bybit, а фильтр по Bybit не будет работать.
Выводы (заданы до запуска):
    данные согласованы   корреляция долей Binance и Bybit ≥ 0.7
    фильтр настоящий      по потоку Bybit сделки с долей ≥ 55% лучше, чем < 55%, во всех периодах IS (с 2023) / VAL /
                          HO / 2026, где в каждой группе ≥ 10 сделок
Плюс разбор несогласных: Binance ≥ 55%, а Bybit нет (и наоборот) — какой из двух потоков «прав».

    python -m research.bybitflow collect --symbols <1h-qualified>
    python -m research.bybitflow report
"""
from __future__ import annotations

import argparse
import gzip
import io
import sys
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import adv30
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _cell
from .tline import tf_frame
from .topcrowd import coin_rows

URL = "https://public.bybit.com/trading/{s}/{s}{d}.csv.gz"
SINCE = pd.Timestamp("2023-01-01", tz="UTC")
MAX_BYTES = 200 * 2 ** 20          # дневной файл больше — пропускаем (BTC / ETH: десятки миллионов сделок)
THR = 0.55
_warned: set[str] = set()


def _get(url: str) -> bytes | None:
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                n = int(r.headers.get("Content-Length") or 0)
                if n > MAX_BYTES:
                    return None
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
        except Exception:
            pass
        time.sleep(2 ** attempt)
    return None


def parse_trades(blob: bytes) -> pd.DataFrame | None:
    """Сделки Bybit: время (с), сторона агрессора (Buy / Sell), объём в монетах. None — формат не узнали."""
    raw = gzip.decompress(blob)
    df = pd.read_csv(io.BytesIO(raw))
    cols = {c.lower(): c for c in df.columns}
    if not {"timestamp", "side", "size"} <= set(cols):
        key = ",".join(df.columns[:12])
        if key not in _warned:
            _warned.add(key)
            print(f"  bybitflow: неожиданные колонки: {key}", flush=True)
        return None
    t = pd.to_numeric(df[cols["timestamp"]], errors="coerce")
    unit = "ms" if t.median() > 1e11 else "s"
    return pd.DataFrame({"t": pd.to_datetime(t, unit=unit, utc=True),
                         "buy": df[cols["side"]].astype(str).str.lower().str.startswith("b"),
                         "size": pd.to_numeric(df[cols["size"]], errors="coerce")}).dropna()


def buy_share(trades: pd.DataFrame, start: pd.Timestamp, hours: int = 4) -> float:
    m = trades[(trades.t >= start) & (trades.t < start + pd.Timedelta(hours=hours))]
    v = m["size"].sum()
    return float(m.loc[m.buy, "size"].sum() / v) if v > 0 else np.nan


def collect(root: Path, syms: list[str]) -> None:
    rows, files, miss = [], 0, 0
    for s in mine(syms):
        if s in ("BTCUSDT", "ETHUSDT"):
            continue
        try:
            d = tf_frame(root, s, "4h")
            if d is None or len(d) < 500:
                continue
            liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
            r = [x for x in coin_rows(d, liq, s, "4h", np.zeros(len(d))) if x["t"] >= SINCE]
            if not r:
                continue
            bn = (d["taker_buy_volume"] / d["volume"].replace(0, np.nan))
            cache: dict[str, pd.DataFrame | None] = {}
            for x in r:
                day = x["t"].strftime("%Y-%m-%d")
                if day not in cache:
                    blob = _get(URL.format(s=s, d=day))
                    cache[day] = parse_trades(blob) if blob else None
                    files += blob is not None
                    miss += blob is None
                tr = cache[day]
                by = buy_share(tr, x["t"]) if tr is not None else np.nan
                b = float(bn.loc[x["t"]])
                rows.append({"symbol": s, "t": x["t"], "side": x["side"], "R": x["R"],
                             "binance": b if x["side"] > 0 else 1 - b,
                             "bybit": by if x["side"] > 0 else 1 - by})
        except Exception as e:
            print(f"  bybitflow {s}: пропуск ({e})", flush=True)
    print(f"  bybitflow: пробоев {len(rows)}, файлов Bybit {files}, нет файла / слишком большой {miss}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("bybitflow"), index=False)


def report() -> None:
    parts = all_parts("bybitflow")
    print("\n===== BYBITFLOW: доля агрессоров в свече пробоя 4h — Binance против Bybit (независимая биржа) =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df.dropna(subset=["R", "binance"])
    z = df.dropna(subset=["bybit"])
    print(f"пробоев с 2023: {len(df)}, с данными Bybit {len(z)} ({len(z) / max(len(df), 1):.0%}), монет {z.symbol.nunique()}")
    if len(z) < 30:
        print("мало данных Bybit")
        return
    corr = z[["binance", "bybit"]].corr().iloc[0, 1]
    diff = (z.binance - z.bybit).abs()
    both = ((z.binance >= THR) == (z.bybit >= THR)).mean()
    print(f"корреляция долей {corr:.2f}; |разница| медиана {diff.median():.1%}, p90 {diff.quantile(.9):.1%}; "
          f"средние: Binance {z.binance.mean():.1%}, Bybit {z.bybit.mean():.1%}; по порогу 55% согласны в {both:.0%} "
          f"→ данные {'согласованы' if corr >= 0.7 else 'НЕ согласованы'} (порог 0.7)")
    per = {k: v for k, v in {**PER_ALL, "2026": ("2026-01-01", "2026-10-01")}.items() if k not in ("2020", "2021")}
    for src in ("binance", "bybit"):
        ok = []
        print(f"\n  фильтр по потоку {src}: доля ≥ 55% против < 55%")
        for p, (a, b) in per.items():
            m = z[(z.t >= a) & (z.t < b)]
            hi, lo = m[m[src] >= THR], m[m[src] < THR]
            if len(hi) >= 10 and len(lo) >= 10:
                ok.append(hi.R.mean() > lo.R.mean())
            print(f"    {p}: ≥ 55% {_cell(hi)} | < 55% {_cell(lo)}")
        print(f"    → лучше во всех периодах: {'да' if ok and all(ok) else f'нет ({sum(ok)} из {len(ok)})'}")
    print("\n  несогласные (все периоды с 2023):")
    for nm, m in (("оба ≥ 55%", z[(z.binance >= THR) & (z.bybit >= THR)]),
                  ("только Binance ≥ 55%", z[(z.binance >= THR) & (z.bybit < THR)]),
                  ("только Bybit ≥ 55%", z[(z.binance < THR) & (z.bybit >= THR)]),
                  ("оба < 55%", z[(z.binance < THR) & (z.bybit < THR)])):
        print(f"    {nm}: {_cell(m)}, всего {m.R.sum():+.1f}R")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x])
    else:
        report()
