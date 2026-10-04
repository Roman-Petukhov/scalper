"""
Ловля прострелов на данных самого Bybit: минутные свечи из всех сделок Bybit (public.bybit.com) для 25 альтов.

Дни на монету: до 30 дней, где на Binance был сигнал (лонг, 4σ, тейк 50%, 60 мин), и 15 случайных дней без
сигнала на Binance (прострелы, которые могли быть только на Bybit). σ считается по предыдущему дню, поэтому
скачиваются оба дня. Автомат тот же (research.spikes.spike_machine), только лонг, без стопа; в отчёт идут
сделки с входом в целевой день. Сравнение с Binance — по тем же монето-дням.

    python -m research.hft.spikes_bybit --jobs research/hft/spikes_bybit_jobs.json --out <dir>
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..spikes import MAKER, TAKER, spike_machine
from .build import BYBIT_TRADES, _get

CONFIGS = [{"m": 4.0, "f": 0.5, "T": 60}, {"m": 6.0, "f": 0.5, "T": 60}]


def minute_bars(blob: bytes) -> pd.DataFrame:
    t = pd.read_csv(io.BytesIO(gzip.decompress(blob)), usecols=["timestamp", "price"])
    t.index = pd.to_datetime(t["timestamp"], unit="s", utc=True)
    b = t["price"].sort_index().resample("1min").ohlc()
    b["close"] = b["close"].ffill()
    for c in ("open", "high", "low"):
        b[c] = b[c].fillna(b["close"])
    return b.dropna()


def coin_day(args) -> list[dict]:
    sym, day, kind = args
    prev = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    parts = []
    for d in (prev, day):
        blob = _get(BYBIT_TRADES.format(s=sym, d=d))
        if blob is None:
            return [{"symbol": sym, "day": day, "kind": kind, "missing": True}]
        parts.append(minute_bars(blob))
    k = pd.concat(parts)
    k = k[~k.index.duplicated()].sort_index()
    sig = (np.log(k["close"]).diff().rolling(1440, min_periods=720).std() * np.sqrt(60)).to_numpy()
    out = [{"symbol": sym, "day": day, "kind": kind, "missing": False}]
    for cfg in CONFIGS:
        i, r, _ = spike_machine(k["open"].to_numpy(), k["high"].to_numpy(), k["low"].to_numpy(),
                                k["close"].to_numpy(), sig, cfg["m"], cfg["f"], cfg["T"], np.inf, MAKER, TAKER, True)
        for ii, rr in zip(i, r):
            if k.index[ii].strftime("%Y-%m-%d") == day:
                out.append({"symbol": sym, "day": day, "kind": kind, "missing": False, "m": cfg["m"],
                            "t": k.index[ii], "r": rr})
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 220)
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(j["symbol"], d, kind) for j in json.load(open(a.jobs))
            for kind, days in (("signal", j["signal_days"]), ("random", j["random_days"])) for d in days]
    rows = []
    with ThreadPoolExecutor(a.workers) as ex:
        for n, res in enumerate(ex.map(coin_day, jobs), 1):
            rows += res
            if n % 100 == 0:
                print(f"  {n}/{len(jobs)} монето-дней", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "spikes_bybit_trades.csv", index=False)
    days = df.drop_duplicates(["symbol", "day"])
    print(f"===== SPIKES on BYBIT: {days['symbol'].nunique()} монет, монето-дней {len(days)}, "
          f"нет данных Bybit {int(days['missing'].sum())} =====")
    tr = df[df["r"].notna()] if "r" in df else pd.DataFrame()
    avail = days[~days["missing"]]
    for m in sorted(tr["m"].unique()) if len(tr) else []:
        print(f"\nm={m:g}σ, только лонг, тейк 50%, 60 мин, без стопа:")
        for kind in ("signal", "random"):
            x = tr[(tr["m"] == m) & (tr["kind"] == kind)]["r"] * 1e4
            nd = int((avail["kind"] == kind).sum())
            if len(x):
                print(f"  дни {'с сигналом Binance' if kind == 'signal' else 'случайные'}: дней {nd}, сделок {len(x)} "
                      f"({len(x) / max(nd, 1):.2f} в день), {x.mean():+.1f} б.п., win {(x > 0).mean():.0%}, "
                      f"медиана {x.median():+.0f}, худшая {x.min():+.0f}")
            else:
                print(f"  дни {kind}: дней {nd}, сделок 0")
        y = tr[tr["m"] == m].copy()
        y["year"] = pd.to_datetime(y["t"]).dt.year
        print("  по годам: " + ", ".join(f"{yr}: {len(g)}, {g['r'].mean() * 1e4:+.0f}" for yr, g in y.groupby("year")))
