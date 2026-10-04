"""
Проверка событийных шортов по анонсам Binance (research.listings) перед тем, как брать их в бота:
    1) funding — результат с учётом funding перпетуала Binance за время удержания (лонг платит положительную
       ставку, шорт получает; после плохих новостей ставка обычно отрицательная — шорт платит);
    2) Bybit — та же сделка по минутным свечам Bybit, собранным из всех сделок (public.bybit.com), funding — Binance
       (архива ставок Bybit с раннера нет; ставки бирж близки);
    3) шорт после листинга в споте (spot_fade) — гипотеза, сформулированная ПОСЛЕ просмотра результатов первого
       теста: вход через 1 / 5 / 60 / 240 мин после анонса, поэтому значимость здесь надо делить на перебор.
Классы: delist, monitoring — шорт; spot_list — лонг (для сравнения) и spot_fade — шорт по тем же событиям.
Вход по open минуты после задержки, выход по close через 4 / 24 / 72 ч; издержки 12 б.п. на круг.

    python -m research.listings2 collect --root <binance data> --symbols <все перпетуалы>   (по частям)
    python -m research.listings2 report
"""
from __future__ import annotations

import argparse
import gzip
import io
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .hft.build import BYBIT_TRADES, _get as _get_raw
from .listings import COST, events, fetch_announcements, minutes
from .shard import all_parts, mine, part_path

CLASSES = {"delist": -1, "monitoring": -1, "spot_list": 1, "spot_fade": -1}
DELAYS = (1, 5, 60, 240)
HOLDS = {"4h": 240, "24h": 1440, "72h": 4320}


def funding(root: Path, sym: str) -> pd.Series:
    p = root / f"{sym}-funding.parquet"
    if not p.exists():
        return pd.Series(dtype="float64")
    f = pd.read_parquet(p)
    return pd.Series(f["rate"].to_numpy(dtype="float64"), index=pd.to_datetime(f["ts"], unit="ms", utc=True)).sort_index()


def bybit_minutes(sym: str, t0: pd.Timestamp, cache: Path) -> pd.DataFrame | None:
    days = pd.date_range((t0 - pd.Timedelta(hours=1)).floor("D"), (t0 + pd.Timedelta(days=3, hours=5)).floor("D"))
    parts = []
    for d in days:
        out = cache / f"{sym}-by1m-{d:%Y-%m-%d}.parquet"
        miss = out.with_suffix(".missing")
        if out.exists():
            try:
                parts.append(pd.read_parquet(out))
                continue
            except Exception:
                out.unlink(missing_ok=True)
        if miss.exists():
            continue
        blob = _get_raw(BYBIT_TRADES.format(s=sym, d=f"{d:%Y-%m-%d}"))
        if blob is None:
            miss.touch()
            continue
        t = pd.read_csv(io.BytesIO(gzip.decompress(blob)), usecols=["timestamp", "price"])
        t.index = pd.to_datetime(t["timestamp"], unit="s", utc=True)
        b = t["price"].sort_index().resample("1min").ohlc()
        b["close"] = b["close"].ffill()
        for c in ("open", "high", "low"):
            b[c] = b[c].fillna(b["close"])
        b = b.dropna()
        tmp = out.with_suffix(".tmp")
        b.to_parquet(tmp)
        tmp.replace(out)
        parts.append(b)
    if not parts:
        return None
    m = pd.concat(parts)
    return m[~m.index.duplicated()].sort_index()


def trades(m: pd.DataFrame, t0: pd.Timestamp, side: int, fund: pd.Series, prefix: str) -> dict:
    """Результаты сделок по минутным свечам m (индекс — начало минуты): g — без funding, d — с funding, б.п."""
    row: dict = {}
    start = m.index
    for dl in DELAYS:
        after = m[start >= t0 + pd.Timedelta(minutes=dl)]
        if not len(after) or after.index[0] > t0 + pd.Timedelta(minutes=dl + 10):
            continue
        e_t, e_px = after.index[0], after["open"].iloc[0]
        for hk, hm in HOLDS.items():
            end = e_t + pd.Timedelta(minutes=hm)
            x = m[start + pd.Timedelta(minutes=1) <= end]
            if x.index[-1] + pd.Timedelta(minutes=1) < end - pd.Timedelta(minutes=5):
                continue
            gross = side * (x["close"].iloc[-1] / e_px - 1) - COST
            paid = fund[(fund.index > e_t) & (fund.index <= end)].sum() if len(fund) else 0.0
            row[f"{prefix}g{dl}_{hk}"] = gross * 1e4
            row[f"{prefix}d{dl}_{hk}"] = (gross - side * paid) * 1e4
    return row


def collect(root: Path, syms: list[str], workers: int) -> None:
    ann = fetch_announcements()
    perps = {}
    for s in syms:
        p = root / f"{s}-1h.parquet"
        if p.exists():
            perps[s] = pd.Timestamp(pd.read_parquet(p, columns=["open_time"])["open_time"].min(), unit="ms", tz="UTC")
    ev = events(ann, perps)
    ev = ev[ev["cls"].isin(["delist", "monitoring", "spot_list"])]
    fade = ev[ev["cls"] == "spot_list"].assign(cls="spot_fade")
    ev = pd.concat([ev, fade], ignore_index=True)
    ev = ev[ev["symbol"].isin(set(mine(sorted(ev["symbol"].unique()))))]
    print(f"анонсов {len(ann)}, событий в части {len(ev)}: " +
          ", ".join(f"{k} {v}" for k, v in ev["cls"].value_counts().items()), flush=True)
    c1, cb = root / "cache" / "1m", root / "cache" / "by1m"
    c1.mkdir(parents=True, exist_ok=True)
    cb.mkdir(parents=True, exist_ok=True)

    def one(r) -> dict | None:
        try:
            side = CLASSES[r.cls]
            fund = funding(root, r.symbol)
            row = {"t": r.t, "cls": r.cls, "symbol": r.symbol, "title": r.title}
            m = minutes(r.symbol, r.t, c1)
            if m is not None:
                row.update(trades(m, r.t, side, fund, ""))
            b = bybit_minutes(r.symbol, r.t, cb)
            row["bybit"] = b is not None
            if b is not None:
                row.update(trades(b, r.t, side, fund, "by_"))
            return row
        except Exception as e:
            print(f"  {r.symbol} {r.t}: пропуск ({e})", flush=True)
            return None

    # одна монета — один поток: файлы дней в кеше общие для событий одной монеты
    groups = [g for _, g in ev.groupby("symbol")]
    with ThreadPoolExecutor(workers) as ex:
        res = [x for rows in ex.map(lambda g: [one(r) for r in g.itertuples()], groups) for x in rows if x is not None]
    if res:
        pd.DataFrame(res).to_parquet(part_path("listings2"), index=False)


def _cell(x: pd.Series, t: pd.Series) -> str:
    ok = x.notna()
    r, d = x[ok].to_numpy(dtype="float64"), t[ok].dt.floor("D").to_numpy()
    if len(r) < 5:
        return f"n={len(r)}"
    g = pd.Series(r - r.mean()).groupby(d).sum()
    td = r.sum() / np.sqrt((g ** 2).sum()) if (g ** 2).sum() > 0 else np.nan
    return f"{r.mean():+.0f} ({td:.1f}, {np.mean(r > 0):.0%}, n={len(r)})"


def report() -> None:
    parts = all_parts("listings2")
    if not parts:
        print("частей нет")
        return
    res = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    res["t"] = pd.to_datetime(res["t"], utc=True)
    print(f"===== LISTINGS2: событий {len(res)} (частей {len(parts)}); с данными Bybit: "
          f"{int(res['bybit'].sum())} =====")
    print("ячейка: средний результат, б.п. (t по дням, доля прибыльных, n)")
    for cls in CLASSES:
        g = res[res["cls"] == cls]
        if not len(g):
            continue
        print(f"\n=== {cls} ({'шорт' if CLASSES[cls] < 0 else 'лонг'}), событий {len(g)}, на Bybit {int(g['bybit'].sum())} ===")
        rows = []
        for dl in DELAYS:
            for hk in HOLDS:
                rows.append({"вход через": f"{dl} мин", "держим": hk,
                             "Binance без funding": _cell(g.get(f"g{dl}_{hk}", pd.Series(dtype=float)), g["t"]),
                             "Binance с funding": _cell(g.get(f"d{dl}_{hk}", pd.Series(dtype=float)), g["t"]),
                             "Bybit с funding": _cell(g.get(f"by_d{dl}_{hk}", pd.Series(dtype=float)), g["t"])})
        print(pd.DataFrame(rows).to_string(index=False))
        col = "by_d5_24h" if cls != "spot_fade" else "by_d60_72h"
        if col in g:
            y = g.groupby(g["t"].dt.year)[col].agg(["size", "mean"]).round(0)
            print(f"  по годам ({col}): " + ", ".join(f"{k}: {v['mean']:+.0f} (n={int(v['size'])})"
                                                   for k, v in y.iterrows()))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 320)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    D.set_host("cdn")
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [s for s in a.symbols.split(",") if s], a.workers)
    else:
        report()
