"""
Скан всех USDT-перпетуалов Bybit за один день: медианный спред, глубина у лучших цен, оборот, число сделок.
Ищем рынки, где спред заметно больше комиссии maker (2 б.п. на сторону) — кандидаты для маркет-мейкинга.

    python -m research.hft.scan --day 2026-09-15 --out <dir>
"""
from __future__ import annotations

import argparse
import gzip
import io
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .build import BYBIT_TRADES, OB_URL, _get, book_seconds


def list_symbols() -> list[str]:
    html = urllib.request.urlopen("https://public.bybit.com/trading/", timeout=60).read().decode()
    return sorted(set(re.findall(r'href="([0-9A-Z]+USDT)/"', html)))


def scan_symbol(sym: str, day: str) -> dict | None:
    tr = _get(BYBIT_TRADES.format(s=sym, d=day))
    if tr is None:
        return None
    t = pd.read_csv(io.BytesIO(gzip.decompress(tr)), usecols=["timestamp", "size", "price"])
    notional = float((t["size"] * t["price"]).sum())
    row = {"symbol": sym, "trades": len(t), "notional_usd": notional}
    if notional < 2e5:                           # совсем мёртвые рынки не интересны
        return row
    ob = None
    for n in (500, 200):
        ob = _get(OB_URL.format(s=sym, d=day, n=n))
        if ob is not None:
            break
    if ob is None:
        return row
    book, _ = book_seconds(ob, day)
    mid = (book["bid"] + book["ask"]) / 2
    spr = (book["ask"] - book["bid"]) / mid * 1e4
    row.update({"spread_med_bps": float(spr.median()), "spread_p25_bps": float(spr.quantile(0.25)),
                "spread_p75_bps": float(spr.quantile(0.75)),
                "top_depth_usd": float(((book["bid_sz1"] * book["bid"] + book["ask_sz1"] * book["ask"]) / 2).median()),
                "depth5_usd": float(((book["bid_dep5"] * book["bid"] + book["ask_dep5"] * book["ask"]) / 2).median()),
                "vol_1m_bps": float(np.log(mid).diff(60).std() * 1e4)})
    return row


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default="2026-09-15")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    syms = list_symbols()
    print(f"символов в архиве сделок: {len(syms)}", flush=True)
    rows = []
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(scan_symbol, s, a.day): s for s in syms}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
                if r:
                    rows.append(r)
            except Exception as e:
                print(f"  {futs[f]}: {e}", flush=True)
            if i % 50 == 0:
                print(f"  {i}/{len(syms)}", flush=True)
    df = pd.DataFrame(rows).sort_values("notional_usd", ascending=False)
    df.to_csv(out / f"bybit_scan_{a.day}.csv", index=False)
    ok = df.dropna(subset=["spread_med_bps"])
    print(f"\nс данными стакана: {len(ok)}")
    print("распределение медианного спреда (б.п.):", ok["spread_med_bps"].describe().round(2).to_dict())
    cand = ok[(ok["spread_med_bps"] >= 6) & (ok["notional_usd"] >= 1e6)].sort_values("notional_usd", ascending=False)
    print(f"\nКандидаты (спред >= 6 б.п., оборот >= $1M/день): {len(cand)}")
    print(cand.round(2).head(60).to_string(index=False))
