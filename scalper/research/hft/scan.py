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

from .build import BYBIT_TRADES, OB_URL, _asof, _get, book_seconds


def list_symbols() -> list[str]:
    html = urllib.request.urlopen("https://public.bybit.com/trading/", timeout=60).read().decode()
    return sorted(set(re.findall(r'href="([0-9A-Z]+USDT)/"', html)))


def scan_symbol(sym: str, day: str) -> dict | None:
    tr = _get(BYBIT_TRADES.format(s=sym, d=day))
    if tr is None:
        return None
    t = pd.read_csv(io.BytesIO(gzip.decompress(tr)), usecols=["timestamp", "side", "size", "price"])
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
    book, tob = book_seconds(ob, day)
    mid = (book["bid"] + book["ask"]) / 2
    spr = (book["ask"] - book["bid"]) / mid * 1e4
    row.update({"spread_med_bps": float(spr.median()), "spread_p25_bps": float(spr.quantile(0.25)),
                "spread_p75_bps": float(spr.quantile(0.75)),
                "top_depth_usd": float(((book["bid_sz1"] * book["bid"] + book["ask_sz1"] * book["ask"]) / 2).median()),
                "depth5_usd": float(((book["bid_dep5"] * book["bid"] + book["ask_dep5"] * book["ask"]) / 2).median()),
                "vol_1m_bps": float(np.log(mid).diff(60).std() * 1e4)})
    # реализованный спред для пассивной стороны: maker продал покупателю (или купил у продавца) по цене сделки,
    # а через h секунд позиция стоит по mid. Взвешено по объёму, в б.п. от цены сделки.
    ok = np.isfinite(tob[:, 1]) & np.isfinite(tob[:, 2])
    tt, tmid = tob[ok, 0], (tob[ok, 1] + tob[ok, 2]) / 2
    ts = t["timestamp"].to_numpy(dtype="float64") * 1000.0
    px = t["price"].to_numpy(dtype="float64")
    q = (t["size"] * t["price"]).to_numpy(dtype="float64")
    taker_buy = np.where(t["side"].to_numpy() == "Buy", 1.0, -1.0)
    for h in (5, 30, 60):
        m = _asof(tt, tmid, ts + h * 1000.0)
        rs = taker_buy * (px - m) / px * 1e4          # >0: maker заработал до комиссии
        good = np.isfinite(rs)
        row[f"rspread_{h}s_bps"] = float(np.average(rs[good], weights=q[good])) if good.any() else np.nan
    m0 = _asof(tt, tmid, ts)
    es = taker_buy * (px - m0) / px * 1e4             # эффективный полуспред, который платит taker
    good = np.isfinite(es)
    row["eff_half_spread_bps"] = float(np.average(es[good], weights=q[good])) if good.any() else np.nan
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
    cand = ok[ok["notional_usd"] >= 1e6].sort_values("rspread_30s_bps", ascending=False)
    print("\nРеализованный спред maker (б.п. на исполнение, до комиссии 2 б.п.), оборот >= $1M/день, топ-60:")
    print(cand.round(2).head(60).to_string(index=False))
    print("\nДоля рынков, где реализованный спред 30с > 2 б.п. (комиссия maker):",
          f"{(cand['rspread_30s_bps'] > 2).mean():.0%} из {len(cand)}")
