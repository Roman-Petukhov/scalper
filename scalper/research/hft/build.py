"""
Посекундные данные микроструктуры для HFT-исследования.

Источники (все публичные, доступны из GitHub Actions):
    стакан Bybit (дельты L2, 500/200 уровней)  quote-saver.bycsi.com/orderbook/linear/<SYM>/<DAY>_<SYM>_ob500.data.zip
    сделки Bybit (тики)                         public.bybit.com/trading/<SYM>/<SYM><DAY>.csv.gz
    сделки Binance USDⓈ-M (aggTrades)          data.binance.vision/data/futures/um/daily/aggTrades/...

Одна строка на секунду UTC. Стакан — состояние на КОНЕЦ секунды (после всех сообщений с ts < конец),
сделки — все сделки внутри секунды. Признаки строго известны к концу своей секунды.

    python -m research.hft.build --symbol SOLUSDT --days 2024-01-08,2024-01-22 --out data_hft
"""
from __future__ import annotations

import argparse
import gzip
import heapq
import io
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import orjson
import pandas as pd

OB_URL = "https://quote-saver.bycsi.com/orderbook/linear/{s}/{d}_{s}_ob{n}.data.zip"
BYBIT_TRADES = "https://public.bybit.com/trading/{s}/{s}{d}.csv.gz"
BINANCE_AGG = "https://data.binance.vision/data/futures/um/daily/aggTrades/{s}/{s}-aggTrades-{d}.zip"
LEVELS = (1, 5, 20)
BAND_BPS = (10, 50)


def _get(url: str, tries: int = 4) -> bytes | None:
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "flow-scalper-research/1.0"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
        except Exception:
            pass
        time.sleep(2 ** k)
    return None


# ---------- стакан ----------

def _book_row(bids: dict, asks: dict) -> list[float]:
    if not bids or not asks:
        return [np.nan] * (4 + 2 * len(LEVELS) + 2 * len(BAND_BPS))
    tb = heapq.nlargest(max(LEVELS), bids.items())
    ta = heapq.nsmallest(max(LEVELS), asks.items())
    b1, a1 = tb[0][0], ta[0][0]
    row = [b1, a1, tb[0][1], ta[0][1]]
    for n in LEVELS:
        row += [sum(q for _, q in tb[:n]), sum(q for _, q in ta[:n])]
    mid = (b1 + a1) / 2
    for bp in BAND_BPS:
        lo, hi = mid * (1 - bp * 1e-4), mid * (1 + bp * 1e-4)
        row += [sum(q for p, q in bids.items() if p >= lo), sum(q for p, q in asks.items() if p <= hi)]
    return row


BOOK_COLS = (["bid", "ask", "bid_sz1", "ask_sz1"] + [f"{s}_dep{n}" for n in LEVELS for s in ("bid", "ask")]
             + [f"{s}_band{bp}" for bp in BAND_BPS for s in ("bid", "ask")])


def book_seconds(blob: bytes, day: str) -> pd.DataFrame:
    """Восстанавливает стакан по снапшотам/дельтам и снимает состояние на конец каждой секунды."""
    t0 = int(pd.Timestamp(day, tz="UTC").timestamp())
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    secs, rows = [], []
    cur_sec = None
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        with z.open(z.namelist()[0]) as fh:
            for line in fh:
                m = orjson.loads(line)
                sec = m["ts"] // 1000
                if cur_sec is not None and sec != cur_sec:
                    secs.append(cur_sec)
                    rows.append(_book_row(bids, asks))
                cur_sec = sec
                d = m["data"]
                if m["type"] == "snapshot":
                    bids = {float(p): float(q) for p, q in d["b"]}
                    asks = {float(p): float(q) for p, q in d["a"]}
                    continue
                for p, q in d["b"]:
                    fp, fq = float(p), float(q)
                    if fq == 0.0:
                        bids.pop(fp, None)
                    else:
                        bids[fp] = fq
                for p, q in d["a"]:
                    fp, fq = float(p), float(q)
                    if fq == 0.0:
                        asks.pop(fp, None)
                    else:
                        asks[fp] = fq
    if cur_sec is not None:
        secs.append(cur_sec)
        rows.append(_book_row(bids, asks))
    df = pd.DataFrame(rows, index=pd.Index(secs, name="sec"), columns=BOOK_COLS)
    grid = pd.RangeIndex(t0, t0 + 86400, name="sec")
    return df[~df.index.duplicated(keep="last")].reindex(grid).ffill()


# ---------- сделки ----------

def bybit_trades_seconds(blob: bytes, day: str) -> pd.DataFrame:
    t0 = int(pd.Timestamp(day, tz="UTC").timestamp())
    t = pd.read_csv(io.BytesIO(gzip.decompress(blob)), usecols=["timestamp", "side", "size", "price"])
    sec = np.floor(t["timestamp"].to_numpy()).astype("int64")
    buy = t["side"].to_numpy() == "Buy"
    q = t["size"].to_numpy(dtype="float64")
    px = t["price"].to_numpy(dtype="float64")
    g = pd.DataFrame({"sec": sec, "buy_vol": np.where(buy, q, 0.0), "sell_vol": np.where(buy, 0.0, q),
                      "n_trades": 1.0, "notional": q * px, "max_px": px, "min_px": px, "last_px": px})
    a = g.groupby("sec").agg({"buy_vol": "sum", "sell_vol": "sum", "n_trades": "sum", "notional": "sum",
                              "max_px": "max", "min_px": "min", "last_px": "last"})
    # крупнейшая сделка секунды со знаком (+ покупка агрессором)
    g["signed"] = np.where(buy, q, -q)
    g["abs"] = q
    a["max_trade"] = g.sort_values(["sec", "abs"], kind="stable").groupby("sec")["signed"].last()
    grid = pd.RangeIndex(t0, t0 + 86400, name="sec")
    a = a.reindex(grid)
    for c in ("buy_vol", "sell_vol", "n_trades", "notional"):
        a[c] = a[c].fillna(0.0)
    a["last_px"] = a["last_px"].ffill()
    return a


def binance_seconds(blob: bytes, day: str) -> pd.DataFrame:
    t0 = int(pd.Timestamp(day, tz="UTC").timestamp())
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0])
    header = not raw[:1].isdigit()
    t = pd.read_csv(io.BytesIO(raw), header=0 if header else None)
    t.columns = ["agg_id", "price", "qty", "first_id", "last_id", "time", "is_buyer_maker"][: t.shape[1]]
    ts = t["time"].to_numpy(dtype="int64")
    if ts.max() > 10**14:                      # микросекунды
        ts = ts // 1000
    sec = ts // 1000
    bm = t["is_buyer_maker"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
    q = t["qty"].to_numpy(dtype="float64")
    g = pd.DataFrame({"sec": sec, "bn_buy": np.where(bm, 0.0, q), "bn_sell": np.where(bm, q, 0.0),
                      "bn_px": t["price"].to_numpy(dtype="float64")})
    a = g.groupby("sec").agg({"bn_buy": "sum", "bn_sell": "sum", "bn_px": "last"})
    grid = pd.RangeIndex(t0, t0 + 86400, name="sec")
    a = a.reindex(grid)
    a[["bn_buy", "bn_sell"]] = a[["bn_buy", "bn_sell"]].fillna(0.0)
    a["bn_px"] = a["bn_px"].ffill()
    return a


def build_day(symbol: str, day: str) -> pd.DataFrame | None:
    t_start = time.time()
    ob = None
    for n in (500, 200):
        ob = _get(OB_URL.format(s=symbol, d=day, n=n))
        if ob is not None:
            break
    tr = _get(BYBIT_TRADES.format(s=symbol, d=day))
    bn = _get(BINANCE_AGG.format(s=symbol, d=day))
    if ob is None or tr is None:
        print(f"  {symbol} {day}: нет данных (стакан={ob is not None}, сделки={tr is not None})", flush=True)
        return None
    t_dl = time.time() - t_start
    df = pd.concat([book_seconds(ob, day), bybit_trades_seconds(tr, day)], axis=1)
    if bn is not None:
        df = df.join(binance_seconds(bn, day))
    df.insert(0, "symbol", symbol)
    f32 = [c for c in df.columns if c not in ("symbol", "bid", "ask", "last_px", "max_px", "min_px", "bn_px")]
    df[f32] = df[f32].astype("float32")
    print(f"  {symbol} {day}: ok, загрузка {t_dl:.0f}s, всего {time.time() - t_start:.0f}s, "
          f"стакан {len(ob) / 1e6:.0f} МБ, спред медиана {((df.ask - df.bid) / df.bid * 1e4).median():.2f} б.п.",
          flush=True)
    return df


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--days", required=True, help="через запятую, YYYY-MM-DD")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for day in a.days.split(","):
        p = out / f"{a.symbol}-{day}.parquet"
        if p.exists():
            continue
        df = build_day(a.symbol, day)
        if df is not None:
            df.to_parquet(p)
