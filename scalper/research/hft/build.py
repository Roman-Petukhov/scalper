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


def book_seconds(blob: bytes, day: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Восстанавливает стакан по снапшотам/дельтам. Возвращает состояние на конец каждой секунды и ленту
    лучших цен (ts_ms, bid, ask) после каждого сообщения — для событийных тестов с миллисекундами."""
    t0 = int(pd.Timestamp(day, tz="UTC").timestamp())
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    bb, ba = -np.inf, np.inf
    secs, rows = [], []
    tob_ts, tob_b, tob_a = [], [], []
    cur_sec = None
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        with z.open(z.namelist()[0]) as fh:
            for line in fh:
                m = orjson.loads(line)
                ts = m["ts"]
                sec = ts // 1000
                if cur_sec is not None and sec != cur_sec:
                    secs.append(cur_sec)
                    rows.append(_book_row(bids, asks))
                cur_sec = sec
                d = m["data"]
                if m["type"] == "snapshot":
                    bids = {float(p): float(q) for p, q in d["b"]}
                    asks = {float(p): float(q) for p, q in d["a"]}
                    bb = max(bids) if bids else -np.inf
                    ba = min(asks) if asks else np.inf
                else:
                    for p, q in d["b"]:
                        fp, fq = float(p), float(q)
                        if fq == 0.0:
                            if bids.pop(fp, None) is not None and fp == bb:
                                bb = max(bids) if bids else -np.inf
                        else:
                            bids[fp] = fq
                            if fp > bb:
                                bb = fp
                    for p, q in d["a"]:
                        fp, fq = float(p), float(q)
                        if fq == 0.0:
                            if asks.pop(fp, None) is not None and fp == ba:
                                ba = min(asks) if asks else np.inf
                        else:
                            asks[fp] = fq
                            if fp < ba:
                                ba = fp
                tob_ts.append(ts)
                tob_b.append(bb)
                tob_a.append(ba)
    if cur_sec is not None:
        secs.append(cur_sec)
        rows.append(_book_row(bids, asks))
    df = pd.DataFrame(rows, index=pd.Index(secs, name="sec"), columns=BOOK_COLS)
    grid = pd.RangeIndex(t0, t0 + 86400, name="sec")
    tob = np.column_stack([np.asarray(tob_ts, dtype="float64"), np.asarray(tob_b), np.asarray(tob_a)])
    return df[~df.index.duplicated(keep="last")].reindex(grid).ffill(), tob


# ---------- сделки ----------

def read_bybit_trades(blob: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(gzip.decompress(blob)), usecols=["timestamp", "side", "size", "price"])


def bybit_trades_seconds(t: pd.DataFrame, day: str) -> pd.DataFrame:
    t0 = int(pd.Timestamp(day, tz="UTC").timestamp())
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


def read_binance(blob: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0])
    header = not raw[:1].isdigit()
    t = pd.read_csv(io.BytesIO(raw), header=0 if header else None)
    t.columns = ["agg_id", "price", "qty", "first_id", "last_id", "time", "is_buyer_maker"][: t.shape[1]]
    ts = t["time"].to_numpy(dtype="int64")
    if ts.max() > 10**14:                      # микросекунды
        ts = ts // 1000
    t["ts_ms"] = ts
    return t


def binance_seconds(t: pd.DataFrame, day: str) -> pd.DataFrame:
    t0 = int(pd.Timestamp(day, tz="UTC").timestamp())
    sec = t["ts_ms"].to_numpy() // 1000
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


# ---------- события «Binance сдвинулся, Bybit ещё нет» с миллисекундами ----------

LATENCIES_MS = (0, 50, 100, 200, 500, 1000)
EXIT_S = (1, 5, 30, 60, 300)
GAP_MIN_BPS = 3.0
COOLDOWN_MS = 2000


def _asof(ts_sorted: np.ndarray, vals: np.ndarray, q: np.ndarray) -> np.ndarray:
    i = np.searchsorted(ts_sorted, q, side="right") - 1
    out = np.where(i >= 0, vals[np.clip(i, 0, None)], np.nan)
    return out


def leadlag_events(bn: pd.DataFrame, tob: np.ndarray, bt: pd.DataFrame) -> pd.DataFrame:
    """Событие: за последнюю секунду Binance сдвинулся сильнее Bybit на >= GAP_MIN_BPS.
    Для каждой задержки L записываем цены, по которым реально торговали на Bybit после tau+L:
    первая покупка агрессора (для входа в лонг) и первая продажа (для шорта) в течение секунды,
    а также bid/ask по стакану. Выход — bid/ask стакана через h секунд."""
    tb = bn["ts_ms"].to_numpy(dtype="float64")
    pb = bn["price"].to_numpy(dtype="float64")
    order = np.argsort(tb, kind="stable")
    tb, pb = tb[order], pb[order]
    tt, bbid, bask = tob[:, 0], tob[:, 1], tob[:, 2]
    ok = np.isfinite(bbid) & np.isfinite(bask)
    tt, bbid, bask = tt[ok], bbid[ok], bask[ok]
    bmid = (bbid + bask) / 2
    # Binance и Bybit доходности за последнюю секунду на моменте каждой сделки Binance
    j = np.searchsorted(tb, tb - 1000.0, side="right") - 1
    valid = j >= 0
    bn_ret = np.where(valid, (pb / pb[np.clip(j, 0, None)] - 1) * 1e4, np.nan)
    by_now = _asof(tt, bmid, tb)
    by_then = _asof(tt, bmid, tb - 1000.0)
    by_ret = (by_now / by_then - 1) * 1e4
    gap = bn_ret - by_ret
    cand = np.flatnonzero(np.isfinite(gap) & (np.abs(gap) >= GAP_MIN_BPS) & (np.sign(gap) == np.sign(bn_ret)))
    keep, last = [], -np.inf
    for i in cand:                                   # не чаще одного события в COOLDOWN_MS
        if tb[i] - last >= COOLDOWN_MS:
            keep.append(i)
            last = tb[i]
    if not keep:
        return pd.DataFrame()
    k = np.asarray(keep)
    tau = tb[k]
    ev = pd.DataFrame({"ts_ms": tau, "side": np.sign(gap[k]), "gap_bps": gap[k], "bn_ret_bps": bn_ret[k],
                       "by_ret_bps": by_ret[k], "spread_bps": (_asof(tt, bask, tau) / _asof(tt, bbid, tau) - 1) * 1e4})
    # сделки Bybit: первые покупка/продажа агрессора после tau+L (в пределах секунды)
    ts_t = (bt["timestamp"].to_numpy(dtype="float64") * 1000.0)
    o = np.argsort(ts_t, kind="stable")
    ts_t, px_t = ts_t[o], bt["price"].to_numpy(dtype="float64")[o]
    buy_t = (bt["side"].to_numpy()[o] == "Buy")
    tb_buy, pb_buy = ts_t[buy_t], px_t[buy_t]
    tb_sell, pb_sell = ts_t[~buy_t], px_t[~buy_t]

    def first_after(ts_arr, px_arr, q):
        i = np.searchsorted(ts_arr, q, side="left")
        hit = (i < len(ts_arr))
        i2 = np.clip(i, 0, len(ts_arr) - 1)
        okk = hit & (ts_arr[i2] - q <= 1000.0)
        return np.where(okk, px_arr[i2], np.nan)

    for L in LATENCIES_MS:
        q = tau + L
        ev[f"ask_{L}"] = _asof(tt, bask, q)
        ev[f"bid_{L}"] = _asof(tt, bbid, q)
        ev[f"tbuy_{L}"] = first_after(tb_buy, pb_buy, q)
        ev[f"tsell_{L}"] = first_after(tb_sell, pb_sell, q)
    for h in EXIT_S:
        q = tau + h * 1000.0
        ev[f"xbid_{h}"] = _asof(tt, bbid, q)
        ev[f"xask_{h}"] = _asof(tt, bask, q)
    return ev


def build_day(symbol: str, day: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
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
    book, tob = book_seconds(ob, day)
    trades = read_bybit_trades(tr)
    df = pd.concat([book, bybit_trades_seconds(trades, day)], axis=1)
    events = pd.DataFrame()
    if bn is not None:
        bnt = read_binance(bn)
        df = df.join(binance_seconds(bnt, day))
        events = leadlag_events(bnt, tob, trades)
        events.insert(0, "symbol", symbol)
    df.insert(0, "symbol", symbol)
    f32 = [c for c in df.columns if c not in ("symbol", "bid", "ask", "last_px", "max_px", "min_px", "bn_px")]
    df[f32] = df[f32].astype("float32")
    print(f"  {symbol} {day}: ok, загрузка {t_dl:.0f}s, всего {time.time() - t_start:.0f}s, "
          f"событий опережения {len(events)}, стакан {len(ob) / 1e6:.0f} МБ, спред медиана {((df.ask - df.bid) / df.bid * 1e4).median():.2f} б.п.",
          flush=True)
    return df, events


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
        res = build_day(a.symbol, day)
        if res is not None:
            res[0].to_parquet(p)
            if len(res[1]):
                res[1].to_parquet(out / f"ev-{a.symbol}-{day}.parquet")
