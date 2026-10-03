"""
Данные для исследования интрадей-стратегий: месячные свечи USDⓈ-M и история funding
с data.binance.vision. Скачивается параллельно, складывается в один parquet на символ.

    python -m research.data --interval 5m --start 2022-01 --end 2026-09
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

BASE = "https://data.binance.vision/data/futures/um/monthly"
UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT",
            "LINKUSDT", "LTCUSDT", "DOTUSDT", "TRXUSDT", "BCHUSDT", "ATOMUSDT", "NEARUSDT", "ETCUSDT"]
KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
         "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def months(start: str, end: str) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def _get(url: str) -> bytes | None:
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "flow-scalper-research/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
        except Exception:
            pass
    return None


def _valid(path: Path) -> bool:
    """Файл кеша существует и читается: защищает от обрезанных записей после прерывания."""
    if not path.exists():
        return False
    try:
        import pyarrow.parquet as pq
        pq.ParquetFile(path).metadata
        return True
    except Exception:
        path.unlink(missing_ok=True)
        return False


def _save(df: pd.DataFrame, out: Path) -> None:
    """Атомарная запись: сначала во временный файл, затем переименование."""
    tmp = out.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out)


def _read_zip_csv(blob: bytes, cols: list[str] | None) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0])
    header = not raw[:1].isdigit()
    df = pd.read_csv(io.BytesIO(raw), header=0 if header else None)
    if cols:
        df.columns = cols[: df.shape[1]]
    return df


def fetch_klines(symbol: str, interval: str, month: str, cache: Path) -> Path | None:
    out = cache / f"{symbol}-{interval}-{month}.parquet"
    if _valid(out):
        return out
    blob = _get(f"{BASE}/klines/{symbol}/{interval}/{symbol}-{interval}-{month}.zip")
    if blob is None:
        return None
    df = _read_zip_csv(blob, KCOLS)
    df = df[["open_time", "open", "high", "low", "close", "volume", "quote_volume", "count",
             "taker_buy_volume", "taker_buy_quote_volume"]].astype("float64")
    df["open_time"] = df["open_time"].astype("int64")
    _save(df, out)
    return out


def fetch_funding(symbol: str, month: str, cache: Path) -> Path | None:
    out = cache / f"{symbol}-funding-{month}.parquet"
    if _valid(out):
        return out
    blob = _get(f"{BASE}/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip")
    if blob is None:
        return None
    df = _read_zip_csv(blob, None)
    df.columns = [c.strip().lower() for c in df.columns] if not str(df.columns[0]).isdigit() \
        else ["calc_time", "funding_interval_hours", "last_funding_rate"][: df.shape[1]]
    df = df.rename(columns={"calc_time": "ts", "last_funding_rate": "rate"})
    _save(df[["ts", "rate"]].astype({"ts": "int64", "rate": "float64"}), out)
    return out


def build(symbols: list[str], interval: str, start: str, end: str, root: Path, workers: int = 12) -> None:
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    jobs = []
    with ThreadPoolExecutor(workers) as ex:
        for s in symbols:
            for m in months(start, end):
                jobs.append(ex.submit(fetch_klines, s, interval, m, cache))
                jobs.append(ex.submit(fetch_funding, s, m, cache))
        done = 0
        for _ in as_completed(jobs):
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)} файлов", flush=True)
    for s in symbols:
        parts = sorted(cache.glob(f"{s}-{interval}-*.parquet"))
        if parts:
            k = pd.concat([pd.read_parquet(p) for p in parts]).drop_duplicates("open_time").sort_values("open_time")
            k.to_parquet(root / f"{s}-{interval}.parquet", index=False)
        fparts = sorted(cache.glob(f"{s}-funding-*.parquet"))
        if fparts:
            f = pd.concat([pd.read_parquet(p) for p in fparts]).drop_duplicates("ts").sort_values("ts")
            f.to_parquet(root / f"{s}-funding.parquet", index=False)
        print(f"{s}: {len(parts)} мес. свечей, {len(fparts)} мес. funding")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--start", default="2022-01")
    ap.add_argument("--end", default="2026-09")
    ap.add_argument("--symbols", default=",".join(UNIVERSE))
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    build(a.symbols.split(","), a.interval, a.start, a.end, Path(a.root))
