"""
Исторические данные.

- aggTrades USDⓈ-M с data.binance.vision (дневные zip + SHA256) -> parquet;
- записи живых потоков (recorder) в jsonl.gz, в них есть стакан и ликвидации.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import urllib.error
import urllib.request
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd

from core.models import Trade
from exchange.binance_ws import parse

log = logging.getLogger("data")

BASE = "https://data.binance.vision/data/futures/um/daily/aggTrades"
COLS = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
        "transact_time", "is_buyer_maker"]


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "flow-scalper/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def day_path(data_dir: Path, symbol: str, d: date) -> Path:
    return data_dir / symbol / f"{d.isoformat()}.parquet"


def download_day(symbol: str, d: date, data_dir: Path, force: bool = False) -> Path | None:
    out = day_path(data_dir, symbol, d)
    if out.exists() and not force:
        return out
    name = f"{symbol}-aggTrades-{d.isoformat()}.zip"
    url = f"{BASE}/{symbol}/{name}"
    try:
        blob = _get(url, timeout=300)
    except urllib.error.HTTPError as e:
        log.warning("%s: HTTP %s (день ещё не опубликован?)", name, e.code)
        return None
    try:
        expected = _get(url + ".CHECKSUM").decode().split()[0]
        actual = hashlib.sha256(blob).hexdigest()
        if expected != actual:
            raise ValueError(f"checksum mismatch for {name}")
    except urllib.error.HTTPError:
        log.warning("%s: нет CHECKSUM, пропускаю проверку", name)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0])
    has_header = not raw[:1].isdigit()
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None,
                     names=None if has_header else COLS)
    df.columns = COLS
    df = pd.DataFrame({
        "ts_ms": df["transact_time"].astype("int64"),
        "price": df["price"].astype("float64"),
        "qty": df["quantity"].astype("float64"),
        "buyer_maker": df["is_buyer_maker"].astype(str).str.lower().isin(["true", "1"]),
    }).sort_values("ts_ms", kind="stable")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info("%s: %d сделок -> %s", name, len(df), out.name)
    return out


def date_range(days: int | None, start: str | None, end: str | None) -> list[date]:
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    e = date.fromisoformat(end) if end else yesterday
    if start:
        s = date.fromisoformat(start)
    else:
        s = e - timedelta(days=(days or 1) - 1)
    return [s + timedelta(days=i) for i in range((e - s).days + 1)]


def download(symbol: str, dates: list[date], data_dir: Path) -> list[Path]:
    paths = []
    for d in dates:
        p = download_day(symbol, d, data_dir)
        if p:
            paths.append(p)
    return paths


def fetch_instrument(symbol: str, data_dir: Path) -> dict | None:
    """tickSize / stepSize / minNotional из публичного exchangeInfo (кешируется)."""
    cache = Path(data_dir) / "instruments.json"
    data = {}
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        if symbol in data:
            return data[symbol]
    try:
        info = json.loads(_get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=30))
    except Exception as e:
        log.warning("exchangeInfo недоступен (%s), беру instrument из конфига", e)
        return None
    for s in info.get("symbols", []):
        f = {x["filterType"]: x for x in s.get("filters", [])}
        try:
            data[s["symbol"]] = {
                "tick_size": float(f["PRICE_FILTER"]["tickSize"]),
                "step_size": float(f["LOT_SIZE"]["stepSize"]),
                "min_notional": float(f.get("MIN_NOTIONAL", {}).get("notional", 5.0)),
            }
        except KeyError:
            continue
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data), encoding="utf-8")
    return data.get(symbol)


MIN_NOTIONAL_DEFAULTS = {"BTCUSDT": 100.0, "ETHUSDT": 20.0}


def _decimals(x: float) -> int:
    s = f"{x:.10f}".rstrip("0")
    return len(s.split(".")[1]) if "." in s else 0


def infer_instrument(symbol: str, paths: list[Path]) -> dict:
    """Запасной вариант без exchangeInfo: tick и шаг лота по самим сделкам."""
    df = pd.read_parquet(sorted(paths)[0]).head(300_000)
    prices = df["price"].drop_duplicates().sort_values()
    diffs = prices.diff().dropna()
    tick = round(float(diffs[diffs > 0].min()), 10)
    tick = 10 ** -_decimals(tick) if tick > 0 else 0.01
    qdec = max(_decimals(q) for q in df["qty"].sample(min(len(df), 5000), random_state=0))
    return {"tick_size": tick, "step_size": 10 ** -qdec,
            "min_notional": MIN_NOTIONAL_DEFAULTS.get(symbol, 5.0)}


def iter_agg_trades(paths: list[Path]) -> Iterator[Trade]:
    for p in sorted(paths):
        df = pd.read_parquet(p)
        ts = (df["ts_ms"].to_numpy() / 1000.0).tolist()
        px = df["price"].tolist()
        qty = df["qty"].tolist()
        bm = df["buyer_maker"].tolist()
        for i in range(len(ts)):
            yield Trade(ts[i], px[i], qty[i], bm[i])


def count_rows(paths: list[Path]) -> int:
    import pyarrow.parquet as pq
    return sum(pq.ParquetFile(p).metadata.num_rows for p in paths)


def iter_recorded(paths: list[Path]) -> Iterator:
    """Реплей файлов recorder: события в порядке биржевого времени внутри файла."""
    for p in sorted(paths):
        events = []
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    m = json.loads(line)
                except json.JSONDecodeError:
                    continue            # оборванная последняя строка
                ev = parse(m["s"], m["d"])
                if ev is not None:
                    events.append(ev)
        events.sort(key=lambda e: e.ts)
        yield from events
