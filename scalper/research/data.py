"""
Данные для исследования интрадей-стратегий: месячные свечи USDⓈ-M и история funding
с data.binance.vision. Скачивается параллельно, складывается в один parquet на символ.

    python -m research.data --interval 5m --start 2022-01 --end 2026-09
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HOSTS = {"cdn": "https://data.binance.vision",
         "s3": "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"}   # тот же архив напрямую из S3
BASE = f"{HOSTS['cdn']}/data/futures/um/monthly"
UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT",
            "LINKUSDT", "LTCUSDT", "DOTUSDT", "TRXUSDT", "BCHUSDT", "ATOMUSDT", "NEARUSDT", "ETCUSDT"]
KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
         "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def months(start: str, end: str) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def set_host(name: str) -> None:
    global BASE, DAILY
    BASE = f"{HOSTS[name]}/data/futures/um/monthly"
    DAILY = f"{HOSTS[name]}/data/futures/um/daily"


def _get(url: str) -> bytes | None:
    url = urllib.parse.quote(url, safe=":/?=&%")          # символы с не-ASCII именами
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
    import pyarrow.parquet as pq
    try:
        with open(path, "rb") as fh:            # явное закрытие: на Windows открытый файл нельзя удалить
            pq.ParquetFile(fh).metadata
        return True
    except Exception:
        pass
    for attempt in range(20):                   # битый файл (например, после аварийного выключения) — удаляем
        try:
            path.unlink(missing_ok=True)
            return False
        except PermissionError:
            time.sleep(0.5 * (attempt + 1))
    return False


def _save(df: pd.DataFrame, out: Path) -> None:
    """Атомарная запись: сначала во временный файл, затем переименование."""
    tmp = out.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    for attempt in range(20):
        try:
            tmp.replace(out)
            return
        except PermissionError:          # Windows: файл держит антивирус или индексатор — ждём и повторяем
            time.sleep(0.5 * (attempt + 1))
    tmp.replace(out)


def _read_zip_csv(blob: bytes, cols: list[str] | None) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0])
    header = not raw[:1].isdigit()
    df = pd.read_csv(io.BytesIO(raw), header=0 if header else None)
    if cols:
        df.columns = cols[: df.shape[1]]
    return df


def fetch_klines(symbol: str, interval: str, month: str, cache: Path, kind: str = "klines") -> Path | None:
    """kind: klines | premiumIndexKlines (премия perp к индексу) | spot (свечи спота, те же колонки)."""
    tag = interval if kind == "klines" else f"{kind}-{interval}"
    out = cache / f"{symbol}-{tag}-{month}.parquet"
    if _valid(out):
        return out
    if kind == "spot":
        url = f"{BASE.replace('/futures/um/', '/spot/')}/klines/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
    else:
        url = f"{BASE}/{kind}/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
    blob = _get(url)
    if blob is None:
        return None
    df = _read_zip_csv(blob, KCOLS)
    df = df[["open_time", "open", "high", "low", "close", "volume", "quote_volume", "count",
             "taker_buy_volume", "taker_buy_quote_volume"]].astype("float64")
    df["open_time"] = df["open_time"].astype("int64")
    big = df["open_time"] > 10**14                     # спотовые архивы с 2025 года — в микросекундах
    df.loc[big, "open_time"] = df.loc[big, "open_time"] // 1000
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


DAILY = f"{HOSTS['cdn']}/data/futures/um/daily"
METRICS_FREQ: str | None = None      # например "1h": хранить последний снимок метрик в каждом часе


def fetch_metrics(symbol: str, month: str, cache: Path) -> Path | None:
    """Метрики позиционирования (OI, long/short ratio) с шагом 5 минут: дневные архивы, склеенные в месяц."""
    out = cache / f"{symbol}-metrics-{month}.parquet"
    if _valid(out):
        return out
    parts = []
    for day in pd.date_range(f"{month}-01", periods=pd.Period(month).days_in_month, freq="D"):
        d = day.strftime("%Y-%m-%d")
        blob = _get(f"{DAILY}/metrics/{symbol}/{symbol}-metrics-{d}.zip")
        if blob is not None:
            parts.append(_read_zip_csv(blob, None))
    if not parts:
        return None
    df = pd.concat(parts)
    df.columns = [str(c).strip().lower() for c in df.columns]
    ts = pd.to_datetime(df["create_time"], utc=True)
    keep = ["sum_open_interest", "sum_open_interest_value", "count_toptrader_long_short_ratio",
            "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]
    m = pd.DataFrame({"ts": ts.dt.as_unit("ms").astype("int64").to_numpy()})
    for c in keep:
        m[c] = pd.to_numeric(df[c], errors="coerce").to_numpy() if c in df else np.nan
    m = m.sort_values("ts")
    if METRICS_FREQ:
        hour = pd.to_datetime(m["ts"], unit="ms").dt.floor(METRICS_FREQ)
        m = m.groupby(hour.to_numpy(), sort=True).tail(1)
    _save(m, out)
    return out


def build(symbols: list[str], interval: str, start: str, end: str, root: Path, workers: int = 12,
          metrics: bool = False, premium: bool = False, spot: bool = False) -> None:
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    jobs = []
    with ThreadPoolExecutor(workers) as ex:
        for s in symbols:
            for m in months(start, end):
                jobs.append(ex.submit(fetch_klines, s, interval, m, cache))
                jobs.append(ex.submit(fetch_funding, s, m, cache))
                if metrics:
                    jobs.append(ex.submit(fetch_metrics, s, m, cache))
                if premium:
                    jobs.append(ex.submit(fetch_klines, s, interval, m, cache, "premiumIndexKlines"))
                if spot:
                    jobs.append(ex.submit(fetch_klines, s, interval, m, cache, "spot"))
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
        sparts = sorted(cache.glob(f"{s}-spot-{interval}-*.parquet"))
        if sparts:
            sk = pd.concat([pd.read_parquet(p) for p in sparts]).drop_duplicates("open_time").sort_values("open_time")
            sk.to_parquet(root / f"{s}-spot-{interval}.parquet", index=False)
        pparts = sorted(cache.glob(f"{s}-premiumIndexKlines-{interval}-*.parquet"))
        if pparts:
            pk = pd.concat([pd.read_parquet(p) for p in pparts]).drop_duplicates("open_time").sort_values("open_time")
            pk.to_parquet(root / f"{s}-premium-{interval}.parquet", index=False)
        mparts = sorted(cache.glob(f"{s}-metrics-*.parquet"))
        if mparts:
            mm = pd.concat([pd.read_parquet(p) for p in mparts]).drop_duplicates("ts").sort_values("ts")
            mm.to_parquet(root / f"{s}-metrics.parquet", index=False)
        print(f"{s}: {len(parts)} мес. свечей, {len(fparts)} мес. funding, {len(mparts)} мес. метрик")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--start", default="2022-01")
    ap.add_argument("--end", default="2026-09")
    ap.add_argument("--symbols", default=",".join(UNIVERSE))
    ap.add_argument("--root", required=True)
    ap.add_argument("--metrics", action="store_true", help="также OI и long/short ratio (дневные архивы)")
    ap.add_argument("--premium", action="store_true", help="также premium index klines")
    ap.add_argument("--spot", action="store_true", help="также свечи спота (тот же символ)")
    ap.add_argument("--host", default="cdn", choices=list(HOSTS))
    ap.add_argument("--metrics-freq", default=None, help="прореживание метрик, например 1h")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    set_host(a.host)
    METRICS_FREQ = a.metrics_freq
    build(a.symbols.split(","), a.interval, a.start, a.end, Path(a.root), a.workers, a.metrics, a.premium, a.spot)
