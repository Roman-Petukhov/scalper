"""
История Bybit USDT-перпетуалов через публичный API v5 (ключи не нужны): свечи, funding, open interest.
Пишет файлы в том же формате, что и загрузчик Binance, поэтому весь research/ работает без изменений.

API Bybit закрыт для IP из США (облака GitHub/Claude), поэтому запускается на своём компьютере:

    python -m research.bybit_data --root data_bybit              # скачать (можно прерывать и перезапускать)
    python -m research.bybit_data --root data_bybit --publish    # скачать и запушить в ветку data-bybit
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .data import UNIVERSE, _save, _valid, months

API = "https://api.bybit.com"         # переопределяется --api (например, https://api.bytick.com)
BRANCH = "data-bybit"
KLINE_MIN = 5
OI_INTERVAL, OI_MIN = "15min", 15


class RateLimiter:
    """Не чаще rps запросов в секунду на все потоки (лимит Bybit 600 запросов за 5 с на IP, берём шестую часть)."""

    def __init__(self, rps: float):
        self.gap, self.lock, self.next = 1.0 / rps, threading.Lock(), 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.gap
        time.sleep(max(0.0, t - now))


LIMITER = RateLimiter(20.0)


def _call(path: str, **params) -> dict:
    url = f"{API}{path}?{urllib.parse.urlencode(params)}"
    last = ""
    for attempt in range(6):
        LIMITER.wait()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "flow-scalper-research/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                body = json.loads(r.read())
            if body.get("retCode") == 0:
                return body["result"]
            last = f"retCode={body.get('retCode')} {body.get('retMsg')}"
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code == 403:
                raise RuntimeError(f"Bybit отклонил запрос (403): регион заблокирован? {url}") from e
        except Exception as e:  # сеть, таймаут, битый JSON — повторяем
            last = repr(e)
        time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"Bybit: не удалось получить {url}: {last}")


def _ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


def _month_bounds(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    a = pd.Timestamp(f"{month}-01", tz="UTC")
    return a, a + pd.offsets.MonthBegin(1)


def _complete(month: str) -> bool:
    """Месяц целиком в прошлом — только такие кешируем навсегда."""
    return _month_bounds(month)[1] <= pd.Timestamp.now(tz="UTC")


def fetch_klines(symbol: str, month: str, cache: Path) -> Path | None:
    out = cache / f"{symbol}-{KLINE_MIN}m-{month}.parquet"
    if _valid(out):
        return out
    a, b = _month_bounds(month)
    rows, step = [], pd.Timedelta(minutes=KLINE_MIN * 1000)
    t = a
    while t < b:
        end = min(t + step, b) - pd.Timedelta(milliseconds=1)
        res = _call("/v5/market/kline", category="linear", symbol=symbol, interval=str(KLINE_MIN),
                    start=_ms(t), end=_ms(end), limit=1000)
        rows.extend(res.get("list", []))
        t += step
    if not rows:
        return None
    k = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume", "quote_volume"])
    k = k.astype("float64")
    k["open_time"] = k["open_time"].astype("int64")
    k = k.drop_duplicates("open_time").sort_values("open_time")
    for c in ("count", "taker_buy_volume", "taker_buy_quote_volume"):
        k[c] = np.nan                                   # Bybit не отдаёт агрессоров в свечах
    if _complete(month):
        _save(k, out)
        return out
    return _save_tmp(k, out)


def fetch_funding(symbol: str, month: str, cache: Path) -> Path | None:
    out = cache / f"{symbol}-funding-{month}.parquet"
    if _valid(out):
        return out
    a, b = _month_bounds(month)
    rows, step = [], pd.Timedelta(hours=200)            # окно на 200 выплат даже при часовом funding
    t = a
    while t < b:
        end = min(t + step, b) - pd.Timedelta(milliseconds=1)
        res = _call("/v5/market/funding/history", category="linear", symbol=symbol,
                    startTime=_ms(t), endTime=_ms(end), limit=200)
        rows.extend(res.get("list", []))
        t += step
    if not rows:
        return None
    f = pd.DataFrame({"ts": [int(r["fundingRateTimestamp"]) for r in rows],
                      "rate": [float(r["fundingRate"]) for r in rows]}).drop_duplicates("ts").sort_values("ts")
    if _complete(month):
        _save(f, out)
        return out
    return _save_tmp(f, out)


def fetch_oi(symbol: str, month: str, cache: Path) -> Path | None:
    out = cache / f"{symbol}-oi-{month}.parquet"
    if _valid(out):
        return out
    a, b = _month_bounds(month)
    rows, step = [], pd.Timedelta(minutes=OI_MIN * 200)
    t = a
    while t < b:
        end = min(t + step, b) - pd.Timedelta(milliseconds=1)
        res = _call("/v5/market/open-interest", category="linear", symbol=symbol, intervalTime=OI_INTERVAL,
                    startTime=_ms(t), endTime=_ms(end), limit=200)
        rows.extend(res.get("list", []))
        t += step
    if not rows:
        return None
    m = pd.DataFrame({"ts": [int(r["timestamp"]) for r in rows],
                      "sum_open_interest": [float(r["openInterest"]) for r in rows]})
    m = m.drop_duplicates("ts").sort_values("ts")
    if _complete(month):
        _save(m, out)
        return out
    return _save_tmp(m, out)


def _save_tmp(df: pd.DataFrame, out: Path) -> Path:
    """Текущий неполный месяц: пишем под отдельным именем, чтобы при следующем запуске скачать заново."""
    p = out.with_name(out.stem + ".partial.parquet")
    _save(df, p)
    return p


def build(symbols: list[str], start: str, end: str, root: Path, workers: int) -> None:
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    for p in cache.glob("*.partial.parquet"):
        p.unlink()
    jobs = {}
    with ThreadPoolExecutor(workers) as ex:
        for s in symbols:
            for m in months(start, end):
                for fn in (fetch_klines, fetch_funding, fetch_oi):
                    jobs[ex.submit(fn, s, m, cache)] = (fn.__name__, s, m)
        done, failed = 0, []
        for fut in as_completed(jobs):
            done += 1
            try:
                fut.result()
            except Exception as e:
                failed.append((jobs[fut], str(e)))
                if len(failed) == 1:
                    print(f"  первая ошибка: {jobs[fut]}: {e}", flush=True)
                if len(failed) >= 20 and len(failed) == done:
                    for f in jobs:
                        f.cancel()
                    raise SystemExit(f"Все первые {done} запросов неудачны — API недоступен. Причина: {e}")
            if done % 100 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} частей, ошибок: {len(failed)}", flush=True)
    if failed:
        for (name, s, m), err in failed[:10]:
            print(f"  ! {name} {s} {m}: {err}")
        raise SystemExit(f"Не скачано {len(failed)} частей — просто запустите команду ещё раз, готовое не перекачается.")

    def concat(pattern: str, key: str) -> pd.DataFrame | None:
        parts = sorted(cache.glob(pattern))
        if not parts:
            return None
        return pd.concat([pd.read_parquet(p) for p in parts]).drop_duplicates(key).sort_values(key)

    for s in symbols:
        k = concat(f"{s}-{KLINE_MIN}m-*.parquet", "open_time")
        f = concat(f"{s}-funding-*.parquet", "ts")
        oi = concat(f"{s}-oi-*.parquet", "ts")
        if k is not None:
            k.to_parquet(root / f"{s}-5m.parquet", index=False)
        if f is not None:
            f.to_parquet(root / f"{s}-funding.parquet", index=False)
        if oi is not None:
            for c in ("sum_open_interest_value", "count_toptrader_long_short_ratio",
                      "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio"):
                oi[c] = np.nan
            oi.to_parquet(root / f"{s}-metrics.parquet", index=False)
        first = pd.to_datetime(k["open_time"].iloc[0], unit="ms").date() if k is not None else "—"
        print(f"{s}: свечей {0 if k is None else len(k)} (с {first}), funding {0 if f is None else len(f)}, "
              f"OI {0 if oi is None else len(oi)}")


def publish(root: Path) -> None:
    """Кладёт итоговые parquet (без кеша) в ветку data-bybit одним коммитом и пушит её."""
    files = sorted(root.glob("*.parquet"))
    if not files:
        raise SystemExit("Нечего публиковать: сначала скачайте данные.")
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    tmp = Path(tempfile.mkdtemp(prefix="data-bybit-"))
    wt = tmp / "wt"
    try:
        subprocess.run(["git", "-C", str(repo), "worktree", "add", "--detach", str(wt)], check=True)
        subprocess.run(["git", "-C", str(wt), "checkout", "--orphan", BRANCH + "-tmp"], check=True)
        subprocess.run(["git", "-C", str(wt), "rm", "-rfq", "."], check=True)
        for p in files:
            shutil.copy2(p, wt / p.name)
        (wt / "README.md").write_text(f"Bybit linear: свечи 5m, funding, OI {OI_INTERVAL}. "
                                      f"Собрано {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC.\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(wt), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(wt), "commit", "-qm", "Bybit market data snapshot"], check=True)
        subprocess.run(["git", "-C", str(wt), "push", "-f", "origin", f"HEAD:refs/heads/{BRANCH}"], check=True)
        print(f"Готово: данные в ветке {BRANCH}.")
    finally:
        subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", str(wt)], check=False)
        subprocess.run(["git", "-C", str(repo), "branch", "-D", BRANCH + "-tmp"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--start", default="2022-01")
    ap.add_argument("--end", default=pd.Timestamp.now(tz="UTC").strftime("%Y-%m"))
    ap.add_argument("--symbols", default=",".join(UNIVERSE))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--publish", action="store_true", help="после загрузки запушить в ветку data-bybit")
    ap.add_argument("--api", default=API, help="адрес API Bybit")
    a = ap.parse_args()
    API = a.api.rstrip("/")
    try:
        print("Bybit время сервера:", _call("/v5/market/time").get("timeSecond"), flush=True)
    except Exception as e:
        raise SystemExit(f"Нет доступа к {API}: {e}")
    root = Path(a.root)
    build(a.symbols.split(","), a.start, a.end, root, a.workers)
    if a.publish:
        publish(root)
