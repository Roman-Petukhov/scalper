"""
Разлоки токенов: сползает ли цена перпетуала перед крупной разблокировкой и после неё.

График разлоков — DefiLlama (defillama-datasets.llama.fi/emissions/<protocol>: накопленный разлок по категориям
по дням). Символ токена — через coins.llama.fi (id CoinGecko из metadata.token). Расписания вестинга публикуются
заранее, поэтому дата и размер разлока известны до события (оговорка: DefiLlama может уточнять старые расписания).

Событие — день T, когда разлок за день >= 1% уже разлоченного предложения и >= 5 x медианы дневного разлока за
30 дней до T (отсекаем линейный вестинг). Размер = разлок за день / разлоченное до T. Монета — USDT-перпетуал Binance,
торговавшийся >= 30 дней до T. Цены — часовые close перпетуала на 00:00 UTC; «рынок» — BTC.
    до:     T-14 -> T, T-7 -> T, T-3 -> T
    после:  T -> T+1, T -> T+3, T -> T+7
    сделка: шорт от T-7 (и T-3) до T+1 — сырой результат, с funding и за вычетом BTC (бета 1); издержки 12 б.п.
Разбивка по размеру разлока и годам; t — кластеризованный по дням T.

    python -m research.unlocks collect --symbols <все перпетуалы>   (по частям, по монете)
    python -m research.unlocks report
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .shard import all_parts, mine, part_path

LIST_URL = "https://defillama-datasets.llama.fi/emissionsProtocolsList"
EMIS_URL = "https://defillama-datasets.llama.fi/emissions/{p}"
COINS_URL = "https://coins.llama.fi/prices/current/{ids}"
MIN_SIZE = 0.01
CLIFF_X = 5.0
COST = 12e-4
SINCE, UNTIL = pd.Timestamp("2022-02-01", tz="UTC"), pd.Timestamp("2026-09-20", tz="UTC")
WINDOWS_PRE = {"pre14": -14, "pre7": -7, "pre3": -3}
WINDOWS_POST = {"post1": 1, "post3": 3, "post7": 7}
PER = {"is": ("2022-01-01", "2024-07-01"), "val": ("2024-07-01", "2025-07-01"), "ho": ("2025-07-01", "2026-10-01")}


def _get_json(url: str, cache: Path) -> object | None:
    f = cache / (url.split("//", 1)[1].replace("/", "_").replace(":", "_").replace(",", "_")[:180] + ".json")
    if f.exists():
        return json.loads(f.read_text())
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
            f.write_text(json.dumps(data))
            return data
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return None


@functools.lru_cache(maxsize=4)
def _schedules(cache_dir: str) -> tuple[list[tuple[str, str, pd.Series]], dict[str, str]]:
    """Накопленный разлок по дням для каждого протокола и тикеры токенов — разбираются один раз на процесс."""
    cache = Path(cache_dir)
    protos = _get_json(LIST_URL, cache) or []
    out = []
    for p in protos:
        d = _get_json(EMIS_URL.format(p=p), cache)
        if not isinstance(d, dict):
            continue
        token = (d.get("metadata") or {}).get("token")
        cats = ((d.get("documentedData") or {}).get("data")) or ((d.get("data") or {}).get("data")) or []
        series = []
        for c in cats:
            pts = c.get("data") or []
            if pts:
                s = pd.Series({pd.Timestamp(x["timestamp"], unit="s", tz="UTC").floor("D"): float(x.get("unlocked") or 0)
                               for x in pts})
                series.append(s)
        if not series or not token:
            continue
        cum = pd.concat(series, axis=1).sort_index().ffill().fillna(0.0).sum(axis=1)
        out.append((p, token, cum[~cum.index.duplicated()].asfreq("D").ffill()))
    sym = {}
    tok = sorted({t for _, t, _ in out if isinstance(t, str) and ":" in t})
    for i in range(0, len(tok), 50):
        d = _get_json(COINS_URL.format(ids=",".join(tok[i:i + 50])), cache) or {}
        for k, v in (d.get("coins") or {}).items():
            sym[k] = str(v.get("symbol", "")).upper()
    return out, sym


def unlock_events(cache: Path) -> pd.DataFrame:
    sched, sym = _schedules(str(cache))
    rows = []
    for p, token, cum in sched:
        inc = cum.diff()
        med = inc.rolling(30, min_periods=10).median().shift(1)
        prev = cum.shift(1)
        ok = (inc >= MIN_SIZE * prev) & (inc >= CLIFF_X * med.clip(lower=1e-12)) & (prev > 0)
        for t in inc.index[ok.fillna(False).to_numpy()]:
            rows.append({"protocol": p, "token": token, "t": t, "size": inc[t] / prev[t]})
    ev = pd.DataFrame(rows)
    if not len(ev):
        return ev
    ev["ticker"] = ev["token"].map(sym)
    return ev[(ev["t"] >= SINCE) & (ev["t"] < UNTIL) & ev["ticker"].notna()].reset_index(drop=True)


def daily_close(root: Path, sym: str) -> pd.Series | None:
    p = root / f"{sym}-1h.parquet"
    if not p.exists():
        return None
    k = pd.read_parquet(p, columns=["open_time", "close"])
    s = pd.Series(k["close"].to_numpy(), index=pd.to_datetime(k["open_time"], unit="ms", utc=True))
    return s.resample("1D").first()                     # цена на 00:00 UTC (open первого часа дня)


def daily_funding(root: Path, sym: str) -> pd.Series:
    p = root / f"{sym}-funding.parquet"
    if not p.exists():
        return pd.Series(dtype="float64")
    f = pd.read_parquet(p)
    return pd.Series(f["rate"].to_numpy(dtype="float64"),
                     index=pd.to_datetime(f["ts"], unit="ms", utc=True)).resample("1D").sum()


def measure(c: pd.Series, btc: pd.Series, fund: pd.Series, t: pd.Timestamp) -> dict | None:
    def px(s: pd.Series, d: int) -> float:
        v = s.get(t + pd.Timedelta(days=d), np.nan)
        return float(v)
    p0 = px(c, 0)
    if not np.isfinite(p0) or c.first_valid_index() is None or c.first_valid_index() > t - pd.Timedelta(days=30):
        return None
    row: dict = {}
    for k, d in {**WINDOWS_PRE, **WINDOWS_POST}.items():
        a, b = (d, 0) if d < 0 else (0, d)
        pa, pb, ba, bb = px(c, a), px(c, b), px(btc, a), px(btc, b)
        row[k] = (pb / pa - 1) * 1e4 if np.isfinite(pa) and np.isfinite(pb) else np.nan
        row[k + "_x"] = row[k] - (bb / ba - 1) * 1e4 if np.isfinite(ba) and np.isfinite(bb) else np.nan
    for start in (-7, -3):
        pa, pb, ba, bb = px(c, start), px(c, 1), px(btc, start), px(btc, 1)
        if not (np.isfinite(pa) and np.isfinite(pb)):
            continue
        f = fund[(fund.index >= t + pd.Timedelta(days=start)) & (fund.index < t + pd.Timedelta(days=1))].sum()
        raw = -(pb / pa - 1) - COST
        row[f"short{-start}"] = raw * 1e4
        row[f"short{-start}_f"] = (raw + f) * 1e4                      # шорт получает положительный funding
        if np.isfinite(ba) and np.isfinite(bb):
            row[f"short{-start}_x"] = (raw + (bb / ba - 1)) * 1e4
    return row


def collect(root: Path, syms: list[str]) -> None:
    cache = root / "cache" / "llama"
    cache.mkdir(parents=True, exist_ok=True)
    ev = unlock_events(cache)
    print(f"разлоков-событий с тикером: {len(ev)}, протоколов {ev['protocol'].nunique() if len(ev) else 0}", flush=True)
    if not len(ev):
        return
    have = set(syms)
    def to_sym(tk: str) -> str | None:
        for s in (f"{tk}USDT", f"1000{tk}USDT"):
            if s in have:
                return s
        return None
    ev["symbol"] = ev["ticker"].map(to_sym)
    ev = ev[ev["symbol"].notna()]
    ev = ev[ev["symbol"].isin(set(mine(sorted(ev["symbol"].unique()))))]
    btc = daily_close(root, "BTCUSDT")
    rows = []
    for s, g in ev.groupby("symbol"):
        c = daily_close(root, s)
        if c is None:
            continue
        fund = daily_funding(root, s)
        for r in g.itertuples():
            m = measure(c, btc, fund, r.t)
            if m is not None:
                rows.append({"symbol": s, "protocol": r.protocol, "t": r.t, "size": r.size, **m})
    print(f"  в части: событий с ценами {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("unlocks"), index=False)


def _cell(x: pd.Series, t: pd.Series) -> str:
    ok = x.notna()
    r, d = x[ok].to_numpy(dtype="float64"), t[ok].dt.floor("D").to_numpy()
    if len(r) < 5:
        return f"n={len(r)}"
    g = pd.Series(r - r.mean()).groupby(d).sum()
    td = r.sum() / np.sqrt((g ** 2).sum()) if (g ** 2).sum() > 0 else np.nan
    return f"{r.mean():+.0f} ({td:.1f}, n={len(r)})"


def report() -> None:
    parts = all_parts("unlocks")
    if not parts:
        print("частей нет")
        return
    res = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    res["t"] = pd.to_datetime(res["t"], utc=True)
    res = res.drop_duplicates(["symbol", "t"])
    res["bucket"] = pd.cut(res["size"], [0, 0.03, 0.10, np.inf], labels=["1–3%", "3–10%", ">10%"])
    print(f"===== UNLOCKS: событий {len(res)}, монет {res.symbol.nunique()}, частей {len(parts)} =====")
    print("ячейка: среднее, б.п. (t по дням, n); _x — за вычетом BTC")
    cols = list(WINDOWS_PRE) + list(WINDOWS_POST)
    rows = []
    for name, g in [("все", res)] + [(f"размер {b}", g) for b, g in res.groupby("bucket", observed=True)]:
        rows.append({"группа": name, **{c: _cell(g[c], g["t"]) for c in cols}})
        rows.append({"группа": name + " _x", **{c: _cell(g[c + "_x"], g["t"]) for c in cols}})
    print("\nДвижение цены вокруг разлока:")
    print(pd.DataFrame(rows).to_string(index=False))
    rows = []
    for name, g in [("все", res)] + [(f"размер {b}", g) for b, g in res.groupby("bucket", observed=True)]:
        for p, (a, b) in PER.items():
            x = g[(g.t >= a) & (g.t < b)]
            rows.append({"группа": name, "период": p,
                         **{c: _cell(x.get(c, pd.Series(dtype=float)), x["t"]) for c in
                            ("short7", "short7_f", "short7_x", "short3", "short3_f", "short3_x")}})
    print("\nШорт от T-7 / T-3 до T+1 (после 12 б.п.; _f — с funding; _x — против BTC):")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nКрупнейшие разлоки:")
    print(res.sort_values("size", ascending=False).head(12)[["t", "symbol", "size", "pre7", "post3", "short7_f"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 30)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x])
    else:
        report()
