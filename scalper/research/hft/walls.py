"""
«Настоящие» и «ложные» участники по историческому стакану Bybit (дельты L2) и тиковым сделкам.

Стена — уровень, объём которого не меньше WALL_MULT медиан объёма уровней в топ-20 своей стороны и который стоит
не дальше MAX_DIST_BPS от средней цены. Её жизнь прослеживается до «смерти» (объём упал ниже половины порога):
    pulled_near  — сняли, когда цена подходила ближе NEAR_BPS (кандидат в спуфинг: «не пускали», а потом убрали)
    pulled_far   — сняли, когда цена была далеко
    consumed     — объём ушёл в сделки агрессоров (настоящая ликвидность, уровень пробит)
    absorbed     — через уровень прошло >= ABSORB_X видимого объёма, а он всё стоит (айсберг/поглощение)
Для каждого события: движение средней цены через 10с..60мин в б.п. Знак «ожидания» задаётся заранее:
    pulled_near bid  -> вниз (сняли фальшивую поддержку),  ask -> вверх
    consumed    bid  -> вниз (поддержку пробили),          ask -> вверх
    absorbed    bid  -> вверх (крупный покупатель держит),  ask -> вниз
    pulled_far       -> без ожидания (контроль)

    python -m research.hft.walls --symbols A,B --days 2024-01-08,... --out <dir>
"""
from __future__ import annotations

import argparse
import gzip
import io
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from array import array
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import orjson
import pandas as pd

from . import luxbook
from .build import BYBIT_TRADES, OB_URL, _asof, _get

WALL_MULT = 5.0
MAX_DIST_BPS = 30.0
NEAR_BPS = 10.0
ABSORB_X = 2.0
BANDS_BPS = luxbook.BANDS_BPS
MIN_SIZE_X = 10.0          # «настоящая» стена — не меньше 10 медиан уровня
MIN_LIFE_MS = 5_000        # и простояла хотя бы 5 с (мерцание котировок HFT не пишем: на BTC его ~10 млн событий/день)
DEDUP_MS = 60_000          # отчёт: одно событие одного типа на стороне за минуту
HORIZONS_S = (10, 60, 300, 900, 3600)
EXPECT = {("pulled_near", 1): -1, ("pulled_near", -1): 1, ("consumed", 1): -1, ("consumed", -1): 1,
          ("absorbed", 1): 1, ("absorbed", -1): -1, ("pulled_far", 1): 0, ("pulled_far", -1): 0}


def _ref_size(book: dict, best: float, side: int) -> float:
    """Медиана объёма уровней в топ-20 стороны (без самой крупной стены, чтобы она не задирала порог)."""
    if not book:
        return np.inf
    prices = sorted(book, reverse=(side == 1))[:20]
    sz = np.array([book[p] for p in prices])
    return float(np.median(sz)) if len(sz) >= 5 else np.inf


def _fetch_to_file(url: str, tries: int = 4) -> str | None:
    """Стакан BTC/ETH за день — сотни мегабайт: качаем на диск, а не в память (раннер private-репо — 7 ГБ)."""
    for k in range(tries):
        fd, path = tempfile.mkstemp(suffix=".zip")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "flow-scalper-research/1.0"})
            with os.fdopen(fd, "wb") as fh, urllib.request.urlopen(req, timeout=300) as r:
                shutil.copyfileobj(r, fh, 1 << 20)
            return path
        except urllib.error.HTTPError as e:
            os.unlink(path)
            if e.code in (403, 404):
                return None
        except Exception:
            if os.path.exists(path):
                os.unlink(path)
        time.sleep(2 ** k)
    return None


def _band_depth(book: dict, mm: float) -> list[float]:
    """Объём лимитных заявок каждой стороны в пределах BANDS_BPS от mid: [bid10, ask10, bid25, ask25]."""
    res = []
    for b in BANDS_BPS:
        lo, hi = mm * (1 - b / 1e4), mm * (1 + b / 1e4)
        res += [sum(q for p, q in book[1].items() if p >= lo), sum(q for p, q in book[-1].items() if p <= hi)]
    return res


def detect(ob_src: bytes | str, trades: pd.DataFrame) -> pd.DataFrame:
    return replay(ob_src, trades)[0]


def replay(ob_src: bytes | str, trades: pd.DataFrame, probes: np.ndarray | None = None
           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """События стен и (если заданы probes — отсортированные ms) снимки глубины на эти моменты."""
    probes = np.empty(0, dtype=np.int64) if probes is None else np.asarray(probes, dtype=np.int64)
    snaps = []
    i_pr = 0
    t_ts = (trades["timestamp"].to_numpy(dtype="float64") * 1000.0)
    o = np.argsort(t_ts, kind="stable")
    t_ts = t_ts[o]
    t_buy = (trades["side"].to_numpy() == "Buy")[o]
    t_sz = trades["size"].to_numpy(dtype="float64")[o]
    t_px = trades["price"].to_numpy(dtype="float64")[o]
    book = {1: {}, -1: {}}
    best = {1: -np.inf, -1: np.inf}
    ref = {1: np.inf, -1: np.inf}
    walls: dict = {}                       # (side, price) -> state
    events = []
    tob_ts, tob_mid = array("d"), array("d")                  # только смены mid: asof по ним даёт тот же ответ
    i_tr, n_tr = 0, len(t_ts)
    last_ref_sec = -1

    def mid():
        return (best[1] + best[-1]) / 2

    src = io.BytesIO(ob_src) if isinstance(ob_src, bytes) else ob_src
    with zipfile.ZipFile(src) as z, z.open(z.namelist()[0]) as fh:
        for line in fh:
            m = orjson.loads(line)
            ts = m["ts"]
            while i_pr < len(probes) and probes[i_pr] < ts:          # состояние после всех обновлений <= probe
                mm = mid()
                if np.isfinite(mm) and best[1] < best[-1]:
                    snaps.append([int(probes[i_pr]), mm, (best[-1] - best[1]) / mm * 1e4, *_band_depth(book, mm)])
                i_pr += 1
            while i_tr < n_tr and t_ts[i_tr] <= ts:                 # сделки до обновления стакана
                side_hit = -1 if t_buy[i_tr] else 1                   # покупатель бьёт в ask (-1), продавец в bid (1)
                w = walls.get((side_hit, t_px[i_tr]))
                if w is not None:
                    w["traded"] += t_sz[i_tr]
                    if not w["absorbed"] and w["traded"] >= ABSORB_X * w["max"] and \
                            book[side_hit].get(t_px[i_tr], 0.0) >= 0.5 * w["thr"]:
                        w["absorbed"] = True
                        if w["max"] >= MIN_SIZE_X * w["ref"] and t_ts[i_tr] - w["born"] >= MIN_LIFE_MS:
                            events.append((t_ts[i_tr], "absorbed", side_hit, w["max"] / w["ref"],
                                           t_ts[i_tr] - w["born"],
                                           abs(t_px[i_tr] / mid() - 1) * 1e4 if np.isfinite(mid()) else np.nan))
                i_tr += 1
            d = m["data"]
            if m["type"] == "snapshot":
                book[1] = {float(p): float(q) for p, q in d["b"]}
                book[-1] = {float(p): float(q) for p, q in d["a"]}
                best[1] = max(book[1]) if book[1] else -np.inf
                best[-1] = min(book[-1]) if book[-1] else np.inf
                walls.clear()
                continue
            for side, key in ((1, "b"), (-1, "a")):
                bk = book[side]
                for p, q in d[key]:
                    fp, fq = float(p), float(q)
                    prev = bk.get(fp, 0.0)
                    if fq == 0.0:
                        bk.pop(fp, None)
                        if fp == best[side]:
                            best[side] = (max(bk) if side == 1 else min(bk)) if bk else (-np.inf if side == 1 else np.inf)
                    else:
                        bk[fp] = fq
                        if (side == 1 and fp > best[side]) or (side == -1 and fp < best[side]):
                            best[side] = fp
                    mm = mid()
                    if not np.isfinite(mm):
                        continue
                    k = (side, fp)
                    w = walls.get(k)
                    thr = WALL_MULT * ref[side]
                    if w is None and fq >= thr and prev < thr and abs(fp / mm - 1) * 1e4 <= MAX_DIST_BPS:
                        walls[k] = {"born": ts, "max": fq, "thr": thr, "ref": ref[side], "traded": 0.0,
                                    "min_dist": abs(fp / mm - 1) * 1e4, "absorbed": False}
                    elif w is not None:
                        w["max"] = max(w["max"], fq)
                        if fq < 0.5 * w["thr"]:
                            drop = max(w["max"] - fq, 1e-12)
                            kind = "consumed" if w["traded"] >= 0.5 * drop else \
                                ("pulled_near" if w["min_dist"] <= NEAR_BPS else "pulled_far")
                            if w["max"] >= MIN_SIZE_X * w["ref"] and ts - w["born"] >= MIN_LIFE_MS:
                                events.append((ts, kind, side, w["max"] / w["ref"], ts - w["born"], w["min_dist"]))
                            del walls[k]
            mm = mid()
            if np.isfinite(mm) and best[1] < best[-1]:
                if not tob_mid or tob_mid[-1] != mm:
                    tob_ts.append(ts)
                    tob_mid.append(mm)
                sec = ts // 1000
                if sec != last_ref_sec:                             # пороги и дистанции — раз в секунду
                    last_ref_sec = sec
                    ref[1] = _ref_size(book[1], best[1], 1)
                    ref[-1] = _ref_size(book[-1], best[-1], -1)
                    for (sd, pr), w in walls.items():
                        w["min_dist"] = min(w["min_dist"], abs(pr / mm - 1) * 1e4)
    snap_cols = ["t_ms", "mid", "spread_bps"] + [f"{s}{b}" for b in BANDS_BPS for s in ("bid", "ask")]
    snaps_df = pd.DataFrame(snaps, columns=snap_cols)
    if not events:
        return pd.DataFrame(), snaps_df
    ev = pd.DataFrame(events, columns=["ts", "kind", "side", "size_x", "life_ms", "dist_bps"])
    tt, tm = np.frombuffer(tob_ts, dtype="float64"), np.frombuffer(tob_mid, dtype="float64")
    m0 = _asof(tt, tm, ev["ts"].to_numpy(dtype="float64"))
    exp = np.array([EXPECT[(k, s)] for k, s in zip(ev["kind"], ev["side"])], dtype="float64")
    ev["expect"] = exp
    for h in HORIZONS_S:
        mh = _asof(tt, tm, ev["ts"].to_numpy(dtype="float64") + h * 1000.0)
        ev[f"fwd{h}"] = (mh / m0 - 1) * 1e4
    return ev, snaps_df


def run_symbol_day(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    sym, day, sig = args
    ob = None
    for n in (500, 200):
        ob = _fetch_to_file(OB_URL.format(s=sym, d=day, n=n))
        if ob is not None:
            break
    tr = _get(BYBIT_TRADES.format(s=sym, d=day))
    if ob is None or tr is None:
        if ob is not None:
            os.unlink(ob)
        return pd.DataFrame(), pd.DataFrame()
    trades = pd.read_csv(io.BytesIO(gzip.decompress(tr)), usecols=["timestamp", "side", "size", "price"])
    del tr
    probes = sig["t_ms"].to_numpy(dtype="int64") if sig is not None and len(sig) else None
    try:
        ev, snaps = replay(ob, trades, probes)
    finally:
        os.unlink(ob)
    lx = luxbook.features(sig, snaps, ev, trades) if probes is not None else pd.DataFrame()
    for df in (ev, lx):
        if len(df):
            df.insert(0, "day", day)
            df.insert(0, "symbol", sym)
    print(f"  {sym} {day}: событий {len(ev)}, пробоев LuxAlgo {len(lx)}", flush=True)
    return ev, lx


def period_of(day: str) -> str:
    return "is" if day < "2024-07-01" else "val" if day < "2025-07-01" else "ho"


def clean(ev: pd.DataFrame) -> pd.DataFrame:
    e = ev[(ev["size_x"] >= MIN_SIZE_X) & (ev["life_ms"] >= MIN_LIFE_MS)].copy()
    e["bucket"] = (e["ts"] // DEDUP_MS).astype("int64")
    return e.sort_values("ts").drop_duplicates(["symbol", "day", "kind", "side", "bucket"])


def _table(ev: pd.DataFrame) -> pd.DataFrame:
    """Среднее движение в ожидаемую сторону: по монето-дням (события внутри дня зависимы), t — по ним же."""
    rows = []
    for (kind, per), g in ev.groupby(["kind", "period"]):
        sgn = g["expect"].where(g["expect"] != 0, 1.0)            # для контроля — просто «вверх»
        r = {"kind": kind, "period": per, "n": len(g), "coin_days": g.groupby(["symbol", "day"]).ngroups}
        for h in HORIZONS_S:
            cd = (g[f"fwd{h}"] * sgn).groupby([g["symbol"], g["day"]]).mean().dropna()
            r[f"{h}s"] = cd.mean()
            r[f"t{h}"] = cd.mean() / (cd.std(ddof=1) / np.sqrt(len(cd))) if len(cd) > 5 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def report(ev: pd.DataFrame) -> None:
    ev = ev.copy()
    ev["period"] = ev["day"].map(period_of)
    print(f"\nсобытий (сырых): {len(ev)}; монет {ev['symbol'].nunique()}, дней {ev['day'].nunique()}")
    diag = ev.groupby(["symbol", "period"]).agg(
        coin_days=("day", "nunique"), events=("kind", "size"), med_size_x=("size_x", "median"),
        life_lt_1s=("life_ms", lambda x: float((x < 1000).mean())))
    diag["per_day"] = diag["events"] / diag["coin_days"]
    print("\nДиагностика сырых событий (сколько, насколько крупные, доля живших < 1 с):")
    print(diag.round(2).to_string())
    ev = clean(ev)
    print(f"\nПосле отбора (>= {MIN_SIZE_X:.0f} медиан, жила >= {MIN_LIFE_MS // 1000} с, "
          f"одно событие типа/стороны в минуту): {len(ev)}")
    print("Движение средней цены в ожидаемую сторону, б.п.: среднее по монето-дням, t — по монето-дням; "
          "издержки круга: maker 4, taker 11")
    print(_table(ev).round(2).to_string(index=False))
    big = ev[ev["size_x"] >= 30]
    if len(big):
        print("\nТолько очень крупные стены (>= 30 медиан):")
        print(_table(big).round(2).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=None, help="папка с walls_*.parquet: только сводка")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--days", default="")
    ap.add_argument("--out", default=".")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lux", action="store_true", help="также пробои LuxAlgo с признаками стакана (luxbook)")
    a = ap.parse_args()
    if a.report:
        walls_files = sorted(Path(a.report).rglob("walls_*.parquet"))
        if walls_files:
            report(pd.concat([pd.read_parquet(p) for p in walls_files]))
        lux_files = sorted(Path(a.report).rglob("luxbook_*.parquet"))
        if lux_files:
            luxbook.report(pd.concat([pd.read_parquet(p) for p in lux_files], ignore_index=True))
        sys.exit(0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    days = a.days.split(",")
    jobs = []
    for s in a.symbols.split(","):
        by_day: dict = {}
        if a.lux:
            k = luxbook.load_klines(s, days, out / "kcache")
            by_day = luxbook.probes_by_day(luxbook.signals(k), days) if k is not None else {}
            print(f"{s}: пробоев LuxAlgo в днях реплея {sum(len(v) for v in by_day.values())}", flush=True)
        jobs += [(s, d, by_day.get(d)) for d in days]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(run_symbol_day, jobs))
    tag = abs(hash(a.symbols + a.days)) % 10**8
    ev_parts = [e for e, _ in res if len(e)]
    lx_parts = [x for _, x in res if len(x)]
    if ev_parts:
        ev = pd.concat(ev_parts)
        ev.to_parquet(out / f"walls_{tag}.parquet")
        report(ev)
    if lx_parts:
        lx = pd.concat(lx_parts, ignore_index=True)
        lx.to_parquet(out / f"luxbook_{tag}.parquet")
        luxbook.report(lx)
