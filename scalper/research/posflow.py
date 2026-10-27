"""Сектора глазами деривативов, корейской розницы и режима рынка (продолжение research/secorder.py и research/flows.py:
по цене, обороту, экосистемам, Википедии и комиссиям следующий сектор не угадывается).

Что ещё не проверяли по СЕКТОРАМ (по отдельным монетам — проверяли, см. docs/knowledge.md):
    P1 новые деньги в плечах   рост открытого интереса (OI, $) монет сектора за 7 дней сверх такого же роста у всех
                               секторных монет; a: наибольший → неделя; b: то же, только если сектор в плюсе за 7 дней.
    P2 фандинг                 средняя ставка funding монет сектора за 7 дней сверх рынка; a: самый высокий (за длинную
                               позицию платят — спрос; funding-rate-alpha 2025: уровень ставки предсказывает доходность
                               после 2024); b: самый низкий (толпа в шорте — топливо сквиза).
    P3 крупные                 топ-трейдеры Binance сильнее своей нормы в лонге (log long/short минус медиана монеты за
                               30 дней, среднее за 7 дней) — наибольший → неделя (у отдельных монет «крупные правы»
                               держалось только на 4h-лонгах).
    P4 агрессоры               доля рыночных покупок в обороте (taker buy) за 7 дней к своей норме за 90 — наибольший
                               → неделя.
    P5 корейская розница       доля сектора в обороте KRW-рынка Upbit за 7 дней к доле за 90 (4pillars 2025: корейцы
                               покупают вершину альтов — знак может быть и обратным); a: наибольшая → неделя;
                               b: то же, только если сектор в плюсе за 7 дней.
    C  сводный                 среднее z-оценок по секторам: цена за 7 дней, оборот (secorder O2), агрессоры, OI — где
                               есть хотя бы два; наибольший → неделя.
    R  режим                   ротация — явление бычьего рынка, а прошлые проверки усредняли все недели. Те же отборы
                               (лучший сектор за 7 дней, O2b «оборот + цена», P1b, C) только в неделях, когда BTC выше
                               недельной EMA50 (R1), и когда ещё и индекс страха и жадности (alternative.me) за 7 дней
                               >= 55 (R2). Для сравнения — недели вне R1.
Цены, оборот, агрессоры и funding — перпетуалы Binance с делистнутыми с 2020 года; OI и топ-трейдеры — дневные архивы
метрик Binance только у секторных монет, с 2022 года. Upbit — сегодняшний список KRW-рынков (снятые рынки выпадают).
Издержки taker 11 б.п. за круг, мерило — сверх рынка (среднее альтов без BTC) той же недели.
Принятие (задано до запуска): > 0 в каждом периоде IS / VAL / HO (недель >= 20; в отборах R — >= 10), t >= 3,
> 0 при издержках ×2.

    python -m research.prepare 1h-all sect-metrics && python -m research.posflow collect --symbols <1h-all>
    python -m research.posflow report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .flows import _get as get_json
from .flows import judge
from .secorder import pick_weekly, sector_turnover
from .sectors import panel, sector_map
from .shard import all_parts, mine, part_path, shard_id
from .tline import tf_frame

ROOT = Path.home() / "bn"
PART = "posd"
FIELDS = ("close", "qv", "tb", "fund", "oi", "top")
SHORT, LONG, TOP_NORM = 7, 90, 30
GREED = 55
UPBIT = "https://api.upbit.com/v1"
UPBIT_SINCE = pd.Timestamp("2020-06-01")
FNG = "https://api.alternative.me/fng/?limit=0&format=json"


# --- сбор: дневки монет своей части ---

def daily_metrics(root: Path, sym: str) -> pd.DataFrame | None:
    """OI ($) на конец дня и среднее за день отношение long/short топ-трейдеров; нет архивов — None."""
    p = root / f"{sym}-metrics.parquet"
    if not p.exists():
        return None
    m = pd.read_parquet(p)
    t = pd.to_datetime(m["ts"], unit="ms", utc=True).dt.tz_localize(None).dt.floor("D")
    x = pd.DataFrame({"day": t.to_numpy(), "oi": pd.to_numeric(m["sum_open_interest_value"], errors="coerce").to_numpy(),
                      "top": pd.to_numeric(m["sum_toptrader_long_short_ratio"], errors="coerce").to_numpy()})
    g = x.groupby("day")
    return pd.DataFrame({"oi": g["oi"].last(), "top": g["top"].mean()})


def collect(root: Path, syms: list[str]) -> None:
    rows = []
    own = mine(syms)
    if shard_id()[0] == 0:
        own = sorted(set(own) | {"BTCUSDT", "ETHUSDT"})
    for s in own:
        try:
            d = tf_frame(root, s, "1d")
            if d is None or len(d) < 60:
                continue
            day = d.index.tz_localize(None)
            f = pd.DataFrame({"day": day, "close": d["close"].to_numpy(dtype="float64"),
                              "qv": d["quote_volume"].to_numpy(dtype="float64"),
                              "tb": d["taker_buy_quote_volume"].to_numpy(dtype="float64"),
                              "fund": d["funding"].to_numpy(dtype="float64"), "symbol": s})
            m = daily_metrics(root, s)
            if m is not None:
                f = f.join(m, on="day")
            else:
                f["oi"], f["top"] = np.nan, np.nan
            rows.append(f)
        except Exception as e:
            print(f"  posflow {s}: пропуск ({e})", flush=True)
    print(f"  posflow: монет {len(rows)}, с метриками {sum(int(r['oi'].notna().any()) for r in rows)}", flush=True)
    if rows:
        pd.concat(rows, ignore_index=True).to_parquet(part_path(PART), index=False)


def load() -> dict[str, pd.DataFrame] | None:
    parts = all_parts(PART)
    if not parts:
        return None
    x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).drop_duplicates(["symbol", "day"])
    x["day"] = pd.to_datetime(x["day"])
    if getattr(x["day"].dt, "tz", None) is not None:
        x["day"] = x["day"].dt.tz_localize(None)
    close = x.pivot_table(index="day", columns="symbol", values="close").sort_index()
    out = {"close": close}
    for c in FIELDS[1:]:
        out[c] = x.pivot_table(index="day", columns="symbol", values=c).reindex_like(close)
    return out


# --- баллы секторов (значение дня d — по данным до конца d) ---

def sector_score(coin: pd.DataFrame, live: pd.DataFrame, sectors: list[str], min_coins: int = 3) -> pd.DataFrame:
    """Среднее по монетам сектора минус среднее по всем секторным монетам; < min_coins монет с данными — NaN."""
    smap = sector_map()
    v = coin.where(live)
    cols_all = [c for c in v.columns if c in smap]
    base = v[cols_all].mean(axis=1)
    out = {}
    for name in sectors:
        cols = [c for c in v.columns if smap.get(c) == name]
        if cols:
            n = v[cols].notna().sum(axis=1)
            out[name] = (v[cols].mean(axis=1) - base).where(n >= min_coins)
    return pd.DataFrame(out, index=v.index)


def oi_growth(oi: pd.DataFrame) -> pd.DataFrame:
    x = oi.where(oi > 0)
    return np.log(x / x.shift(SHORT))


def funding_7d(fund: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    return fund.where(close.notna()).rolling(SHORT, min_periods=SHORT - 1).sum()


def top_tilt(top: pd.DataFrame) -> pd.DataFrame:
    lg = np.log(top.where(top > 0))
    return (lg - lg.rolling(TOP_NORM, min_periods=20).median()).rolling(SHORT, min_periods=5).mean()


def taker_tilt(tb: pd.DataFrame, qv: pd.DataFrame) -> pd.DataFrame:
    share = (tb / qv.where(qv > 0)).clip(0.01, 0.99)
    lg = np.log(share / (1 - share))
    return lg.rolling(SHORT, min_periods=6).mean() - lg.rolling(LONG, min_periods=60).mean()


def turnover_score(qv: pd.DataFrame, sectors: list[str]) -> pd.DataFrame:
    """secorder O2: log(оборот сектора за 7 дней к 90) минус то же у рынка без BTC."""
    turn = pd.DataFrame(sector_turnover(qv)).reindex(columns=sectors)
    tot = qv.drop(columns=["BTCUSDT"], errors="ignore").sum(axis=1, min_count=1)
    rel = turn.rolling(SHORT, min_periods=6).mean() / turn.rolling(LONG, min_periods=60).mean()
    mrel = tot.rolling(SHORT, min_periods=6).mean() / tot.rolling(LONG, min_periods=60).mean()
    return np.log(rel.div(mrel, axis=0))


def momentum(ex: pd.DataFrame) -> pd.DataFrame:
    return np.log1p(ex.fillna(0.0)).rolling(SHORT, min_periods=6).sum().where(ex.notna())


def composite(parts: list[pd.DataFrame], min_parts: int = 2) -> pd.DataFrame:
    """Среднее z-оценок по секторам за день; сектор без хотя бы min_parts признаков — NaN."""
    zs = []
    for p in (q.reindex(index=parts[0].index, columns=parts[0].columns) for q in parts):
        mu, sd = p.mean(axis=1), p.std(axis=1)
        zs.append(p.sub(mu, axis=0).div(sd.where(sd > 0), axis=0))
    stack = np.stack([z.to_numpy(dtype="float64") for z in zs])
    cnt = np.isfinite(stack).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(stack, axis=0)
    return pd.DataFrame(np.where(cnt >= min_parts, mean, np.nan), index=parts[0].index, columns=parts[0].columns)


# --- Upbit ---

def _upbit(url: str) -> list | None:
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
            time.sleep(0.13)                                    # лимит Upbit ~10 запросов/с
            return data if isinstance(data, list) else None
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(2 + 2 * attempt)
                continue
            if 400 <= e.code < 500:
                return None
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return None


def upbit_turnover(until: pd.Timestamp) -> pd.DataFrame:
    """Оборот KRW-рынков Upbit по дням (UTC), столбцы — <BASE>USDT."""
    markets = _upbit(f"{UPBIT}/market/all") or []
    krw = [m["market"] for m in markets if isinstance(m, dict) and str(m.get("market", "")).startswith("KRW-")]
    cols = {}
    for mk in krw:
        rows, to = [], until.strftime("%Y-%m-%dT%H:%M:%S")
        for _ in range(20):
            got = _upbit(f"{UPBIT}/candles/days?market={mk}&count=200&to={to}")
            if not got:
                break
            rows += got
            first = min(pd.Timestamp(r["candle_date_time_utc"]) for r in got)
            if len(got) < 200 or first <= UPBIT_SINCE:
                break
            to = first.strftime("%Y-%m-%dT%H:%M:%S")
        if rows:
            s = pd.Series({pd.Timestamp(r["candle_date_time_utc"]).floor("D"): float(r["candle_acc_trade_price"])
                           for r in rows})
            cols[mk.split("-", 1)[1] + "USDT"] = s
    print(f"  Upbit: KRW-рынков {len(krw)}, с историей {len(cols)}", flush=True)
    return pd.DataFrame(cols).sort_index()


def upbit_share_score(krw: pd.DataFrame, sectors: list[str], index: pd.DatetimeIndex) -> pd.DataFrame:
    """log(доля сектора в обороте KRW за 7 дней / доля за 90 дней); сектор — >= 2 монеты на Upbit."""
    smap = sector_map()
    tot = krw.drop(columns=["BTCUSDT"], errors="ignore").sum(axis=1, min_count=1)
    out = {}
    for name in sectors:
        cols = [c for c in krw.columns if smap.get(c) == name]
        if len(cols) < 2:
            continue
        share = krw[cols].sum(axis=1, min_count=1) / tot
        out[name] = np.log(share.rolling(SHORT, min_periods=6).mean() / share.rolling(LONG, min_periods=60).mean())
    return pd.DataFrame(out).reindex(index)


# --- режим ---

def btc_bull(close: pd.Series) -> pd.Series:
    """BTC выше недельной EMA50 (закрытие воскресенья) — по дням, значение недели держится до следующего воскресенья."""
    w = close.dropna().resample("W-SUN").last()
    ema = w.ewm(span=50, adjust=False, min_periods=30).mean()
    bull = (w > ema).astype("float64").where(ema.notna())
    return bull.reindex(close.index, method="ffill")


def fear_greed(raw: object) -> pd.Series:
    data = raw.get("data") if isinstance(raw, dict) else None
    if not isinstance(data, list):
        return pd.Series(dtype="float64")
    s = pd.Series({pd.Timestamp(int(r["timestamp"]), unit="s").floor("D"): float(r["value"])
                   for r in data if isinstance(r, dict) and "timestamp" in r and "value" in r})
    return s.sort_index()


def in_regime(z: pd.DataFrame, flag: pd.Series) -> pd.DataFrame:
    """Сделки, у которых на воскресенье перед входом флаг режима True."""
    if z.empty:
        return z
    prev = (z.t - pd.Timedelta(days=1))
    ok = flag.reindex(prev).fillna(False).astype(bool).to_numpy()
    return z[ok].reset_index(drop=True)


def by_year(z: pd.DataFrame) -> str:
    if z.empty:
        return "—"
    g = z.groupby(z.t.dt.year).X
    return "  ".join(f"{y} {m:+.2%} ({n})" for y, m, n in zip(g.mean().index, g.mean(), g.size()))


# --- отчёт ---

def report() -> None:
    print("\n===== POSFLOW: сектора по деривативам, Upbit и режиму рынка (перпетуалы Binance с делистнутыми) =====")
    got = load()
    if got is None:
        print("  posflow: нет данных")
        return
    close, qv = got["close"], got["qv"]
    r, mkt, idx = panel(close, qv)
    live = r.notna()
    sectors = list(idx)
    ex = pd.DataFrame(idx).sub(mkt, axis=0)
    mom = momentum(ex)
    print(f"  монет {close.shape[1]} (с OI {int(got['oi'].notna().any().sum())}), секторов {len(sectors)}, "
          f"дни {close.index[0].date()} … {close.index[-1].date()}")
    passed: list[str] = []
    pos = mom > 0

    def gated(score: pd.DataFrame) -> pd.DataFrame:
        return score.where(pos.reindex_like(score).fillna(False))

    print("\n  --- P1: новые деньги в плечах (рост OI сектора) ---")
    p1 = sector_score(oi_growth(got["oi"]), live, sectors)
    z_p1b = pick_weekly(gated(p1), ex)
    judge("a: наибольший рост OI за 7 дн. → неделя", pick_weekly(p1, ex), passed)
    judge("b: рост OI + сектор в плюсе за 7 дн. → неделя", z_p1b, passed)

    print("\n  --- P2: фандинг ---")
    p2 = sector_score(funding_7d(got["fund"], close), live, sectors)
    judge("a: самый высокий funding за 7 дн. → неделя", pick_weekly(p2, ex), passed)
    judge("b: самый низкий funding за 7 дн. → неделя", pick_weekly(-p2, ex), passed)

    print("\n  --- P3: топ-трейдеры в лонге сильнее нормы ---")
    judge("наибольший перекос топ-трейдеров в лонг → неделя",
          pick_weekly(sector_score(top_tilt(got["top"]), live, sectors), ex), passed)

    print("\n  --- P4: агрессивные покупки ---")
    p4 = sector_score(taker_tilt(got["tb"], qv), live, sectors)
    judge("наибольший рост доли рыночных покупок → неделя", pick_weekly(p4, ex), passed)

    print("\n  --- P5: корейская розница (доля сектора в обороте Upbit KRW) ---")
    krw = upbit_turnover(close.index[-1] + pd.Timedelta(days=1))
    if krw.empty:
        print("    Upbit недоступен")
    else:
        p5 = upbit_share_score(krw, sectors, close.index)
        judge("a: наибольший рост доли → неделя", pick_weekly(p5, ex), passed)
        judge("b: рост доли + сектор в плюсе за 7 дн. → неделя", pick_weekly(gated(p5), ex), passed)
        judge("c: (справка о знаке) наибольший рост доли → против рынка, неделя",
              pick_weekly(p5, -ex), [])

    print("\n  --- C: сводный балл (цена + оборот + агрессоры + OI) ---")
    att = turnover_score(qv, sectors)
    z_c = pick_weekly(composite([mom, att, p4, p1]), ex)
    judge("наибольший сводный балл → неделя", z_c, passed)

    print("\n  --- R: только бычьи недели ---")
    bull = btc_bull(close["BTCUSDT"]) if "BTCUSDT" in close else pd.Series(dtype="object")
    fng = fear_greed(get_json(FNG))
    greed = fng.rolling(SHORT, min_periods=5).mean().reindex(close.index, method="ffill") >= GREED
    print(f"    недель BTC выше EMA50: {int((bull.resample('W-SUN').last() > 0.5).sum())}; "
          f"индекс страха и жадности: {'дней ' + str(len(fng)) if len(fng) else 'недоступен'}")
    r1 = bull.fillna(0.0) > 0.5
    r2 = r1 & greed.fillna(False) if len(fng) else None
    cands = {"лучший сектор за 7 дн.": pick_weekly(mom, ex),
             "O2b оборот + цена": pick_weekly(gated(att), ex),
             "P1b OI + цена": z_p1b, "C сводный": z_c}
    for name, z in cands.items():
        print(f"\n    [{name}]  по годам: {by_year(z)}")
        judge(f"{name} | R1 BTC > EMA50 недельной", in_regime(z, r1), passed, min_n=10)
        if r2 is not None:
            judge(f"{name} | R2 R1 и жадность >= {GREED}", in_regime(z, r2), passed, min_n=10)
        judge(f"{name} | вне R1 (для сравнения)", in_regime(z, ~r1 & bull.notna()), [], min_n=10)
    print("\n  ИТОГ: " + ("прошли — " + "; ".join(passed) if passed else
                          "ни деривативы, ни Upbit, ни режим не дают отбора сектора с запасом на издержки"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 300)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--symbols", default="")
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root), [s for s in a.symbols.split(",") if s])
    else:
        report()
