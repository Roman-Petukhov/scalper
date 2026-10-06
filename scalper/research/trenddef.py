"""Какое определение тренда лучше выбирает сторону: сравнение на одних и тех же сигналах 4h.

Сигналы: (1) правило бота без фильтра тренда — линии по значимым точкам, агрессоры >= 55%, закрытие в верхней половине
свечи, ретест, 3R; (2) пробой линии коррекции (research/pullback.py), 4h, по рынку, агрессоры >= 55%, 3R.
Тренд по дневкам, известный на закрытии свечи сигнала (последняя закрытая дневка); сделка «по тренду», если знак
тренда совпадает со стороной. Определения заданы заранее:
    ema50       закрытие выше / ниже EMA50 (бот сейчас)
    ema50_slope EMA50 выше / ниже, чем 5 дней назад
    ema20_50    EMA20 выше / ниже EMA50
    mom7/14/28  знак изменения цены за 7 / 14 / 28 дней (time-series momentum, Moskowitz–Ooi–Pedersen)
    er20        эффективность движения Кауфмана за 20 дней: (закрытие − закрытие 20 дней назад) / сумма |изменений|;
                тренд — знак, если |ER| > 0.3, иначе тренда нет
    adx14       ADX > 25, направление — +DI против −DI; иначе тренда нет
    don20/55    закрытие в верхней / нижней половине канала Дончиана за 20 / 55 дней
    vote        большинство из ema50, ema50_slope, mom28, don55 (3 из 4)
Периоды 2020, 2021 (RESEARCH_START=2020-01), IS, VAL, HO.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import tline
from .broad import ADV_MIN, adv30
from .oos import PER_ALL
from .pullback import coin_pullbacks
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import _htf_bars, _years, coin_trades, market_context, tf_frame

DEFS = ("ema50", "ema50_slope", "ema20_50", "mom7", "mom14", "mom28", "er20", "adx14", "don20", "don55", "vote")
COLS = ["symbol", "t", "side", "line", "entry", "confirm", "aggr", "with_trend", "close_loc", "adv", "risk_pct", "R3"]


def _adx(day: pd.DataFrame, n: int = 14) -> tuple[pd.Series, pd.Series]:
    h, l_, c = day["high"], day["low"], day["close"]
    up, dn = h.diff(), -l_.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    ndm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = pd.concat([h - l_, (h - c.shift()).abs(), (l_ - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=1 / n, adjust=False).mean() / atr
    ndi = 100 * ndm.ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), np.sign(pdi - ndi)


def daily_trends(d: pd.DataFrame) -> pd.DataFrame:
    """Знак тренда (+1 / −1 / 0) по каждому определению на каждой закрытой дневке; индекс — время закрытия дневки."""
    day = _htf_bars(d, "1D")
    c = day["close"]
    e50, e20 = c.ewm(span=50, adjust=False).mean(), c.ewm(span=20, adjust=False).mean()
    out = pd.DataFrame(index=day.index)
    out["ema50"] = np.sign(c - e50)
    out["ema50_slope"] = np.sign(e50 - e50.shift(5))
    out["ema20_50"] = np.sign(e20 - e50)
    for n in (7, 14, 28):
        out[f"mom{n}"] = np.sign(c - c.shift(n))
    er = (c - c.shift(20)) / c.diff().abs().rolling(20).sum()
    out["er20"] = np.where(er.abs() > 0.3, np.sign(er), 0.0)
    adx, dirn = _adx(day)
    out["adx14"] = np.where(adx > 25, dirn, 0.0)
    for n in (20, 55):
        hi, lo = c.rolling(n).max(), c.rolling(n).min()
        out[f"don{n}"] = np.sign(c - (hi + lo) / 2)
    votes = out[["ema50", "ema50_slope", "mom28", "don55"]].sum(axis=1)
    out["vote"] = np.where(votes >= 3, 1.0, np.where(votes <= -3, -1.0, 0.0))
    warm = pd.Series(np.arange(len(out)) >= 60, index=out.index)
    out = out.where(warm)                                            # без 60 дней истории тренд не определён
    out.index = out.index + pd.Timedelta(days=1)
    return out


def _attach(x: pd.DataFrame, tr: pd.DataFrame) -> pd.DataFrame:
    close_t = pd.to_datetime(x["t"], utc=True) + pd.Timedelta(hours=4)
    k = tr.index.searchsorted(close_t, side="right") - 1
    vals = tr.to_numpy()
    sel = np.where(k[:, None] >= 0, vals[np.clip(k, 0, None)], np.nan) if len(vals) else np.full((len(x), len(DEFS)), np.nan)
    side = x["side"].to_numpy()[:, None]
    agree = pd.DataFrame(np.where(np.isnan(sel), np.nan, (sel * side > 0).astype(float)), columns=[f"tr_{c}" for c in DEFS],
                         index=x.index)
    return pd.concat([x, agree], axis=1)


def collect(root: Path, syms: list[str]) -> None:
    ctx = market_context(root)
    tline.LINES, tline.SCALE_LINES = ("zz",), {}
    bots, pulls = [], []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, "4h")
            if d is None or len(d) < 500:
                continue
            tr = daily_trends(d)[list(DEFS)]
            x = coin_trades(root, s, "4h", ctx)
            if len(x):
                x = x[(x.line == "zz") & ~x.confirm & (x.entry == "retest")][COLS].reset_index(drop=True)
                bots.append(_attach(x, tr))
            pb = coin_pullbacks(d, "4h", adv30(root, s).reindex(d.index.floor("D")).to_numpy(), s)
            if len(pb):
                pb = pb[(pb.entry == "market") & (pb.aggr >= 0.55)].reset_index(drop=True)
                pulls.append(_attach(pb, tr))
        except Exception as e:
            print(f"  trenddef {s}: пропуск ({e})", flush=True)
    print(f"  trenddef: монет с ботом {len(bots)}, с откатами {len(pulls)}", flush=True)
    if bots:
        pd.concat(bots, ignore_index=True).to_parquet(part_path("trenddef_bot"), index=False)
    if pulls:
        pd.concat(pulls, ignore_index=True).to_parquet(part_path("trenddef_pull"), index=False)


def _per(df: pd.DataFrame) -> pd.DataFrame:
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["per"] = ""
    for p, (a, b) in PER_ALL.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    return df


def _pm(z: pd.DataFrame) -> str:
    return " / ".join(f"{z[z.per == p].R3.sum() / ((pd.Timestamp(b) - pd.Timestamp(a)).days / 30.4):+.1f}"
                      for p, (a, b) in PER_ALL.items())


def _block(title: str, g: pd.DataFrame) -> None:
    print(f"\n  --- {title}: сделки по тренду (знак совпал со стороной) для каждого определения ---")
    rows = [{"тренд": "без фильтра тренда", **{p: _cell(g.assign(R=g.R3)[g.per == p]) for p in PER_ALL},
             "R/мес " + " / ".join(PER_ALL): _pm(g)}]
    for dname in DEFS:
        z = g[g[f"tr_{dname}"] == 1]
        rows.append({"тренд": dname + (" (бот)" if dname == "ema50" else ""),
                     **{p: _cell(z.assign(R=z.R3)[z.per == p]) for p in PER_ALL}, "R/мес " + " / ".join(PER_ALL): _pm(z)})
    print(pd.DataFrame(rows).to_string(index=False))
    rows = []
    for dname in DEFS:
        z = g[g[f"tr_{dname}"] == 0]
        rows.append({"против тренда / нет тренда": dname, **{p: _cell(z.assign(R=z.R3)[z.per == p]) for p in PER_ALL}})
    print(pd.DataFrame(rows).to_string(index=False))


def report() -> None:
    bp, pp = all_parts("trenddef_bot"), all_parts("trenddef_pull")
    print("\n===== TRENDDEF: определения тренда по дневкам на сигналах 4h; ячейка — R на сделку (t по дням, прибыльных, "
          "сделок в месяц), всё на 3R =====")
    if not bp:
        print("частей нет")
        return
    bot = _per(pd.concat([pd.read_parquet(p) for p in bp], ignore_index=True))
    bot = bot[(bot.adv.isna() | (bot.adv >= ADV_MIN)) & (bot.aggr >= 0.55) & (bot.close_loc >= 0.5)]
    m = bot.tr_ema50.notna()
    print(f"сверка: ema50 здесь совпадает с фильтром бота в {(bot[m].tr_ema50 == bot[m].with_trend).mean():.1%} сделок")
    _block("правило бота без фильтра тренда (ретест)", bot)
    for dname in ("ema50", "vote", "mom28"):
        z = bot[bot[f"tr_{dname}"] == 1]
        print(f"  бот + {dname}, по годам: {_years(z)}")
    if pp:
        pull = _per(pd.concat([pd.read_parquet(p) for p in pp], ignore_index=True))
        _block("пробой линии коррекции, 4h, рынок, агрессоры >= 55%", pull)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 400)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x])
    else:
        report()
