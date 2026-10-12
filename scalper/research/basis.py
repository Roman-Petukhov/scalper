"""Базис перпа (премия к индексу спота) как фактор доходности и как фильтр к сигналам бота 4h.

Повод: Cao, Luo, Cheng, Dong «Anatomy of Cryptocurrency Perpetual Futures Returns» (SSRN 6365329) — из 170 признаков
доходности перпетуалов 63 значимы, и их объясняют два фактора, главный — лог-базис. Funding в разрезе монет мы
проверяли (fe_xs) — пусто; funding — это базис, усреднённый за 8 ч и с ограничителями, а не сам базис.

Данные: premium index klines Binance 1h (close — премия перпа к индексу спота, доля), свечи и funding 1h.
Базис на момент D — среднее premium close за 24 часа до D (часы с open_time в [D − 24 ч, D − 1 ч]), только прошлое.

A. Фактор (все монеты 1h-qualified с оборотом от $20M на день D): каждый день / каждый понедельник в 00:00 UTC монеты
   сортируются по базису на квинтили (от 20 монет); доходность держателя лонга за 1 / 7 дней = изменение цены −
   выплаченный funding. Портфель «лонг дешёвых (Q1) − шорт дорогих (Q5)», равные веса; издержки — тейкер на долю
   сменившихся монет в каждой ноге (выход + вход). Отдельно — вклад цены и funding.
B. Фильтр к боту (первые пробои, агрессоры >= 55%, закрытие в своей половине, тренд, оборот; ретест и рынок):
   «базис за нас» = −сторона × базис (для шорта — перп дороже спота, толпа в лонгах). Правило зафиксировано до
   запуска: базис за нас > 0 — принимается, если лучше дополнения во всех шести периодах. Плюс квинтили по IS.

    python -m research.prepare 1h-premium   (RESEARCH_START=2020-01 RESEARCH_METRICS=0)
    python -m research.basis collect --symbols <монеты с premium>
    python -m research.basis report
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30
from .oos import PER_ALL
from .rebreak import coin_rows
from .shard import all_parts, mine, part_path
from .smc import _cell
from .tline import TAKER, tf_frame

H = pd.Timedelta(hours=1)
HORIZONS = {"день": 24, "неделя": 168}
MIN_COINS = 20
COLS = list(PER_ALL) + ["2026"]


def premium(root: Path, sym: str) -> pd.Series | None:
    p = root / f"{sym}-premium-1h.parquet"
    if not p.exists():
        return None
    k = pd.read_parquet(p, columns=["open_time", "close"])
    s = pd.Series(k["close"].to_numpy(dtype="float64"), index=pd.to_datetime(k["open_time"], unit="ms", utc=True))
    return s[~s.index.duplicated()].sort_index()


def basis_at(prem: pd.Series, at: pd.DatetimeIndex) -> np.ndarray:
    """Среднее premium за 24 часа до момента at (часы с open_time в [at − 24 ч, at − 1 ч]); NaN, если часов < 20."""
    hourly = prem.resample("1h").mean()
    roll = hourly.rolling(24, min_periods=20).mean()                   # на open_time X — часы [X − 23 ч, X]
    return roll.reindex(at - H).to_numpy()


def daily_rows(d1: pd.DataFrame, prem: pd.Series, adv: pd.Series, sym: str) -> pd.DataFrame:
    c, f = d1["close"], d1["funding"]
    days = pd.date_range(d1.index[0].ceil("D"), d1.index[-1].floor("D"), freq="D", tz="UTC")
    if len(days) < 30:
        return pd.DataFrame()
    out = {"symbol": sym, "day": days, "basis": basis_at(prem, days), "adv": adv.reindex(days).to_numpy(),
           "p0": c.reindex(days - H).to_numpy()}
    cf = f.cumsum()
    for name, h in HORIZONS.items():
        p1 = c.reindex(days + (h - 1) * H).to_numpy()
        fund = cf.reindex(days + (h - 1) * H).to_numpy() - cf.reindex(days - H).to_numpy()
        out[f"ret_{h}"] = p1 / out["p0"] - 1
        out[f"fund_{h}"] = fund
    return pd.DataFrame(out).drop(columns="p0")


def bot_rows(root: Path, sym: str, prem: pd.Series, adv: pd.Series) -> pd.DataFrame:
    d = tf_frame(root, sym, "4h")
    if d is None or len(d) < 500:
        return pd.DataFrame()
    ev = coin_rows(d, adv.reindex(d.index.floor("D")).to_numpy(), sym, "4h", ("market", "retest"))
    if not len(ev):
        return ev
    ev = ev[(ev.kind == "first") & ev.with_trend & (ev.adv >= ADV_MIN) & (ev.close_loc >= 0.5) & (ev.aggr >= 0.55)]
    if not len(ev):
        return ev
    t = pd.DatetimeIndex(ev["t"]) + pd.Timedelta(hours=4)              # сигнал известен на закрытии свечи
    b = basis_at(prem, t)
    return pd.DataFrame({"symbol": sym, "t": ev["t"].to_numpy(), "side": ev["side"].to_numpy(),
                         "entry": ev["entry"].to_numpy(), "R3": ev["R3"].to_numpy(),
                         "basis": b, "fav": -ev["side"].to_numpy() * b})


def collect(root: Path, syms: list[str]) -> None:
    xs, bot = [], []
    for s in mine(syms):
        try:
            prem = premium(root, s)
            d1 = tf_frame(root, s, "1h")
            if prem is None or d1 is None or len(d1) < 24 * 60:
                continue
            adv = adv30(root, s)
            a = daily_rows(d1, prem, adv, s)
            if len(a):
                xs.append(a)
            b = bot_rows(root, s, prem, adv)
            if len(b):
                bot.append(b)
        except Exception as e:
            print(f"  basis {s}: пропуск ({e})", flush=True)
    print(f"  basis: монет с днями {len(xs)}, с сигналами бота {len(bot)}", flush=True)
    if xs:
        pd.concat(xs, ignore_index=True).to_parquet(part_path("basis_xs"), index=False)
    if bot:
        pd.concat(bot, ignore_index=True).to_parquet(part_path("basis_bot"), index=False)


# ---------- отчёт ----------

def _period(t: pd.Series) -> pd.Series:
    per = pd.Series("", index=t.index)
    for p, (a, b) in PER_ALL.items():
        per[(t >= a) & (t < b)] = p
    return per


def portfolio(df: pd.DataFrame, h: int, weekly: bool) -> pd.DataFrame:
    """По датам ребалансировки: средняя доходность лонга по квинтилям, ноги Q1 / Q5, оборот ног."""
    z = df.dropna(subset=["basis", f"ret_{h}", f"fund_{h}"])
    z = z[z.adv >= ADV_MIN]
    if weekly:
        z = z[z.day.dt.dayofweek == 0]
    z = z.assign(tot=z[f"ret_{h}"] - z[f"fund_{h}"], px=z[f"ret_{h}"], fd=-z[f"fund_{h}"])
    rows, prev = [], {1: set(), 5: set()}
    for day, g in z.groupby("day"):
        if len(g) < MIN_COINS:
            continue
        q = pd.qcut(g["basis"].rank(method="first"), 5, labels=False) + 1
        row = {"day": day, "n": len(g)}
        for k in range(1, 6):
            row[f"q{k}"] = g.loc[q == k, "tot"].mean()
        for k in (1, 5):
            names = set(g.loc[q == k, "symbol"])
            row[f"turn{k}"] = len(names - prev[k]) / max(len(names), 1)
            row[f"px{k}"], row[f"fd{k}"] = g.loc[q == k, "px"].mean(), g.loc[q == k, "fd"].mean()
            prev[k] = names
        rows.append(row)
    p = pd.DataFrame(rows)
    if not len(p):
        return p
    p["ls"] = p.q1 - p.q5
    p["ls_net"] = p.ls - 2 * TAKER * (p.turn1 + p.turn5)
    p["ls_px"], p["ls_fd"] = p.px1 - p.px5, p.fd1 - p.fd5
    p["per"] = _period(p.day)
    return p


def _stat(x: pd.Series, per_year: float) -> str:
    x = x.dropna()
    if len(x) < 10:
        return f"n={len(x)}"
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    sr = x.mean() / x.std(ddof=1) * np.sqrt(per_year) if x.std(ddof=1) > 0 else np.nan
    return f"{x.mean() * 1e4:+.0f} б.п. (t {t:.1f}, SR {sr:.1f}, {np.mean(x > 0):.0%})"


def report_xs() -> None:
    parts = all_parts("basis_xs")
    if not parts:
        print("basis_xs: частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["day"] = pd.to_datetime(df["day"], utc=True)
    print("\n===== BASIS A: фактор базиса (лонг дешёвых Q1 − шорт дорогих Q5, равные веса, оборот от $20M); "
          "доходность держателя = цена − funding =====")
    for name, h in HORIZONS.items():
        weekly = h == 168
        p = portfolio(df, h, weekly)
        if not len(p):
            continue
        per_year = 52 if weekly else 365
        print(f"\n  --- ребалансировка: {name} (дат {len(p)}, монет на дату — медиана {int(p.n.median())}) ---")
        rows = []
        for col, nm in (("q1", "Q1 (дешёвые) лонг"), ("q2", "Q2"), ("q3", "Q3"), ("q4", "Q4"), ("q5", "Q5 (дорогие) лонг"),
                        ("ls", "Q1 − Q5 до издержек"), ("ls_px", "  из них цена"), ("ls_fd", "  из них funding"),
                        ("ls_net", "Q1 − Q5 после тейкера")):
            rows.append({"ряд": nm, **{per: _stat(p.loc[p.per == per, col], per_year) for per in PER_ALL},
                         "2026": _stat(p.loc[p.day.dt.year == 2026, col], per_year)})
        print(pd.DataFrame(rows)[["ряд"] + COLS].to_string(index=False))
        print(f"  оборот ног за ребалансировку: Q1 {p.turn1.mean():.0%}, Q5 {p.turn5.mean():.0%}")


def report_bot() -> None:
    parts = all_parts("basis_bot")
    if not parts:
        print("basis_bot: частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["per"] = _period(df.t)
    print("\n===== BASIS B: базис на сигнале бота 4h («за нас» = −сторона × базис за 24 ч); R на сделку, 3R =====")

    def row(nm: str, g: pd.DataFrame) -> dict:
        g = g.assign(R=g.R3)
        return {"группа": nm, **{p: _cell(g[g.per == p]) for p in PER_ALL}, "2026": _cell(g[g.t.dt.year == 2026])}

    def means(g: pd.DataFrame) -> list[float]:
        return [g.loc[g.per == p, "R3"].mean() for p in PER_ALL] + [g.loc[g.t.dt.year == 2026, "R3"].mean()]

    for entry in ("retest", "market"):
        z = df[(df.entry == entry) & df.fav.notna()]
        print(f"\n  --- вход: {'ретест' if entry == 'retest' else 'по рынку'}; базис известен у {len(z)} из "
              f"{(df.entry == entry).sum()} сигналов; медиана базиса {z.basis.median() * 1e4:+.1f} б.п. ---")
        isz = z[z.per == "is"]
        edges = np.unique(np.quantile(isz.fav, [0.2, 0.4, 0.6, 0.8])) if len(isz) >= 50 else np.array([])
        rows = [row("сигналы бота", z)]
        if len(edges):
            q = np.digitize(z.fav.to_numpy(), edges)
            rows += [row(f"базис за нас Q{k + 1}" + (f" (< {edges[k] * 1e4:+.1f} б.п.)" if k < len(edges) else ""), z[q == k])
                     for k in range(len(edges) + 1)]
        yes, no = z[z.fav > 0], z[z.fav <= 0]
        rows += [row("правило: базис за нас > 0", yes), row("   базис против нас", no)]
        print(pd.DataFrame(rows)[["группа"] + COLS].to_string(index=False))
        diff = [a - b for a, b in zip(means(yes), means(no))]
        ok = all(np.isfinite(v) and v > 0 for v in diff)
        print(f"  доля «за нас» {len(yes) / max(len(z), 1):.0%}; «за нас − против» "
              + ", ".join(f"{p} {v:+.2f}" for p, v in zip(COLS, diff))
              + f" → {'лучше во всех периодах' if ok else 'не во всех периодах'}")


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
        report_xs()
        report_bot()
