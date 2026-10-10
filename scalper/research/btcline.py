"""Направление BTC по пробоям его трендовых линий (а не по EMA) и сделки бота 4h в альтах.

Сделки — правило бота без изменений (линия, ретест, 3R — research/noline.py). Для каждой сделки берётся направление
BTC, известное на закрытии свечи сигнала:
  btc4     — сторона последнего пробоя линии BTC на 4h (линии бота: зигзаг 3 ATR), любой пробой;
  btc4bot  — то же, но только пробои, прошедшие фильтры бота (агрессоры >= 55%, закрытие в своей половине свечи);
  btc1d    — сторона последнего пробоя линии BTC на дневках;
  btcema   — для сравнения: дневной BTC выше / ниже EMA50.
Группы: «по BTC» (сторона сделки = направление BTC), «против BTC», и свежесть пробоя (4h — до 5 дней, 15m — до суток).
15m (collect15 / report15): правило бота 15m, плюс направление по пробоям линий BTC на 15m.
Правило «пропускать сделки против BTC» смотрится по R на сделку и R в месяц.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30
from .noline import TF, coin_trades
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _cell
from .tline import BAR_MIN, PER, _bot_base, _years, htf_trend, market_context, tf_frame, zz_lines
from .tline import coin_trades as tline_trades
from .wide15 import MAX_SLOPE_ATR

TOP15 = 150
KEEP15 = ["symbol", "t", "side", "entry", "aggr", "with_trend", "close_loc", "slope_atr", "R3", "confirm"]

WAYS15 = {"btc15": "пробой линии BTC 15m (любой)"}
WAYS = {"btc4": "пробой линии BTC 4h (любой)", "btc4bot": "пробой линии BTC 4h с фильтрами бота",
        "btc1d": "пробой линии BTC 1d", "btcema": "BTC 1d против EMA50 (для сравнения)"}


def break_series(d: pd.DataFrame, tf: str, bot_filter: bool) -> pd.DataFrame:
    """Пробои линий: время, когда пробой известен (закрытие свечи), и сторона; по возрастанию времени."""
    c, lo, h = (d[k].to_numpy(dtype="float64") for k in ("close", "low", "high"))
    buy = (d["taker_buy_volume"] / d["volume"].replace(0, np.nan)).to_numpy()
    rng = h - lo
    loc = (c - lo) / np.where(rng > 0, rng, np.nan)
    rows = []
    for r in zz_lines(d):
        t, side = r["t"], r["side"]
        if t <= 0:
            continue
        if bot_filter:
            aggr = buy[t] if side > 0 else 1 - buy[t]
            cl = loc[t] if side > 0 else 1 - loc[t]
            if not (aggr >= 0.55 and cl >= 0.5):
                continue
        rows.append((d.index[t] + pd.Timedelta(minutes=BAR_MIN[tf]), side))
    return pd.DataFrame(rows, columns=["known", "side"]).sort_values("known").reset_index(drop=True)


def btc_context(root: Path, with15: bool = False) -> dict[str, pd.DataFrame] | None:
    d4, d1 = tf_frame(root, "BTCUSDT", "4h"), tf_frame(root, "BTCUSDT", "1d")
    if d4 is None or d1 is None:
        return None
    ema = pd.DataFrame({"known": d4.index + pd.Timedelta(hours=4), "side": htf_trend(d4, "4h")})
    out = {"btc4": break_series(d4, "4h", False), "btc4bot": break_series(d4, "4h", True),
           "btc1d": break_series(d1, "1d", False), "btcema": ema}
    if with15:
        d15 = tf_frame(root, "BTCUSDT", "15m")
        if d15 is None:
            return None
        out = {"btc15": break_series(d15, "15m", False)} | out
    return out


def direction(ctx: pd.DataFrame, at: pd.Timestamp) -> tuple[int, float]:
    """Сторона последнего события, известного к моменту at, и сколько часов прошло (0, nan — событий не было)."""
    i = int(ctx["known"].searchsorted(at, side="right")) - 1
    if i < 0:
        return 0, np.nan
    return int(ctx["side"].iloc[i]), (at - ctx["known"].iloc[i]).total_seconds() / 3600


def collect(root: Path, syms: list[str]) -> None:
    ctx = btc_context(root)
    if ctx is None:
        print("  btcline: нет данных BTCUSDT", flush=True)
        return
    parts = []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, TF)
            if d is None or len(d) < 500:
                continue
            liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
            x = coin_trades(d, liq, s)
            if not len(x):
                continue
            x = x[(x.trig == "line") & (x.entry == "retest")]
            rows = []
            for r in x.itertuples(index=False):
                at = r.t + pd.Timedelta(hours=4)
                row = {"symbol": s, "t": r.t, "side": int(r.side), "R3": r.R3}
                for k, c in ctx.items():
                    side, age = direction(c, at)
                    row[k], row[f"{k}_age"] = side, age
                rows.append(row)
            if rows:
                parts.append(pd.DataFrame(rows))
        except Exception as e:
            print(f"  btcline {s}: пропуск ({e})", flush=True)
    print(f"  btcline: монет со сделками {len(parts)}", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path("btcline"), index=False)


def collect15(root: Path, syms: list[str]) -> None:
    """Правило бота 15m (research/hedge15.py): шорт от пологой линии, по рынку; фильтры оборота — в отчёте."""
    ctx = btc_context(root, with15=True)
    if ctx is None:
        print("  btcline15: нет данных BTCUSDT", flush=True)
        return
    mctx = market_context(root)
    parts, advs = [], []
    for s in mine(syms):
        try:
            a = adv30(root, s).dropna()
            advs.append(pd.DataFrame({"symbol": s, "day": a.index, "adv": a.to_numpy()}))
            x = tline_trades(root, s, "15m", mctx)
            if not len(x):
                continue
            x = x[~x.confirm & (x.line == "zz") & (x.side == -1) & (x.entry == "market")][KEEP15].copy()
            rows = []
            for r in x.itertuples(index=False):
                at = r.t + pd.Timedelta(minutes=15)
                row = r._asdict()
                for k, c in ctx.items():
                    side, age = direction(c, at)
                    row[k], row[f"{k}_age"] = side, age
                rows.append(row)
            if rows:
                parts.append(pd.DataFrame(rows))
        except Exception as e:
            print(f"  btcline15 {s}: пропуск ({e})", flush=True)
    print(f"  btcline15: монет со сделками {len(parts)}", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path("btcline15"), index=False)
    if advs:
        pd.concat(advs, ignore_index=True).to_parquet(part_path("btcline15_adv"), index=False)


def _months(a: str, b: str) -> float:
    return (pd.Timestamp(b) - pd.Timestamp(a)).days / 30.4


def report() -> None:
    parts = all_parts("btcline")
    print("\n===== BTCLINE: направление BTC по пробоям его линий и сделки бота 4h (линия, ретест, 3R); ячейка — R на "
          "сделку (t по дням, прибыльных, сделок в месяц) =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    tables(df, PER_ALL, WAYS, 30 * 4)


def report15() -> None:
    tp, ap = all_parts("btcline15"), all_parts("btcline15_adv")
    print(f"\n===== BTCLINE15: направление BTC по пробоям его линий и сделки бота 15m (пологие шорты, по рынку, "
          f"топ-{TOP15}, 3R); ячейка — R на сделку (t по дням, прибыльных, сделок в месяц) =====")
    if not tp or not ap:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in tp], ignore_index=True)
    adv = pd.concat([pd.read_parquet(p) for p in ap], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    adv["day"] = pd.to_datetime(adv["day"], utc=True)
    adv["rank"] = adv.groupby("day")["adv"].rank(ascending=False, method="first")
    df["day"] = df.t.dt.floor("D")
    df = df.merge(adv[["symbol", "day", "adv", "rank"]], on=["symbol", "day"], how="left")
    base = _bot_base(df.assign(confirm=False))
    g = base[(base.adv >= ADV_MIN) & (base.slope_atr <= MAX_SLOPE_ATR) & (base["rank"] <= TOP15)].copy()
    tables(g.reset_index(drop=True), PER, WAYS15 | WAYS, 24)


def tables(df: pd.DataFrame, periods: dict[str, tuple[str, str]], ways: dict[str, str], fresh_h: float) -> None:
    df = df.copy()
    df["per"] = ""
    for p, (a, b) in periods.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    per_m = "R/мес " + " / ".join(periods)

    def row(nm: str, z: pd.DataFrame) -> dict:
        return {"группа": nm, **{p: _cell(z.assign(R=z.R3)[z.per == p]) for p in periods},
                per_m: " / ".join(f"{z[z.per == p].R3.sum() / _months(a, b):+.1f}" for p, (a, b) in periods.items())}

    print(f"  сделок {len(df)}, шортов {(df.side < 0).mean():.0%}; «свежий» пробой BTC — не старше {fresh_h:.0f} ч")
    for k, nm in ways.items():
        same, opp = df[df[k] == df.side], df[(df[k] != 0) & (df[k] != df.side)]
        rows = [row("все (бот)", df), row("по BTC", same), row("против BTC", opp)]
        if k != "btcema":
            fresh = df[k + "_age"] <= fresh_h
            rows += [row("по BTC, пробой BTC свежий", same[fresh.loc[same.index]]),
                     row("против BTC, пробой BTC свежий", opp[fresh.loc[opp.index]])]
        print(f"\n  --- {nm} ---")
        print(pd.DataFrame(rows).to_string(index=False))
        print(f"  против BTC по годам: {_years(opp)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 400)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report", "collect15", "report15"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    syms = [x for x in a.symbols.split(",") if x]
    {"collect": lambda: collect(Path(a.root).expanduser(), syms), "report": report,
     "collect15": lambda: collect15(Path(a.root).expanduser(), syms), "report15": report15}[a.mode]()
