"""Вход по рынку, на ретесте или гибрид по длине свечи пробоя — на текущих правилах бота, 4h и 15m.

4h — правило бота (research/noline.py: линия, оборот от $20M, агрессоры >= 55%, закрытие в своей половине свечи, дневной
тренд, стоп за свингом, 3R). 15m — правило бота 15m (research/hedge15.py: шорт от пологой линии, топ-150).
Каждый сигнал считается дважды: по рынку на закрытии пробоя и лимиткой на линии (12 свечей). Сравнение на сигнал:
неисполненный ретест — 0R (сделки нет). Гибрид с порогом k: свеча пробоя короче k ATR — рынок, длиннее — ретест (так
работает настройка «гибрид» в панели). Квинтили длины свечи — по IS.
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
from .smc import _atr, _cell
from .tline import PER, _bot_base, _years, market_context, tf_frame
from .tline import coin_trades as tline_trades
from .wide15 import MAX_SLOPE_ATR

THRESHOLDS = (1.0, 1.5, 2.0, 2.5, 3.0)
TOP15 = 150
KEY = ["symbol", "t", "side"]


def collect(root: Path, syms: list[str]) -> None:
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
            x = x[x.trig == "line"].copy()
            rng = ((d["high"] - d["low"]) / _atr(d)).to_numpy()
            x["rng_atr"] = rng[d.index.get_indexer(x["t"])]
            parts.append(x[KEY + ["entry", "R3", "rng_atr"]])
        except Exception as e:
            print(f"  entrycandle {s}: пропуск ({e})", flush=True)
    print(f"  entrycandle: монет со сделками {len(parts)}", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path("entrycandle"), index=False)


def collect15(root: Path, syms: list[str]) -> None:
    ctx = market_context(root)
    parts, advs = [], []
    for s in mine(syms):
        try:
            a = adv30(root, s).dropna()
            advs.append(pd.DataFrame({"symbol": s, "day": a.index, "adv": a.to_numpy()}))
            x = tline_trades(root, s, "15m", ctx)
            if not len(x):
                continue
            x = x[~x.confirm & (x.line == "zz") & (x.side == -1)]
            parts.append(x[KEY + ["entry", "R3", "rng_atr", "aggr", "with_trend", "close_loc", "slope_atr"]].copy())
        except Exception as e:
            print(f"  entrycandle15 {s}: пропуск ({e})", flush=True)
    print(f"  entrycandle15: монет со сделками {len(parts)}", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path("entrycandle15"), index=False)
    if advs:
        pd.concat(advs, ignore_index=True).to_parquet(part_path("entrycandle15_adv"), index=False)


def per_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Одна строка на сигнал: R по рынку, R ретеста (0 — не исполнился), исполнился ли ретест, длина свечи."""
    mk = df[df.entry == "market"].drop_duplicates(KEY)[KEY + ["R3", "rng_atr"]]
    rt = df[df.entry == "retest"].drop_duplicates(KEY)[KEY + ["R3"]].rename(columns={"R3": "R_rt"})
    x = mk.merge(rt, on=KEY, how="left")
    x["filled"] = x["R_rt"].notna()
    x["R_rt"] = x["R_rt"].fillna(0.0)
    return x.rename(columns={"R3": "R_mk"})


def tables(x: pd.DataFrame, periods: dict[str, tuple[str, str]], title: str) -> None:
    x = x.copy()
    x["per"] = ""
    for p, (a, b) in periods.items():
        x.loc[(x.t >= a) & (x.t < b), "per"] = p
    cols = list(periods) + ["2026"]

    def cells(col: str, z: pd.DataFrame) -> dict:
        zz = z.assign(R=z[col])
        return {**{p: _cell(zz[zz.per == p]) for p in periods}, "2026": _cell(zz[zz.t.dt.year == 2026])}

    edges = x.loc[x.per == "is", "rng_atr"].quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
    x["q"] = np.digitize(x["rng_atr"], edges) + 1
    print(f"\n===== ENTRYCANDLE {title}: {len(x)} сигналов; R на сигнал (неисполненный ретест = 0R); квинтили длины "
          f"свечи пробоя, ATR, границы по IS {np.round(edges, 2).tolist()} =====")
    rows = []
    for q, g in x.groupby("q"):
        lo = "<" if q == 1 else f"{edges[q - 2]:.2f}–"
        hi = f"{edges[q - 1]:.2f}" if q <= len(edges) else "∞"
        for nm, col in (("рынок", "R_mk"), ("ретест", "R_rt")):
            rows.append({"свеча, ATR": f"{lo}{hi}", "вход": nm,
                         "исполнено": f"{g.filled.mean():.0%}" if col == "R_rt" else "100%", **cells(col, g)})
    print(pd.DataFrame(rows)[["свеча, ATR", "вход", "исполнено"] + cols].to_string(index=False))
    rows = [{"вариант": "всё по рынку", **cells("R_mk", x)}, {"вариант": "всё ретест", **cells("R_rt", x)}]
    for k in THRESHOLDS:
        h = x.assign(R_h=np.where(x.rng_atr < k, x.R_mk, x.R_rt))
        rows.append({"вариант": f"гибрид {k:.1f} ATR (рынок у {(x.rng_atr < k).mean():.0%})", **cells("R_h", h)})
    print()
    print(pd.DataFrame(rows)[["вариант"] + cols].to_string(index=False))
    for nm, col in (("всё по рынку", "R_mk"), ("всё ретест", "R_rt")):
        print(f"  {nm} по годам: {_years(x, col)}")


def report() -> None:
    parts = all_parts("entrycandle")
    if parts:
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        df["t"] = pd.to_datetime(df["t"], utc=True)
        tables(per_signal(df), PER_ALL, "4h (правило бота)")


def report15() -> None:
    tp, ap = all_parts("entrycandle15"), all_parts("entrycandle15_adv")
    if not tp or not ap:
        print("entrycandle15: частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in tp], ignore_index=True)
    adv = pd.concat([pd.read_parquet(p) for p in ap], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    adv["day"] = pd.to_datetime(adv["day"], utc=True)
    adv["rank"] = adv.groupby("day")["adv"].rank(ascending=False, method="first")
    df["day"] = df.t.dt.floor("D")
    df = df.merge(adv[["symbol", "day", "adv", "rank"]], on=["symbol", "day"], how="left")
    base = _bot_base(df.assign(confirm=False))
    g = base[(base.adv >= ADV_MIN) & (base.slope_atr <= MAX_SLOPE_ATR) & (base["rank"] <= TOP15)]
    tables(per_signal(g), PER, f"15m (правило бота: пологие шорты, топ-{TOP15})")


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
