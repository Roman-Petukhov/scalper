"""Какую цель ставить на 15m: правило бота для 15m без изменений, меняется только выход (выходы — research/targets4).

Правило: шорт от пологой линии (наклон <= 0.025 ATR на свечу), агрессоры >= 55%, тренд старшего ТФ, закрытие в верхней
половине свечи, вход по рынку, оборот от $20M и место в рейтинге оборота на день сигнала <= 150 (настройка бота
«топ-150»; рейтинг — среди 725 монет, adv30 без заглядывания вперёд, как в research/wide15.py). Срок сделки 200 свечей.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

from .broad import ADV_MIN, adv30
from .shard import all_parts, mine, part_path
from .targets4 import EXITS, exit_tables
from .tline import PER, _bot_base, coin_trades, market_context
from .wide15 import MAX_SLOPE_ATR

TOP = 150
COLS = ["symbol", "t", "side", "entry", "aggr", "with_trend", "close_loc", "slope_atr", "risk_pct", *(c for c, _ in EXITS)]


def collect(root: Path, syms: list[str]) -> None:
    ctx = market_context(root)
    trades, advs = [], []
    for s in mine(syms):
        try:
            a = adv30(root, s).dropna()
            advs.append(pd.DataFrame({"symbol": s, "day": a.index, "adv": a.to_numpy()}))
            x = coin_trades(root, s, "15m", ctx)
            if len(x):
                trades.append(x[~x.confirm & (x.line == "zz") & (x.side == -1)][COLS])
        except Exception as e:
            print(f"  targets15 {s}: пропуск ({e})", flush=True)
    print(f"  targets15: монет со сделками {len(trades)}, с оборотом {len(advs)}", flush=True)
    if trades:
        pd.concat(trades, ignore_index=True).to_parquet(part_path("targets15"), index=False)
    if advs:
        pd.concat(advs, ignore_index=True).to_parquet(part_path("targets15_adv"), index=False)


def report() -> None:
    tp, ap = all_parts("targets15"), all_parts("targets15_adv")
    print(f"\n===== TARGETS15: правило бота 15m (пологие шорты, по рынку, топ-{TOP} по обороту на день сигнала), меняется "
          "только выход; ячейка — R на сделку (t по дням, прибыльных, сделок в месяц) =====")
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
    df["per"] = ""
    for p, (a, b) in PER.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    base = _bot_base(df.assign(confirm=False))
    g = base[(base.entry == "market") & (base.adv >= ADV_MIN) & (base.slope_atr <= MAX_SLOPE_ATR) & (base["rank"] <= TOP)]
    exit_tables(g, PER, f"15m, топ-{TOP}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 400)
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("mode", choices=["collect", "report"])
    ap_.add_argument("--root", default="~/bn")
    ap_.add_argument("--symbols", default="")
    a_ = ap_.parse_args()
    if a_.mode == "collect":
        collect(Path(a_.root).expanduser(), [x for x in a_.symbols.split(",") if x])
    else:
        report()
