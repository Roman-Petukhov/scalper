"""То же, что research/extension.py, для 15m: правило бота 15m (пологие шорты, по рынку, топ-150 по обороту на день
сигнала, как в research/wide15.py) — насколько цена уже ушла в сторону сделки к сигналу. Признаки те же (сутки, 7 и
30 дней, EMA50 и RSI дневок, место в диапазоне 90 дней, BTC за 7 дней), квинтили и правило — по IS. Данные с 2022.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

from .broad import ADV_MIN, adv30
from .extension import features, quintile_report
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .tline import _bot_base, coin_trades, market_context, tf_frame
from .wide15 import MAX_SLOPE_ATR

TOP = 150
COLS = ["symbol", "t", "side", "entry", "aggr", "with_trend", "close_loc", "slope_atr", "risk_pct", "R3", "btc_ret7"]


def collect(root: Path, syms: list[str]) -> None:
    ctx = market_context(root)
    trades, advs = [], []
    for s in mine(syms):
        try:
            a = adv30(root, s).dropna()
            advs.append(pd.DataFrame({"symbol": s, "day": a.index, "adv": a.to_numpy()}))
            x = coin_trades(root, s, "15m", ctx)
            if not len(x):
                continue
            x = x[~x.confirm & (x.line == "zz") & (x.side == -1) & (x.entry == "market")][COLS].reset_index(drop=True)
            d = tf_frame(root, s, "15m")
            pos = d.index.get_indexer(pd.to_datetime(x["t"], utc=True))
            x = pd.concat([x, features(d, pos, x["side"].to_numpy(), "15m")], axis=1)
            x["btc7"] = x["side"] * x["btc_ret7"]
            trades.append(x)
        except Exception as e:
            print(f"  extension15 {s}: пропуск ({e})", flush=True)
    print(f"  extension15: монет со сделками {len(trades)}", flush=True)
    if trades:
        pd.concat(trades, ignore_index=True).to_parquet(part_path("extension15"), index=False)
    if advs:
        pd.concat(advs, ignore_index=True).to_parquet(part_path("extension15_adv"), index=False)


def report() -> None:
    tp, ap = all_parts("extension15"), all_parts("extension15_adv")
    print(f"\n===== EXTENSION15: 15m, правило бота (пологие шорты, по рынку, топ-{TOP}, 3R) — насколько цена уже ушла "
          "в сторону сделки к сигналу; ячейка — R на сделку (t по дням, прибыльных, сделок в месяц) =====")
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
    for p, (a, b) in PER_ALL.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    g = _bot_base(df.assign(confirm=False))
    quintile_report(g[(g.adv >= ADV_MIN) & (g.slope_atr <= MAX_SLOPE_ATR) & (g["rank"] <= TOP)])


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
