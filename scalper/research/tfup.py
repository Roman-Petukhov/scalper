"""Правило бота (линия зигзага 3 ATR, фильтры, ретест, 3R — research/noline.py) на старших таймфреймах: 4h (контроль),
12h и 1d. Всё как у 4h, меняется только таймфрейм; срок сделки и тренд старшего ТФ берутся из research/tline.py
(HOLD: 12h — 40, 1d — 30 свечей; тренд — неделя против EMA50)."""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

from . import noline
from .broad import adv30
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _cell
from .tline import _years, tf_frame

TFS = tuple(__import__("os").environ.get("TFUP_TFS", "4h,12h,1d").split(","))


def collect(root: Path, syms: list[str]) -> None:
    parts = []
    for s in mine(syms):
        for tf in TFS:
            try:
                d = tf_frame(root, s, tf)
                if d is None or len(d) < {"2h": 1000, "4h": 500}.get(tf, 150):
                    continue
                noline.TF = tf
                liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
                x = noline.coin_trades(d, liq, s)
                if len(x):
                    x = x[(x.trig == "line") & (x.entry == "retest")]
                if len(x):
                    parts.append(x.assign(tf=tf)[["symbol", "tf", "t", "side", "R3", "risk_pct"]])
            except Exception as e:
                print(f"  tfup {s} {tf}: пропуск ({e})", flush=True)
    print(f"  tfup: частей {len(parts)}", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path("tfup"), index=False)


def report() -> None:
    parts = all_parts("tfup")
    print("\n===== TFUP: правило бота (линия, ретест, 3R) на старших ТФ; ячейка — R на сделку (t по дням, "
          "прибыльных, сделок в месяц) =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["per"] = ""
    for p, (a, b) in PER_ALL.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    for tf in TFS:
        z = df[df.tf == tf]
        print(f"\n  --- {tf}: всего {len(z)} сделок, шортов {(z.side < 0).mean():.0%}, стоп медиана "
              f"{z.risk_pct.median() * 100:.1f}% ---")
        print("  " + "  ".join(f"{p}: {_cell(z.assign(R=z.R3)[z.per == p])}" for p in PER_ALL))
        print(f"  по годам: {_years(z)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x])
    else:
        report()
