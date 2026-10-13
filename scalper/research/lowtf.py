"""Правило бота (линия зигзага 3 ATR, агрессоры от 55%, закрытие в своей половине, тренд старшего ТФ, стоп за свингом
PIV5 + 0.1 ATR, 3R — research/noline.py) на младших ТФ: 15m (контроль), 5m и 1m. Меняется только таймфрейм.

Задано до запуска: все параметры в свечах те же, что на 15m (срок сделки 200 свечей, ретест 12 свечей); тренд — close
старшего ТФ против EMA50: 15m и 5m — 1h, 1m — 15 минут; оборот монеты от $20M за 30 дней; издержки — taker 0.055% /
maker 0.02% на сторону, как у бота на Bybit. Монеты — core16 + ext54. Периоды: 15m и 5m — IS / VAL / HO (с 2022), 1m —
только HO (2025-07 … 2026-09, ~650 тыс. свечей на монету). Чем младше ТФ, тем короче стоп в % цены и тем большую долю R
съедают комиссии — это и проверяется: колонка «издержки» — медиана (вход + выход) / стоп в R.

    python -m research.lowtf collect --symbols <core70>   (по частям)
    python -m research.lowtf report
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import noline
from .broad import adv30
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _cell
from .tline import MAKER, TAKER, _years, tf_frame

TFS = tuple(os.environ.get("LOWTF_TFS", "15m,5m,1m").split(","))


def collect(root: Path, syms: list[str]) -> None:
    parts = []
    for s in mine(syms):
        for tf in TFS:
            try:
                d = tf_frame(root, s, tf)
                if d is None or len(d) < 2000:
                    continue
                noline.TF = tf
                liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
                x = noline.coin_trades(d, liq, s, lines_only=True)
                if len(x):
                    parts.append(x.assign(tf=tf)[["symbol", "tf", "t", "side", "entry", "R3", "risk_pct"]])
            except Exception as e:
                print(f"  lowtf {s} {tf}: пропуск ({e})", flush=True)
    print(f"  lowtf: частей {len(parts)}", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path("lowtf"), index=False)


def report() -> None:
    parts = all_parts("lowtf")
    print("\n===== LOWTF: правило бота (линия, фильтры, стоп за свингом, 3R) на 15m / 5m / 1m; ячейка — R на сделку "
          "(t по дням, прибыльных, сделок в месяц) =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["R"] = df["R3"]
    for tf in TFS:
        for entry in ("retest", "market"):
            z = df[(df.tf == tf) & (df.entry == entry)]
            if z.empty:
                continue
            fee = (MAKER if entry == "retest" else TAKER) + TAKER
            cut = z.R.quantile(0.95)
            print(f"\n  --- {tf}, вход {entry}: {len(z)} сделок на {z.symbol.nunique()} монетах, шортов "
                  f"{(z.side < 0).mean():.0%}, стоп медиана {z.risk_pct.median() * 100:.2f}% цены, издержки ≈ "
                  f"{(fee / z.risk_pct).median():.2f}R, без 5% лучших {z.R[z.R < cut].mean():+.3f}R ---")
            print("  " + "  ".join(f"{p}: {_cell(z[(z.t >= a) & (z.t < b)])}" for p, (a, b) in PER_ALL.items()
                                   if ((z.t >= a) & (z.t < b)).any()))
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
