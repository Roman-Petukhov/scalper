"""Какую цель ставить на 4h: правило бота без изменений (вход на ретесте и по рынку), меняется только выход.

Цели: всё на 1.5 / 2 / 2.5 / 3 (бот) / 4R; половина на 1R, 1.5R или 2R со стопом в безубыток и вторая половина на 2R,
3R или 4R. Стоп проверяется раньше тейка, срок сделки 60 свечей, комиссии и funding — как в research/tline.py.
По каждому выходу: R на сделку, доля прибыльных, R в месяц, худшая просадка в R, доля убыточных месяцев и разброс
месяца — то, что видно на счёте. Периоды 2020, 2021 (данные с RESEARCH_START=2020-01), IS, VAL, HO.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import tline
from .broad import ADV_MIN
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _cell
from .tline import TAKER, _bot_base, _years, coin_trades, market_context

EXITS = (("R15", "всё на 1.5R"), ("R2", "всё на 2R"), ("R25", "всё на 2.5R"), ("R3", "всё на 3R (бот)"),
         ("R4", "всё на 4R"), ("R1_2be", "1/2 на 1R, б/у, 1/2 на 2R"), ("R15_3be", "1/2 на 1.5R, б/у, 1/2 на 3R"),
         ("R2_4be", "1/2 на 2R, б/у, 1/2 на 4R"))
COLS = ["symbol", "t", "side", "line", "entry", "confirm", "aggr", "with_trend", "close_loc", "adv", "risk_pct",
        *(c for c, _ in EXITS)]


def collect(root: Path, syms: list[str]) -> None:
    ctx = market_context(root)
    tline.LINES, tline.SCALE_LINES = ("zz",), {}
    parts = []
    for s in mine(syms):
        try:
            x = coin_trades(root, s, "4h", ctx)
            if len(x):
                parts.append(x[(x.line == "zz") & ~x.confirm][COLS].copy())
        except Exception as e:
            print(f"  targets4 {s}: пропуск ({e})", flush=True)
    print(f"  targets4: монет со сделками {len(parts)}", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path("targets4"), index=False)


def _dd(z: pd.DataFrame, col: str) -> float:
    eq = z.sort_values("t")[col].cumsum().to_numpy()
    return float((np.maximum.accumulate(np.r_[0.0, eq]) - np.r_[0.0, eq]).max()) if len(eq) else 0.0


def _months(z: pd.DataFrame, col: str, a: str, b: str) -> pd.Series:
    idx = pd.period_range(pd.Timestamp(a), pd.Timestamp(b) - pd.Timedelta(days=1), freq="M")
    return z.groupby(z.t.dt.tz_localize(None).dt.to_period("M"))[col].sum().reindex(idx, fill_value=0.0)


def report() -> None:
    parts = all_parts("targets4")
    print("\n===== TARGETS4: правило бота 4h, меняется только выход; ячейка — R на сделку (t по дням, прибыльных, "
          "сделок в месяц) =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["per"] = ""
    for p, (a, b) in PER_ALL.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    base = _bot_base(df[df.adv.isna() | (df.adv >= ADV_MIN)])
    for entry in ("retest", "market"):
        exit_tables(base[base.entry == entry], PER_ALL, "ретест (бот)" if entry == "retest" else "рынок")


def exit_tables(g: pd.DataFrame, periods: dict[str, tuple[str, str]], title: str) -> None:
    """Таблицы по выходам EXITS для сделок g (колонки t, per, risk_pct и EXITS)."""
    print(f"\n  --- {title}: R на сделку ---")
    rows = [{"выход": nm, **{p: _cell(g[g.per == p].assign(R=g[col])) for p in periods}} for col, nm in EXITS]
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n  --- {title}: как на счёте (R в месяц / худшая просадка R / убыточных месяцев / разброс месяца R) ---")
    rows = []
    for col, nm in EXITS:
        row = {"выход": nm}
        for p, (a, b) in periods.items():
            z = g[g.per == p]
            m = _months(z, col, a, b)
            row[p] = f"{m.mean():+.1f} / {_dd(z, col):.1f} / {(m < 0).mean():.0%} / {m.std():.1f}" if len(z) else "—"
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    for col, nm in EXITS:
        print(f"  {nm}: по годам {_years(g, col)}; издержки x2: " + ", ".join(
            f"{p}: {(z[col] - 2 * TAKER / z.risk_pct).mean():+.3f}" for p, z in g.groupby("per") if p))

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
