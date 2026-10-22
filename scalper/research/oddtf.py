"""Нестандартные ТФ: 3m, 30m, 8h (по вопросу владельца: «а мы проверяли такие ТФ, как 3m и т. п.?»). Проверены были 1m,
5m, 15m, 1h, 2h, 4h, 6h, 12h, 1d (docs/knowledge.md); здесь — промежуточные. Правило бота без изменений
(research/noline.py: линии зигзага, агрессоры ≥ 55%, закрытие в своей половине, тренд старшего ТФ, стоп за свингом,
цель 3R); свечи 30m собираются из 15m, 8h — из 1h, 3m — из 1m (1m есть только с 2025-07, поэтому 3m — только HO).
Срок сделки в свечах: 3m 200, 30m 150, 8h 45; тренд: 3m — 1h, 30m — 2h, 8h — день.

Варианты по каждому ТФ: ретест, обе стороны, все линии (как 4h); по рынку, обе стороны; пологие (наклон ≤ 0.025 ATR
за свечу) шорты по рынку (как 15m). Основной вариант, заданный до запуска: 8h — ретест, все линии, обе стороны;
30m и 3m — пологие шорты по рынку. Принятие основного: R на сделку > 0 в каждом периоде 2020 / 2021 / IS / VAL / HO,
где сделок ≥ 20, и за всё время t ≥ 3 (три новых ТФ сразу — повышенный порог). Остальные варианты — справочно.

    python -m research.oddtf collect --symbols <монеты>      # ODD_TFS=8h,30m или 3m
    python -m research.oddtf report
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
from .smc import _atr, _cell
from .tline import MID_TFS, tf_frame, zz_lines

TFS = tuple(os.environ.get("ODD_TFS", "8h,30m,3m").split(","))
MAX_SLOPE = 0.025
MAIN = {"8h": "retest_all", "30m": "market_flat_short", "3m": "market_flat_short"}
TITLES = {"retest_all": "ретест, все линии, обе стороны", "market_all": "по рынку, все линии, обе стороны",
          "market_flat_short": "пологие шорты по рынку"}


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str) -> list[dict]:
    noline.TF = tf
    x = noline.coin_trades(d, liq, sym, lines_only=True)
    if not len(x):
        return []
    atr = _atr(d).to_numpy()
    slope = {(int(r["t"]), int(r["side"])): abs(float(r["slope"])) for r in zz_lines(d) if r["t"] > 0}
    rows = []
    for r in x.itertuples(index=False):
        t, side = d.index.get_loc(r.t), int(r.side)
        if not (atr[t] > 0):
            continue
        rows.append({"symbol": sym, "tf": tf, "t": r.t, "side": side, "entry": r.entry, "R": float(r.R3),
                     "slope_atr": slope.get((t, side), np.inf) / atr[t]})
    return rows


def collect(root: Path, syms: list[str]) -> None:
    for tf in TFS:
        rows = []
        src = MID_TFS[tf][0] if tf in MID_TFS else "1h"
        for s in mine(syms):
            try:
                if not (root / f"{s}-{src}.parquet").exists():
                    continue
                d = tf_frame(root, s, tf)
                if d is None or len(d) < 500:
                    continue
                rows += coin_rows(d, adv30(root, s).reindex(d.index.floor("D")).to_numpy(), s, tf)
            except Exception as e:
                print(f"  oddtf {s} {tf}: пропуск ({e})", flush=True)
        print(f"  oddtf {tf}: сделок {len(rows)}", flush=True)
        if rows:
            pd.DataFrame(rows).to_parquet(part_path(f"oddtf_{tf}"), index=False)


def variant(x: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "retest_all":
        return x[x.entry == "retest"]
    if name == "market_all":
        return x[x.entry == "market"]
    return x[(x.entry == "market") & (x.side < 0) & (x.slope_atr <= MAX_SLOPE)]


def tstat(a: np.ndarray) -> float:
    return float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))) if len(a) > 2 and a.std(ddof=1) > 0 else float("nan")


def verdict(z: pd.DataFrame) -> tuple[bool, float]:
    ok = []
    for _, (a, b) in PER_ALL.items():
        m = z[(z.t >= a) & (z.t < b)]
        if len(m) >= 20:
            ok.append(m.R.mean() > 0)
    t = tstat(z.R.to_numpy())
    return bool(ok) and all(ok) and t >= 3, t


def report() -> None:
    print("\n===== ODDTF: нестандартные ТФ 3m / 30m / 8h — правило бота; R на сделку (t по дням, прибыльных, сделок в месяц) "
          "=====")
    out = []
    for tf in TFS:
        parts = all_parts(f"oddtf_{tf}")
        if not parts:
            print(f"  oddtf {tf}: нет данных")
            continue
        x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
        x = x.dropna(subset=["R"])
        print(f"\n  --- {tf} ---")
        for name, title in TITLES.items():
            z = variant(x, name)
            if z.empty:
                continue
            cells = "  ".join(f"{p}: {_cell(z[(z.t >= a) & (z.t < b)])}" for p, (a, b) in PER_ALL.items()
                              if ((z.t >= a) & (z.t < b)).any())
            main = name == MAIN[tf]
            ok, t = verdict(z)
            tail = f"  → {'принято' if ok else 'нет'} (основной, t {t:.1f})" if main else ""
            print(f"    {title}{' [основной]' if main else ''}: {cells}  всё время: {_cell(z)}{tail}")
            if main:
                out.append(f"{tf}: {'принято' if ok else 'нет'}")
    print("\n  ИТОГ: " + "; ".join(out))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [s for s in a.symbols.split(",") if s])
    else:
        report()
