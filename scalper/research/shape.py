"""Форма консолидации до пробоя линии: ширина, кучность, длина свечей (по вопросу владельца — влияют ли они на силу
пробоя). Направление пробоя у бота задают линия и тренд старшего ТФ, поэтому меряется сила: R сделки.

Сделки — правило бота без изменений (research/noline.py: линия зигзага 3 ATR, агрессоры ≥ 55%, закрытие в своей
половине, тренд, стоп за свингом, 3R; 4h — ретест, 15m — по рынку и ретест). Фигура — свечи от первой опоры линии
(не включая) до свечи пробоя (не включая); «обычная свеча» — средний размах high − low за 100 свечей до опоры.
Признаки и ожидаемое направление заданы до запуска (группа «да» — нижняя треть по IS):
  S1 узкая фигура     (max high − min low) фигуры / обычная свеча / √длины (ширина с поправкой на длительность:
                      у случайного блуждания ширина растёт как √n); ждём, что узкие лучше
  S2 кучная           СКО закрытий вокруг прямой (линейная регрессия по фигуре) / обычная свеча; ждём, что лучше
  S3 мелкие свечи     средний размах свечей фигуры / обычная свеча (сжатие); ждём, что лучше
  S4 сужение к концу  средний размах последней трети фигуры / первой трети < 0.8 (VCP-подобно); ждём, что лучше
Справочно — длина фигуры в свечах (терцили). Принятие (как research/footprint.py): «да» лучше остальных в каждом
периоде, где в обеих группах ≥ 20 сделок, и разница за всё время с t ≥ 2.

    python -m research.shape collect --symbols <1h-qualified>     (4h; 15m — где есть 15m-свечи)
    python -m research.shape report
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
from .footprint import verdict
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _cell
from .tline import PER, tf_frame, zz_lines

TFS = tuple(os.environ.get("SHAPE_TFS", "4h,15m").split(","))
BASE, MIN_SPAN = 100, 6
SPLITS = {
    "S1": "узкая фигура (ширина / обычная свеча / √длины)",
    "S2": "кучная (разброс закрытий вокруг прямой)",
    "S3": "мелкие свечи в фигуре (сжатие)",
    "S4": "сужение к концу (последняя треть < 0.8 первой)",
}
METRIC = {"S1": "width", "S2": "scatter", "S3": "bars"}


def features(hi: np.ndarray, lo: np.ndarray, c: np.ndarray, a: int, t: int) -> dict:
    """Признаки фигуры по свечам a+1 … t−1 и обычной свече за BASE свечей до a."""
    out: dict = {"span": t - a - 1}
    if a < BASE or t - a - 1 < MIN_SPAN:
        return out
    base = float(np.mean(hi[a - BASE:a] - lo[a - BASE:a]))
    h, l, cc = hi[a + 1:t], lo[a + 1:t], c[a + 1:t]
    if not base > 0:
        return out
    n = len(cc)
    x = np.arange(n, dtype="float64")
    resid = cc - np.polyval(np.polyfit(x, cc, 1), x)
    rng = h - l
    third = n // 3
    out.update({"width": float((h.max() - l.min()) / base / np.sqrt(n)), "scatter": float(resid.std() / base),
                "bars": float(rng.mean() / base),
                "narrowing": float(rng[-third:].mean() / rng[:third].mean()) if rng[:third].mean() > 0 else np.nan})
    return out


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str) -> list[dict]:
    noline.TF = tf
    x = noline.coin_trades(d, liq, sym, lines_only=True)
    if not len(x):
        return []
    anchor = {(int(r["t"]), int(r["side"])): int(r["a"]) for r in zz_lines(d) if r["t"] > 0}
    hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("high", "low", "close"))
    rows = []
    for r in x.itertuples(index=False):
        t, side = d.index.get_loc(r.t), int(r.side)
        a = anchor.get((t, side))
        if a is None:
            continue
        rows.append({"symbol": sym, "tf": tf, "t": r.t, "side": side, "entry": r.entry, "R": float(r.R3),
                     **features(hi, lo, c, a, t)})
    return rows


def collect(root: Path, syms: list[str]) -> None:
    rows = []
    for s in mine(syms):
        for tf in TFS:
            try:
                if tf == "15m" and not (root / f"{s}-15m.parquet").exists():
                    continue
                d = tf_frame(root, s, tf)
                if d is None or len(d) < 500:
                    continue
                liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
                rows += coin_rows(d, liq, s, tf)
            except Exception as e:
                print(f"  shape {s} {tf}: пропуск ({e})", flush=True)
    print(f"  shape: сделок {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("shape"), index=False)


def label(z: pd.DataFrame) -> pd.DataFrame:
    """S1–S3: нижняя треть по IS (границы — по сделкам IS этой выборки); S4 — порог 0.8; длина — терцили IS."""
    z = z.copy()
    a, b = PER["is"]
    ins = z[(z.t >= a) & (z.t < b)]
    for key, col in METRIC.items():
        q = ins[col].quantile(1 / 3)
        z[key] = np.where(z[col].isna(), None, np.where(z[col] <= q, "да", "нет"))
    z["S4"] = np.where(z["narrowing"].isna(), None, np.where(z["narrowing"] < 0.8, "да", "нет"))
    q1, q2 = ins["span"].quantile([1 / 3, 2 / 3]).to_numpy()
    z["len"] = np.where(z.span <= q1, f"короткая ≤ {q1:.0f}", np.where(z.span <= q2, "средняя", f"длинная > {q2:.0f}"))
    return z


def report() -> None:
    parts = all_parts("shape")
    print("\n===== SHAPE: ширина, кучность, длина свечей фигуры до пробоя линии и R сделки бота (признаки до запуска; "
          "ячейка — R на сделку, t по дням, прибыльных, сделок в месяц) =====")
    if not parts:
        print("  shape: нет данных")
        return
    x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
    for tf in TFS:
        for entry in ("retest", "market"):
            z0 = x[(x.tf == tf) & (x.entry == entry)]
            if tf == "4h" and entry == "market" or z0.empty:
                continue
            z = label(z0)
            print(f"\n  --- {tf}, вход {entry}: сделок {len(z)}, с признаками {z.S1.notna().sum()}; фигура в медиане "
                  f"{z.span.median():.0f} свечей ---")
            for nm, zz in (("все", z), ("шорты", z[z.side < 0])):
                print(f"   {nm}:")
                for key, title in {**SPLITS, "len": "длина фигуры (справочно)"}.items():
                    print(f"    {key} {title}:")
                    for g in sorted(zz[key].dropna().unique()):
                        m = zz[zz[key] == g]
                        cells = "  ".join(f"{p}: {_cell(m[(m.t >= a) & (m.t < b)])}" for p, (a, b) in PER_ALL.items()
                                          if ((m.t >= a) & (m.t < b)).any())
                        print(f"       {g}: доля {len(m) / max(1, zz[key].notna().sum()):.0%}  {cells}  всё время: {_cell(m)}")
                    if key != "len":
                        line, _ = verdict(zz, key, "да")
                        print(f"       «да» минус остальные: {line}")


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
