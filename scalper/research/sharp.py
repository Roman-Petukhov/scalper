"""Резкость фигуры у пробоя линии: отдельные выносы вместо средних по фигуре (research/figure.py мерил среднее тело и
долю длинных свечей — среднее по 50–300 свечам гасит резкие свечи). Плюс сжатие волатильности на 4h как фильтр.
По просьбе владельца: «какие свечи в фигуре — пологие, длинные, резкие, короткие».

Признаки на отрезке опора A … свеча перед пробоем (ATR — своей свечи, без заглядывания вперёд):
  max_bar   — самая большая свеча фигуры: размах, ATR;
  spikes    — сколько выносов (размах > SPIKE ATR) в фигуре;
  late      — 1, если вынос был в последней трети фигуры (резкое движение прямо перед пробоем), иначе 0;
  squeeze   — ATR на пробое / ATR на опоре A (research/formation.py; < 1 — волатильность сжалась, VCP).
Принятие (задано до запуска; как в formation.py / figure.py): max_bar и spikes — срезаем нижний или верхний квинтиль
(границы по IS); late — убираем фигуры с выносом в конце или без него; squeeze — оставляем только сильное сжатие
(нижний квинтиль IS) или нижние два квинтиля. Отрезанные сделки убыточны в каждом периоде, где их >= 20, t по дням
<= −3, R в месяц с фильтром не ниже. Решают 4h (обе стороны) и 15m шорты.

    python -m research.sharp collect --symbols <монеты>      # FORM_TFS=4h или 15m
    python -m research.sharp report
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import formation as F
from .oos import PER_ALL
from .shard import all_parts
from .smc import _cell
from .spec import cut_verdicts

SPIKE = 2.0
FEATS = ("max_bar", "spikes")
NAMES = {"max_bar": "самая большая свеча фигуры, ATR", "spikes": f"выносов > {SPIKE:g} ATR в фигуре"}


def sharp_features(d: pd.DataFrame, atr: np.ndarray, rec: dict, t: int) -> dict:
    a = int(rec["a"])
    rng = (d["high"].to_numpy(dtype="float64") - d["low"].to_numpy(dtype="float64"))[a:t]
    ak = atr[a:t]
    ok = ak > 0
    x = np.where(ok, rng / np.where(ok, ak, 1.0), np.nan)
    third = max((t - a) // 3, 1)
    return {"max_bar": float(np.nanmax(x)) if ok.any() else np.nan,
            "spikes": int(np.nansum(x > SPIKE)),
            "late": int(np.nansum(x[-third:] > SPIKE) > 0)}


def _load(tf: str) -> pd.DataFrame | None:
    parts = all_parts(f"sharp_{tf}")
    if not parts:
        return None
    x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
    return x.dropna(subset=["R"])


def _periods(m: pd.DataFrame) -> str:
    return "  ".join(f"{p}: {_cell(m[(m.t >= a) & (m.t < b)])}" for p, (a, b) in PER_ALL.items()
                     if ((m.t >= a) & (m.t < b)).any())


def _verdict(z: pd.DataFrame, col: str, thr: float, what: str, primary: bool, tag: str, passed: list[str]) -> None:
    for _, (ok, txt) in cut_verdicts(z, col, (thr,)).items():
        print(f"      {what}: {'принято' if ok and primary else 'нет'}\n{txt}")
        if ok and primary:
            passed.append(f"{tag}: {what}")


def report() -> None:
    print("\n===== SHARP: резкость фигуры у пробоя — самая большая свеча, выносы, вынос перед пробоем; сжатие ATR как "
          "фильтр (t по дням, прибыльных, сделок в месяц) =====")
    passed: list[str] = []
    for tf in F.TFS:
        x = _load(tf)
        if x is None:
            print(f"  sharp {tf}: нет данных")
            continue
        slices = (("обе стороны", x),) if tf == "4h" else (("шорты", x[x.side < 0]), ("обе стороны", x))
        for label, z in slices:
            primary = tf == "4h" or label == "шорты"
            tag = f"{tf} {label}"
            print(f"\n  --- {tf}, вход {F.ENTRY[tf]}, {label}: сделок {len(z)}, {_cell(z)} ---")
            for f in FEATS:
                e = F.is_edges(z, f)
                if not len(e):
                    continue
                print(f"    {NAMES[f]} (границы IS: {', '.join(f'{v:.3g}' for v in e)}):")
                q = np.searchsorted(e, z[f].to_numpy(), side="right")
                for i in range(len(e) + 1):
                    print(f"      квинтиль {i + 1}: {_periods(z[q == i])}  всё время: {_cell(z[q == i])}")
                zz = z.assign(neg=-z[f])
                _verdict(zz, f, e[0], "режем нижний квинтиль", primary, f"{tag}: {NAMES[f]}", passed)
                _verdict(zz, "neg", -e[-1], "режем верхний квинтиль", primary, f"{tag}: {NAMES[f]}", passed)
            print(f"    вынос > {SPIKE:g} ATR в последней трети фигуры:")
            for v, nm in ((1, "был"), (0, "не было")):
                m = z[z.late == v]
                print(f"      {nm} ({len(m) / max(len(z), 1):.0%}): {_periods(m)}  всё время: {_cell(m)}")
            zz = z.assign(neg=-z.late)
            _verdict(zz, "neg", -0.5, "убираем фигуры с выносом в конце", primary, tag, passed)
            _verdict(zz, "late", 0.5, "убираем фигуры без выноса в конце", primary, tag, passed)
            e = F.is_edges(z, "squeeze")
            if len(e) >= 2:
                print(f"    сжатие ATR (пробой / опора A; границы IS: {', '.join(f'{v:.3g}' for v in e)}):")
                q = np.searchsorted(e, z["squeeze"].to_numpy(), side="right")
                for i in range(len(e) + 1):
                    print(f"      квинтиль {i + 1}: {_periods(z[q == i])}  всё время: {_cell(z[q == i])}")
                zz = z.assign(neg=-z["squeeze"])
                _verdict(zz, "neg", -e[0], "оставляем только сильное сжатие (квинтиль 1)", primary, tag, passed)
                _verdict(zz, "neg", -e[1], "оставляем квинтили 1–2", primary, tag, passed)
    print("\n  ИТОГ: " + ("прошли — " + "; ".join(passed) if passed else "ни резкость свечей фигуры, ни сжатие ATR "
                                                                          "не приняты как фильтр"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        F.collect(Path(a.root).expanduser(), [s for s in a.symbols.split(",") if s], "sharp", sharp_features)
    else:
        report()
