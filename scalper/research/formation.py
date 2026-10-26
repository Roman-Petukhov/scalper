"""Паспорт формации у пробоя трендовой линии: какие фигуры (длина, ширина, касания, возраст, сжатие, стоп) работают лучше.
По просьбе владельца — сверка с его журналом; частично проверялось раньше (research/shape.py, tlcanon.py), здесь — всё
вместе, на данных до 2026-10 и со срезом живых настроек 15m (наклон <= 0.04, шорты и обе стороны).

Сделки — правило бота (4h ретест, обе стороны; 15m по рынку). Признаки на свече пробоя t (линия от опоры A до B):
  length   — свечей от A до пробоя;            span   — свечей между опорами A и B;
  age      — свечей от B до пробоя;            height — ширина фигуры: самое далёкое закрытие по другую сторону линии
                                                         между A и пробоем, в ATR свечи пробоя;
  touches  — подходов к линии (как метка бота); slope — наклон линии, ATR за свечу;
  squeeze  — ATR на пробое / ATR на опоре A (меньше 1 — волатильность сжалась, как VCP);
  stop     — расстояние до стопа, ATR.
Принятие (задано до запуска; 8 признаков × 2 хвоста = 16 проверок на срез, t <= −3): срезаем нижний или верхний квинтиль
признака (границы — по IS); отрезанные сделки убыточны в каждом периоде, где их >= 20, t по дням <= −3, R в месяц с
фильтром не ниже. Справочно — R по квинтилям и таблица длина × ширина.

    python -m research.formation collect --symbols <монеты>      # FORM_TFS=4h или 15m
    python -m research.formation report
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
from .oos import PER, PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .spec import cut_verdicts
from .tline import line_touches, tf_frame, zz_lines

TFS = tuple(os.environ.get("FORM_TFS", "4h,15m").split(","))
ENTRY = {"4h": "retest", "15m": "market"}
MAX_SLOPE = 0.04
FEATS = ("length", "span", "age", "height", "touches", "slope", "squeeze", "stop")
NAMES = {"length": "длина A→пробой, свечей", "span": "между опорами, свечей", "age": "от B до пробоя, свечей",
         "height": "ширина фигуры, ATR", "touches": "касаний линии", "slope": "наклон, ATR/свечу",
         "squeeze": "сжатие ATR (пробой / опора A)", "stop": "стоп, ATR"}


def features(c: np.ndarray, atr: np.ndarray, rec: dict, t: int, stop: float, px: float) -> dict:
    a, b, side, slope = int(rec["a"]), int(rec["b"]), int(rec["side"]), float(rec["slope"])
    k = np.arange(a, t)
    line = c[a] + slope * (k - a)
    return {"length": t - a, "span": b - a, "age": t - b,
            "height": float(np.max(-side * (c[a:t] - line)) / atr[t]),
            "touches": line_touches(c, atr, a, t, side, float(c[a]), slope),
            "slope": abs(slope) / atr[t], "squeeze": atr[t] / atr[a] if atr[a] > 0 else np.nan,
            "stop": abs(px - stop) / atr[t]}


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str) -> list[dict]:
    noline.TF = tf
    x = noline.coin_trades(d, liq, sym, lines_only=True)
    if not len(x):
        return []
    x = x[x.entry == ENTRY[tf]]
    c = d["close"].to_numpy(dtype="float64")
    atr = _atr(d).to_numpy()
    recs = {(int(r["t"]), int(r["side"])): r for r in zz_lines(d) if r["t"] > 0}
    rows = []
    for r in x.itertuples(index=False):
        t, side = d.index.get_loc(r.t), int(r.side)
        rec = recs.get((t, side))
        if rec is None or rec.get("log") or not (atr[t] > 0) or not np.isfinite(r.R3):
            continue
        if tf != "4h" and abs(float(rec["slope"])) / atr[t] > MAX_SLOPE:
            continue
        rows.append({"symbol": sym, "tf": tf, "t": r.t, "side": side, "R": float(r.R3),
                     **features(c, atr, rec, t, float(r.stop), float(r.px))})
    return rows


def collect(root: Path, syms: list[str]) -> None:
    for tf in TFS:
        rows = []
        for s in mine(syms):
            try:
                if tf != "4h" and not (root / f"{s}-{tf}.parquet").exists():
                    continue
                d = tf_frame(root, s, tf)
                if d is None or len(d) < 500:
                    continue
                rows += coin_rows(d, adv30(root, s).reindex(d.index.floor("D")).to_numpy(), s, tf)
            except Exception as e:
                print(f"  formation {s} {tf}: пропуск ({e})", flush=True)
        print(f"  formation {tf}: сделок {len(rows)}", flush=True)
        if rows:
            pd.DataFrame(rows).to_parquet(part_path(f"formation_{tf}"), index=False)


def is_edges(z: pd.DataFrame, f: str) -> np.ndarray:
    """Границы квинтилей признака по IS (без заглядывания в VAL / HO)."""
    a, b = PER["is"]
    q = z[(z.t >= a) & (z.t < b)][f].dropna()
    e = np.unique(np.quantile(q, [0.2, 0.4, 0.6, 0.8])) if len(q) >= 50 else np.array([])
    return e


def report() -> None:
    print("\n===== FORMATION: паспорт фигуры у пробоя линии — длина, ширина, касания, возраст, сжатие, стоп (квинтили по IS; "
          "t по дням, прибыльных, сделок в месяц) =====")
    passed = []
    for tf in TFS:
        parts = all_parts(f"formation_{tf}")
        if not parts:
            print(f"  formation {tf}: нет данных")
            continue
        x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
        x = x.dropna(subset=["R"])
        slices = (("обе стороны", x),) if tf == "4h" else (("шорты", x[x.side < 0]), ("обе стороны", x))
        for label, z in slices:
            primary = tf == "4h" or label == "шорты"
            print(f"\n  --- {tf}, вход {ENTRY[tf]}, {label}: сделок {len(z)}, {_cell(z)} ---")
            for f in FEATS:
                e = is_edges(z, f)
                if not len(e):
                    continue
                print(f"    {NAMES[f]} (границы IS: {', '.join(f'{v:.3g}' for v in e)}):")
                q = np.searchsorted(e, z[f].to_numpy(), side="right")
                for i in range(len(e) + 1):
                    m = z[q == i]
                    cells = "  ".join(f"{p}: {_cell(m[(m.t >= a) & (m.t < b)])}" for p, (a, b) in PER_ALL.items()
                                      if ((m.t >= a) & (m.t < b)).any())
                    print(f"      квинтиль {i + 1}: {cells}  всё время: {_cell(m)}")
                for tail, col, thr in (("нижний", f, e[0]), ("верхний", "neg", -e[-1])):
                    zz = z.assign(neg=-z[f])
                    for _, (ok, txt) in cut_verdicts(zz, col, (thr,)).items():
                        print(f"      режем {tail} квинтиль: {'принято' if ok and primary else 'нет'}\n{txt}")
                        if ok and primary:
                            passed.append(f"{tf} {label}: {NAMES[f]}, режем {tail}")
            if len(z) >= 200:
                le, he = is_edges(z, "length"), is_edges(z, "height")
                if len(le) >= 2 and len(he) >= 2:
                    lq = pd.cut(z.length, [-np.inf, le[1], le[-2], np.inf], labels=["короткая", "средняя", "длинная"])
                    hq = pd.cut(z.height, [-np.inf, he[1], he[-2], np.inf], labels=["узкая", "средняя", "широкая"])
                    tab = z.groupby([lq, hq], observed=True).R.agg(["mean", "size"]).unstack()
                    print("    длина × ширина (R на сделку / сделок за всё время):")
                    for li in tab.index:
                        print("      " + f"{li:<9}" + "  ".join(
                            f"{hi}: {tab.loc[li, ('mean', hi)]:+.2f}R / {int(tab.loc[li, ('size', hi)])}"
                            for hi in tab.columns.get_level_values(1).unique() if np.isfinite(tab.loc[li, ('mean', hi)])))
    print("\n  ИТОГ: " + ("прошли — " + "; ".join(passed) if passed else "ни один признак формы фигуры не принят как фильтр"))


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
