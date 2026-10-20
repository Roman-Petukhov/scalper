"""Стоп и тейк по волатильности (ATR) вместо стопа за свингом (по вопросу владельца). Повод: на дневках стоп 1.5 ATR от
линии был лучшим и самым ровным вариантом (docs/knowledge.md), на 4h и 15m бота не проверялся.

Сигналы и вход — правило бота без изменений (research/noline.py; 4h — ретест на уровне линии, 15m — пологие шорты по
рынку); свеча исполнения та же, меняются только стоп и цель. ATR — свечи пробоя. Варианты (заданы до запуска):
  base       стоп за свингом PIV5 ∓ 0.1 ATR, цель 3R — бот
  atr1 / atr15 / atr2 / atr3   стоп 1 / 1.5 / 2 / 3 ATR от цены входа (у ретеста — от линии), цель 3R
  floor1     стоп за свингом, но не ближе 1 ATR
  cap2       стоп за свингом, но не дальше 2 ATR
  atr15_tp2 / atr15_tp4        стоп 1.5 ATR, цель 2R / 4R
  tp3atr / tp6atr              стоп за свингом, цель 3 / 6 ATR от входа (не в R)
Основной вариант — atr15 (повод с дневок); принятие: R на сделку выше бота в каждом периоде 2020 / 2021 / IS / VAL / HO
(≥ 20 сделок) и парная разница за всё время с t ≥ 2. Остальные — справочно, для них при таком числе вариантов нужен t ≥ 3
(Harvey, Liu, Zhu 2016). Риск везде в R своего стопа — размер позиции бот считает от стопа, поэтому R сравнимы.

    python -m research.volstop collect --symbols <1h-qualified>
    python -m research.volstop report
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
from .tline import HOLD, MAKER, TAKER, tf_frame, two_targets, zz_lines

TFS = tuple(os.environ.get("VOLSTOP_TFS", "4h,15m").split(","))
BOT = {"4h": ("retest", False), "15m": ("market", True)}
MAX_SLOPE, MAIN = 0.025, "atr15"
TITLES = {"base": "свинг, 3R (бот)", "atr1": "стоп 1 ATR, 3R", "atr15": "стоп 1.5 ATR, 3R (основной)",
          "atr2": "стоп 2 ATR, 3R", "atr3": "стоп 3 ATR, 3R", "floor1": "свинг, не ближе 1 ATR",
          "cap2": "свинг, не дальше 2 ATR", "atr15_tp2": "стоп 1.5 ATR, 2R", "atr15_tp4": "стоп 1.5 ATR, 4R",
          "tp3atr": "свинг, цель 3 ATR", "tp6atr": "свинг, цель 6 ATR"}


def variants(px: float, stop: float, side: int, atr: float) -> dict[str, tuple[float, float]]:
    """Вариант → (стоп, цель в R своего стопа)."""
    dist = side * (px - stop)
    at = lambda k: px - side * k * atr
    out = {"base": (stop, 3.0), "atr1": (at(1.0), 3.0), "atr15": (at(1.5), 3.0), "atr2": (at(2.0), 3.0),
           "atr3": (at(3.0), 3.0), "floor1": (stop if dist >= atr else at(1.0), 3.0),
           "cap2": (stop if dist <= 2 * atr else at(2.0), 3.0), "atr15_tp2": (at(1.5), 2.0),
           "atr15_tp4": (at(1.5), 4.0), "tp3atr": (stop, 3 * atr / dist), "tp6atr": (stop, 6 * atr / dist)}
    return out


def sim(o, hi, lo, c, f, fill, side, px, stop, k, fee, hold) -> float:
    risk = side * (px - stop)
    if not (risk > 0):
        return np.nan
    if fee == MAKER and ((side > 0 and lo[fill] <= stop) or (side < 0 and hi[fill] >= stop)):
        return (side * (stop - px) - (fee + TAKER) * px) / risk                # стоп в свече исполнения лимитки
    return float(two_targets(o, hi, lo, c, f, fill, side, px, stop, k, k, False, hold, fee)[0])


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str) -> list[dict]:
    entry, shorts_flat = BOT[tf]
    noline.TF = tf
    x = noline.coin_trades(d, liq, sym, lines_only=True)
    if not len(x):
        return []
    x = x[x.entry == entry]
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    atr = _atr(d).to_numpy()
    slope = {(int(r["t"]), int(r["side"])): abs(float(r["slope"])) for r in zz_lines(d) if r["t"] > 0}
    fee = MAKER if entry == "retest" else TAKER
    rows = []
    for r in x.itertuples(index=False):
        t, side, fill, px, stop = d.index.get_loc(r.t), int(r.side), int(r.fill_i), float(r.px), float(r.stop)
        if not (atr[t] > 0):
            continue
        if shorts_flat and not (side < 0 and slope.get((t, side), np.inf) / atr[t] <= MAX_SLOPE):
            continue
        row = {"symbol": sym, "tf": tf, "t": r.t, "side": side, "risk_atr": side * (px - stop) / atr[t]}
        for v, (sp, k) in variants(px, stop, side, atr[t]).items():
            row[v] = sim(o, hi, lo, c, f, fill, side, px, sp, k, fee, HOLD[tf])
        rows.append(row)
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
                print(f"  volstop {s} {tf}: пропуск ({e})", flush=True)
    print(f"  volstop: сделок {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("volstop"), index=False)


def paired(z: pd.DataFrame, v: str) -> tuple[str, bool, float]:
    cells, ok = [], []
    for p, (a, b) in PER_ALL.items():
        m = z[(z.t >= a) & (z.t < b)]
        if len(m) >= 20:
            dlt = m[v].mean() - m["base"].mean()
            ok.append(dlt > 0)
            cells.append(f"{p}: {dlt:+.3f}")
    dd = (z[v] - z["base"]).to_numpy()
    sd = dd.std(ddof=1) if len(dd) > 2 else 0.0
    t = dd.mean() / (sd / np.sqrt(len(dd))) if sd > 0 else np.nan
    return f"{'  '.join(cells)}  всё время t {t:.1f}", bool(ok) and all(ok), float(t)


def report() -> None:
    parts = all_parts("volstop")
    print("\n===== VOLSTOP: стоп и цель по ATR против стопа за свингом (сигналы и вход бота); R на сделку в R своего стопа "
          "(t по дням, прибыльных, сделок в месяц) =====")
    if not parts:
        print("  volstop: нет данных")
        return
    x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
    x = x.dropna(subset=list(TITLES))
    verdicts = []
    for tf in TFS:
        z = x[x.tf == tf]
        if len(z) < 20:
            continue
        q = z.risk_atr.quantile([0.1, 0.5, 0.9]).to_numpy()
        print(f"\n  --- {tf}, вход {BOT[tf][0]}: сделок {len(z)}; стоп бота в ATR p10 / медиана / p90 "
              f"{q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f} ---")
        for v, title in TITLES.items():
            cells = "  ".join(f"{p}: {_cell(z[(z.t >= a) & (z.t < b)].assign(R=z[v]))}"
                              for p, (a, b) in PER_ALL.items() if ((z.t >= a) & (z.t < b)).any())
            print(f"    {v:10s} {title}: {cells}  всё время: {_cell(z.assign(R=z[v]))}")
            if v == "base":
                continue
            txt, all_ok, t = paired(z, v)
            need = 2.0 if v == MAIN else 3.0
            passed = all_ok and t >= need
            print(f"               минус бот: {txt}  → {'принято' if passed else 'нет'} (нужно t ≥ {need:g})")
            if passed:
                verdicts.append(f"{tf} {v}")
    print("\n  ИТОГ: " + ("прошли — " + ", ".join(verdicts) if verdicts else "ни один вариант не лучше стопа за свингом"))


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
