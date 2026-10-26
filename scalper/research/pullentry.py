"""Вход на откате после длинной свечи пробоя (идея владельца: длинная свеча — вход по рынку далеко от стопа, лучше
дождаться отката). Сейчас бот после свечи длиннее 2.15 ATR входит по рынку («обратный гибрид»), иначе — ретест линии.

Сигналы — правило бота со свечой пробоя от 2.15 ATR: 4h (обе стороны), 15m (шорты, наклон <= 0.04). Стоп — тот же
(свинг ± 0.1 ATR), цель 3R от своей цены входа, срок как у бота. Способы входа на одних и тех же сигналах:
  рынок     — на закрытии свечи пробоя (как сейчас);
  линия     — лимитка на линии (обычный ретест), 12 свечей;
  откат 30% / 50% — лимитка на 30 / 50% диапазона свечи пробоя от её закрытия назад, 12 свечей.
Лимитка снимается, если цена раньше дошла до первой цели; неисполненная = 0R (R на сигнал). Принятие (задано до запуска):
откат лучше рынка по R на сигнал в каждом периоде 2021 / IS / VAL / HO (где сигналов >= 20) и t по дням разности >= 2.5.

    python -m research.pullentry collect --symbols <монеты>      # PULL_TFS=4h или 15m
    python -m research.pullentry report
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
from .tline import HOLD, MAKER, RETEST_BARS, TAKER, retest_fill, tf_frame, two_targets, zz_lines

TFS = tuple(os.environ.get("PULL_TFS", "4h,15m").split(","))
LONG = 2.15
MAX_SLOPE = 0.04
DEPTHS = (0.3, 0.5)
WAYS = ("рынок", "линия", "откат 30%", "откат 50%")


def limit_r(o, hi, lo, c, f, t: int, side: int, level: float, stop: float, atr: float, tf: str) -> float:
    """R лимитки на уровне level после свечи t (0 — не исполнилась или вне границ риска бота)."""
    risk = side * (level - stop)
    if not (0.3 * atr <= risk <= 4.0 * atr):
        return 0.0
    fill = retest_fill(hi, lo, t, side, level, 3 * risk, RETEST_BARS)
    if fill < 0:
        return 0.0
    if (side > 0 and lo[fill] <= stop) or (side < 0 and hi[fill] >= stop):
        return (side * (stop - level) - (MAKER + TAKER) * level) / risk
    r, _ = two_targets(o, hi, lo, c, f, fill, side, level, stop, 3.0, 3.0, False, HOLD[tf], MAKER)
    return float(r)


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str) -> list[dict]:
    noline.TF = tf
    x = noline.coin_trades(d, liq, sym, lines_only=True)
    if not len(x):
        return []
    x = x[x.entry == "market"]
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    atr = _atr(d).to_numpy()
    recs = {(int(r["t"]), int(r["side"])): r for r in zz_lines(d) if r["t"] > 0}
    rows = []
    for r in x.itertuples(index=False):
        t, side = d.index.get_loc(r.t), int(r.side)
        rec = recs.get((t, side))
        if rec is None or not (atr[t] > 0) or not np.isfinite(r.R3):
            continue
        if tf != "4h" and (side > 0 or abs(float(rec["slope"])) / atr[t] > MAX_SLOPE):
            continue
        rng = hi[t] - lo[t]
        if rng < LONG * atr[t]:
            continue
        stop = float(r.stop)
        res = {"рынок": float(r.R3),
               "линия": limit_r(o, hi, lo, c, f, t, side, float(rec["line_t"]), stop, atr[t], tf)}
        for k in DEPTHS:
            res[f"откат {k:.0%}"] = limit_r(o, hi, lo, c, f, t, side, c[t] - side * k * rng, stop, atr[t], tf)
        rows.append({"symbol": sym, "tf": tf, "t": r.t, "side": side, "range_atr": rng / atr[t], **res})
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
                print(f"  pullentry {s} {tf}: пропуск ({e})", flush=True)
        print(f"  pullentry {tf}: сигналов с длинной свечой {len(rows)}", flush=True)
        if rows:
            pd.DataFrame(rows).to_parquet(part_path(f"pullentry_{tf}"), index=False)


def diff_t(z: pd.DataFrame, a: str, b: str) -> float:
    """t по дням разности R (a − b) на одних и тех же сигналах."""
    dd = (z[a] - z[b]).to_numpy(dtype="float64")
    if len(dd) < 10:
        return float("nan")
    g = pd.Series(dd - dd.mean()).groupby(z.t.dt.floor("D").to_numpy()).sum()
    den = float(np.sqrt((g ** 2).sum()))
    return float(dd.sum() / den) if den > 0 else float("nan")


def report() -> None:
    print("\n===== PULLENTRY: вход после длинной свечи пробоя (>= 2.15 ATR) — рынок, ретест линии, откат 30 / 50% свечи "
          "(R на сигнал, неисполненная лимитка = 0R) =====")
    passed = []
    for tf in TFS:
        parts = all_parts(f"pullentry_{tf}")
        if not parts:
            print(f"  pullentry {tf}: нет данных")
            continue
        x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
        print(f"\n  --- {tf}{' шорты' if tf != '4h' else ''}: сигналов {len(x)}, медиана свечи {x.range_atr.median():.2f} ATR ---")
        for w in WAYS:
            filled = (x[w] != 0).mean()
            cells = "  ".join(f"{p}: {_cell(x[(x.t >= a) & (x.t < b)].assign(R=x[w]))}" for p, (a, b) in PER_ALL.items()
                              if ((x.t >= a) & (x.t < b)).any())
            print(f"    {w:<10} (исполнено {filled:.0%}): {cells}  всё время: {_cell(x.assign(R=x[w]))}")
        for w in WAYS[2:]:
            wins = []
            for p, (a, b) in PER_ALL.items():
                m = x[(x.t >= a) & (x.t < b)]
                if p != "2020" and len(m) >= 20:
                    wins.append(m[w].mean() > m["рынок"].mean())
            t = diff_t(x, w, "рынок")
            ok = bool(wins) and all(wins) and t >= 2.5
            print(f"    {w} против рынка: лучше по периодам {wins}, t разности {t:.1f} → {'принято' if ok else 'нет'}")
            if ok:
                passed.append(f"{tf} {w}")
    print("\n  ИТОГ: " + ("прошли — " + ", ".join(passed) if passed else "откат после длинной свечи не лучше входа по рынку"))


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
