"""Горизонтальные уровни на лонгах — вопрос владельца. Уровни проверялись раньше на обеих сторонах сразу (research/noline.py
zzpiv / piv5, research/levels.py, range4, boxbreak) — шортов там большинство; лонги отдельно не смотрели.

Уровни по закрытиям: вершины / впадины зигзага 3 ATR (zz) и свинги PIV=5 (piv5), известны с бара подтверждения.
Фильтры бота: оборот от $20M, тренд старшего ТФ (дневки для 4h, недели для 1d) в сторону сделки; цель 3R, срок бота.
Режимы лонга (шорт — зеркально, для сравнения):
  пробой      закрытие над последней вершиной, лимитка ретеста на уровне (research/noline.py, свеча в верхней
              половине, стоп за свингом PIV=5 − 0.1 ATR); агрессоры: без фильтра / >= 52.5% / >= 55%
  отбой       лимитка на впадине с бара после подтверждения, пока уровень не тронут (срок 120 свечей); первое
              касание — вход (цена — min(открытие, уровень)), стоп уровень − 0.5 ATR
  ложный пробой  закрытие под впадиной, затем закрытие над ней в течение 3 свечей — вход по рынку, стоп под минимумом
              выноса − 0.1 ATR (риск 0.3–4 ATR)
Варианты (заданы до запуска): 4h и 1d × (пробой × 3 порога + отбой + ложный пробой) × zz / piv5 = 20 на лонгах.
Принятие (лонги): R на сделку > 0 в каждом периоде 2020 / 2021 / IS / VAL / HO, где сделок >= 20 (таких >= 3), t по дням
>= 3. Одиночный проход при 20 вариантах на t около 3 может быть случайным.

    python -m research.longlevels collect --symbols <монеты>      # LL_TFS=4h или 1d
    python -m research.longlevels report
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
from .broad import ADV_MIN, adv30
from .oos import PER_ALL
from .ppr import day_t
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .spec import months
from .tline import HOLD, MAKER, PIV, STOP_BUF, TAKER, ZZ_K, htf_trend, pivots, tf_frame, two_targets, zigzag

TFS = tuple(os.environ.get("LL_TFS", "4h,1d").split(","))
VALID = 120
BOUNCE_STOP = 0.5
BACK_BARS = 3
AGGR = {"без фильтра": 0.0, "агрессоры >= 52.5%": 0.525, "агрессоры >= 55%": 0.55}
LEVELS = ("zz", "piv5")


def level_points(c: np.ndarray, a: np.ndarray) -> dict[tuple[str, int], np.ndarray]:
    """(уровень, сторона) → массив (индекс экстремума, бар подтверждения). Сторона +1 — впадины (поддержка для
    лонга), −1 — вершины (сопротивление для шорта)."""
    zh, zl = zigzag(c, a, ZZ_K)
    return {("zz", 1): zl, ("zz", -1): zh, ("piv5", 1): pivots(c, PIV, False), ("piv5", -1): pivots(c, PIV, True)}


def _trade(o, hi, lo, c, f, j: int, side: int, px: float, stop: float, fee: float, hold: int) -> float:
    risk = side * (px - stop)
    if not risk > 0:
        return np.nan
    if (side > 0 and lo[j] <= stop) or (side < 0 and hi[j] >= stop):
        return (side * (stop - px) - (fee + TAKER) * px) / risk
    r, _ = two_targets(o, hi, lo, c, f, j, side, px, stop, 3.0, 3.0, False, hold, fee)
    return float(r)


def bounce_rows(o, hi, lo, c, f, a, trend, liq, pts, side: int, hold: int) -> list[tuple[int, float]]:
    """Отбой: лимитка на уровне с бара после подтверждения до первого касания (не дольше VALID свечей)."""
    out, used = [], set()
    n = len(c)
    for p, conf in pts:
        lv = c[p]
        for j in range(conf + 1, min(conf + 1 + VALID, n - 1)):
            if (side > 0 and lo[j] <= lv) or (side < 0 and hi[j] >= lv):
                if j in used or trend[j - 1] != side or not (liq[j] >= ADV_MIN) or not (a[j - 1] > 0):
                    break
                px = min(o[j], lv) if side > 0 else max(o[j], lv)
                stop = lv - side * BOUNCE_STOP * a[j - 1]
                r = _trade(o, hi, lo, c, f, j, side, px, stop, MAKER, hold)
                if np.isfinite(r):
                    out.append((j, r))
                    used.add(j)
                break
    return out


def false_break_rows(o, hi, lo, c, f, a, trend, liq, pts, side: int, hold: int) -> list[tuple[int, float]]:
    """Ложный пробой: закрытие за уровнем против стороны, затем закрытие обратно в течение BACK_BARS свечей."""
    out, used = [], set()
    n = len(c)
    for p, conf in pts:
        lv = c[p]
        for t in range(conf + 1, min(conf + 1 + VALID, n - 1)):
            if side * (c[t] - lv) < 0:
                ext = lo[t] if side > 0 else hi[t]
                for u in range(t + 1, min(t + 1 + BACK_BARS, n - 1)):
                    ext = min(ext, lo[u]) if side > 0 else max(ext, hi[u])
                    if side * (c[u] - lv) > 0:
                        if u not in used and trend[u] == side and liq[u] >= ADV_MIN and a[u] > 0:
                            stop = ext - side * STOP_BUF * a[u]
                            risk = side * (c[u] - stop)
                            if 0.3 * a[u] <= risk <= 4.0 * a[u]:
                                r, _ = two_targets(o, hi, lo, c, f, u, side, c[u], stop, 3.0, 3.0, False, hold, TAKER)
                                if np.isfinite(r):
                                    out.append((u, float(r)))
                                    used.add(u)
                        break
                break
    return out


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str) -> list[dict]:
    noline.TF = tf
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].fillna(0.0).to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    a = _atr(d).to_numpy()
    trend = htf_trend(d, tf)
    hold = HOLD[tf]
    rows = []
    x = noline.coin_trades(d, liq, sym, min_aggr=0.0, kinds=("zzpiv", "piv5"))
    if len(x):
        x = x[x.trig.isin(("zzpiv", "piv5")) & (x.entry == "retest")]
        for r in x.itertuples(index=False):
            if np.isfinite(r.R3):
                rows.append({"symbol": sym, "tf": tf, "t": r.t, "side": int(r.side), "mode": "пробой",
                             "level": "zz" if r.trig == "zzpiv" else "piv5", "aggr": float(r.aggr), "R": float(r.R3)})
    for (lvl, side), pts in level_points(c, a).items():
        for mode, fn in (("отбой", bounce_rows), ("ложный пробой", false_break_rows)):
            for j, r in fn(o, hi, lo, c, f, a, trend, liq, pts, side, hold):
                rows.append({"symbol": sym, "tf": tf, "t": d.index[j], "side": side, "mode": mode, "level": lvl,
                             "aggr": np.nan, "R": r})
    return rows


def collect(root: Path, syms: list[str]) -> None:
    for tf in TFS:
        rows = []
        for s in mine(syms):
            try:
                d = tf_frame(root, s, tf)
                if d is None or len(d) < 300:
                    continue
                rows += coin_rows(d, adv30(root, s).reindex(d.index.floor("D")).to_numpy(), s, tf)
            except Exception as e:
                print(f"  longlevels {s} {tf}: пропуск ({e})", flush=True)
        print(f"  longlevels {tf}: сделок {len(rows)}", flush=True)
        if rows:
            pd.DataFrame(rows).to_parquet(part_path(f"longlevels_{tf}"), index=False)


def verdict(v: pd.DataFrame) -> tuple[bool, str]:
    cells, pos = [], []
    for p, (a, b) in PER_ALL.items():
        m = v[(v.t >= a) & (v.t < b)]
        if m.empty:
            continue
        cells.append(f"{p} {m.R.mean():+.2f}R ({len(m)}, {m.R.sum() / months(a, b):+.1f}/мес)")
        if len(m) >= 20:
            pos.append(m.R.mean() > 0)
    t = day_t(v, "R") if len(v) > 10 else float("nan")
    return len(pos) >= 3 and all(pos) and np.isfinite(t) and t >= 3, \
        f"{'  '.join(cells)}\n          всё время {_cell(v)}"


def variants(x: pd.DataFrame):
    for lvl in LEVELS:
        z = x[x.level == lvl]
        b = z[z["mode"] == "пробой"]
        for fname, thr in AGGR.items():
            yield f"пробой {lvl}, {fname}", b[b.aggr >= thr]
        for mode in ("отбой", "ложный пробой"):
            yield f"{mode} {lvl}", z[z["mode"] == mode]


def report() -> None:
    print("\n===== LONGLEVELS: горизонтальные уровни на лонгах (шорты зеркально — для сравнения; R на сделку, сделок, "
          "R в месяц) =====")
    passed = []
    for tf in TFS:
        parts = all_parts(f"longlevels_{tf}")
        if not parts:
            print(f"  longlevels {tf}: нет данных")
            continue
        x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
        for label, side in (("лонги", 1), ("шорты", -1)):
            print(f"\n  ===== {tf}, {label} =====")
            for name, z in variants(x[x.side == side]):
                if len(z) == 0:
                    print(f"    {name}: сделок нет")
                    continue
                ok, txt = verdict(z)
                ok = ok and side > 0
                tag = (" → принято" if ok else " → нет") if side > 0 else ""
                print(f"    {name}{tag}\n          {txt}")
                if ok:
                    passed.append(f"{tf} {name}")
    print("\n  ИТОГ (лонги): " + ("прошли — " + "; ".join(passed) if passed else
                              "горизонтальные уровни лонги не вытягивают ни в одном варианте"))


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
