"""5m: сопровождение сделки (по наблюдению владельца: «на 5m цена часто доходит до 2–2.9R, не берёт 3R и уходит
в минус» и «пробой закрепился, а потом свеча хорошо закрылась обратно за трендовой — и сделка ушла в стоп»).
На 4h и 15m оба вопроса закрыты (research/latebe.py, research/lineexit15.py, docs/knowledge.md) — здесь 5m.

Сделки — правило бота (research/noline.py: линия зигзага 3 ATR по закрытиям, агрессоры ≥ 55%, закрытие в своей
половине, тренд старшего ТФ, стоп за свингом, вход по рынку на закрытии свечи пробоя), все ~725 монет с 2022
(как research/holdlen.py), срок 800 свечей. Срез «как бот»: шорты, линия не круче 0.025 ATR за свечу; справочно — все.
Меняется только сопровождение. Варианты (заданы до запуска):
  base     цель 3R, стоп на месте — бот, контроль
  tp2      цель 2R
  tp25     цель 2.5R
  be2      после +2R стоп в точку входа (со следующей свечи)
  lock2    после +2R стоп на +1R (со следующей свечи)
  half2    половина на +2R, остаток к 3R со стопом в точке входа
  back0    выход по закрытию свечи обратно за линией (для шорта — выше линии), всё время сделки
  back05   то же, но закрытие за линией дальше 0.5 ATR («хорошо закрылась»)
  back05h  то же, только первые 12 свечей (час) после входа
Стоп внутри свечи раньше тейка и раньше выхода по линии, как у бота; перенос стопа — только со следующей свечи.
Справочно — разбор наблюдения: сколько сделок дошли до +2R, не взяли 3R и ушли в стоп; сколько закрылись обратно за
линией и чем кончили у бота.
Принятие: вариант лучше бота по R на сделку в каждом периоде (IS / VAL / HO), где ≥ 20 сделок, и парная разница по тем
же сделкам за всё время с t ≥ 2.

    python -m research.exit5 collect --symbols <hot-wide>
    python -m research.exit5 report
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from . import noline
from .broad import adv30
from .oos import PER
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import MAKER, TAKER, tf_frame, zz_lines

TFS = tuple(os.environ.get("EXIT5_TFS", "5m").split(","))
HOLD5, MAX_SLOPE = 800, 0.025
NO = -1.0
#           цель, б/у после, б/у на, часть на, доля части, буфер линии (ATR), окно линии (свечей)
VARIANTS = {"base": (3.0, NO, 0.0, NO, 0.0, NO, 0),
            "tp2": (2.0, NO, 0.0, NO, 0.0, NO, 0),
            "tp25": (2.5, NO, 0.0, NO, 0.0, NO, 0),
            "be2": (3.0, 2.0, 0.0, NO, 0.0, NO, 0),
            "lock2": (3.0, 2.0, 1.0, NO, 0.0, NO, 0),
            "half2": (3.0, 2.0, 0.0, 2.0, 0.5, NO, 0),
            "back0": (3.0, NO, 0.0, NO, 0.0, 0.0, HOLD5),
            "back05": (3.0, NO, 0.0, NO, 0.0, 0.5, HOLD5),
            "back05h": (3.0, NO, 0.0, NO, 0.0, 0.5, 12)}
TITLES = {"base": "3R, стоп на месте (бот)", "tp2": "цель 2R", "tp25": "цель 2.5R", "be2": "б/у после +2R",
          "lock2": "+1R после +2R", "half2": "½ на +2R, остаток к 3R с б/у",
          "back0": "выход по закрытию за линией", "back05": "то же, за линией > 0.5 ATR",
          "back05h": "то же, только первый час"}


@njit(cache=True)
def walk(o, h, lo, c, fund, e, side, entry, stop, k, be_at, be_to, part_at, part_frac, line_e, slope, buf, window,
         max_hold, entry_fee):
    """Сделка с входа на закрытии свечи e. Цель k·R; после +be_at·R (если ≥ 0) со следующей свечи стоп на
    entry + be_to·R; на +part_at·R (если ≥ 0) закрывается доля part_frac (лимитка, maker); выход по закрытию, если
    свеча j ≤ e + window закрылась за линией (line_e + slope·(j − e)) дальше buf (в цене; buf < 0 — выключено).
    Порядок в свече: стоп, часть, цель, линия, срок. Результат: R, наибольший ход в R до выхода, причина
    (0 стоп, 1 цель, 2 срок, 3 линия)."""
    n = len(c)
    risk = side * (entry - stop)
    if not (risk > 0):
        return np.nan, np.nan, -1
    tp = entry + side * k * risk
    sp, armed, part_done = stop, False, False
    rem, booked, fees, paid, mfe = 1.0, 0.0, entry_fee * entry, 0.0, 0.0
    for j in range(e + 1, n):
        paid += fund[j] * rem
        if (side > 0 and lo[j] <= sp) or (side < 0 and h[j] >= sp):
            ex = min(sp, o[j]) if side > 0 else max(sp, o[j])
            pnl = booked + rem * side * (ex - entry) - fees - rem * TAKER * entry
            return pnl / risk - side * paid * entry / risk, mfe, 0
        best = (h[j] - entry) / risk if side > 0 else (entry - lo[j]) / risk
        if best > mfe:
            mfe = best
        if part_at >= 0 and not part_done and best >= part_at:
            booked += part_frac * part_at * risk
            fees += part_frac * MAKER * entry
            rem -= part_frac
            part_done = True
        if (side > 0 and h[j] >= tp) or (side < 0 and lo[j] <= tp):
            pnl = booked + rem * k * risk - fees - rem * MAKER * entry
            return pnl / risk - side * paid * entry / risk, mfe, 1
        back = buf >= 0 and j - e <= window and side * (c[j] - (line_e + slope * (j - e))) < -buf
        if back or j - e >= max_hold or j == n - 1:
            pnl = booked + rem * side * (c[j] - entry) - fees - rem * TAKER * entry
            return pnl / risk - side * paid * entry / risk, mfe, 3 if back else 2
        if be_at >= 0 and not armed and best >= be_at:
            sp, armed = entry + side * be_to * risk, True
    return np.nan, mfe, -1


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str) -> list[dict]:
    noline.TF = tf
    x = noline.coin_trades(d, liq, sym, lines_only=True)
    if not len(x):
        return []
    x = x[x.entry == "market"]
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    atr = _atr(d).to_numpy()
    lines = {(int(r["t"]), int(r["side"])): r for r in zz_lines(d) if r["t"] > 0}
    rows = []
    for r in x.itertuples(index=False):
        t, side, fill, px, stop = d.index.get_loc(r.t), int(r.side), int(r.fill_i), float(r.px), float(r.stop)
        ln = lines.get((t, side))
        if ln is None or not (atr[t] > 0):
            continue
        slope = float(ln["slope"])
        line_e = float(ln["line_t"]) + slope * (fill - t)
        out = {"symbol": sym, "tf": tf, "t": r.t, "side": side, "slope_atr": abs(slope) / atr[t]}
        for v, (k, be_at, be_to, part_at, frac, buf, window) in VARIANTS.items():
            res, m, why = walk(o, hi, lo, c, f, fill, side, px, stop, k, be_at, be_to, part_at, frac, line_e, slope,
                               buf * atr[t] if buf >= 0 else NO, window, HOLD5, TAKER)
            out[v], out[f"why_{v}"] = float(res), int(why)
            if v == "base":
                out["mfe"] = float(m)
        rows.append(out)
    return rows


def collect(root: Path, syms: list[str]) -> None:
    rows = []
    for s in mine(syms):
        for tf in TFS:
            try:
                d = tf_frame(root, s, tf)
                if d is None or len(d) < 2000:
                    continue
                liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
                rows += coin_rows(d, liq, s, tf)
            except Exception as e:
                print(f"  exit5 {s} {tf}: пропуск ({e})", flush=True)
    print(f"  exit5: сделок {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("exit5"), index=False)


def paired(z: pd.DataFrame, v: str) -> tuple[str, bool]:
    cells, ok = [], []
    for p, (a, b) in PER.items():
        m = z[(z.t >= a) & (z.t < b)]
        if len(m) >= 20:
            dlt = m[v].mean() - m["base"].mean()
            ok.append(dlt > 0)
            cells.append(f"{p}: {dlt:+.3f}")
    dd = (z[v] - z["base"]).to_numpy()
    sd = dd.std(ddof=1) if len(dd) > 2 else 0.0
    t = dd.mean() / (sd / np.sqrt(len(dd))) if sd > 0 else np.nan
    passed = bool(ok) and all(ok) and t >= 2
    return f"{'  '.join(cells)}  всё время t {t:.1f}  → {'принято' if passed else 'нет'}", passed


def report() -> None:
    parts = all_parts("exit5")
    print("\n===== EXIT5: сопровождение на 5m — ближе цель, б/у или часть на +2R, выход по закрытию обратно за линией; "
          "R на сделку (t по дням, прибыльных, сделок в месяц) =====")
    if not parts:
        print("  exit5: нет данных")
        return
    x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
    x = x.dropna(subset=list(VARIANTS))
    for tf in TFS:
        z0 = x[x.tf == tf]
        for nm, z in (("как бот: шорты, пологие", z0[(z0.side < 0) & (z0.slope_atr <= MAX_SLOPE)]), ("все", z0)):
            if len(z) < 20:
                continue
            near = z[(z.mfe >= 2.0) & (z.why_base != 1)]
            back = z[z.why_back0 == 3]
            print(f"\n  --- {tf}, {nm}: сделок {len(z)} ---")
            print(f"    дошли до +2R {(z.mfe >= 2).mean():.0%}, взяли 3R {(z.why_base == 1).mean():.0%}; дошли до +2R и не "
                  f"взяли 3R — {len(near) / len(z):.1%} сделок (в стоп {(near.why_base == 0).sum()}, по сроку "
                  f"{(near.why_base == 2).sum()}; в среднем {near.base.mean():+.2f}R)")
            print(f"    закрывались обратно за линией {len(back) / len(z):.0%} сделок; у бота они дали {back.base.mean():+.2f}R "
                  f"(взяли 3R {(back.why_base == 1).mean():.0%}), остальные {z[z.why_back0 != 3].base.mean():+.2f}R")
            for v in VARIANTS:
                cells = "  ".join(f"{p}: {_cell(z[(z.t >= a) & (z.t < b)].assign(R=z[v]))}" for p, (a, b) in PER.items())
                print(f"    {v:8s} {TITLES[v]}: {cells}  всё время: {_cell(z.assign(R=z[v]))}")
                if v != "base":
                    print(f"             минус бот: {paired(z, v)[0]}")


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
