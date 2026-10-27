"""Пробой коробки вместе с пробоем EMA10 / EMA20, выход по отрыву от EMA20, EMA200 как тренд и препятствие — стратегия
владельца с графика AVAX 15m (2026-10-27). Части проверялись по отдельности: коробка 15m (research/boxbreak.py —
провал на VAL), EMA-пробои (research/emasys.py — слабый затухающий кандидат), выход по отрыву от EMA вместо 3R на
сделках бота (research/emaexit.py — хуже), EMA200 как тренд (research/ema200.py — хуже EMA50 старшего ТФ). Новое —
связка и фильтр «EMA200 рядом — ход будет маленький».

Свечи 15m и 1h, 70 монет core70, обе стороны; правила заданы до запуска:
    коробка   L = 16 / 32 свечей до пробоя, высота <= 3 ATR, у верха и у низа по >= 2 свечи (тень в пределах 15% высоты
              от границы), прошлое закрытие — внутри коробки
    вход      закрытие за границей коробки, и та же свеча закрывается за EMA10 или EMA20, прошлая — нет (пробой средней);
              по рынку на закрытии; стоп — противоположная граница ∓ 0.1 ATR (риск 0.3–4 ATR)
    EMA200    нет / по тренду (лонг выше EMA200) / не ближе 2 ATR от EMA200 / тренд и не ближе 2 ATR
    выход     отрыв: закрытие дальше k ATR от EMA20 в сторону сделки, k = 2 / 3 / 4 (идея владельца); цель 3R;
              цель — высота коробки × 1 / × 2; везде стоп и срок (15m 200, 1h 120 свечей)
Издержки: taker 5.5 б.п. на вход, выход по стопу / отрыву / сроку, maker 2 б.п. на цель; funding. Вариантов 2 × 2 × 4 × 6
= 96. Принятие: R на сделку > 0 в каждом периоде 2020 / 2021 / IS / VAL / HO, где сделок >= 20 (таких >= 3), t по дням
>= 3 и плюс при издержках x2. Лонги и шорты — для справки по отдельности.

    python -m research.boxema collect --symbols <core70>      # 15m-core70 и 1h
    python -m research.boxema report
"""
from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from .oos import PER_ALL
from .ppr import day_t
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .spec import months
from .tline import HOLD, MAKER, TAKER, tf_frame

TFS = ("15m", "1h")
LS = (16, 32)
MAX_H = 3.0
TOUCH = 0.15
ROOM = 2.0
FILTERS = ("нет", "по тренду EMA200", "не ближе 2 ATR к EMA200", "тренд и не ближе 2 ATR")
EXITS = (("отрыв 2 ATR", 0, 2.0), ("отрыв 3 ATR", 0, 3.0), ("отрыв 4 ATR", 0, 4.0), ("цель 3R", 1, 3.0),
         ("высота × 1", 2, 1.0), ("высота × 2", 2, 2.0))
VARIANTS = list(itertools.product(TFS, LS, FILTERS, range(len(EXITS))))


@njit(cache=True)
def exit_r(o, hi, lo, c, f, e, side, px, stop, mode, k, height, ema20, atr, hold, fee_in):
    """(R, издержки в R) сделки, вход на закрытии бара e. mode 0 — отрыв от EMA20 на k ATR, 1 — цель k·R,
    2 — цель k × высота коробки. Стоп раньше цели в той же свече."""
    risk = side * (px - stop)
    if not risk > 0:
        return np.nan, np.nan
    tp = np.inf
    if mode == 1:
        tp = px + side * k * risk
    elif mode == 2:
        tp = px + side * k * height
    paid = 0.0
    n = len(c)
    for j in range(e + 1, n):
        paid += f[j]
        if (side > 0 and lo[j] <= stop) or (side < 0 and hi[j] >= stop):
            ex = min(stop, o[j]) if side > 0 else max(stop, o[j])
            cost = (fee_in + TAKER) * px / risk
            return side * (ex - px) / risk - cost - side * paid * px / risk, cost
        if mode > 0 and ((side > 0 and hi[j] >= tp) or (side < 0 and lo[j] <= tp)):
            cost = (fee_in + MAKER) * px / risk
            return side * (tp - px) / risk - cost - side * paid * px / risk, cost
        if (mode == 0 and atr[j] > 0 and side * (c[j] - ema20[j]) >= k * atr[j]) or j - e >= hold or j == n - 1:
            cost = (fee_in + TAKER) * px / risk
            return side * (c[j] - px) / risk - cost - side * paid * px / risk, cost
    return np.nan, np.nan


def signals(d: pd.DataFrame, L: int) -> list[tuple[int, int, float, float]]:
    """(бар пробоя, сторона, стоп, высота коробки)."""
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    atr = _atr(d).to_numpy()
    e10 = d["close"].ewm(span=10, adjust=False).mean().to_numpy()
    e20 = d["close"].ewm(span=20, adjust=False).mean().to_numpy()
    top = pd.Series(hi).shift(1).rolling(L).max().to_numpy()
    bot = pd.Series(lo).shift(1).rolling(L).min().to_numpy()
    out = []
    for t in range(L + 1, len(c) - 1):
        a = atr[t - 1]
        if not (a > 0) or not np.isfinite(top[t]):
            continue
        h = top[t] - bot[t]
        if not (0 < h <= MAX_H * a) or not (bot[t] <= c[t - 1] <= top[t]):
            continue
        side = 1 if c[t] > top[t] else -1 if c[t] < bot[t] else 0
        if side == 0:
            continue
        crossed = any(side * (c[t] - e[t]) > 0 and side * (c[t - 1] - e[t - 1]) <= 0 for e in (e10, e20))
        if not crossed:
            continue
        w = slice(t - L, t)
        n_top = int((hi[w] >= top[t] - TOUCH * h).sum())
        n_bot = int((lo[w] <= bot[t] + TOUCH * h).sum())
        if n_top < 2 or n_bot < 2:
            continue
        stop = (bot[t] - 0.1 * a) if side > 0 else (top[t] + 0.1 * a)
        risk = side * (c[t] - stop)
        if 0.3 * a <= risk <= 4.0 * a:
            out.append((t, side, float(stop), float(h)))
    return out


def coin_rows(d: pd.DataFrame, sym: str, tf: str) -> list[dict]:
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].fillna(0.0).to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    atr = _atr(d).to_numpy()
    e20 = d["close"].ewm(span=20, adjust=False).mean().to_numpy()
    e200 = d["close"].ewm(span=200, adjust=False, min_periods=200).mean().to_numpy()
    rows = []
    for L in LS:
        for t, side, stop, h in signals(d, L):
            if not np.isfinite(e200[t]):
                continue
            trend = side * (c[t] - e200[t]) > 0
            room = abs(c[t] - e200[t]) >= ROOM * atr[t]
            row = {"symbol": sym, "tf": tf, "L": L, "t": d.index[t], "side": side,
                   "f0": True, "f1": bool(trend), "f2": bool(room), "f3": bool(trend and room)}
            for i, (_, mode, k) in enumerate(EXITS):
                r, cost = exit_r(o, hi, lo, c, f, t, side, c[t], stop, mode, k, h, e20, atr, HOLD[tf], TAKER)
                row[f"R{i}"], row[f"C{i}"] = r, cost
            rows.append(row)
    return rows


def collect(root: Path, syms: list[str]) -> None:
    for tf in TFS:
        rows = []
        for s in mine(syms):
            try:
                d = tf_frame(root, s, tf)
                if d is None or len(d) < 2000:
                    continue
                rows += coin_rows(d, s, tf)
            except Exception as e:
                print(f"  boxema {s} {tf}: пропуск ({e})", flush=True)
        print(f"  boxema {tf}: сигналов {len(rows)}", flush=True)
        if rows:
            pd.DataFrame(rows).to_parquet(part_path(f"boxema_{tf}"), index=False)


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
    x2 = (v.R - v.C).mean()
    ok = len(pos) >= 3 and all(pos) and np.isfinite(t) and t >= 3 and x2 > 0
    return ok, f"{'  '.join(cells)}\n        всё время {_cell(v)}, издержки x2 {x2:+.3f}R"


def report() -> None:
    print("\n===== BOXEMA: пробой коробки вместе с пробоем EMA10 / EMA20, выход по отрыву от EMA20, EMA200 — тренд и "
          "препятствие (core70; R на сделку, сделок, R в месяц) =====")
    passed = []
    for tf in TFS:
        parts = all_parts(f"boxema_{tf}")
        if not parts:
            print(f"  boxema {tf}: нет данных")
            continue
        x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
        print(f"\n  ===== {tf}: сигналов {len(x)} (лонгов {(x.side > 0).mean():.0%}) =====")
        for L, fi, ei in itertools.product(LS, range(len(FILTERS)), range(len(EXITS))):
            z = x[(x.L == L) & x[f"f{fi}"]].rename(columns={f"R{ei}": "R", f"C{ei}": "C"}).dropna(subset=["R"])
            if len(z) < 20:
                continue
            ok, txt = verdict(z)
            sides = "  ".join(f"{nm} {z[z.side == s].R.mean():+.3f}R ({(z.side == s).sum()})"
                              for nm, s in (("лонги", 1), ("шорты", -1)) if (z.side == s).any())
            print(f"    L {L}, {FILTERS[fi]}, {EXITS[ei][0]}: {'принято' if ok else 'нет'}\n        {txt}\n        {sides}")
            if ok:
                passed.append(f"{tf} L {L} {FILTERS[fi]} {EXITS[ei][0]}")
    print("\n  ИТОГ: " + ("прошли — " + "; ".join(passed) if passed else "связка коробка + EMA не прошла ни в одном варианте"))


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
