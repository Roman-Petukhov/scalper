"""Длинные выходы только на лонгах. Проверки целей (4–6R, лестница, трейлинг, ранний выход) шли на всех сделках бота —
это на 90% шорты; на лонгах отдельно не проверялись. Гипотеза владельца и асимметрия рынка: падения резкие (3R берётся
быстро), рост медленный — 3R его обрезает (разбор NEAR 2026: ×3.9 за 3 месяца).

Сделки — лонговые пробои линий бота (зигзаг 3 ATR) на 4h и 2h: оборот от $20M, дневной тренд EMA50 вверх, закрытие в
верхней половине свечи, стоп за свингом PIV=5 (0.3–4 ATR), лимитка ретеста 12 свечей. Фильтр агрессоров — три уровня:
без фильтра / >= 52.5% / >= 55% (бот). Выходы, заданы до запуска (стоп бота действует всегда; срок 90 дней):
  3R       — контроль, выход бота (лимитка 3R, срок бота);
  5R / 8R  — лимитка дальше;
  trail3   — стоп подтягивается за максимумом с начала сделки на 3 ATR (ATR прошлой свечи);
  ema20d   — выход по закрытию свечи, на которой известно, что последняя дневка закрылась ниже дневной EMA20;
  low10d   — стоп поднимается до минимума 10 последних закрытых дневок (выход Дончиана).
Принятие: R на сделку > 0 в каждом периоде 2020 / 2021 / IS / VAL / HO, где сделок >= 20 (таких периодов >= 3), и t по
дням >= 3. Вариантов 5 × 3 × 2 = 30 — одиночный проход на t около 3 может быть случайным. Для справки: R в месяц и R
на сделку без 5% лучших сделок (трендовые выходы держатся на редких крупных выигрышах).

    python -m research.longexit collect --symbols <монеты>      # LX_TFS=4h или 2h
    python -m research.longexit report
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
from .ppr import day_t
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .spec import months
from .tline import BAR_MIN, MAKER, TAKER, _htf_bars, _to_bars, tf_frame

TFS = tuple(os.environ.get("LX_TFS", "4h,2h").split(","))
HOLD_DAYS = 90
TRAIL_ATR = 3.0
EMA_D, LOW_D = 20, 10
FILTERS = {"без фильтра": 0.0, "агрессоры >= 52.5%": 0.525, "агрессоры >= 55% (бот)": 0.55}
EXITS = ("5R", "8R", "trail3", "ema20d", "low10d")


def daily_levels(d: pd.DataFrame, tf: str) -> tuple[np.ndarray, np.ndarray]:
    """На закрытии каждой свечи: (последняя закрытая дневка ниже своей EMA20 — 1/0, минимум 10 последних дневок)."""
    day = _htf_bars(d, "1D")
    below = (day["close"] < day["close"].ewm(span=EMA_D, adjust=False, min_periods=EMA_D).mean()).astype(float)
    low = day["low"].rolling(LOW_D, min_periods=LOW_D).min()
    return _to_bars(below, d, "1D", tf), _to_bars(low, d, "1D", tf)


def long_exit(o, hi, lo, c, f, fill: int, entry: float, stop: float, fee: float, kind: str, hold: int,
              atr: np.ndarray, below: np.ndarray, low10: np.ndarray) -> float:
    """R лонга, вход на баре fill по entry (лимитка, комиссия fee), со стопом и выходом kind; сделка с бара fill + 1."""
    risk = entry - stop
    if not risk > 0:
        return np.nan
    tp = entry + (5.0 if kind == "5R" else 8.0) * risk if kind in ("5R", "8R") else np.inf
    st, peak, paid, n = stop, hi[fill], 0.0, len(c)
    for j in range(fill + 1, n):
        paid += f[j]
        if lo[j] <= st:
            return (min(st, o[j]) - entry - (fee + TAKER) * entry) / risk - paid * entry / risk
        if hi[j] >= tp:
            return (tp - entry - (fee + MAKER) * entry) / risk - paid * entry / risk
        if (kind == "ema20d" and below[j] == 1.0) or j - fill >= hold or j == n - 1:
            return (c[j] - entry - (fee + TAKER) * entry) / risk - paid * entry / risk
        peak = max(peak, hi[j])
        if kind == "trail3" and atr[j] > 0:
            st = max(st, peak - TRAIL_ATR * atr[j])
        elif kind == "low10d" and np.isfinite(low10[j]):
            st = max(st, low10[j])
    return np.nan


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str) -> list[dict]:
    noline.TF = tf
    x = noline.coin_trades(d, liq, sym, lines_only=True, min_aggr=0.0)
    if not len(x):
        return []
    x = x[(x.side > 0) & (x.entry == "retest")]
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].fillna(0.0).to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    atr = _atr(d).to_numpy()
    below, low10 = daily_levels(d, tf)
    hold = HOLD_DAYS * 24 * 60 // BAR_MIN[tf]
    rows = []
    for r in x.itertuples(index=False):
        if not np.isfinite(r.R3):
            continue
        fill, px, stop = int(r.fill_i), float(r.px), float(r.stop)
        row = {"symbol": sym, "tf": tf, "t": r.t, "aggr": float(r.aggr), "R": float(r.R3)}
        immediate = lo[fill] <= stop                                 # стоп в свече входа — выход уже посчитан
        for k in EXITS:
            row[k] = float(r.R3) if immediate else long_exit(o, hi, lo, c, f, fill, px, stop, MAKER, k, hold, atr,
                                                             below, low10)
        rows.append(row)
    return rows


def collect(root: Path, syms: list[str]) -> None:
    for tf in TFS:
        rows = []
        for s in mine(syms):
            try:
                d = tf_frame(root, s, tf)
                if d is None or len(d) < 500:
                    continue
                rows += coin_rows(d, adv30(root, s).reindex(d.index.floor("D")).to_numpy(), s, tf)
            except Exception as e:
                print(f"  longexit {s} {tf}: пропуск ({e})", flush=True)
        print(f"  longexit {tf}: лонгов {len(rows)}", flush=True)
        if rows:
            pd.DataFrame(rows).to_parquet(part_path(f"longexit_{tf}"), index=False)


def verdict(z: pd.DataFrame, col: str) -> tuple[bool, str]:
    v = z[["t", col]].rename(columns={col: "R"}).dropna()
    cells, pos = [], []
    for p, (a, b) in PER_ALL.items():
        m = v[(v.t >= a) & (v.t < b)]
        if m.empty:
            continue
        cells.append(f"{p} {m.R.mean():+.2f}R ({len(m)}, {m.R.sum() / months(a, b):+.1f}/мес)")
        if len(m) >= 20:
            pos.append(m.R.mean() > 0)
    t = day_t(v, "R") if len(v) > 10 else float("nan")
    cut = v.R[v.R < v.R.quantile(0.95)].mean() if len(v) >= 20 else float("nan")
    ok = len(pos) >= 3 and all(pos) and np.isfinite(t) and t >= 3
    return ok, f"{'  '.join(cells)}\n        всё время {_cell(v)}, без 5% лучших {cut:+.3f}R, лучшая {v.R.max():+.1f}R"


def report() -> None:
    print("\n===== LONGEXIT: длинные выходы на лонгах бота (4h / 2h, ретест; R на сделку, сделок, R в месяц) =====")
    passed = []
    for tf in TFS:
        parts = all_parts(f"longexit_{tf}")
        if not parts:
            print(f"  longexit {tf}: нет данных")
            continue
        x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
        for fname, thr in FILTERS.items():
            z = x[x.aggr >= thr]
            print(f"\n  --- {tf}, {fname}: лонгов {len(z)} ---")
            for col in ("R",) + EXITS:
                ok, txt = verdict(z, col)
                name = "3R (контроль)" if col == "R" else col
                ok = ok and col != "R"
                print(f"    {name}: {'принято' if ok else 'нет'}\n        {txt}")
                if ok:
                    passed.append(f"{tf} {fname} {col}")
    print("\n  ИТОГ: " + ("прошли — " + "; ".join(passed) if passed else
                         "длинные выходы лонги не вытягивают ни в одном варианте"))


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
