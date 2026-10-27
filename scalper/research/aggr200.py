"""EMA200 вместо фильтра агрессоров — идея владельца: «если трендовая пробилась над EMA200 — лонг, иначе нет; так же
для шорта». Фильтр агрессоров (>= 55%) режет 95% лонговых пробоев 4h (research/sidefunnel.py); здесь он снимается, а
сторону задаёт EMA200.

Сделки — правило бота без агрессоров и без тренда (оба навешиваются здесь): 4h ретест (лонги и шорты отдельно), 15m по
рынку с наклоном <= 0.025 (шорты — как в живых настройках, лонги — для справки и на решение). Признаки на свече пробоя:
  ema50   — тренд бота: закрытие старшего ТФ (дневки для 4h, часа для 15m) против его EMA50;
  e200    — закрытие свечи пробоя за EMA200 того же ТФ (для лонга выше, для шорта ниже);
  e200l   — и уровень линии в момент пробоя за EMA200 (весь пробой над / под средней, а не пересечение снизу);
  d200    — закрытие последней дневки против дневной EMA200.
Варианты, заданы до запуска (контроль — бот: агрессоры >= 55% и ema50):
  ema50 + e200      — EMA200 вместо агрессоров, тренд бота остаётся;
  ema50 + e200l     — то же, пробой целиком за EMA200;
  e200              — EMA200 вместо агрессоров и тренда;
  ema50 + d200      — дневная EMA200 вместо агрессоров;
  d200              — дневная EMA200 вместо агрессоров и тренда.
Сравнение на одних и тех же барах (после прогрева нужной EMA). Принятие: R в месяц не ниже, чем у бота, в каждом периоде
2020 / 2021 / IS / VAL / HO, R на сделку > 0 в каждом из них, где сделок >= 20, и t по дням >= 3. 5 вариантов × 4 среза —
одиночный проход на t около 3 может быть случайным.

    python -m research.aggr200 collect --symbols <монеты>      # A200_TFS=4h или 15m
    python -m research.aggr200 report
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
from .ema200 import htf_sign, replace_verdict
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import tf_frame, zz_lines

TFS = tuple(os.environ.get("A200_TFS", "4h,15m").split(","))
ENTRY = {"4h": "retest", "15m": "market"}
MAX_SLOPE_15M = 0.025
SPAN = 200
VARIANTS: dict[str, tuple[str, ...]] = {
    "ema50 + e200": ("ema50", "e200"),
    "ema50 + e200l": ("ema50", "e200", "e200l"),
    "e200": ("e200",),
    "ema50 + d200": ("ema50", "d200"),
    "d200": ("d200",),
}


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str) -> list[dict]:
    noline.TF = tf
    saved = noline.htf_trend
    parts = []
    try:
        for s in (1, -1):                                  # без фильтра тренда: обе стороны «по тренду»
            noline.htf_trend = (lambda d_, tf_, s=s: np.full(len(d_), float(s)))
            x = noline.coin_trades(d, liq, sym, lines_only=True, min_aggr=0.0)
            if len(x):
                parts.append(x[(x.side == s) & (x.entry == ENTRY[tf])])
    finally:
        noline.htf_trend = saved
    if not parts:
        return []
    x = pd.concat(parts, ignore_index=True)
    c = d["close"].to_numpy(dtype="float64")
    ema = d["close"].ewm(span=SPAN, adjust=False, min_periods=SPAN).mean().to_numpy()
    atr = _atr(d).to_numpy()
    recs = {(int(r["t"]), int(r["side"])): r for r in zz_lines(d) if r["t"] > 0}
    tr = {"ema50": htf_sign(d, tf, 50), "d200": htf_sign(d, tf, SPAN, "1D")}
    rows = []
    for r in x.itertuples(index=False):
        t, side = d.index.get_loc(r.t), int(r.side)
        rec = recs.get((t, side))
        if rec is None or not (atr[t] > 0) or not np.isfinite(r.R3) or not np.isfinite(ema[t]):
            continue
        if tf == "15m" and abs(float(rec["slope"])) / atr[t] > MAX_SLOPE_15M:
            continue
        rows.append({"symbol": sym, "tf": tf, "t": r.t, "side": side, "R": float(r.R3), "aggr": float(r.aggr),
                     "e200": float(np.sign(side * (c[t] - ema[t]))),
                     "e200l": float(np.sign(side * (r.level - ema[t]))),
                     **{k: float(v[t] * side) if np.isfinite(v[t]) else np.nan for k, v in tr.items()}})
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
                print(f"  aggr200 {s} {tf}: пропуск ({e})", flush=True)
        print(f"  aggr200 {tf}: пробоев {len(rows)}", flush=True)
        if rows:
            pd.DataFrame(rows).to_parquet(part_path(f"aggr200_{tf}"), index=False)


def pick(z: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    m = np.ones(len(z), dtype=bool)
    for k in cols:
        m &= (z[k] > 0).to_numpy()
    return z[m]


def verdict(v: pd.DataFrame, base: pd.DataFrame) -> tuple[bool, str]:
    """replace_verdict (R в месяц не ниже бота в каждом периоде, t >= 3) и R на сделку > 0 в периодах с >= 20 сделок."""
    ok, txt = replace_verdict(v, base)
    pos = [v[(v.t >= a) & (v.t < b)].R.mean() > 0 for a, b in PER_ALL.values()
           if ((v.t >= a) & (v.t < b)).sum() >= 20]
    return ok and bool(pos) and all(pos), txt


def report() -> None:
    print("\n===== AGGR200: EMA200 вместо фильтра агрессоров (сделки бота без агрессоров, t по дням, прибыльных, "
          "сделок в месяц) =====")
    passed = []
    for tf in TFS:
        parts = all_parts(f"aggr200_{tf}")
        if not parts:
            print(f"  aggr200 {tf}: нет данных")
            continue
        x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
        x = x.dropna(subset=["R", "ema50"])
        for label, side in (("лонги", 1), ("шорты", -1)):
            sub = x[x.side == side]
            if sub.empty:
                continue
            bot = lambda z: z[(z.aggr >= 0.55) & (z.ema50 > 0)]
            print(f"\n  --- {tf}, вход {ENTRY[tf]}, {label}: пробоев {len(sub)}, бот {_cell(bot(sub))}, "
                  f"без агрессоров с трендом {_cell(sub[sub.ema50 > 0])}, без обоих {_cell(sub)} ---")
            for name, cols in VARIANTS.items():
                z = sub.dropna(subset=["d200"]) if "d200" in cols else sub
                ok, txt = verdict(pick(z, cols), bot(z))
                print(f"    {name}: {'принято' if ok else 'нет'}\n{txt}")
                if ok:
                    passed.append(f"{tf} {label} {name}")
    print("\n  ИТОГ: " + ("прошли — " + ", ".join(passed) if passed else
                         "EMA200 не заменяет фильтр агрессоров ни в одном варианте"))


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
