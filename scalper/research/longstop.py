"""Выбивают ли лонги по стопам — гипотеза владельца: «в лонги залетает больше людей, их чаще выносят». Проверки стопа
(за свечой пробоя, свинг без потолка, 2.5 ATR) шли на всех сделках бота — это на 90% шорты; на лонгах отдельно нет.

Сделки — пробои линий бота 4h (зигзаг 3 ATR), ретест, 3R, срок бота: оборот от $20M, дневной тренд, закрытие в своей
половине свечи, стоп за свингом PIV=5 − 0.1 ATR (0.3–4 ATR). Агрессоры: без фильтра / >= 52.5% / >= 55% (бот).
1. Выбивание: из сделок, закрытых стопом, доля тех, где цена в течение 1 / 3 / 10 дней после стопа всё же дошла до цели
   3R (и до цены входа). Лонги против шортов — если у лонгов заметно чаще, стопы лонгов снимают.
2. Варианты, заданы до запуска (на тех же сделках, бар исполнения тот же — цель дальше лимитку не отменяет):
     buf0.5 / buf1   — стоп за свингом − 0.5 / − 1 ATR (цель 3R от нового риска);
     line1.5         — стоп 1.5 ATR от линии (лучший на дневках, research/htftrend.py);
     reentry         — после стопа, если закрытие в течение 3 свечей вернулось за уровень стопа: вход по рынку на этом
                       закрытии, стоп за минимумом выноса − 0.1 ATR (0.3–4 ATR), 3R; R сделки = сигнал + перезаход.
   Принятие (лонги): R на сделку > 0 в каждом периоде 2020 / 2021 / IS / VAL / HO, где сделок >= 20 (таких >= 3), t по
   дням >= 3. Вариантов 4 × 3 = 12. Шорты — для сравнения, бот на них не меняется.

    python -m research.longstop collect --symbols <монеты>
    python -m research.longstop report
"""
from __future__ import annotations

import argparse
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
from .tline import HOLD, MAKER, STOP_BUF, TAKER, tf_frame, two_targets

TF = "4h"
DAYS = (1, 3, 10)
REENTRY_BARS = 3
FILTERS = {"без фильтра": 0.0, "агрессоры >= 52.5%": 0.525, "агрессоры >= 55% (бот)": 0.55}
VARIANTS = ("buf0.5", "buf1", "line1.5", "reentry")


def trade_r(o, hi, lo, c, f, fill: int, side: int, px: float, stop: float, fee: float) -> tuple[float, int]:
    """R и бар выхода сделки бота (3R, срок бота); стоп в свече входа — сразу убыток."""
    risk = side * (px - stop)
    if not risk > 0:
        return np.nan, fill
    if (side > 0 and lo[fill] <= stop) or (side < 0 and hi[fill] >= stop):
        return (side * (stop - px) - (fee + TAKER) * px) / risk, fill
    r, ex = two_targets(o, hi, lo, c, f, fill, side, px, stop, 3.0, 3.0, False, HOLD[TF], fee)
    return float(r), int(ex)


def stopped(hi, lo, ex: int, side: int, stop: float) -> bool:
    return bool(lo[ex] <= stop) if side > 0 else bool(hi[ex] >= stop)


def after_stop(hi, lo, ex: int, side: int, px: float, stop: float, bars: int) -> tuple[bool, bool]:
    """(дошла ли цена до цели 3R, вернулась ли к цене входа) за bars свечей после свечи стопа."""
    tp = px + 3.0 * (px - stop)
    j = slice(ex + 1, min(ex + 1 + bars, len(hi)))
    if side > 0:
        return bool((hi[j] >= tp).any()), bool((hi[j] >= px).any())
    return bool((lo[j] <= tp).any()), bool((lo[j] <= px).any())


def reentry(o, hi, lo, c, f, atr, ex: int, side: int, stop: float) -> float:
    """Перезаход после выноса: закрытие за уровнем стопа в течение REENTRY_BARS свечей; 0 — перезахода не было."""
    n = len(c)
    ext = lo[ex] if side > 0 else hi[ex]
    for j in range(ex + 1, min(ex + 1 + REENTRY_BARS, n - 1)):
        ext = min(ext, lo[j]) if side > 0 else max(ext, hi[j])
        if side * (c[j] - stop) > 0:
            st = ext - side * STOP_BUF * atr[j]
            risk = side * (c[j] - st)
            if not (0.3 * atr[j] <= risk <= 4.0 * atr[j]):
                return 0.0
            r, _ = two_targets(o, hi, lo, c, f, j, side, c[j], st, 3.0, 3.0, False, HOLD[TF], TAKER)
            return float(r) if np.isfinite(r) else 0.0
    return 0.0


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str) -> list[dict]:
    noline.TF = TF
    x = noline.coin_trades(d, liq, sym, lines_only=True, min_aggr=0.0)
    if not len(x):
        return []
    x = x[x.entry == "retest"]
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].fillna(0.0).to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    atr = _atr(d).to_numpy()
    per_day = 24 * 60 // 240
    rows = []
    for r in x.itertuples(index=False):
        if not np.isfinite(r.R3):
            continue
        t, side, px, stop, fill = d.index.get_loc(r.t), int(r.side), float(r.px), float(r.stop), int(r.fill_i)
        a = atr[t]
        ex = int(r.exit_i)
        hit = stopped(hi, lo, ex, side, stop) and r.R3 < 0
        row = {"symbol": sym, "t": r.t, "side": side, "aggr": float(r.aggr), "R": float(r.R3), "stopped": hit}
        for k in DAYS:
            tp, back = after_stop(hi, lo, ex, side, px, stop, k * per_day) if hit else (False, False)
            row[f"tp{k}d"], row[f"back{k}d"] = tp, back
        for name, st in (("buf0.5", stop - side * 0.4 * a), ("buf1", stop - side * 0.9 * a),
                         ("line1.5", px - side * 1.5 * a)):
            row[name] = trade_r(o, hi, lo, c, f, fill, side, px, st, MAKER)[0]
        row["reentry"] = float(r.R3) + (reentry(o, hi, lo, c, f, atr, ex, side, stop) if hit else 0.0)
        rows.append(row)
    return rows


def collect(root: Path, syms: list[str]) -> None:
    rows = []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, TF)
            if d is None or len(d) < 500:
                continue
            rows += coin_rows(d, adv30(root, s).reindex(d.index.floor("D")).to_numpy(), s)
        except Exception as e:
            print(f"  longstop {s}: пропуск ({e})", flush=True)
    print(f"  longstop: сделок {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("longstop"), index=False)


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
    return len(pos) >= 3 and all(pos) and np.isfinite(t) and t >= 3, f"{'  '.join(cells)}\n        всё время {_cell(v)}"


def report() -> None:
    print("\n===== LONGSTOP: выбивают ли лонги по стопам (4h, ретест, 3R; R на сделку, сделок, R в месяц) =====")
    parts = all_parts("longstop")
    if not parts:
        print("  longstop: нет данных")
        return
    x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
    passed = []
    for fname, thr in FILTERS.items():
        print(f"\n  --- {fname} ---")
        for label, side in (("лонги", 1), ("шорты", -1)):
            z = x[(x.side == side) & (x.aggr >= thr)]
            s = z[z.stopped]
            if len(s) < 10:
                print(f"    {label}: стопом закрыто {len(s)} — мало")
                continue
            hunt = "  ".join(f"{k}д: до 3R {s[f'tp{k}d'].mean():.0%}, к входу {s[f'back{k}d'].mean():.0%}" for k in DAYS)
            print(f"    {label}: сделок {len(z)}, стопом {len(s)} ({len(s) / len(z):.0%}); после стопа — {hunt}")
            for col in ("R",) + VARIANTS:
                ok, txt = verdict(z, col)
                ok = ok and col != "R" and side > 0
                print(f"      {'бот (контроль)' if col == 'R' else col}: {'принято' if ok else 'нет'}\n        {txt}")
                if ok:
                    passed.append(f"{fname} {col}")
    print("\n  ИТОГ: " + ("прошли — " + "; ".join(passed) if passed else
                         "ни шире стоп, ни стоп от линии, ни перезаход лонги не вытягивают"))


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
