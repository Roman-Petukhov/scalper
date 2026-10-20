"""Можно ли в момент +2R угадать, что цена не дойдёт до 3R (по вопросу владельца после research/exit5.py).

Сделки — правило бота (research/noline.py), все ~725 монет с 2022 (как research/holdlen.py). Срезы как у бота:
5m и 15m — пологие шорты по рынку (срок 800 / 200 свечей), 4h — ретест, обе стороны (срок 60 свечей).
Момент решения — закрытие первой свечи, в которой ход дошёл до +2R (стоп в ней не задет, 3R не взят). Два исхода
одной сделки: «держать» — бот как есть (стоп / 3R / срок); «выйти» — по закрытию этой свечи (тейкер). У бота на 5m из
дошедших до +2R 3R берут ~75% — ровно точка безразличия (0.75 · 3 − 0.25 · 1 = 2), поэтому ищем признак, который делит
их на группы, где 3R берут заметно реже (там выходить) и чаще (держать).
Признаки на закрытии свечи +2R (заданы до запуска; непрерывные — крайние трети по IS):
  speed     свечей от входа до +2R: «быстро» (нижняя треть) / «медленно» (верхняя)
  aggr      доля агрессоров в сторону сделки от входа до +2R: «слабые» (нижняя треть)
  aggr3     то же за последние 3 свечи: «слабые» (нижняя треть)
  vfade     средний объём последних 3 свечей / объём свечи пробоя: «затух» (нижняя треть)
  wick      свеча +2R закрылась дальше середины от своего экстремума в сторону сделки (отбой): да / нет
  round     крупный круглый уровень между ценой 2R и 3R (research/roundnum.py): да / нет
  swing     экстремум 500 свечей до входа (лоу для шорта) между ценой 2R и 3R: да / нет
  stretch   отрыв закрытия от EMA50 в ATR в сторону сделки: «растянуто» (верхняя треть)
Принятие: в группе выход лучше удержания по R на сделку в каждом периоде IS / VAL / HO (≥ 20 сделок) и парная
разница за всё время с t ≥ 2. Признаков восемь — одна группа с t ≈ 2 может быть случайной; смотрим на все периоды.

    python -m research.at2r collect --symbols <hot-wide>
    python -m research.at2r report
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
from .roundnum import level_ahead
from .shard import all_parts, mine, part_path
from .smc import _atr
from .tline import HOLD, MAKER, TAKER, tf_frame, zz_lines

TFS = tuple(os.environ.get("AT2R_TFS", "5m,15m,4h").split(","))
SLICE = {"5m": ("market", True, 800), "15m": ("market", True, HOLD["15m"]), "4h": ("retest", False, HOLD["4h"])}
MAX_SLOPE, SWING_BACK, AT = 0.025, 500, 2.0
FEATS = {"speed": ("свечей до +2R", "q"), "aggr": ("агрессоры по ходу", "lo"), "aggr3": ("агрессоры последних 3 свечей", "lo"),
         "vfade": ("объём последних 3 / объём пробоя", "lo"), "wick": ("отбой на свече +2R", "bin"),
         "round": ("круглый уровень между 2R и 3R", "bin"), "swing": ("экстремум 500 свечей между 2R и 3R", "bin"),
         "stretch": ("отрыв от EMA50 в ATR", "hi")}


@njit(cache=True)
def path(o, h, lo, c, fund, e, side, entry, stop, k, at, max_hold, entry_fee):
    """Бот: стоп / k·R / срок. Возвращает (R бота, свеча первого хода до +at·R или −1, R выхода по её закрытию)."""
    n = len(c)
    risk = side * (entry - stop)
    if not (risk > 0):
        return np.nan, -1, np.nan
    tp = entry + side * k * risk
    paid, j2, r2 = 0.0, -1, np.nan
    for j in range(e + 1, n):
        paid += fund[j]
        if (side > 0 and lo[j] <= stop) or (side < 0 and h[j] >= stop):
            ex = min(stop, o[j]) if side > 0 else max(stop, o[j])
            return (side * (ex - entry) - (entry_fee + TAKER) * entry) / risk - side * paid * entry / risk, j2, r2
        if (side > 0 and h[j] >= tp) or (side < 0 and lo[j] <= tp):
            return (k * risk - (entry_fee + MAKER) * entry) / risk - side * paid * entry / risk, j2, r2
        best = (h[j] - entry) / risk if side > 0 else (entry - lo[j]) / risk
        if j2 < 0 and best >= at:
            j2 = j
            r2 = (side * (c[j] - entry) - (entry_fee + TAKER) * entry) / risk - side * paid * entry / risk
        if j - e >= max_hold or j == n - 1:
            return (side * (c[j] - entry) - (entry_fee + TAKER) * entry) / risk - side * paid * entry / risk, j2, r2
    return np.nan, j2, r2


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str) -> list[dict]:
    entry_kind, bot_slice, hold = SLICE[tf]
    noline.TF = tf
    x = noline.coin_trades(d, liq, sym, lines_only=True)
    if not len(x):
        return []
    x = x[x.entry == entry_kind]
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    vol = d["volume"].to_numpy(dtype="float64")
    buy = d["taker_buy_volume"].to_numpy(dtype="float64")
    atr = _atr(d).to_numpy()
    ema = d["close"].ewm(span=50, adjust=False).mean().to_numpy()
    slope = {(int(r["t"]), int(r["side"])): abs(float(r["slope"])) for r in zz_lines(d) if r["t"] > 0}
    rows = []
    for r in x.itertuples(index=False):
        t, side, fill, px, stop = d.index.get_loc(r.t), int(r.side), int(r.fill_i), float(r.px), float(r.stop)
        if not (atr[t] > 0):
            continue
        sl = slope.get((t, side), np.nan) / atr[t]
        if bot_slice and not (side < 0 and sl <= MAX_SLOPE):
            continue
        fee = MAKER if entry_kind == "retest" else TAKER
        if entry_kind == "retest" and ((side > 0 and lo[fill] <= stop) or (side < 0 and hi[fill] >= stop)):
            continue                                                   # стоп в свече исполнения: до +2R не дошла
        hold_r, j2, exit_r = path(o, hi, lo, c, f, fill, side, px, stop, 3.0, AT, hold, fee)
        if j2 < 0 or not np.isfinite(hold_r):
            continue
        risk = side * (px - stop)
        p2, p3 = px + side * AT * risk, px + side * 3.0 * risk
        s = slice(fill + 1, j2 + 1)
        s3 = slice(max(fill + 1, j2 - 2), j2 + 1)
        share = lambda sl_: (buy[sl_].sum() / vol[sl_].sum()) if vol[sl_].sum() > 0 else np.nan
        aggr, aggr3 = share(s), share(s3)
        if side < 0:
            aggr, aggr3 = 1 - aggr, 1 - aggr3
        rng = hi[j2] - lo[j2]
        far = (lo[j2] if side < 0 else hi[j2])
        ext = lo[max(0, fill - SWING_BACK):fill].min() if side < 0 else hi[max(0, fill - SWING_BACK):fill].max()
        rows.append({"symbol": sym, "tf": tf, "t": r.t, "side": side, "hold": float(hold_r), "exit": float(exit_r),
                     "took3": bool(hold_r > 2.5),
                     "speed": j2 - fill, "aggr": aggr, "aggr3": aggr3,
                     "vfade": vol[s3].mean() / vol[t] if vol[t] > 0 else np.nan,
                     "wick": "да" if rng > 0 and abs(c[j2] - far) / rng > 0.5 else "нет",
                     "round": "да" if level_ahead(p2, side, abs(p3 - p2), True) else "нет",
                     "swing": "да" if min(p2, p3) < ext < max(p2, p3) else "нет",
                     "stretch": side * (c[j2] - ema[j2]) / atr[j2] if atr[j2] > 0 else np.nan})
    return rows


def collect(root: Path, syms: list[str]) -> None:
    rows = []
    for s in mine(syms):
        for tf in TFS:
            try:
                d = tf_frame(root, s, tf)
                if d is None or len(d) < 500:
                    continue
                liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
                rows += coin_rows(d, liq, s, tf)
            except Exception as e:
                print(f"  at2r {s} {tf}: пропуск ({e})", flush=True)
    print(f"  at2r: сделок, дошедших до +2R, {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("at2r"), index=False)


def label(z: pd.DataFrame) -> pd.DataFrame:
    """Непрерывные признаки — в «да» (выходить) по крайней трети IS; speed — «быстро» / «средне» / «медленно»."""
    z = z.copy()
    a, b = PER["is"]
    ins = z[(z.t >= a) & (z.t < b)]
    for key, (_, kind) in FEATS.items():
        if kind == "bin":
            continue
        q1, q2 = ins[key].quantile([1 / 3, 2 / 3]).to_numpy()
        if kind == "q":
            z[key] = np.where(z[key] <= q1, f"быстро ≤ {q1:.0f}", np.where(z[key] > q2, f"медленно > {q2:.0f}", "средне"))
        elif kind == "lo":
            z[key] = np.where(z[key].isna(), None, np.where(z[key] <= q1, "да", "нет"))
        else:
            z[key] = np.where(z[key].isna(), None, np.where(z[key] > q2, "да", "нет"))
    return z


def group_line(m: pd.DataFrame) -> tuple[str, bool]:
    cells, ok = [], []
    for p, (a, b) in PER.items():
        g = m[(m.t >= a) & (m.t < b)]
        if len(g) >= 20:
            dlt = g["exit"].mean() - g["hold"].mean()
            ok.append(dlt > 0)
            cells.append(f"{p}: {dlt:+.2f} (3R {g.took3.mean():.0%}, {len(g)})")
        elif len(g):
            cells.append(f"{p}: мало ({len(g)})")
    dd = (m["exit"] - m["hold"]).to_numpy()
    sd = dd.std(ddof=1) if len(dd) > 2 else 0.0
    t = dd.mean() / (sd / np.sqrt(len(dd))) if sd > 0 else np.nan
    passed = bool(ok) and len(ok) == len(PER) and all(ok) and t >= 2
    return f"{'  '.join(cells)}  всё время {dd.mean():+.2f}R t {t:.1f}" + ("  → ВЫХОДИТЬ" if passed else ""), passed


def report() -> None:
    parts = all_parts("at2r")
    print("\n===== AT2R: угадать в момент +2R, что 3R не будет — выход по закрытию свечи +2R минус удержание (бот), "
          "R на сделку по группам признака; в скобках доля взявших 3R и число сделок =====")
    if not parts:
        print("  at2r: нет данных")
        return
    x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
    x = x.dropna(subset=["hold", "exit"])
    winners = []
    for tf in TFS:
        z0 = x[x.tf == tf]
        if len(z0) < 60:
            continue
        z = label(z0)
        line, _ = group_line(z)
        print(f"\n  --- {tf} ({SLICE[tf][0]}{', пологие шорты' if SLICE[tf][1] else ''}): дошли до +2R {len(z)}, "
              f"взяли 3R {z.took3.mean():.0%}; выход по закрытию минус удержание: {line}")
        for key, (title, _) in FEATS.items():
            print(f"    {key} {title}:")
            for g in sorted(z[key].dropna().unique()):
                txt, ok = group_line(z[z[key] == g])
                print(f"       {g}: {txt}")
                if ok:
                    winners.append(f"{tf} {key}={g}")
    print("\n  ИТОГ: " + ("прошли порог — " + "; ".join(winners) if winners else
                         "ни одна группа не прошла порог: в момент +2R по этим признакам не угадать"))


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
