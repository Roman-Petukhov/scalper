"""Относительная сила монеты к BTC на момент сигнала: фильтр и очередность при лимите позиций (идея «шортить слабых»).

RS = доходность монеты за окно (24 ч и 72 ч) минус доходность BTC за то же окно, по закрытию свечи пробоя. Для шорта
«слабая» монета — RS ниже, для лонга — выше; признак sd = сторона · RS (чем больше, тем «правильнее» для стороны).
Сделки — как у бота: 4h ретест (обе стороны), 15m по рынку (шорты, наклон <= 0.025); контроль — те же без отбора.

Принятие, задано до запуска:
  фильтр  — отрезаем сделки с sd < T (T = −2% / 0 / +2%, окна 24 ч и 72 ч, шесть комбинаций): отрезанные убыточны в каждом
            периоде 2020 / 2021 / IS / VAL / HO, где их >= 20, за всё время t по дням <= −3, R в месяц с фильтром не ниже
            без него в каждом периоде.
  очередь — на 15m при лимите 3 позиции (как у бота) из сигналов одной свечи берём слабейших по RS (24 ч) вместо случайных
            (среднее по 30 перемешиваниям): R в месяц выше в каждом периоде и t по дням разности >= 2.

    python -m research.rs collect --symbols <монеты>      # RS_TFS=4h или 15m
    python -m research.rs report
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
from .spec import cut_verdicts, months
from .tline import tf_frame, zz_lines

TFS = tuple(os.environ.get("RS_TFS", "4h,15m").split(","))
ENTRY = {"4h": "retest", "15m": "market"}
BARS = {"4h": {"24": 6, "72": 18}, "15m": {"24": 96, "72": 288}}      # окна RS в свечах
MAX_SLOPE = 0.025
LIMIT = 3
THRESHOLDS = (-0.02, 0.0, 0.02)
SHUFFLES = 30
BASE = "BTCUSDT"


def rel_strength(close: pd.Series, btc: pd.Series, n: int) -> np.ndarray:
    """Доходность монеты за n свечей минус доходность BTC (по времени свечи); NaN, если у BTC нет данных."""
    b = btc.reindex(close.index)
    return (close.pct_change(n) - b.pct_change(n)).to_numpy()


def coin_rows(d: pd.DataFrame, btc: pd.Series, liq: np.ndarray, sym: str, tf: str) -> list[dict]:
    noline.TF = tf
    x = noline.coin_trades(d, liq, sym, lines_only=True)
    if not len(x):
        return []
    x = x[x.entry == ENTRY[tf]]
    atr = _atr(d).to_numpy()
    recs = {(int(r["t"]), int(r["side"])): r for r in zz_lines(d) if r["t"] > 0}
    rs = {k: rel_strength(d["close"], btc, n) for k, n in BARS[tf].items()}
    rows = []
    for r in x.itertuples(index=False):
        t, side = d.index.get_loc(r.t), int(r.side)
        rec = recs.get((t, side))
        if rec is None or not (atr[t] > 0):
            continue
        if tf != "4h" and abs(float(rec["slope"])) / atr[t] > MAX_SLOPE:
            continue
        rows.append({"symbol": sym, "tf": tf, "t": r.t, "fill": d.index[int(r.fill_i)], "exit": d.index[int(r.exit_i)],
                     "side": side, "R": float(r.R3), **{f"rs{k}": float(v[t]) for k, v in rs.items()}})
    return rows


def collect(root: Path, syms: list[str]) -> None:
    for tf in TFS:
        btc = tf_frame(root, BASE, tf)
        if btc is None:
            print(f"  rs {tf}: нет данных {BASE}", flush=True)
            continue
        rows = []
        for s in mine(syms):
            try:
                if s == BASE or (tf != "4h" and not (root / f"{s}-{tf}.parquet").exists()):
                    continue
                d = tf_frame(root, s, tf)
                if d is None or len(d) < 500:
                    continue
                rows += coin_rows(d, btc["close"], adv30(root, s).reindex(d.index.floor("D")).to_numpy(), s, tf)
            except Exception as e:
                print(f"  rs {s} {tf}: пропуск ({e})", flush=True)
        print(f"  rs {tf}: сделок {len(rows)}", flush=True)
        if rows:
            pd.DataFrame(rows).to_parquet(part_path(f"rs_{tf}"), index=False)


def pick(z: pd.DataFrame, order: np.ndarray, limit: int = LIMIT) -> pd.DataFrame:
    """Жадный отбор при лимите позиций: сигналы по времени входа, внутри одной свечи — по порядку order (меньше — раньше)."""
    z = z.assign(_o=order).sort_values(["fill", "_o"])
    open_exits: list[pd.Timestamp] = []
    take = []
    for fill, g in z.groupby("fill", sort=True):
        open_exits = [e for e in open_exits if e > fill]
        for i, e in zip(g.index, g["exit"]):
            if len(open_exits) >= limit:
                break
            open_exits.append(e)
            take.append(i)
    return z.loc[take]


def queue_verdict(z: pd.DataFrame, seed: int = 0) -> tuple[bool, str]:
    """Слабейшие по RS(24 ч) вперёд против случайного порядка: R в месяц по периодам и t по дням разности."""
    rng = np.random.default_rng(seed)
    z = z.dropna(subset=["rs24"]).reset_index(drop=True)
    by_rs = pick(z, (-z.side * z.rs24).to_numpy())          # «правильные» (sd больше) первыми
    rnd = [pick(z, rng.permutation(len(z))) for _ in range(SHUFFLES)]
    cells, wins = [], []
    for p, (a, b) in PER_ALL.items():
        m = months(a, b)
        sel = by_rs[(by_rs.fill >= a) & (by_rs.fill < b)]
        if sel.empty:
            continue
        base = np.mean([r[(r.fill >= a) & (r.fill < b)].R.sum() for r in rnd]) / m
        cells.append(f"{p}: по RS {sel.R.sum() / m:+.1f}R в месяц ({len(sel)} сделок, {sel.R.mean():+.3f}R) | "
                     f"случайно {base:+.1f}")
        wins.append(sel.R.sum() / m > base)
    days = pd.date_range(z.fill.min().floor("D"), z.fill.max().floor("D"))
    daily = lambda s: s.groupby(s.fill.dt.floor("D")).R.sum().reindex(days, fill_value=0.0)
    diff = daily(by_rs) - sum(daily(r) for r in rnd) / len(rnd)
    t = float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))) if diff.std(ddof=1) > 0 else float("nan")
    return bool(wins) and all(wins) and t >= 2, "\n".join(f"      {c}" for c in cells) + f"\n      разность по дням: t {t:.1f}"


def report() -> None:
    print("\n===== RS: относительная сила к BTC на момент сигнала — фильтр и очередность при лимите позиций "
          "(t по дням, прибыльных, сделок в месяц) =====")
    passed = []
    for tf in TFS:
        parts = all_parts(f"rs_{tf}")
        if not parts:
            print(f"  rs {tf}: нет данных")
            continue
        x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        for c in ("t", "fill", "exit"):
            x[c] = pd.to_datetime(x[c], utc=True).dt.tz_localize(None)
        x = x.dropna(subset=["R"])
        sub = x if tf == "4h" else x[x.side < 0]
        print(f"\n  --- {tf}, вход {ENTRY[tf]}, {'обе стороны' if tf == '4h' else 'шорты'}: сделок {len(sub)} ---")
        for w in BARS[tf]:
            col = f"rs{w}"
            z = sub.dropna(subset=[col]).assign(sd=lambda q: q.side * q[col])
            for iv, m in z.groupby(pd.qcut(z.sd, 5, duplicates="drop"), observed=True):
                cells = "  ".join(f"{p}: {_cell(m[(m.t >= a) & (m.t < b)].assign(R=m.R))}" for p, (a, b) in PER_ALL.items()
                                  if ((m.t >= a) & (m.t < b)).any())
                print(f"    RS {w} ч, «правильность» sd {iv.left:+.3f}…{iv.right:+.3f}: {cells}  всё время: {_cell(m)}")
            for T, (ok, txt) in cut_verdicts(z, "sd", THRESHOLDS).items():
                print(f"    RS {w} ч: режем sd < {T:+.2f}: {'принято' if ok else 'нет'}\n{txt}")
                if ok:
                    passed.append(f"{tf} фильтр RS{w} sd<{T:+.2f}")
        if tf == "15m":
            ok, txt = queue_verdict(sub)
            print(f"    очередь при лимите {LIMIT} (слабейшие по RS 24 ч вперёд): {'принято' if ok else 'нет'}\n{txt}")
            if ok:
                passed.append("15m очередь по RS")
    print("\n  ИТОГ: " + ("прошли — " + ", ".join(passed) if passed else "RS к BTC не принят ни как фильтр, ни как очередь"))


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
