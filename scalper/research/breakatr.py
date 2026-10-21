"""Закрытие свечи за линией в ATR (по просьбе владельца: «проверь тесты на разные значения закрытия свечи за линией ATR»).
В боте min_break_atr — минимум, на сколько ATR свеча пробоя должна закрыться за линией: меньше — сигнала нет (в бэктестах
бота фильтра не было, в живых настройках владельца стоит 0.3 на 4h и 0.4 на 15m / 5m). Здесь — сигналы бота без этого
фильтра и итог в R по корзинам расстояния закрытия от линии: brk = сторона · (закрытие − линия на свече пробоя) / ATR.

Правило остальное как у бота: 4h — ретест, 15m и 5m — по рынку, пологие линии (наклон ≤ 0.025), обе стороны; вердикт и
для шортов отдельно. Корзины: <0.1, 0.1–0.2, 0.2–0.3, 0.3–0.4, 0.4–0.5, 0.5–0.75, 0.75–1, 1–1.5, ≥1.5.
Принятие порога T (0.1 / 0.2 / 0.3 / 0.4 / 0.5 / 0.75 / 1), задано до запуска: сделки, которые порог отрезает (brk < T),
в среднем убыточны в каждом периоде 2020 / 2021 / IS / VAL / HO, где их ≥ 20, за всё время с t ≤ −3 (семь порогов сразу —
Harvey, Liu, Zhu 2016), а R в месяц с порогом не ниже, чем без него, в каждом периоде. Справочно — R по корзинам и
то, что отрезают пороги владельца 0.3 и 0.4.

    python -m research.breakatr collect --symbols <монеты>      # BRK_TFS=4h, 15m или 5m
    python -m research.breakatr report
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
from .tline import tf_frame, zz_lines

TFS = tuple(os.environ.get("BRK_TFS", "4h,15m,5m").split(","))
ENTRY = {"4h": "retest", "15m": "market", "5m": "market"}
MAX_SLOPE = 0.025
EDGES = (-np.inf, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, np.inf)
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0)
OWNER = (0.3, 0.4)


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str) -> list[dict]:
    noline.TF = tf
    x = noline.coin_trades(d, liq, sym, lines_only=True)
    if not len(x):
        return []
    x = x[x.entry == ENTRY[tf]]
    c, atr = d["close"].to_numpy(dtype="float64"), _atr(d).to_numpy()
    recs = {(int(r["t"]), int(r["side"])): r for r in zz_lines(d) if r["t"] > 0}
    rows = []
    for r in x.itertuples(index=False):
        t, side = d.index.get_loc(r.t), int(r.side)
        rec = recs.get((t, side))
        if rec is None or not (atr[t] > 0):
            continue
        if tf != "4h" and abs(float(rec["slope"])) / atr[t] > MAX_SLOPE:
            continue
        rows.append({"symbol": sym, "tf": tf, "t": r.t, "side": side, "R": float(r.R3),
                     "brk_atr": side * (c[t] - float(rec["line_t"])) / atr[t]})
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
                print(f"  breakatr {s} {tf}: пропуск ({e})", flush=True)
        print(f"  breakatr {tf}: сделок {len(rows)}", flush=True)
        if rows:
            pd.DataFrame(rows).to_parquet(part_path(f"breakatr_{tf}"), index=False)


def tstat(a: np.ndarray) -> float:
    return float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))) if len(a) > 2 and a.std(ddof=1) > 0 else float("nan")


def months(a, b) -> float:
    return max(1.0, (pd.Timestamp(b) - pd.Timestamp(a)).days / 30.4)


def verdicts(z: pd.DataFrame, thresholds=THRESHOLDS) -> dict[float, tuple[bool, str]]:
    """Порог T → (принят, строки по периодам): отрезанные сделки (brk < T) и R в месяц без порога → с порогом."""
    out = {}
    for T in thresholds:
        cut = z.brk_atr < T
        ok_cut, ok_month, cells = [], [], []
        for p, (a, b) in PER_ALL.items():
            m = z[(z.t >= a) & (z.t < b)]
            if m.empty:
                continue
            ct = m[m.brk_atr < T]
            base_m, new_m = m.R.sum() / months(a, b), m.R[m.brk_atr >= T].sum() / months(a, b)
            cells.append(f"{p}: отрезано {_cell(ct) if len(ct) else '—'}; R в месяц {base_m:+.1f} → {new_m:+.1f}")
            if len(ct) >= 20:
                ok_cut.append(ct.R.mean() < 0)
            ok_month.append(new_m >= base_m)
        t = tstat(z.R[cut].to_numpy())
        out[T] = (bool(ok_cut) and all(ok_cut) and all(ok_month) and t <= -3,
                  "\n".join(f"      {c}" for c in cells)
                  + f"\n      отрезанные за всё время: {z.R[cut].mean():+.3f}R на {int(cut.sum())}, t {t:.1f}")
    return out


def report() -> None:
    print("\n===== BREAKATR: закрытие свечи пробоя за линией в ATR — R по корзинам и пороги min_break_atr (сигналы бота без этого "
          "фильтра; t по дням, прибыльных, сделок в месяц) =====")
    passed = []
    for tf in TFS:
        parts = all_parts(f"breakatr_{tf}")
        if not parts:
            print(f"  breakatr {tf}: нет данных")
            continue
        x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
        x = x.dropna(subset=["R", "brk_atr"])
        for label, z in (("обе стороны", x), ("шорты", x[x.side < 0]), ("лонги", x[x.side > 0])):
            print(f"\n  --- {tf}, вход {ENTRY[tf]}, {label}: сделок {len(z)} ---")
            for iv, m in z.groupby(pd.cut(z.brk_atr, EDGES, right=False), observed=True):
                cells = "  ".join(f"{p}: {_cell(m[(m.t >= a) & (m.t < b)])}" for p, (a, b) in PER_ALL.items()
                                  if ((m.t >= a) & (m.t < b)).any())
                print(f"    закрытие за линией {iv.left:g}–{iv.right:g} ATR: {cells}  всё время: {_cell(m)}")
            if label == "лонги":
                continue
            for T, (ok, txt) in verdicts(z).items():
                tag = " (порог владельца)" if T in OWNER else ""
                print(f"    порог {T:g}{tag}: {'принято' if ok else 'нет'}\n{txt}")
                if ok:
                    passed.append(f"{tf} {label} {T:g}")
    print("\n  ИТОГ: " + ("прошли — " + ", ".join(passed) if passed else "ни один порог min_break_atr не принят"))


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
