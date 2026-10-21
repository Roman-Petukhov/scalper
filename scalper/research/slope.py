"""Порог наклона линии выше 0.025 ATR за свечу (по вопросу владельца: «давай проверку выше 0.025»). В боте max_slope_atr —
максимально допустимый наклон: линия круче порога сигнала не даёт. Бэктестный порог 0.025 (нижняя треть по IS), у
владельца в живых настройках стоит 0.005. Здесь — сигналы бота на 15m и 5m без ограничения по наклону (вход по рынку,
обе стороны, остальные фильтры бота) и итог в R по корзинам наклона; цена порога — что получают сделки, которые он
добавляет или отрезает.

Корзины, ATR за свечу: <0.005, 0.005–0.01, 0.01–0.025, 0.025–0.03, 0.03–0.04, 0.04–0.05, 0.05–0.1, 0.1–0.2, ≥0.2.
Второй заход (владелец: «а если более гибко, 0.03, 0.04»): корзины 0.03 и 0.04 добавлены, порог проверяется и отдельно на шортах.
Принятие повышения порога T (0.03 / 0.04 / 0.05 / 0.1 / 0.2 / без порога), задано до запуска: сделки, которые порог T добавляет к
порогу 0.025 (наклон от 0.025 до T), в среднем в плюсе в каждом периоде 2020 / 2021 / IS / VAL / HO, где их ≥ 20, и
за всё время с t ≥ 3 (несколько порогов сразу — Harvey, Liu, Zhu 2016); при этом R в месяц с порогом T не ниже, чем с 0.025,
в каждом периоде. Справочно — то же для стороны отдельно и для понижения порога (0.005 против 0.025).

    python -m research.slope collect --symbols <монеты>      # SLOPE_TFS=15m или 5m
    python -m research.slope report
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

TFS = tuple(os.environ.get("SLOPE_TFS", "15m,5m").split(","))
EDGES = (0.0, 0.005, 0.01, 0.025, 0.03, 0.04, 0.05, 0.1, 0.2, np.inf)
BASE, LOW, UP = 0.025, 0.005, (0.03, 0.04, 0.05, 0.1, 0.2, np.inf)


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str) -> list[dict]:
    noline.TF = tf
    x = noline.coin_trades(d, liq, sym, lines_only=True)
    if not len(x):
        return []
    x = x[x.entry == "market"]
    atr = _atr(d).to_numpy()
    slope = {(int(r["t"]), int(r["side"])): abs(float(r["slope"])) for r in zz_lines(d) if r["t"] > 0}
    rows = []
    for r in x.itertuples(index=False):
        t, side = d.index.get_loc(r.t), int(r.side)
        sl = slope.get((t, side))
        if sl is None or not (atr[t] > 0):
            continue
        rows.append({"symbol": sym, "tf": tf, "t": r.t, "side": side, "R": float(r.R3), "slope_atr": sl / atr[t]})
    return rows


def collect(root: Path, syms: list[str]) -> None:
    for tf in TFS:
        rows = []
        for s in mine(syms):
            try:
                if not (root / f"{s}-{tf}.parquet").exists():
                    continue
                d = tf_frame(root, s, tf)
                if d is None or len(d) < 500:
                    continue
                rows += coin_rows(d, adv30(root, s).reindex(d.index.floor("D")).to_numpy(), s, tf)
            except Exception as e:
                print(f"  slope {s} {tf}: пропуск ({e})", flush=True)
        print(f"  slope {tf}: сделок {len(rows)}", flush=True)
        if rows:
            pd.DataFrame(rows).to_parquet(part_path(f"slope_{tf}"), index=False)


def tstat(a: np.ndarray) -> float:
    return float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))) if len(a) > 2 and a.std(ddof=1) > 0 else float("nan")


def months(a, b) -> float:
    return max(1.0, (pd.Timestamp(b) - pd.Timestamp(a)).days / 30.4)


def verdicts(z: pd.DataFrame) -> dict[float, tuple[bool, str]]:
    """Порог T → (принят, строка по периодам): добавленные сделки и R в месяц против порога 0.025."""
    out = {}
    for T in UP:
        add = (z.slope_atr > BASE) & (z.slope_atr <= T)
        ok_add, ok_month, cells = [], [], []
        for p, (a, b) in PER_ALL.items():
            m = z[(z.t >= a) & (z.t < b)]
            if m.empty:
                continue
            ad = m[(m.slope_atr > BASE) & (m.slope_atr <= T)]
            base_m = m.R[m.slope_atr <= BASE].sum() / months(a, b)
            new_m = m.R[m.slope_atr <= T].sum() / months(a, b)
            cells.append(f"{p}: добавлено {_cell(ad) if len(ad) else '—'}; R в месяц {base_m:+.1f} → {new_m:+.1f}")
            if len(ad) >= 20:
                ok_add.append(ad.R.mean() > 0)
            ok_month.append(new_m >= base_m)
        t = tstat(z.R[add].to_numpy())
        out[T] = (bool(ok_add) and all(ok_add) and all(ok_month) and t >= 3, "\n".join(f"      {c}" for c in cells)
                  + f"\n      добавленные за всё время: {z.R[add].mean():+.3f}R на {int(add.sum())}, t {t:.1f}")
    return out


def report() -> None:
    print("\n===== SLOPE: порог наклона линии (ATR за свечу) — R по корзинам наклона, вход по рынку, обе стороны, фильтры бота "
          "(t по дням, прибыльных, сделок в месяц) =====")
    passed = []
    for tf in TFS:
        parts = all_parts(f"slope_{tf}")
        if not parts:
            print(f"  slope {tf}: нет данных")
            continue
        x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
        x = x.dropna(subset=["R", "slope_atr"])
        for label, z in (("обе стороны", x), ("шорты", x[x.side < 0]), ("лонги", x[x.side > 0])):
            print(f"\n  --- {tf}, {label}: сделок {len(z)} ---")
            bk = pd.cut(z.slope_atr, EDGES, right=False)
            for iv, m in z.groupby(bk, observed=True):
                cells = "  ".join(f"{p}: {_cell(m[(m.t >= a) & (m.t < b)])}" for p, (a, b) in PER_ALL.items()
                                  if ((m.t >= a) & (m.t < b)).any())
                print(f"    наклон {iv.left:g}–{iv.right:g}: {cells}  всё время: {_cell(m)}")
            low = z[(z.slope_atr >= LOW) & (z.slope_atr < BASE)]
            print(f"    справочно, то, что режет понижение 0.025 → 0.005 (наклон 0.005–0.025): "
                  f"{_cell(low) if len(low) else '—'}")
            if label == "лонги":
                continue
            for T, (ok, txt) in verdicts(z).items():
                print(f"    порог {BASE:g} → {T:g}: {'принято' if ok else 'нет'}\n{txt}")
                if ok:
                    passed.append(f"{tf} {label} {T:g}")
    print("\n  ИТОГ: " + ("прошли — " + ", ".join(passed) if passed else "ни одно повышение порога выше 0.025 не принято"))


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
