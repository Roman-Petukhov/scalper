"""Наши линии против канона (Bulkowski, Edwards & Magee, Murphy): по вопросу владельца «правильно ли мы строим линии».

Сделки — правило бота без изменений (research/noline.py: линия зигзага 3 ATR по закрытиям, агрессоры ≥ 55%, закрытие
в своей половине, тренд, стоп за свингом, 3R; 4h — ретест, 15m — по рынку и ретест). Отличия от канона, заданы до
запуска (группа «да» — канонический вариант; принятие — как research/footprint.py: «да» лучше остальных в каждом
периоде, где в обеих группах ≥ 20 сделок, и t ≥ 2):
  K1 касания      ≥ 3 отдельных касаний линии (закрытие — или тень в режиме по теням — в пределах 0.25 ATR от линии;
                  касания разделены ≥ 3 свечами), включая две опорные точки; Murphy / Bulkowski: третье касание
                  подтверждает линию, у Bulkowski 4 касания лучше 3. Справочно — ≥ 4
  K2 длина        от первой опоры до пробоя дольше медианы IS (Bulkowski: длинные линии лучше коротких)
  K3 глубина      закрытие пробоя за линией на ≥ 0.5 ATR (решительный пробой); справочно ≥ 1% и ≥ 3% цены
                  (Edwards & Magee: «около 3%»)
  K4 по теням     вся система с линией по high / low (как рисует Bulkowski; research/tline.py, zz_lines wicks=True),
                  пробой — закрытием. Отдельный набор сделок: принимается, если R на сделку выше, чем у линии по
                  закрытиям, в каждом периоде и сумма R в месяц не ниже в каждом периоде.

    python -m research.tlcanon collect --symbols <1h-qualified>
    python -m research.tlcanon report
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from . import noline
from .broad import adv30
from .footprint import verdict
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import PER, tf_frame, zz_lines

TFS = tuple(os.environ.get("TLCANON_TFS", "4h,15m").split(","))
TOUCH_ATR, TOUCH_GAP = 0.25, 3
SPLITS = {"K1": "≥ 3 касаний (Murphy, Bulkowski)", "K1b": "≥ 4 касаний (справочно)",
          "K2": "длиннее медианы IS (Bulkowski)", "K3": "пробой ≥ 0.5 ATR за линией",
          "K3b": "пробой ≥ 1% цены (справочно)", "K3c": "пробой ≥ 3% цены (Edwards & Magee, справочно)"}


def touches(y: np.ndarray, atr: np.ndarray, a: int, t: int, side: int, base: float, slope: float) -> int:
    """Отдельные подходы y к линии base + slope·(j − a) на свечах a … t−1 (в пределах TOUCH_ATR ATR с внутренней
    стороны или за линией не дальше того же допуска); подходы разделены ≥ TOUCH_GAP свечами."""
    j = np.arange(a, t)
    dist = side * (base + slope * (j - a) - y[a:t])
    near = np.abs(dist) <= TOUCH_ATR * np.where(atr[a:t] > 0, atr[a:t], np.inf)
    n, last = 0, -10 ** 9
    for k in np.flatnonzero(near):
        if k - last >= TOUCH_GAP:
            n += 1
        last = k
    return n


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str, wicks: bool) -> list[dict]:
    noline.TF = tf
    lines = partial(zz_lines, wicks=wicks)
    orig, noline.zz_lines = noline.zz_lines, lines
    try:
        x = noline.coin_trades(d, liq, sym, lines_only=True)
    finally:
        noline.zz_lines = orig
    if not len(x):
        return []
    info = {(int(r["t"]), int(r["side"])): r for r in lines(d) if r["t"] > 0}
    c = d["close"].to_numpy(dtype="float64")
    hi, lo = d["high"].to_numpy(dtype="float64"), d["low"].to_numpy(dtype="float64")
    atr = _atr(d).to_numpy()
    rows = []
    for r in x.itertuples(index=False):
        t, side = d.index.get_loc(r.t), int(r.side)
        ln = info.get((t, side))
        if ln is None:
            continue
        a, slope = int(ln["a"]), float(ln["slope"])
        y = (hi if side > 0 else lo) if wicks else c
        base = float(y[a])
        line_t = base + slope * (t - a)
        rows.append({"kind": "wick" if wicks else "close", "symbol": sym, "tf": tf, "t": r.t, "side": side,
                     "entry": r.entry, "R": float(r.R3), "touches": touches(y, atr, a, t, side, base, slope),
                     "length": t - a, "depth_atr": side * (c[t] - line_t) / atr[t] if atr[t] > 0 else np.nan,
                     "depth_pct": side * (c[t] - line_t) / line_t})
    return rows


def collect(root: Path, syms: list[str]) -> None:
    rows = []
    for s in mine(syms):
        for tf in TFS:
            try:
                if tf == "15m" and not (root / f"{s}-15m.parquet").exists():
                    continue
                d = tf_frame(root, s, tf)
                if d is None or len(d) < 500:
                    continue
                liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
                for wicks in (False, True):
                    rows += coin_rows(d, liq, s, tf, wicks)
            except Exception as e:
                print(f"  tlcanon {s} {tf}: пропуск ({e})", flush=True)
    print(f"  tlcanon: сделок {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("tlcanon"), index=False)


def label(z: pd.DataFrame) -> pd.DataFrame:
    z = z.copy()
    a, b = PER["is"]
    med = z[(z.t >= a) & (z.t < b)].length.median()
    yes = lambda m: np.where(m, "да", "нет")
    z["K1"], z["K1b"] = yes(z.touches >= 3), yes(z.touches >= 4)
    z["K2"] = yes(z.length > med)
    z["K3"], z["K3b"], z["K3c"] = yes(z.depth_atr >= 0.5), yes(z.depth_pct >= 0.01), yes(z.depth_pct >= 0.03)
    z.attrs["median_len"] = med
    return z


def per_month(z: pd.DataFrame) -> dict[str, float]:
    return {p: z[(z.t >= a) & (z.t < b)].R.sum() / max(1.0, (pd.Timestamp(b) - pd.Timestamp(a)).days / 30.4)
            for p, (a, b) in PER_ALL.items()}


def report() -> None:
    parts = all_parts("tlcanon")
    print("\n===== TLCANON: линии бота против канона — касания, длина, глубина пробоя, линия по теням (признаки до запуска; "
          "ячейка — R на сделку, t по дням, прибыльных, сделок в месяц) =====")
    if not parts:
        print("  tlcanon: нет данных")
        return
    x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
    for tf, entry in (("4h", "retest"), ("15m", "market"), ("15m", "retest")):
        if tf not in TFS:
            continue
        base = x[(x.tf == tf) & (x.entry == entry)]
        cl, wk = base[base.kind == "close"], base[base.kind == "wick"]
        if cl.empty:
            continue
        print(f"\n  === {tf}, вход {entry} ===")
        print("   K4 линия по теням против линии по закрытиям (бот):")
        ok = []
        for nm, z in (("по закрытиям", cl), ("по теням", wk)):
            cells = "  ".join(f"{p}: {_cell(z[(z.t >= a) & (z.t < b)])}" for p, (a, b) in PER_ALL.items()
                              if ((z.t >= a) & (z.t < b)).any())
            pm = per_month(z)
            print(f"     {nm} ({len(z)}): {cells}  всё время: {_cell(z)}")
            print(f"       R в месяц: " + " / ".join(f"{p} {v:+.1f}" for p, v in pm.items()))
        for p, (a, b) in PER_ALL.items():
            mc, mw = cl[(cl.t >= a) & (cl.t < b)], wk[(wk.t >= a) & (wk.t < b)]
            if len(mc) >= 20 and len(mw) >= 20:
                ok.append(mw.R.mean() > mc.R.mean() and per_month(wk)[p] >= per_month(cl)[p])
        print(f"     → по теням {'принято' if ok and all(ok) else 'не принято'} "
              f"(лучше в {sum(ok)} из {len(ok)} периодов по R на сделку и R в месяц)")
        for kind, z0 in (("по закрытиям (бот)", cl), ("по теням", wk)):
            if z0.empty:
                continue
            z = label(z0)
            print(f"   {kind}: касаний медиана {z.touches.median():.0f} (≥ 3 у {(z.touches >= 3).mean():.0%}), "
                  f"длина медиана IS {z.attrs['median_len']:.0f} свечей, пробой медиана {z.depth_atr.median():.2f} ATR / "
                  f"{z.depth_pct.median():.1%}")
            for nm, zz in (("все", z), ("шорты", z[z.side < 0])):
                print(f"    {nm}:")
                for key, title in SPLITS.items():
                    print(f"     {key} {title}:")
                    for g in ("да", "нет"):
                        m = zz[zz[key] == g]
                        cells = "  ".join(f"{p}: {_cell(m[(m.t >= a) & (m.t < b)])}" for p, (a, b) in PER_ALL.items()
                                          if ((m.t >= a) & (m.t < b)).any())
                        print(f"       {g}: доля {len(m) / max(1, len(zz)):.0%}  {cells}  всё время: {_cell(m)}")
                    print(f"       «да» минус «нет»: {verdict(zz, key, 'да')[0]}")


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
