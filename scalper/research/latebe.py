"""Поздний безубыток у самой цели (по наблюдению владельца: «цена часто разворачивается у нашего тейка 3R»).

Сигналы и входы — правило бота без изменений (research/noline.py: линия зигзага 3 ATR, агрессоры ≥ 55%, закрытие в своей
половине, тренд старшего ТФ, стоп за свингом PIV5 ± 0.1 ATR, 4h — ретест на уровне, 15m — по рынку и ретест). Меняется
только сопровождение. Варианты (заданы до запуска):
  base      цель 3R, стоп не двигается — бот, контроль;
  be25      цена прошла +2.5R → со следующей свечи стоп на цену входа (0R до комиссий);
  be20      то же после +2.0R;
  lock25    после +2.5R стоп на +1R (справочно);
  tp28      цель 2.8R без переноса (справочно: «не дотягиваем 3R»).
Свеча, в которой цена дошла до порога, ещё со старым стопом: порядок цен внутри свечи неизвестен, переносим только со
следующей (осторожно для вариантов с переносом). Стоп внутри свечи проверяется раньше тейка, как у бота.
Справочно — сколько сделок дошли до 2.5R, не взяли 3R и чем закончились.

Принятие: be25 лучше base по R на сделку в каждом периоде (2020 / 2021 / IS / VAL / HO), где ≥ 20 сделок, и парная
разница по тем же сделкам за всё время с t ≥ 2. Отдельно для 4h (основной бот) и 15m.

    python -m research.latebe collect --symbols <1h-qualified>
    python -m research.latebe report
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
from .smc import _cell
from .tline import HOLD, MAKER, TAKER, tf_frame

TFS = tuple(os.environ.get("LATEBE_TFS", "4h,15m").split(","))
VARIANTS = {"base": (3.0, None, 0.0), "be25": (3.0, 2.5, 0.0), "be20": (3.0, 2.0, 0.0), "lock25": (3.0, 2.5, 1.0),
            "tp28": (2.8, None, 0.0)}
TITLES = {"base": "3R, стоп на месте (бот)", "be25": "б/у после +2.5R", "be20": "б/у после +2.0R",
          "lock25": "+1R после +2.5R", "tp28": "цель 2.8R"}


def walk(o, h, lo, c, fund, e, side, entry, stop, k, be_at, be_to, max_hold, entry_fee) -> tuple[float, float]:
    """Вся позиция на k·R; после +be_at·R (если задан) со следующей свечи стоп на entry + be_to·R. Стоп раньше тейка.
    Возвращает (R, наибольший достигнутый ход в R до выхода, без свечи выхода по стопу)."""
    n = len(c)
    risk = side * (entry - stop)
    if not (risk > 0):
        return np.nan, np.nan
    tp = entry + side * k * risk
    sp, armed, paid, mfe = stop, False, 0.0, 0.0
    for j in range(e + 1, n):
        paid += fund[j]
        if (side > 0 and lo[j] <= sp) or (side < 0 and h[j] >= sp):
            ex = min(sp, o[j]) if side > 0 else max(sp, o[j])
            return (side * (ex - entry) - (entry_fee + TAKER) * entry) / risk - side * paid * entry / risk, mfe
        best = (h[j] - entry) / risk if side > 0 else (entry - lo[j]) / risk
        mfe = max(mfe, best)
        if (side > 0 and h[j] >= tp) or (side < 0 and lo[j] <= tp):
            return (k * risk - (entry_fee + MAKER) * entry) / risk - side * paid * entry / risk, mfe
        if j - e >= max_hold or j == n - 1:
            return (side * (c[j] - entry) - (entry_fee + TAKER) * entry) / risk - side * paid * entry / risk, mfe
        if be_at is not None and not armed and best >= be_at:
            sp, armed = entry + side * be_to * risk, True
    return np.nan, mfe


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str) -> list[dict]:
    noline.TF = tf
    x = noline.coin_trades(d, liq, sym, lines_only=True)
    if not len(x):
        return []
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    rows = []
    for r in x.itertuples(index=False):
        fill, side, px, stop = int(r.fill_i), int(r.side), float(r.px), float(r.stop)
        fee = MAKER if r.entry == "retest" else TAKER
        out = {"symbol": sym, "tf": tf, "t": r.t, "side": side, "entry": r.entry}
        if r.entry == "retest" and ((side > 0 and lo[fill] <= stop) or (side < 0 and hi[fill] >= stop)):
            loss = (side * (stop - px) - (fee + TAKER) * px) / (side * (px - stop))
            out.update({v: loss for v in VARIANTS}, mfe=0.0)          # стоп в свече исполнения — у всех одинаково
        else:
            for v, (k, be_at, be_to) in VARIANTS.items():
                out[v], m = walk(o, hi, lo, c, f, fill, side, px, stop, k, be_at, be_to, HOLD[tf], fee)
                if v == "base":
                    out["mfe"] = m
        rows.append(out)
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
                rows += coin_rows(d, liq, s, tf)
            except Exception as e:
                print(f"  latebe {s} {tf}: пропуск ({e})", flush=True)
    print(f"  latebe: сделок {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("latebe"), index=False)


def paired(z: pd.DataFrame, v: str) -> tuple[str, bool]:
    cells, ok = [], []
    for p, (a, b) in PER_ALL.items():
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
    parts = all_parts("latebe")
    print("\n===== LATEBE: безубыток у самой цели (после +2.5R / +2.0R) против цели 3R без переноса; R на сделку "
          "(t по дням, доля > 0, сделок в месяц) =====")
    if not parts:
        print("  latebe: нет данных")
        return
    x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
    x = x.dropna(subset=list(VARIANTS))
    for tf in TFS:
        for entry in ("retest", "market"):
            for nm, z in (("обе стороны", x[(x.tf == tf) & (x.entry == entry)]),
                          ("только шорты", x[(x.tf == tf) & (x.entry == entry) & (x.side < 0)])):
                if len(z) < 20:
                    continue
                near = z[(z.mfe >= 2.5) & (z.base < 2.5)]
                print(f"\n  --- {tf}, вход {entry}, {nm}: сделок {len(z)}; дошли до +2.5R {(z.mfe >= 2.5).mean():.0%}, "
                      f"из них не взяли 3R {len(near)} ({len(near) / max(1, (z.mfe >= 2.5).sum()):.0%}; в стоп "
                      f"{(near.base < 0).sum()}, по сроку {(near.base >= 0).sum()}); дошли до +2.0R "
                      f"{(z.mfe >= 2.0).mean():.0%} ---")
                for v in VARIANTS:
                    cells = "  ".join(f"{p}: {_cell(z[(z.t >= a) & (z.t < b)].assign(R=z[v]))}"
                                      for p, (a, b) in PER_ALL.items() if ((z.t >= a) & (z.t < b)).any())
                    print(f"    {v:7s} {TITLES[v]}: {cells}  всё время: {_cell(z.assign(R=z[v]))}")
                    if v != "base":
                        line, _ = paired(z, v)
                        print(f"            минус бот: {line}")


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
