"""Перезаход в шорт после выноса стопа — проверка на свежих данных. Находка research/longstop.py (не заданная заранее,
4h, на тех же данных): шорты бота с перезаходом +0.556R против +0.481R, не ниже во всех периодах. На дневках тот же
перезаход в минусе (research/htftrend.py). Здесь — другие ТФ, где правило бота уже есть, а перезаход не смотрели.

Правило перезахода (как в longstop.reentry): сделка бота закрыта стопом; если в течение 3 свечей после свечи стопа
закрытие вернулось за уровень стопа (для шорта — ниже), вход по рынку на этом закрытии, стоп за экстремумом выноса
+ 0.1 ATR (риск 0.3–4 ATR), цель 3R, срок бота. Перезаход один на сигнал.
Наборы (заданы до запуска; агрессоры >= 55%, тренд, свеча, оборот — как у бота):
  15m шорты   — по рынку, наклон линии <= 0.025 ATR (как в живых настройках) — главный;
  2h шорты    — ретест, дневной тренд (research/tfup.py) — второй;
  4h шорты    — ретест — для справки (на них находка, не решают).
Принятие (на 15m и 2h отдельно): сами перезаходы — R на сделку > 0 в каждом периоде 2020 / 2021 / IS / VAL / HO, где
их >= 20 (таких периодов >= 3), t по дням >= 3. Внедрять — только если пройдут оба набора (сначала тень в журнале).

    python -m research.reentry collect --symbols <монеты>      # RE_TFS=15m / 2h / 4h
    python -m research.reentry report
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import longstop, noline
from .broad import adv30
from .oos import PER_ALL
from .ppr import day_t
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .spec import months
from .tline import tf_frame, zz_lines

TFS = tuple(os.environ.get("RE_TFS", "15m,2h,4h").split(","))
ENTRY = {"15m": "market", "2h": "retest", "4h": "retest"}
PRIMARY = ("15m", "2h")
MAX_SLOPE_15M = 0.025


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str) -> list[dict]:
    noline.TF = longstop.TF = tf
    x = noline.coin_trades(d, liq, sym, lines_only=True)
    if not len(x):
        return []
    x = x[(x.side < 0) & (x.entry == ENTRY[tf])]
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].fillna(0.0).to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    atr = _atr(d).to_numpy()
    recs = {(int(r["t"]), int(r["side"])): r for r in zz_lines(d) if r["t"] > 0} if tf == "15m" else {}
    rows = []
    for r in x.itertuples(index=False):
        if not np.isfinite(r.R3):
            continue
        t, side, stop, ex = d.index.get_loc(r.t), int(r.side), float(r.stop), int(r.exit_i)
        if tf == "15m":
            rec = recs.get((t, side))
            if rec is None or not (atr[t] > 0) or abs(float(rec["slope"])) / atr[t] > MAX_SLOPE_15M:
                continue
        hit = r.R3 < 0 and longstop.stopped(hi, lo, ex, side, stop)
        re = longstop.reentry(o, hi, lo, c, f, atr, ex, side, stop) if hit else 0.0
        rows.append({"symbol": sym, "tf": tf, "t": r.t, "R": float(r.R3), "stopped": bool(hit),
                     "re": re if re != 0.0 else np.nan})
    return rows


def collect(root: Path, syms: list[str]) -> None:
    for tf in TFS:
        rows = []
        for s in mine(syms):
            try:
                if tf == "15m" and not (root / f"{s}-15m.parquet").exists():
                    continue
                d = tf_frame(root, s, tf)
                if d is None or len(d) < 500:
                    continue
                rows += coin_rows(d, adv30(root, s).reindex(d.index.floor("D")).to_numpy(), s, tf)
            except Exception as e:
                print(f"  reentry {s} {tf}: пропуск ({e})", flush=True)
        print(f"  reentry {tf}: шортов {len(rows)}", flush=True)
        if rows:
            pd.DataFrame(rows).to_parquet(part_path(f"reentry_{tf}"), index=False)


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
    return len(pos) >= 3 and all(pos) and np.isfinite(t) and t >= 3, f"{'  '.join(cells)}\n      всё время {_cell(v)}"


def report() -> None:
    print("\n===== REENTRY: перезаход в шорт после выноса стопа (сделки бота; R на сделку, сделок, R в месяц) =====")
    passed = []
    for tf in TFS:
        parts = all_parts(f"reentry_{tf}")
        if not parts:
            print(f"  reentry {tf}: нет данных")
            continue
        x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
        re = x.dropna(subset=["re"])
        both = x.assign(R=x.R + x.re.fillna(0.0))
        role = "главный" if tf == PRIMARY[0] else "второй" if tf in PRIMARY else "справка, на нём находка"
        print(f"\n  --- {tf} шорты ({role}): сигналов {len(x)}, стопом {x.stopped.mean():.0%}, перезаходов {len(re)} "
              f"({len(re) / max(1, x.stopped.sum()):.0%} стопов) ---")
        print(f"    бот: {_cell(x)}\n    бот + перезаход: {_cell(both)}")
        ok, txt = verdict(re[["t", "re"]].rename(columns={"re": "R"}))
        ok = ok and tf in PRIMARY
        print(f"    сами перезаходы: {'принято' if ok else 'нет'}\n      {txt}")
        if ok:
            passed.append(tf)
    good = all(tf in passed for tf in PRIMARY)
    print("\n  ИТОГ: " + ("перезаход прошёл на 15m и 2h — кандидат в тень журнала" if good else
                         f"прошли: {', '.join(passed) or 'ни один набор'} — не внедряем"))


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
