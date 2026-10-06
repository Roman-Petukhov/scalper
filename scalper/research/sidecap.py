"""Риск портфеля 4h: несколько шортов (или лонгов) альтов сразу — по сути одна ставка на рынок.

Правило бота (линии по значимым точкам, агрессоры >= 55%, тренд старшего ТФ, закрытие в верхней половине свечи,
оборот от $20M), всё на 3R, 1R на сделку. Сделки берутся по времени входа, как это делал бы бот: общий лимит
12 позиций, дневной стоп 4R (4% при риске 1%). Варианты поверх:
    лимит по стороне   не больше N позиций в одну сторону (остальные сигналы этой стороны пропускаются)
    серия убытков      после K стопов подряд (по времени выхода) риск — половина, пока не будет прибыльной сделки
Время выхода — по выходу лесенки (hold_bars): для цели 3R это верхняя граница, пересечений позиций не меньше, чем на
самом деле. Порог «заметно лучше» — ниже просадка и худшие 3 дня без потери больше ~10% R в месяц на HOLDOUT.
"""
from __future__ import annotations

import argparse
import heapq
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN
from .shard import all_parts, mine, part_path
from .tline import BAR_MIN, PER, _bot_base, coin_trades, market_context

COLS = ["symbol", "t", "side", "tf", "line", "entry", "confirm", "aggr", "with_trend", "close_loc", "adv",
        "wait_bars", "hold_bars", "R3", "R3_old", "stop_in_fill"]
LIMIT, DAILY_STOP = 12, 4.0
SIDE_CAPS = (np.inf, 8, 6, 5, 4, 3)
STREAKS = (None, 5, 4)


def collect(root: Path, syms: list[str]) -> None:
    ctx = market_context(root)
    parts = []
    for s in mine(syms):
        try:
            x = coin_trades(root, s, "4h", ctx)
            if len(x):
                parts.append(x[x.line == "zz"][COLS])
        except Exception as e:
            print(f"  sidecap {s}: пропуск ({e})", flush=True)
    print(f"  sidecap: монет со сделками {len(parts)}", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path("sidecap"), index=False)


def simulate(g: pd.DataFrame, side_cap: float, streak: int | None) -> pd.DataFrame:
    """Сделки, которые взял бы бот, с весом риска w (1 или 0.5 после серии убытков); Rw — результат с весом.
    Закрытия становятся известны по мере наступления времени выхода (куча по t_out)."""
    bar = pd.Timedelta(minutes=BAR_MIN["4h"])
    x = g.assign(t_in=g.t + g.wait_bars.fillna(0) * bar)
    x = x.assign(t_out=x.t_in + x.hold_bars * bar).sort_values("t_in")
    pending: list[tuple[pd.Timestamp, int, float, int]] = []    # куча: (выход, порядковый №, R с весом, сторона)
    day_loss: dict[pd.Timestamp, float] = {}
    run, seq = 0, 0
    keep, weight, same_at_entry = [], [], []
    for i, r in zip(x.index, x.itertuples()):
        while pending and pending[0][0] <= r.t_in:               # закрылось к моменту входа — итог известен
            t_out, _, v, _ = heapq.heappop(pending)
            run = run + 1 if v < 0 else 0
            if v < 0:
                d = t_out.floor("D")
                day_loss[d] = day_loss.get(d, 0.0) - v
        same = sum(1 for o in pending if o[3] == r.side)
        if len(pending) >= LIMIT or same >= side_cap or day_loss.get(r.t_in.floor("D"), 0.0) >= DAILY_STOP:
            continue
        w = 0.5 if streak is not None and run >= streak else 1.0
        keep.append(i)
        weight.append(w)
        same_at_entry.append(same)
        seq += 1
        heapq.heappush(pending, (r.t_out, seq, r.R3 * w, r.side))
    k = x.loc[keep].copy()
    k["w"], k["same"] = weight, same_at_entry
    k["Rw"] = k["R3"] * k["w"]
    return k


def _stats(k: pd.DataFrame, n_all: int) -> dict:
    out = {"взято": f"{len(k) / n_all:.0%}"}
    for p, (a, b) in PER.items():
        z = k[(k.t_out >= a) & (k.t_out < b)]
        months = (pd.Timestamp(b) - pd.Timestamp(a)).days / 30.4
        eq = z.sort_values("t_out").Rw.cumsum().to_numpy()
        dd = float((np.maximum.accumulate(np.r_[0.0, eq]) - np.r_[0.0, eq]).max()) if len(eq) else 0.0
        days = z.groupby(z.t_out.dt.floor("D")).Rw.sum()
        days = days.reindex(pd.date_range(days.index.min(), days.index.max(), freq="D"), fill_value=0.0) \
            if len(days) else days
        worst3 = float(days.rolling(3).sum().min()) if len(days) >= 3 else float("nan")
        out[p] = f"{z.Rw.sum() / months:+.1f}R/мес, просадка {dd:.1f}, худшие 3 дня {worst3:+.1f}"
    return out


def report() -> None:
    parts = all_parts("sidecap")
    print("\n===== SIDECAP: 4h, лимит позиций в одну сторону и снижение риска после серии убытков; правило бота, "
          f"всё на 3R, общий лимит {LIMIT}, дневной стоп {DAILY_STOP:g}R; по каждому периоду — R в месяц, "
          "макс. просадка (R), худшие 3 дня подряд (R) =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    base = _bot_base(df[df.adv.isna() | (df.adv >= ADV_MIN)])
    r = base[base.entry == "retest"]
    if "R3_old" in r.columns and len(r):
        print("\n  --- ПРОВЕРКА БЭКТЕСТА: ретест исполняется внутри свечи, а стоп в свече исполнения раньше не проверялся ---")
        print(f"  стоп задет в свече исполнения: {r.stop_in_fill.mean():.1%} сделок ретеста")
        for p, (a, b) in PER.items():
            z = r[(r.t >= a) & (r.t < b)]
            months = (pd.Timestamp(b) - pd.Timestamp(a)).days / 30.4
            print(f"  {p}: R на сделку было {z.R3_old.mean():+.3f}, стало {z.R3.mean():+.3f}; "
                  f"R в месяц было {z.R3_old.sum() / months:+.1f}, стало {z.R3.sum() / months:+.1f} (сделок {len(z)})")
    for entry in ("retest", "market"):
        g = base[base.entry == entry]
        if g.empty:
            continue
        full = simulate(g, np.inf, None)
        sh = full[full.side == -1]
        print(f"\n  --- {'ретест' if entry == 'retest' else 'рынок'}: сделок {len(g)}, шортов {np.mean(g.side == -1):.0%}; "
              f"при входе уже открыто в ту же сторону — медиана {np.median(full.same):.0f}, 90% до "
              f"{np.percentile(full.same, 90):.0f}, максимум {full.same.max()} ---")
        print("  R на сделку по числу уже открытых в ту же сторону (0, 1–2, 3–5, 6+): " + ", ".join(
            f"{nm}: {z.R3.mean():+.2f} ({len(z)})" for nm, z in (
                ("0", full[full.same == 0]), ("1–2", full[full.same.between(1, 2)]),
                ("3–5", full[full.same.between(3, 5)]), ("6+", full[full.same >= 6])) if len(z)))
        print(f"  то же для шортов: " + ", ".join(
            f"{nm}: {z.R3.mean():+.2f} ({len(z)})" for nm, z in (
                ("0", sh[sh.same == 0]), ("1–2", sh[sh.same.between(1, 2)]),
                ("3–5", sh[sh.same.between(3, 5)]), ("6+", sh[sh.same >= 6])) if len(z)))
        rows = []
        for cap in SIDE_CAPS:
            for st in STREAKS:
                k = simulate(g, cap, st)
                rows.append({"в одну сторону": "без лимита" if cap == np.inf else f"до {cap:g}",
                             "после серии": "—" if st is None else f"{st} стопов → риск x0.5", **_stats(k, len(g))})
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 400)
    pd.set_option("display.max_colwidth", 80)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x])
    else:
        report()
