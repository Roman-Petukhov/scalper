"""Как выжать больше из 4h-правила бота, не меняя само правило. Три рычага:

1. Ликвидность. Бот берёт монеты с оборотом от $20M в день (adv30 на момент сигнала). Все перпетуалы архива (1h-all,
   включая делистнутые — без отбора «выросших» монет): корзины $5–10M, 10–20M, 20–50M, 50–150M, >150M и пороги
   «от $5M / от $10M». Издержки x2 / x3 — запас на тонкий стакан Bybit.
2. Лимитка ретеста: сдвиг от линии к цене −0.2 / −0.1 / 0 (бот) / +0.1 / +0.25 ATR (плюс — ближе к цене: чаще
   исполняется, вход хуже). Монеты 1h-qualified, оборот от $20M.
3. Размер по качеству сигнала (не фильтр): доля агрессоров, глубина закрытия за линией (ATR), место закрытия в свече.
   Терцили — по IS; вес 1.5 / 1 / 0.5 для верхней / средней / нижней трети. Сравнение по R в месяц и просадке на
   единицу среднего веса (тот же средний риск).
Всё на 3R, правило бота: агрессоры >= 55%, тренд старшего ТФ, закрытие в верхней половине свечи, без закрепления.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import tline
from .shard import all_parts, mine, part_path
from .smc import _cell
from .tline import PER, TAKER, _bot_base, _per_month, _years, coin_trades, market_context

OFFSETS = (-0.2, -0.1, 0.0, 0.1, 0.25)
COLS = ["symbol", "t", "side", "line", "entry", "confirm", "aggr", "with_trend", "close_loc", "adv", "risk_pct", "R3",
        "brk_atr"]
BUCKETS = ((5e6, 10e6), (10e6, 20e6), (20e6, 50e6), (50e6, 150e6), (150e6, np.inf))


def collect(root: Path, syms_all: list[str], syms_q: list[str]) -> None:
    ctx = market_context(root)
    tline.LINES, tline.SCALE_LINES = ("zz",), {}
    q = set(syms_q)
    rows = []
    for s in mine(syms_all):
        for off in OFFSETS if s in q else (0.0,):
            tline.RETEST_OFF = off
            try:
                x = coin_trades(root, s, "4h", ctx)
            except Exception as e:
                print(f"  improve {s} {off}: пропуск ({e})", flush=True)
                continue
            if not len(x):
                continue
            x = x[(x.line == "zz") & ~x.confirm][COLS].copy()
            if off != 0.0:
                x = x[x.entry == "retest"]
            x["off"], x["qualified"] = off, s in q
            rows.append(x)
    tline.RETEST_OFF = 0.0
    print(f"  improve: частей {len(rows)}", flush=True)
    if rows:
        pd.concat(rows, ignore_index=True).to_parquet(part_path("improve"), index=False)


def _table(items: list[tuple[str, pd.DataFrame]], col: str = "R3") -> None:
    rows = [{"вариант": nm, **{p: _cell(z[z.per == p].assign(R=z[col])) for p in PER}, "R/мес IS / VAL / HO": _per_month(z, col)}
            for nm, z in items]
    print(pd.DataFrame(rows).to_string(index=False))


def _costs(z: pd.DataFrame, k: int) -> pd.DataFrame:
    return z.assign(R3=z.R3 - k * 2 * TAKER / z.risk_pct)


def _dd(z: pd.DataFrame, col: str) -> float:
    eq = z.sort_values("t")[col].cumsum().to_numpy()
    return float((np.maximum.accumulate(np.r_[0.0, eq]) - np.r_[0.0, eq]).max()) if len(eq) else 0.0


def report() -> None:
    parts = all_parts("improve")
    print("\n===== IMPROVE: 4h, правило бота — ликвидность, место лимитки ретеста, размер по качеству сигнала; ячейка — "
          "R на сделку (t по дням, прибыльных, сделок в месяц), всё на 3R =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["per"] = ""
    for p, (a, b) in PER.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    base = _bot_base(df)
    b0 = base[base.off == 0.0]

    for entry in ("retest", "market"):
        g = b0[b0.entry == entry]
        print(f"\n  --- 1. {'ретест (бот)' if entry == 'retest' else 'рынок'}: оборот монеты на момент сигнала, все "
              f"перпетуалы архива ({g.symbol.nunique()} монет со сделками) ---")
        items = []
        for lo, hi in BUCKETS:
            z = g[(g.adv >= lo) & (g.adv < hi)]
            nm = f"${lo / 1e6:.0f}–{hi / 1e6:.0f}M" if np.isfinite(hi) else f"от ${lo / 1e6:.0f}M"
            items += [(nm, z), ("  издержки x2", _costs(z, 1)), ("  издержки x3", _costs(z, 2))]
        for lo in (5e6, 10e6, 20e6):
            z = g[g.adv >= lo]
            items += [(f"порог от ${lo / 1e6:.0f}M{' (бот)' if lo == 20e6 else ''}", z), ("  издержки x2", _costs(z, 1))]
        _table(items)
        print(f"  $5–20M по годам: {_years(g[(g.adv >= 5e6) & (g.adv < 20e6)])}")

    q = base[base.qualified & (base.adv >= 20e6) & (base.entry == "retest")]
    print("\n  --- 2. лимитка ретеста: сдвиг от линии к цене (ATR), монеты от $20M ---")
    _table([(f"{off:+.2f} ATR{' (бот)' if off == 0 else ''}", q[q.off == off]) for off in OFFSETS])
    for off in OFFSETS:
        z = q[q.off == off]
        print(f"  {off:+.2f}: по годам {_years(z)}; просадка R " +
              " / ".join(f"{_dd(z[z.per == p], 'R3'):.1f}" for p in PER))

    g = b0[(b0.entry == "retest") & (b0.adv >= 20e6)].copy()
    print("\n  --- 3. размер по качеству сигнала: терцили по IS, вес 1.5 / 1 / 0.5 ---")
    is_ = g[g.per == "is"]
    for feat, nm in (("aggr", "доля агрессоров"), ("brk_atr", "закрытие за линией, ATR"), ("close_loc", "место закрытия")):
        lo, hi = is_[feat].quantile([1 / 3, 2 / 3])
        t3 = np.where(g[feat] >= hi, 2, np.where(g[feat] >= lo, 1, 0))
        print(f"\n  {nm}: границы по IS {lo:.3f} / {hi:.3f}")
        _table([(f"нижняя треть", g[t3 == 0]), ("средняя", g[t3 == 1]), ("верхняя", g[t3 == 2])])
        w = np.choose(t3, [0.5, 1.0, 1.5])
        z = g.assign(Rw=g.R3 * w, w=w)
        rows = []
        for p in PER:
            zz = z[z.per == p]
            k = zz.w.mean() if len(zz) else 1.0
            rows.append(f"{p}: R/сделку {zz.R3.mean():+.3f} → {zz.Rw.mean() / k:+.3f}, просадка {_dd(zz, 'R3'):.1f} → "
                        f"{_dd(zz.assign(Rw=zz.Rw / k), 'Rw'):.1f}")
        print("  с весами (на единицу среднего веса): " + "; ".join(rows))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 360)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--qualified", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x],
                [x for x in a.qualified.split(",") if x])
    else:
        report()
