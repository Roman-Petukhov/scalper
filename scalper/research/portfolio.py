"""
Итоговый портфель: обе проверенные стратегии на всей вселенной (16 исходных + новые монеты).
Плечо каждой ноги подбирается ТОЛЬКО по IS (цель волатильности TARGET_VOL), с потолком пиковой экспозиции.

    python -m research.portfolio --root <data> --out <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data import UNIVERSE
from .engine import BAR_MINUTES, metrics, run
from .search import COST_BPS, STRESS_COST_BPS
from .universe import CANDIDATES, EXTRA, available
from .wave2 import Data2, _cut

TARGET_VOL = 0.10
MAX_GROSS = 3.0


def leg(data: Data2, fn, p: dict, cost: float):
    pnls, poss = {}, {}
    for s in data.symbols:
        df = data.get(s, "1h", "full")
        pos, ex = fn(df, **p)
        r = run(df, np.asarray(pos, float), cost, BAR_MINUTES["1h"], ex)
        pnls[s], poss[s] = r.pnl, r.pos
    pnl = pd.DataFrame(pnls).fillna(0.0)
    pos = pd.DataFrame(poss).fillna(0.0)
    return pnl.mean(axis=1), pos


def entries_per_day(pos: pd.DataFrame) -> pd.Series:
    prev = pos.shift(1).fillna(0.0)
    e = ((pos != 0) & (pos != prev)).sum(axis=1)
    return e.resample("1D").sum()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 250)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    syms = available(root, UNIVERSE + EXTRA)
    data = Data2(root, syms)
    n = len(syms)
    print(f"Вселенная: {n} монет")
    for cost in (COST_BPS, STRESS_COST_BPS):
        legs, poss = {}, {}
        for name, (fn, p) in CANDIDATES.items():
            legs[name], poss[name] = leg(data, fn, p, cost)
        if cost == COST_BPS:
            lev = {}
            for name in legs:
                vol_lev = TARGET_VOL / metrics(_cut(legs[name], "is"))["ann_vol"]
                gross_is = _cut(poss[name].abs().sum(axis=1) / n, "is")
                lev[name] = min(vol_lev, MAX_GROSS / gross_is.max())
        port = sum(legs[k] * lev[k] for k in legs)
        gross = sum(poss[k].abs().sum(axis=1) / n * lev[k] for k in legs)
        print(f"\n=== издержки {cost:g} б.п. на сторону; плечо ног: "
              + ", ".join(f"{k} x{v:.1f} (на позицию {v / n:.1%} капитала)" for k, v in lev.items()) + " ===")
        rows = []
        for per in ("is", "val", "ho", "oos1"):
            x = _cut(port, per)
            m = metrics(x)
            rows.append({"period": per, "sharpe": m["sharpe"], "ann_ret": m["ann_ret"], "ann_vol": m["ann_vol"],
                         "max_dd": m["max_dd"], "worst_day": x.resample("1D").sum().min(),
                         "pos_months": float((x.resample("ME").sum() > 0).mean())})
        print(pd.DataFrame(rows).round(3).to_string(index=False))
        if cost == COST_BPS:
            ent = sum(entries_per_day(poss[k]) for k in poss)
            conc = sum((poss[k] != 0).sum(axis=1) for k in poss)
            y = pd.DataFrame({"доходность": port.groupby(port.index.year).sum(),
                              "сделок/день": ent.groupby(ent.index.year).mean(),
                              "макс. одновременных позиций": conc.groupby(conc.index.year).max(),
                              "пик экспозиции": gross.groupby(gross.index.year).max()})
            print("\nпо годам:")
            print(y.round(3).to_string())
            port.resample("1D").sum().to_frame("pnl").to_csv(out / "portfolio_daily.csv")
            mon = pd.DataFrame({k: (legs[k] * lev[k]).resample("ME").sum() for k in legs})
            mon["портфель"] = mon.sum(axis=1)
            mon["сделок"] = ent.resample("ME").sum()
            print("\nпо месяцам с 2025-01:")
            print(mon[mon.index >= "2025-01-01"].round(4).to_string())
