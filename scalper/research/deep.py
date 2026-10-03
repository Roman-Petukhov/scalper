"""
Углублённая проверка кандидатов, прошедших IS -> VAL -> HOLDOUT, и их портфель.

Всё здесь — диагностика устойчивости, а не отбор: параметры кандидатов не меняются по итогам
этих таблиц (иначе holdout перестаёт быть holdout).

    python -m research.deep --root <data> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import strategies as S
from .data import UNIVERSE
from .engine import BAR_MINUTES, metrics, run
from .search import COST_BPS, STRESS_COST_BPS, grid
from .wave2 import SINGLE, Data2, _cut, eval_single, evaluate, oi_price

CANDIDATES = {
    "oi_liq": ("oi_price", "1h", {"k": 12, "thr": 2.5, "hold": 4, "mode": "liq", "sign": 1}),
    "fe_q99": ("funding_extreme", "1h", {"q": 0.99, "hold": 8, "win_days": 30}),
}
NEIGHBORS = {
    "oi_liq": ("oi_price", grid(k=[4, 8, 12, 24], thr=[2.0, 2.5, 3.0], hold=[2, 4, 8, 12], mode=["liq"], sign=[1])),
    "fe_q99": ("funding_extreme", grid(q=[0.98, 0.99, 0.995], hold=[4, 8, 12, 16], win_days=[14, 30, 60])),
}
TARGET_VOL = 0.10          # годовая волатильность каждой ноги портфеля, плечо считается только по IS
MAX_GROSS = 2.0            # потолок пиковой валовой экспозиции ноги на IS (доли капитала)


def neighbors(data: Data2, out: Path) -> None:
    for name, (fam, g) in NEIGHBORS.items():
        rows = []
        for p in g:
            r = {"params": json.dumps(p)}
            for per in ("is", "val", "ho"):
                _, s, _ = evaluate(data, fam, "1h", p, per, COST_BPS)
                r[f"{per}_sharpe"], r[f"{per}_trades"] = s["sharpe"], s["trades"]
            rows.append(r)
        df = pd.DataFrame(rows)
        df.to_csv(out / f"deep_neighbors_{name}.csv", index=False)
        oos_pos = ((df["val_sharpe"] > 0) & (df["ho_sharpe"] > 0)).mean()
        print(f"\n{name}: соседи ({len(df)} конфигов), доля с VAL>0 и HO>0: {oos_pos:.0%}; "
              f"медиана Sharpe IS {df['is_sharpe'].median():.2f} / VAL {df['val_sharpe'].median():.2f} / "
              f"HO {df['ho_sharpe'].median():.2f}")
        print(df.round(2).to_string(index=False))


def trades_table(data: Data2, fam: str, p: dict) -> pd.DataFrame:
    fn = S.funding_extreme if fam == "funding_extreme" else next(f for _, f, _ in SINGLE[fam])
    rows = []
    for s in data.symbols:
        df = data.get(s, "1h", "full")
        pos, ex = fn(df, **p)
        res = run(df, np.asarray(pos, float), COST_BPS, 60, ex)
        pv = res.pos.to_numpy()
        prev = np.concatenate([[0.0], pv[:-1]])
        entry = (pv != 0) & (pv != prev)
        tid = np.cumsum(entry)
        owner = np.where(prev != 0, np.concatenate([[0], tid[:-1]]), np.where(pv != 0, tid, 0))
        pnl = pd.Series(res.pnl.to_numpy()).groupby(owner).sum().drop(0, errors="ignore")
        t_entry = res.pos.index[entry]
        side = pv[entry]
        for i, (t, sd) in enumerate(zip(t_entry, side), start=1):
            rows.append({"symbol": s, "entry": t, "side": int(sd), "pnl_bps": float(pnl.get(i, 0.0)) * 1e4})
    return pd.DataFrame(rows)


def trade_report(data: Data2, out: Path) -> None:
    for name, (fam, _, p) in CANDIDATES.items():
        t = trades_table(data, fam, p)
        t.to_csv(out / f"deep_trades_{name}.csv", index=False)
        t["year"] = t["entry"].dt.year
        t["entry"] = t["entry"].dt.strftime("%Y-%m-%d %H:%M")
        g = t.groupby("year")["pnl_bps"]
        print(f"\n{name}: сделки по годам (б.п. на сделку, после 6 б.п./сторону)")
        print(pd.DataFrame({"n": g.size(), "mean": g.mean(), "median": g.median(), "win": g.apply(lambda x: (x > 0).mean()),
                            "long_n": t[t.side > 0].groupby("year").size(),
                            "short_n": t[t.side < 0].groupby("year").size()}).round(1).to_string())
        print(f"  худшие 5: {t.nsmallest(5, 'pnl_bps')[['symbol', 'entry', 'side', 'pnl_bps']].round(0).to_dict('records')}")
        print(f"  лучшие 5: {t.nlargest(5, 'pnl_bps')[['symbol', 'entry', 'side', 'pnl_bps']].round(0).to_dict('records')}")
        top = t["pnl_bps"].sort_values(ascending=False)
        share = top.head(max(1, len(top) // 20)).sum() / top.sum() if top.sum() > 0 else np.nan
        print(f"  доля прибыли от лучших 5% сделок: {share:.0%}")


def delay_and_cost(data: Data2) -> None:
    print("\nЧувствительность: задержка входа и издержки (OOS 2024-07..2026-09)")
    for name, (fam, tf, p) in CANDIDATES.items():
        parts = []
        for delay in (0, 1):
            for cost in (COST_BPS, STRESS_COST_BPS, 15.0):
                _, s, _ = evaluate(data, fam, tf, p, "oos1", cost, delay=delay)
                parts.append(f"delay={delay} cost={cost:g}: {s['sharpe']:.2f}")
        print(f"  {name}: " + "; ".join(parts))


def portfolio(data: Data2, out: Path) -> None:
    legs, gross = {}, {}
    for name, (fam, tf, p) in CANDIDATES.items():
        fn = S.funding_extreme if fam == "funding_extreme" else oi_price
        legs[name], _, _, gross[name] = eval_single(data, tf, fn, p, "full", COST_BPS)
    lev = {n: min(TARGET_VOL / metrics(_cut(legs[n], "is"))["ann_vol"], MAX_GROSS / _cut(gross[n], "is").max())
           for n in legs}
    comb = sum(legs[n] * lev[n] for n in legs)
    exp = sum(gross[n] * lev[n] for n in legs)
    print(f"\nПОРТФЕЛЬ: плечо ног по IS-волатильности (цель {TARGET_VOL:.0%} на ногу, пик экспозиции <= {MAX_GROSS:g}): "
          + ", ".join(f"{n} x{v:.1f}" for n, v in lev.items()))
    print(f"  нужная валовая экспозиция: средняя {exp.mean():.2f}, 99-й перцентиль {exp.quantile(0.99):.2f}, "
          f"максимум {exp.max():.2f} (доли капитала)")
    rows = []
    for per in ("is", "val", "ho", "oos1"):
        for n, x in list(legs.items()) + [("portfolio", None)]:
            series = _cut(comb if x is None else x * lev[n], per)
            m = metrics(series)
            d = series.resample("1D").sum()
            rows.append({"period": per, "leg": n, "sharpe": m["sharpe"], "ann_ret": m["ann_ret"], "ann_vol": m["ann_vol"],
                         "max_dd": m["max_dd"], "worst_day": d.min(), "pos_months": float((series.resample("ME").sum() > 0).mean())})
    t = pd.DataFrame(rows)
    print(t.round(3).to_string(index=False))
    y = comb.groupby(comb.index.year).sum()
    print("  портфель по годам: " + ", ".join(f"{k}: {v:+.1%}" for k, v in y.items()))
    print(f"  корреляция ног (дневная, весь период): {legs['oi_liq'].resample('1D').sum().corr(legs['fe_q99'].resample('1D').sum()):.2f}")
    comb.resample("1D").sum().to_frame("pnl").to_csv(out / "deep_portfolio_daily.csv")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 250)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--symbols", default=",".join(UNIVERSE))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Data2(Path(a.root), a.symbols.split(","))
    delay_and_cost(data)
    trade_report(data, out)
    portfolio(data, out)
    neighbors(data, out)
