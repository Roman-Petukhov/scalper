"""
Экзамен «ложного пробоя» уровней LuxAlgo на всех USDT-перпетуалах Binance (часовые свечи), параметры не меняются.

Конфигурация зафиксирована по 16 монетам Bybit (IS -> VAL -> HOLDOUT пройдены): вершины/впадины 10/10, пробой при
osc объёма > 20, возврат close за уровень в течение 3 часов -> позиция против пробоя, выход по противоположному
сигналу, без фильтра. Соседние настройки печатаются для устойчивости, но вердикт — только по зафиксированной.
Группы монет: core16 (подбор), ext54 (прошлые проверки), fresh (не участвовали нигде). Монета торгуется, только
пока её средний оборот за 30 прошлых дней >= $20 млн (как в research.broad). Портфель — 2% капитала на позицию.

    python -m research.levels_exam --root <binance 1h data>
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, POS_FRACTION, adv30, group_of
from .engine import load, metrics, run
from .levels import sr_levels, sr_signals, volume_osc, MODES
from .lux import PERIODS, positions

FIXED = {"n": 10, "thr": 20.0, "mode": "fade3"}
NEIGHBOURS = [{"n": n, "thr": thr, "mode": m} for n in (5, 10, 15, 20) for thr in (20.0, 50.0)
              for m in ("fade3", "fade6")]
COSTS = (6.0, 10.0, 15.0)


def symbol_pos(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    h, lo = df["high"].to_numpy(), df["low"].to_numpy()
    res, sup = sr_levels(h, lo, cfg["n"])
    osc = np.nan_to_num(volume_osc(df["volume"]), nan=-1e9)
    up, dn = sr_signals(df["open"].to_numpy(), h, lo, df["close"].to_numpy(), osc, res, sup, cfg["thr"],
                        MODES.index(cfg["mode"]))
    return positions(up, dn, np.zeros(len(df)), False, 0)


def cut(s: pd.Series, per: str) -> pd.Series:
    a, b = PERIODS[per]
    return s[(s.index >= a) & (s.index < b)]


def run_all(root: Path, syms: list[str], cfgs: list[dict]) -> dict:
    """{(cfg_idx, cost): {group: [pnl series]}}, плюс статистика сделок для зафиксированной конфигурации."""
    out: dict = {}
    stats = []
    for i, s in enumerate(syms, 1):
        try:
            df = load(s, root, "1h")
            allowed = (adv30(root, s).reindex(df.index, method="ffill") >= ADV_MIN).to_numpy()
        except Exception as e:                              # битые ряды не валят прогон
            print(f"  {s}: пропуск ({e})", flush=True)
            continue
        g = group_of(s)
        for j, cfg in enumerate(cfgs):
            pos = np.where(allowed, symbol_pos(df, cfg), 0.0)
            if not pos.any():
                continue
            for cost in COSTS:
                r = run(df, pos, cost, 60, None)
                daily = r.pnl.resample("1D").sum().rename(s)       # метрики всё равно дневные; так влезает в память
                out.setdefault((j, cost), {}).setdefault(g, []).append(daily)
                if j == 0 and cost == COSTS[0]:
                    ps = r.pos
                    entries = (ps != 0) & (ps != ps.shift(1).fillna(0.0))
                    stats.append({"symbol": s, "group": g, "trades": int(entries.sum()),
                                  "hours_in_pos": int((ps != 0).sum()), "total_ret": float(r.pnl.sum()),
                                  "first_entry": ps.index[entries.to_numpy()][0] if entries.any() else pd.NaT})
        if i % 100 == 0:
            print(f"  {i}/{len(syms)} монет", flush=True)
    return {"pnl": out, "stats": pd.DataFrame(stats)}


def table(pnl: dict, j: int) -> pd.DataFrame:
    rows = []
    for cost in COSTS:
        groups = pnl.get((j, cost), {})
        for g in ("core16", "ext54", "fresh", "all"):
            parts = sum(groups.values(), []) if g == "all" else groups.get(g, [])
            if not parts:
                continue
            port = pd.concat(parts, axis=1).fillna(0.0).sum(axis=1) * POS_FRACTION
            for per in ("is", "val", "ho"):
                m = metrics(cut(port, per))
                rows.append({"cost": cost, "group": g, "coins": len(parts), "period": per, "sharpe": m["sharpe"],
                             "ann_ret": m["ann_ret"], "max_dd": m["max_dd"]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 250)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.root)
    syms = [s for s in json.load(open(Path(__file__).with_name("symbols_qualified.json")))
            if (root / f"{s}-1h.parquet").exists()]
    cfgs = [FIXED] + [c for c in NEIGHBOURS if c != FIXED]
    print(f"===== ЭКЗАМЕН ложного пробоя уровней (n=10, объём > 20%, возврат за 3 ч): {len(syms)} монет =====")
    res = run_all(root, syms, cfgs)
    st = res["stats"]
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        st.to_csv(Path(a.out) / "levels_exam_per_symbol.csv", index=False)
    act = st[st["trades"] > 0]
    print("\nМонеты со сделками и доля в плюсе:")
    print(act.groupby("group").agg(coins=("symbol", "size"), positive=("total_ret", lambda x: float((x > 0).mean())),
                                   trades=("trades", "sum"),
                                   avg_hold_h=("hours_in_pos", "sum")).assign(
        avg_hold_h=lambda d: d["avg_hold_h"] / d["trades"]).round(2).to_string())
    t = table(res["pnl"], 0)
    print("\nЗАФИКСИРОВАННАЯ конфигурация — Sharpe по группам и периодам:")
    print(t.pivot_table(index=["group", "cost"], columns="period", values="sharpe").round(2).to_string())
    print("\nДоходность в год / просадка (6 б.п., 2% на позицию):")
    print(t[t["cost"] == 6.0].pivot_table(index="group", columns="period", values=["ann_ret", "max_dd"]).round(3)
          .to_string())
    print("\nСоседние настройки (fresh, 10 б.п.) — устойчивость, не вердикт:")
    rows = []
    for j, cfg in enumerate(cfgs):
        tj = table(res["pnl"], j)
        x = tj[(tj["group"] == "fresh") & (tj["cost"] == 10.0)].set_index("period")["sharpe"]
        rows.append({**cfg, **{p: x.get(p, np.nan) for p in ("is", "val", "ho")}})
    print(pd.DataFrame(rows).round(2).to_string(index=False))
