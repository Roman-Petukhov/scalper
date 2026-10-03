"""
Перебор гипотез с жёстким разделением in-sample / out-of-sample.

Шаг 1 (search):  все конфиги считаются ТОЛЬКО на IS, результаты в is_results.csv.
Шаг 2 (select):  лучший конфиг каждого семейства по Sharpe портфеля на IS
                 (с минимальным числом сделок и проверкой устойчивости соседей).
Шаг 3 (oos):     финалисты один раз прогоняются на OOS; отчёт oos_results.csv.

    python -m research.search --root <data> --out <dir>
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import strategies as S
from .data import UNIVERSE
from .engine import BAR_MINUTES, load, metrics, run

IS_START, IS_END, OOS_END = "2022-01-01", "2024-07-01", "2026-10-01"
COST_BPS = 6.0          # на сторону: тейкер 5 б.п. + проскальзывание 1 б.п.
STRESS_COST_BPS = 10.0


def grid(**axes):
    keys = list(axes)
    for vals in itertools.product(*axes.values()):
        yield dict(zip(keys, vals))


# семейство -> (таймфрейм, функция, сетка). Таймфрейм тоже часть сетки.
SINGLE = {
    "donchian":       [(tf, S.donchian, grid(n=[20, 50, 100, 200], exit_frac=[0.25, 0.5, 1.0], long_only=[False, True]))
                       for tf in ("1h", "4h")],
    "tsmom":          [(tf, S.tsmom, grid(lookback=lb, band=[0.5, 1.0, 1.5]))
                       for tf, lb in (("1h", [24, 72, 168, 336]), ("4h", [6, 18, 42, 84]))],
    "meanrev_z":      [(tf, S.meanrev_z, grid(k=[1, 3, 6, 12], thr=[2.5, 3.0, 4.0], hold=[6, 12, 24],
                                              volwin=[vw]))
                       for tf, vw in (("5m", 288), ("15m", 96 * 3), ("1h", 24 * 7))],
    "vwap_rev":       [(tf, S.vwap_rev, grid(win=w, thr=[2.0, 2.5, 3.0], hold=[24, 96]))
                       for tf, w in (("15m", [48, 96, 192]), ("1h", [24, 72]))],
    "flow":           [(tf, S.flow, grid(k=[3, 12, 48], thr=[2.0, 3.0], hold=[6, 24], sign=[1, -1], zwin=[zw]))
                       for tf, zw in (("5m", 288 * 7), ("15m", 96 * 14), ("1h", 24 * 30))],
    "absorption":     [(tf, S.absorption, grid(k=[3, 6, 12], thr=[2.0, 3.0], move=[0.2, 0.5], hold=[6, 24],
                                               zwin=[zw], volwin=[vw]))
                       for tf, zw, vw in (("15m", 96 * 14, 96 * 3), ("1h", 24 * 30, 24 * 7))],
    "funding_extreme": [("1h", S.funding_extreme, grid(q=[0.9, 0.95, 0.99], hold=[8, 24, 72], win_days=[30, 90]))],
    "opening_range":  [("15m", S.opening_range, grid(range_hours=[1, 2, 4, 8], stop_mult=[0.5, 1.0], rr=[1.0, 2.0, 3.0]))],
    "vol_breakout":   [("15m", S.vol_breakout, grid(k=[0.3, 0.5, 0.7, 1.0]))],
}


class Data:
    """Кеш данных по таймфреймам, только IS или полный период."""

    def __init__(self, root: Path, symbols: list[str]):
        self.root, self.symbols = Path(root), symbols
        self.cache: dict[tuple[str, str], pd.DataFrame] = {}

    def get(self, sym: str, tf: str, period: str) -> pd.DataFrame:
        key = (sym, tf)
        if key not in self.cache:
            df = load(sym, self.root, tf)
            keep = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume", "funding"]
            self.cache[key] = df[keep].astype("float64")
        df = self.cache[key]
        return df[(df.index >= IS_START) & (df.index < IS_END)] if period == "is" else \
            df[(df.index >= IS_START) & (df.index < OOS_END)]

    def panel(self, tf: str, period: str, col: str = "close") -> pd.DataFrame:
        return pd.DataFrame({s: self.get(s, tf, period)[col] for s in self.symbols})


def _kw(fn, params, tf):
    kw = dict(params)
    if fn in (S.opening_range, S.vol_breakout):
        kw["bar_min"] = BAR_MINUTES[tf]
    return kw


def eval_single(data: Data, tf: str, fn, params: dict, period: str, cost: float, mask=None):
    pnls, poss = {}, {}
    for s in data.symbols:
        df = data.get(s, tf, period)
        if fn is S.seasonal:
            pos, ex = fn(df, mask)
        else:
            pos, ex = fn(df, **_kw(fn, params, tf))
        res = run(df, np.asarray(pos, dtype=float), cost, BAR_MINUTES[tf], ex)
        pnls[s], poss[s] = res.pnl, res.pos
    port = pd.DataFrame(pnls).fillna(0.0).mean(axis=1)
    per_sym = {s: metrics(p)["sharpe"] for s, p in pnls.items()}
    trades = sum(metrics(pnls[s], poss[s]).get("trades", 0) for s in data.symbols)
    return port, per_sym, trades


def eval_panel(weights: pd.DataFrame, closes: pd.DataFrame, funding: pd.DataFrame, cost: float) -> tuple[pd.Series, int]:
    r = closes.pct_change().fillna(0.0)
    w_prev = weights.shift(1).fillna(0.0)
    turnover = (weights - w_prev).abs().sum(axis=1)
    pnl = (w_prev * r).sum(axis=1) - turnover * cost / 1e4 - (w_prev * funding.fillna(0.0)).sum(axis=1)
    rebalances = int((turnover > 1e-9).sum())
    return pnl, rebalances


def panel_configs():
    for p in grid(lookback=[1, 4, 12, 24, 72, 168], rebalance=[1, 4, 24], k=[3], sign=[1, -1]):
        yield "xs_momentum" if p["sign"] > 0 else "xs_reversal", "1h", p
    for p in grid(k=[2, 3, 4], rebalance=[8, 24]):
        yield "funding_carry", "1h", p
    for p in grid(k=[1, 3, 6], thr=[1.5, 2.0, 3.0], hold=[3, 6, 12]):
        yield "leadlag", "5m", p


def panel_weights(data: Data, family: str, tf: str, p: dict, period: str):
    closes = data.panel(tf, period)
    if family.startswith("xs_"):
        w = S.xs_weights(closes, p["lookback"], p["rebalance"], p["k"], p["sign"])
    elif family == "funding_carry":
        f = data.panel(tf, period, "funding").replace(0, np.nan).ffill()
        w = S.funding_carry_weights(f, p["k"], p["rebalance"])
    else:
        w = S.leadlag_weights(closes, p["k"], p["thr"], p["hold"])
    return w, closes, data.panel(tf, period, "funding")


def summarize(port: pd.Series, trades: int, per_sym: dict | None = None) -> dict:
    m = metrics(port)
    out = {k: m[k] for k in ("sharpe", "ann_ret", "ann_vol", "max_dd", "pos_days")}
    out["trades"] = trades
    out["trades_per_day"] = trades / max(m["days"], 1)
    if per_sym:
        v = np.array(list(per_sym.values()))
        out["sym_pos_frac"] = float((v > 0).mean())
    return out


def search(data: Data, out: Path) -> pd.DataFrame:
    rows = []
    t0 = time.time()
    is_dfs = {s: data.get(s, "1h", "is") for s in data.symbols}
    for fam, specs in SINGLE.items():
        for tf, fn, g in specs:
            for p in g:
                port, per_sym, trades = eval_single(data, tf, fn, p, "is", COST_BPS)
                rows.append({"family": fam, "tf": tf, "params": json.dumps(p), **summarize(port, trades, per_sym)})
        print(f"  {fam}: готово ({time.time() - t0:.0f}s)", flush=True)
    for t_thr in (2.0, 3.0):
        mask, _ = S.hour_mask_from_is(is_dfs, t_thr)
        port, per_sym, trades = eval_single(data, "1h", S.seasonal, {}, "is", COST_BPS, mask)
        rows.append({"family": "seasonal", "tf": "1h", "params": json.dumps({"t_thr": t_thr}),
                     **summarize(port, trades, per_sym)})
    print(f"  seasonal: готово ({time.time() - t0:.0f}s)", flush=True)
    for fam, tf, p in panel_configs():
        if fam == "leadlag" and "BTCUSDT" not in data.symbols:
            continue
        w, closes, fund = panel_weights(data, fam, tf, p, "is")
        port, reb = eval_panel(w, closes, fund, COST_BPS)
        rows.append({"family": fam, "tf": tf, "params": json.dumps(p), **summarize(port, reb)})
    print(f"  панельные: готово ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "is_results.csv", index=False)
    return df


def select(df: pd.DataFrame, min_trades: int = 200) -> pd.DataFrame:
    """Лучший по IS Sharpe в каждом семействе + медиана Sharpe по семейству/таймфрейму как мера устойчивости."""
    df = df[df["trades"] >= min_trades].copy()
    df["family_tf_median_sharpe"] = df.groupby(["family", "tf"])["sharpe"].transform("median")
    best = df.sort_values("sharpe", ascending=False).groupby("family").head(1)
    return best.sort_values("sharpe", ascending=False)


def oos(data: Data, finalists: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    is_dfs = {s: data.get(s, "1h", "is") for s in data.symbols}
    for _, f in finalists.iterrows():
        p = json.loads(f["params"])
        for cost in (COST_BPS, STRESS_COST_BPS):
            if f["family"] in SINGLE or f["family"] == "seasonal":
                if f["family"] == "seasonal":
                    mask, _ = S.hour_mask_from_is(is_dfs, p["t_thr"])
                    port, per_sym, trades = eval_single(data, "1h", S.seasonal, {}, "full", cost, mask)
                else:
                    fn = next(fn for tf, fn, _ in SINGLE[f["family"]] if tf == f["tf"])
                    port, per_sym, trades = eval_single(data, f["tf"], fn, p, "full", cost)
            else:
                w, closes, fund = panel_weights(data, f["family"], f["tf"], p, "full")
                port, trades = eval_panel(w, closes, fund, cost)
            port_oos = port[port.index >= IS_END]
            port_oos.to_frame("pnl").to_parquet(out / f"oos_{f['family']}_{int(cost)}.parquet")
            m = metrics(port_oos)
            yearly = port.groupby(port.index.year).sum().round(4).to_dict()
            rows.append({"family": f["family"], "tf": f["tf"], "params": f["params"], "cost_bps": cost,
                         "is_sharpe": f["sharpe"], "oos_sharpe": m["sharpe"], "oos_ann_ret": m["ann_ret"],
                         "oos_max_dd": m["max_dd"], "oos_pos_days": m["pos_days"], "yearly": json.dumps(yearly)})
    res = pd.DataFrame(rows)
    res.to_csv(out / "oos_results.csv", index=False)
    return res


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--symbols", default=",".join(UNIVERSE))
    ap.add_argument("--stage", default="all", choices=["search", "oos", "all"])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Data(Path(a.root), a.symbols.split(","))
    if a.stage in ("search", "all"):
        res = search(data, out)
        print(f"\nВсего конфигураций: {len(res)}")
    else:
        res = pd.read_csv(out / "is_results.csv")
    fin = select(res)
    cols = ["family", "tf", "params", "sharpe", "ann_ret", "max_dd", "trades_per_day", "sym_pos_frac",
            "family_tf_median_sharpe"]
    print("\nФИНАЛИСТЫ (по IS):")
    print(fin[cols].to_string(index=False))
    if a.stage in ("oos", "all"):
        o = oos(data, fin, out)
        print("\nOUT-OF-SAMPLE (2024-07 .. 2026-09):")
        print(o.drop(columns=["yearly"]).to_string(index=False))
