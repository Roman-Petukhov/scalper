"""
Волна 5: стратегии для спокойных периодов (дополнение к oi_liq/fe_q99, которые зарабатывают в обвалы).

Протокол (зафиксирован до запуска; holdout 2025-07..2026-09 уже многократно использован, поэтому главный экзамен —
монеты, на которых новые гипотезы никогда не подбирались):
    1. перебор: 16 исходных монет, IS 2022-01..2024-06, издержки 6 б.п.; финалисты — 2 лучших на семейство
    2. ворота VAL: те же 16 монет, 2024-07..2025-06 — Sharpe >= 0.5 при 6 б.п. и > 0 при 10 б.п.
    3. экзамен: 54 новые монеты за весь период 2022-01..2026-09 (Sharpe >= 0.5 при 10 б.п.)
       и 16 исходных монет на HOLDOUT 2025-07..2026-09 (Sharpe > 0 при 10 б.п.)
    4. справочно: корреляция с текущим портфелем и результат в месяцы без обвалов

    python -m research.wave5 --root <data> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import strategies as S
from .data import UNIVERSE
from .engine import metrics, state_machine
from .search import COST_BPS, STRESS_COST_BPS, eval_panel, grid
from .universe import CANDIDATES, EXTRA, available
from .wave2 import Data2, _cut, _z, _zeros, eval_single

MIN_TRADES_IS = 200
VAL_MIN = 0.5
EXAM_NEW_MIN = 0.5


# ---------- гипотезы на один символ ----------

def smart_div(df, W=24 * 30, thr=2.0, H=12, sign=1):
    """Топ-трейдеры (по объёму позиций) смещены в лонг сильнее, чем толпа (по числу аккаунтов) -> за умными."""
    top = np.log(df["top_lsr"].where(df["top_lsr"] > 0))
    crowd = np.log(df["lsr"].where(df["lsr"] > 0))
    d = (_z(top, W) - _z(crowd, W)) * sign
    return state_machine(S._b(d > thr), S._b(d < -thr), _zeros(len(df)), _zeros(len(df)), H), None


def oi_build(df, k=24, thr=2.0, H=24, sign=1, win=24 * 30):
    """OI резко растёт при сжатой волатильности: толпа набирает позиции в тишине.
    Сторона толпы — по знаку funding; sign=+1 — против толпы (ждём сквиза)."""
    oi = df["oi"].replace(0, np.nan)
    doi = _z(np.log(oi / oi.shift(k)), win)
    r = np.log(df["close"]).diff()
    rv = r.rolling(k).std()
    calm = _z(np.log(rv), win) < -0.5
    f = df["funding"].replace(0, np.nan).ffill()
    fz = _z(f, win)
    gate = (doi > thr) & calm
    crowd_long = fz > 0.5
    crowd_short = fz < -0.5
    el = gate & (crowd_short if sign > 0 else crowd_long)
    es = gate & (crowd_long if sign > 0 else crowd_short)
    return state_machine(S._b(el), S._b(es), _zeros(len(df)), _zeros(len(df)), H), None


SINGLE = {
    "smart_div": (smart_div, grid(W=[24 * 7, 24 * 30], thr=[1.5, 2.0, 2.5], H=[4, 12, 24], sign=[1, -1])),
    "oi_build": (oi_build, grid(k=[12, 24, 48], thr=[1.5, 2.0], H=[8, 24], sign=[1, -1])),
}


# ---------- панельные гипотезы ----------

_RESID_CACHE: dict = {}


def _residuals(close: pd.DataFrame, hedge: str):
    """Остатки доходностей альтов после вычета беты к BTC (бета по прошлой неделе, со сдвигом). Кешируется."""
    key = (close.index[0], close.index[-1], tuple(close.columns), hedge)
    if key not in _RESID_CACHE:
        r = np.log(close).diff()
        btc = r[hedge]
        var = btc.rolling(168, min_periods=72).var()
        out = {}
        for s in close.columns:
            if s == hedge:
                continue
            beta = (r[s].rolling(168, min_periods=72).cov(btc) / var).shift(1).clip(-3, 3)
            resid = r[s] - beta * btc
            out[s] = (beta, resid, resid.rolling(168, min_periods=72).std())
        _RESID_CACHE.clear()
        _RESID_CACHE[key] = out
    return _RESID_CACHE[key]


def idio_rev_weights(close: pd.DataFrame, k=12, thr=2.5, H=12, hedge="BTCUSDT") -> pd.DataFrame:
    """Монета ушла от своей «беты к BTC» на thr сигм за k часов -> ставка на возврат, хедж бетой в BTC."""
    res = _residuals(close, hedge)
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    hedge_w = pd.Series(0.0, index=close.index)
    n = max(len(res), 1)
    for s, (beta, resid, sd) in res.items():
        z = resid.rolling(k).sum() / (sd * np.sqrt(k))
        pos = state_machine(S._b(z < -thr), S._b(z > thr), S._b(z > 0), S._b(z < 0), H)
        w[s] = pos / n
        hedge_w -= pd.Series(pos / n, index=close.index) * beta.fillna(0.0)
    w[hedge] = hedge_w
    return w


def crowd_xs_weights(close: pd.DataFrame, funding: pd.DataFrame, oi: pd.DataFrame, k=5, H=8, L=24, sign=1):
    """Каждые H часов: «перегретость» = ранг роста за L + ранг funding (z к своей истории) + ранг роста OI за L.
    sign=+1: шорт k самых перегретых, лонг k самых «холодных» (рыночно-нейтрально, 0.5/0.5 капитала)."""
    ret = np.log(close / close.shift(L))
    f = funding.replace(0, np.nan).ffill()
    fz = (f - f.rolling(24 * 30, min_periods=24 * 10).mean()) / f.rolling(24 * 30, min_periods=24 * 10).std()
    doi = np.log(oi.replace(0, np.nan) / oi.replace(0, np.nan).shift(L))
    score = ret.rank(axis=1) + fz.rank(axis=1) + doi.rank(axis=1)
    valid = ret.notna() & fz.notna() & doi.notna()
    score = score.where(valid)
    rank = score.rank(axis=1, ascending=False)
    n = valid.sum(axis=1)
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    w[rank <= k] = -0.5 / k * sign
    w[rank.gt(n - k, axis=0) & valid] = 0.5 / k * sign
    w[n < 2 * k + 2] = 0.0
    reb = (close.index.hour % H == 0) if H < 24 else (close.index.hour == 0)
    return w.where(pd.Series(reb, index=w.index), np.nan).ffill().fillna(0.0)


PANEL = {
    "idio_rev": grid(k=[4, 12, 24], thr=[2.0, 2.5, 3.0], H=[4, 12, 24]),
    "crowd_xs": grid(k=[3, 5], H=[4, 8, 24], L=[24, 72], sign=[1, -1]),
}


# ---------- оценка ----------

class Universe:
    def __init__(self, root: Path, symbols: list[str], hedge: str = "BTCUSDT"):
        self.data = Data2(root, symbols)
        self.hedge_data = Data2(root, [hedge])
        self.symbols, self.hedge = symbols, hedge

    def panel(self, col: str, period: str, with_hedge: bool = False) -> pd.DataFrame:
        full = "is" if period == "is" else "full"
        d = {s: self.data.get(s, "1h", full)[col] for s in self.symbols}
        if with_hedge and self.hedge not in d:
            d[self.hedge] = self.hedge_data.get(self.hedge, "1h", full)[col]
        return pd.DataFrame(d)


def evaluate(u: Universe, family: str, p: dict, period: str, cost: float):
    if family in SINGLE:
        port, per_sym, trades, _ = eval_single(u.data, "1h", SINGLE[family][0], p, period, cost)
        return port, trades
    hedge = family == "idio_rev"
    close = u.panel("close", period, hedge)
    fund = u.panel("funding", period, hedge)
    if family == "idio_rev":
        w = idio_rev_weights(close, **p, hedge=u.hedge)
    else:
        w = crowd_xs_weights(close, fund, u.panel("oi", period), **p)
    pnl, reb = eval_panel(w, close, fund, cost)
    return _cut(pnl, period), int((_cut(w.diff().abs().sum(axis=1), period) > 1e-9).sum())


def configs():
    for fam, (_, g) in SINGLE.items():
        for p in g:
            yield fam, p
    for fam, g in PANEL.items():
        for p in g:
            yield fam, p


def summarize(port: pd.Series, trades: int) -> dict:
    m = metrics(port)
    return {"sharpe": m["sharpe"], "ann_ret": m["ann_ret"], "max_dd": m["max_dd"], "trades": trades,
            "trades_per_day": trades / max(m["days"], 1)}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 250)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    core = Universe(root, available(root, UNIVERSE))
    new = Universe(root, available(root, EXTRA))
    print(f"исходных монет: {len(core.symbols)}, новых: {len(new.symbols)}")

    rows, t0 = [], time.time()
    for fam, p in configs():
        port, tr = evaluate(core, fam, p, "is", COST_BPS)
        rows.append({"family": fam, "params": json.dumps(p), **summarize(port, tr)})
    res = pd.DataFrame(rows)
    res.to_csv(out / "wave5_is.csv", index=False)
    print(f"\nВолна 5: {len(res)} конфигураций ({time.time() - t0:.0f}s)")
    print(res.groupby("family")["sharpe"].describe()[["count", "50%", "max"]].round(2).to_string())
    fin = res[res["trades"] >= MIN_TRADES_IS].sort_values("sharpe", ascending=False).groupby("family").head(2)
    print("\nФИНАЛИСТЫ (IS, 16 монет):")
    print(fin.round(3).to_string(index=False))

    # текущий портфель для корреляции и «месяцев без обвалов»
    base = []
    for u in (core, new):
        for fn, p in CANDIDATES.values():
            base.append(eval_single(u.data, "1h", fn, p, "full", COST_BPS)[0])
    base_m = (sum(base) / len(base)).resample("ME").sum()
    crash = base_m > base_m.quantile(0.85)          # месяцы, где текущий портфель заработал больше всего

    gate_rows = []
    for _, f in fin.iterrows():
        p = json.loads(f["params"])
        r = {"family": f["family"], "params": f["params"], "is": f["sharpe"]}
        for cost in (COST_BPS, STRESS_COST_BPS):
            r[f"val_{int(cost)}"] = metrics(evaluate(core, f["family"], p, "val", cost)[0])["sharpe"]
        r["passed_val"] = bool(r["val_6"] >= VAL_MIN and r["val_10"] > 0)
        if r["passed_val"]:
            for cost in (COST_BPS, STRESS_COST_BPS):
                pn, trn = evaluate(new, f["family"], p, "full", cost)
                r[f"new54_{int(cost)}"] = metrics(pn)["sharpe"]
                r[f"core_ho_{int(cost)}"] = metrics(evaluate(core, f["family"], p, "ho", cost)[0])["sharpe"]
                if cost == COST_BPS:
                    r["new54_ret"] = metrics(pn)["ann_ret"]
                    r["new54_trades_day"] = trn / max(metrics(pn)["days"], 1)
                    pm = pn.resample("ME").sum().reindex(base_m.index).fillna(0.0)
                    r["corr_with_base"] = float(pm.corr(base_m))
                    calm = pm[~crash]
                    r["calm_months_sharpe"] = float(calm.mean() / calm.std() * np.sqrt(12)) if calm.std() > 0 else 0.0
            r["passed_exam"] = bool(r["new54_10"] >= EXAM_NEW_MIN and r["core_ho_10"] > 0)
        gate_rows.append(r)
    g = pd.DataFrame(gate_rows)
    g.to_csv(out / "wave5_gates.csv", index=False)
    print("\nВОРОТА: VAL (16 монет) -> ЭКЗАМЕН (54 новые монеты, весь период; 16 монет HOLDOUT)")
    print(g.round(3).to_string(index=False))
