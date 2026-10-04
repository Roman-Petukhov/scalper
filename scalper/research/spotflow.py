"""
Спот против фьючерсов (Binance, часовые бары): кто двигает цену — плечевые спекулянты или реальные покупатели.

Гипотезы (зафиксированы до запуска):
    perp_led_fade   цена резко ушла, а доля спота в обороте аномально мала -> движение «на плечах», против него
    spot_led_follow цена резко ушла, спот участвует сильнее обычного и агрессоры спота в ту же сторону -> по движению
    flow_div        агрессоры фьючерсов продают (z < -thr), агрессоры спота покупают (z > thr) -> лонг; зеркально шорт
Протокол: перебор на 16 исходных монетах (IS 2022-01..2024-06) -> 2 лучших на семейство -> VAL на тех же монетах
(Sharpe >= 0.5 при 6 б.п. и > 0 при 10 б.п.) -> экзамен: 54 новые монеты за весь период и 16 монет на HOLDOUT.

    python -m research.spotflow --root <data> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data import UNIVERSE
from .engine import _to_utc, metrics, run, state_machine
from .search import Data, grid
from .universe import EXTRA

PERIODS = {"is": ("2022-01-01", "2024-07-01"), "val": ("2024-07-01", "2025-07-01"),
           "ho": ("2025-07-01", "2026-10-01"), "full": ("2022-01-01", "2026-10-01")}
COST, STRESS = 6.0, 10.0
VAL_MIN, EXAM_MIN = 0.5, 0.5
MIN_TRADES = 150
W = 24 * 30


def _z(x: pd.Series, w: int = W) -> pd.Series:
    return (x - x.rolling(w, min_periods=w // 3).mean()) / x.rolling(w, min_periods=w // 3).std()


def load_pair(root: Path, data: Data, sym: str) -> pd.DataFrame | None:
    p = root / f"{sym}-spot-1h.parquet"
    if not p.exists():
        return None
    perp = data.get(sym, "1h", "full")
    sp = pd.read_parquet(p, columns=["open_time", "quote_volume", "volume", "taker_buy_volume"])
    sp.index = _to_utc(sp["open_time"])
    sp = sp[~sp.index.duplicated()].sort_index().reindex(perp.index)
    df = perp.copy()
    df["s_qv"], df["s_v"], df["s_tb"] = sp["quote_volume"], sp["volume"], sp["taker_buy_volume"]
    return df


def features(df: pd.DataFrame, k: int) -> dict:
    c = df["close"]
    lr = np.log(c).diff()
    sig = lr.rolling(W, min_periods=W // 3).std()
    ret_z = np.log(c / c.shift(k)) / (sig * np.sqrt(k))
    pq, sq = df["quote_volume"].rolling(k).sum(), df["s_qv"].rolling(k).sum()
    share_z = _z(np.log(sq / pq))
    tfi_p = (2 * df["taker_buy_volume"].rolling(k).sum() - df["volume"].rolling(k).sum()) / df["volume"].rolling(k).sum()
    tfi_s = (2 * df["s_tb"].rolling(k).sum() - df["s_v"].rolling(k).sum()) / df["s_v"].rolling(k).sum()
    return {"ret_z": ret_z, "share_z": share_z, "tfi_p_z": _z(tfi_p), "tfi_s_z": _z(tfi_s), "tfi_s": tfi_s}


def positions(df: pd.DataFrame, fam: str, p: dict) -> np.ndarray:
    f = features(df, p["k"])
    z0 = np.zeros(len(df), np.bool_)
    if fam == "perp_led_fade":
        gate = (f["ret_z"].abs() > p["thr"]) & (f["share_z"] < -p["s"])
        el, es = gate & (f["ret_z"] < 0), gate & (f["ret_z"] > 0)
    elif fam == "spot_led_follow":
        gate = (f["ret_z"].abs() > p["thr"]) & (f["share_z"] > p["s"])
        el = gate & (f["ret_z"] > 0) & (f["tfi_s"] > 0)
        es = gate & (f["ret_z"] < 0) & (f["tfi_s"] < 0)
    else:
        el = (f["tfi_p_z"] < -p["thr"]) & (f["tfi_s_z"] > p["thr"])
        es = (f["tfi_p_z"] > p["thr"]) & (f["tfi_s_z"] < -p["thr"])
    b = lambda x: np.array(x.fillna(False), dtype=np.bool_)
    return state_machine(b(el), b(es), z0, z0, p["H"])


FAMILIES = {
    "perp_led_fade": grid(k=[1, 4, 12], thr=[2.0, 3.0], s=[0.5, 1.0], H=[4, 12, 24]),
    "spot_led_follow": grid(k=[1, 4, 12], thr=[2.0, 3.0], s=[0.5, 1.0], H=[4, 12, 24]),
    "flow_div": grid(k=[1, 4, 12], thr=[1.5, 2.0, 2.5], H=[4, 12, 24]),
}


def cut(s: pd.Series, per: str) -> pd.Series:
    a, b = PERIODS[per]
    return s[(s.index >= a) & (s.index < b)]


def evaluate(frames: dict, fam: str, p: dict, per: str, cost: float) -> dict:
    pnls, trades = [], 0
    for sym, df in frames.items():
        pos = positions(df, fam, p)
        r = run(df, pos, cost, 60, None)
        pn, ps = cut(r.pnl, per), cut(r.pos, per)
        pnls.append(pn)
        prev = ps.shift(1).fillna(0.0)
        trades += int(((ps != 0) & (ps != prev)).sum())
    port = pd.concat(pnls, axis=1).fillna(0.0).mean(axis=1)
    m = metrics(port)
    return {"sharpe": m["sharpe"], "ann_ret": m["ann_ret"], "max_dd": m["max_dd"], "trades": trades,
            "bps_per_trade": float(port.sum() * len(frames) / max(trades, 1) * 1e4)}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 250)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    syms = [s for s in UNIVERSE + EXTRA if (root / f"{s}-1h.parquet").exists() or (root / f"{s}-5m.parquet").exists()]
    data = Data(root, syms)
    core = {s: f for s in UNIVERSE if (f := load_pair(root, data, s)) is not None}
    new = {s: f for s in EXTRA if (f := load_pair(root, data, s)) is not None}
    core_is = {s: cut(f, "is") for s, f in core.items()}
    print(f"монет со спотом: исходных {len(core)}, новых {len(new)}")
    rows = []
    for fam, g in FAMILIES.items():
        for p in g:
            rows.append({"family": fam, "params": json.dumps(p), **evaluate(core_is, fam, p, "is", COST)})
    res = pd.DataFrame(rows)
    res.to_csv(out / "spotflow_is.csv", index=False)
    print(f"\nконфигураций: {len(res)}")
    print(res.groupby("family")["sharpe"].describe()[["count", "50%", "max"]].round(2).to_string())
    fin = res[res["trades"] >= MIN_TRADES].sort_values("sharpe", ascending=False).groupby("family").head(2)
    print("\nФИНАЛИСТЫ (IS, 16 монет):")
    print(fin.round(3).to_string(index=False))
    gates = []
    for _, f in fin.iterrows():
        p = json.loads(f["params"])
        r = {"family": f["family"], "params": f["params"], "is": f["sharpe"]}
        r["val_6"] = evaluate(core, f["family"], p, "val", COST)["sharpe"]
        r["val_10"] = evaluate(core, f["family"], p, "val", STRESS)["sharpe"]
        r["passed_val"] = bool(r["val_6"] >= VAL_MIN and r["val_10"] > 0)
        if r["passed_val"]:
            for cost in (COST, STRESS):
                e = evaluate(new, f["family"], p, "full", cost)
                r[f"new54_{int(cost)}"], r[f"new54_bps_{int(cost)}"] = e["sharpe"], e["bps_per_trade"]
                r[f"core_ho_{int(cost)}"] = evaluate(core, f["family"], p, "ho", cost)["sharpe"]
            r["passed_exam"] = bool(r["new54_10"] >= EXAM_MIN and r["core_ho_10"] > 0)
        gates.append(r)
    g = pd.DataFrame(gates)
    g.to_csv(out / "spotflow_gates.csv", index=False)
    print("\nВОРОТА VAL -> ЭКЗАМЕН (54 новые монеты весь период; 16 монет HOLDOUT):")
    print(g.round(3).to_string(index=False))
