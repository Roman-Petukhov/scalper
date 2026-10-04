"""
Экзамен лесенки лимитов для oi_liq на USDT-перпетуалах Binance (часовые свечи достаточно: лимит исполнен, если
минимум (максимум для шорта) 4 часов после сигнала прошёл сквозь уровень; выход — по close через 4 ч, рыночно).

Основной вариант зафиксирован по IS и VAL на 16 монетах Bybit (HOLDOUT не использовался): 5 лимитов по цене сигнала
и ниже с шагом 0.5 * ATR(14, 1h), каждый 1/5 капитала. Критерий: на монетах вне подбора (ext54 + fresh)
доходность на задействованный капитал выше рыночного входа во всех трёх периодах. Остальные лесенки — для справки.
Дополнительно — разрез по фильтру выноса уровня (research.sweepfilter, n=10).
Комиссии: лимит maker 2 б.п., рыночный вход/выход taker 5.5 б.п. Монета торгуется при обороте >= $20 млн/день.

    python -m research.ladder_exam --root <binance 1h data with metrics> --symbols ...
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30, group_of
from .grid import LADDERS, MAKER, P_OI, TAKER
from .levels import sr_levels
from .sweepfilter import LOOKBACK, PER
from .wave2 import Data2, oi_price

PRIMARY = "L5x0.5"
HOLD_H = 4


def signal_rows(root: Path, syms: list[str]) -> pd.DataFrame:
    d = Data2(root, syms)
    rows = []
    for s in syms:
        try:
            h = d.get(s, "1h", "full")
        except Exception as e:
            print(f"  {s}: пропуск ({e})", flush=True)
            continue
        if h["oi"].notna().sum() < 24 * 60:
            continue
        pos, _ = oi_price(h, **P_OI)
        pos = np.asarray(pos, float)
        allowed = (adv30(root, s).reindex(h.index, method="ffill") >= ADV_MIN).to_numpy()
        pos = np.where(allowed, pos, 0.0)
        prev = np.concatenate([[0.0], pos[:-1]])
        hi, lo, c = h["high"].to_numpy(), h["low"].to_numpy(), h["close"].to_numpy()
        tr = np.maximum.reduce([hi - lo, np.abs(hi - np.roll(c, 1)), np.abs(lo - np.roll(c, 1))])
        atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
        res, sup = sr_levels(hi, lo, 10)
        for e in np.flatnonzero((pos != 0) & (prev == 0)):
            if e + HOLD_H >= len(c) or e < LOOKBACK:
                continue
            d_, p0, a = pos[e], c[e], atr[e]
            w = slice(e + 1, e + HOLD_H + 1)
            ex = c[e + HOLD_H]
            lvl_s = sup[e - LOOKBACK] if d_ > 0 else res[e - LOOKBACK]
            sw = slice(e - LOOKBACK + 1, e + 1)
            swept = (lo[sw].min() < lvl_s) if d_ > 0 else (hi[sw].max() > lvl_s)
            row = {"symbol": s, "group": group_of(s), "t": h.index[e],
                   "sweep": "no_level" if np.isnan(lvl_s) else ("sweep" if swept else "no_sweep"),
                   "market": d_ * (ex / p0 - 1) - 2 * TAKER * 1e-4, "market_fill": 1.0}
            for kk, sp in LADDERS:
                if kk == 1:
                    continue
                r, nf = 0.0, 0
                for i in range(kk):
                    lv = p0 - d_ * i * sp * a
                    if (lo[w].min() < lv) if d_ > 0 else (hi[w].max() > lv):
                        nf += 1
                        r += (d_ * (ex / lv - 1) - (MAKER + TAKER) * 1e-4) / kk
                row[f"L{kk}x{sp}"], row[f"L{kk}x{sp}_fill"] = r, nf / kk
            rows.append(row)
    return pd.DataFrame(rows)


def summary(x: pd.DataFrame, variants: list[str]) -> dict:
    out = {"signals": len(x)}
    for v in variants:
        dep = x[f"{v}_fill"].mean()
        out[f"{v}_sig"] = x[v].mean() * 1e4
        out[f"{v}_cap"] = x[v].mean() / dep * 1e4 if dep > 0 else np.nan     # на задействованный капитал
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 260)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--symbols", required=True)
    a = ap.parse_args()
    r = signal_rows(Path(a.root), a.symbols.split(","))
    variants = ["market"] + [f"L{k}x{s}" for k, s in LADDERS if k > 1]
    print(f"===== LADDER EXAM: oi_liq, рыночный вход против лесенки лимитов (сигналов {len(r)}) =====")
    print(f"Основной вариант (зафиксирован до экзамена): {PRIMARY}; sig — б.п. на сигнал, cap — на задействованный капитал")
    groups = {"core16": ["core16"], "ext54": ["ext54"], "fresh": ["fresh"], "new": ["ext54", "fresh"]}
    rows = []
    for g, members in groups.items():
        for per, (pa, pb) in PER.items():
            x = r[r["group"].isin(members) & (r.t >= pa) & (r.t < pb)]
            if len(x):
                rows.append({"group": g, "period": per, **summary(x, ["market", PRIMARY])})
    print(pd.DataFrame(rows).round(1).to_string(index=False))
    print("\nВсе лесенки, монеты вне подбора (new), б.п. на задействованный капитал:")
    rows = []
    for per, (pa, pb) in PER.items():
        x = r[r["group"].isin(groups["new"]) & (r.t >= pa) & (r.t < pb)]
        s_ = summary(x, variants)
        rows.append({"period": per, "signals": s_["signals"], **{v: s_[f"{v}_cap"] for v in variants}})
    print(pd.DataFrame(rows).round(1).to_string(index=False))
    print("\nЛесенка × вынос уровня (new, б.п. на сигнал):")
    rows = []
    for per, (pa, pb) in PER.items():
        for sw in ("sweep", "no_sweep"):
            x = r[r["group"].isin(groups["new"]) & (r.t >= pa) & (r.t < pb) & (r["sweep"] == sw)]
            if len(x):
                rows.append({"period": per, "sweep": sw, "n": len(x), "market": x["market"].mean() * 1e4,
                             PRIMARY: x[PRIMARY].mean() * 1e4, "fill": x[f"{PRIMARY}_fill"].mean()})
    print(pd.DataFrame(rows).round(1).to_string(index=False))
