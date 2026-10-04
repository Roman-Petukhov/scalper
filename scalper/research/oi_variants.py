"""
Частота oi_liq: более мягкие пороги каскада + фильтр выноса уровня (725 монет Binance, 1h, оборот >= $20 млн/день).

Сетка: окно каскада k = 4/6/12 ч × порог z = 1.5/2.0/2.5 (цена и OI), выход через 4 ч. Вынос уровня — как в
research.sweepfilter (последняя подтверждённая впадина/вершина n=10), но известная до начала каскада: за k часов
до входа, проверка выноса — за эти k часов. Сделка берётся только при выносе.
Выбор — ТОЛЬКО по IS (все монеты): наибольшая t-статистика среди вариантов, дающих сделок в день не меньше
двойной базовой (k=12, z=2.5). VAL и HOLDOUT печатаются для всех вариантов, вердикт — по выбранному.
Издержки 12 б.п. на круг. Плюс распределение числа сделок по дням и Sharpe дневного портфеля (2% на сделку).

    python -m research.oi_variants --root <binance 1h data with metrics> --symbols ...
"""
from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, POS_FRACTION, adv30, group_of
from .engine import metrics
from .levels import sr_levels
from .sweepfilter import PER
from .wave2 import Data2, oi_price

KS = (4, 6, 12)
THRS = (1.5, 2.0, 2.5)
HOLD = 4
BASE = (12, 2.5)
COST_RT = 12e-4


def collect(root: Path, syms: list[str]) -> pd.DataFrame:
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
        allowed = (adv30(root, s).reindex(h.index, method="ffill") >= ADV_MIN).to_numpy()
        hi, lo, c = h["high"].to_numpy(), h["low"].to_numpy(), h["close"].to_numpy()
        res, sup = sr_levels(hi, lo, 10)
        g = group_of(s)
        for k, thr in itertools.product(KS, THRS):
            pos, _ = oi_price(h, k=k, thr=thr, hold=HOLD, mode="liq", sign=1)
            pos = np.where(allowed, np.asarray(pos, float), 0.0)
            prev = np.concatenate([[0.0], pos[:-1]])
            for e in np.flatnonzero((pos != 0) & (prev == 0)):
                if e + HOLD >= len(c) or e < k:
                    continue
                dirn = pos[e]
                lvl = sup[e - k] if dirn > 0 else res[e - k]
                w = slice(e - k + 1, e + 1)
                swept = (not np.isnan(lvl)) and ((lo[w].min() < lvl) if dirn > 0 else (hi[w].max() > lvl))
                rows.append({"k": k, "thr": thr, "symbol": s, "group": g, "t": h.index[e], "sweep": bool(swept),
                             "r": dirn * (c[e + HOLD] / c[e] - 1) - COST_RT})
    return pd.DataFrame(rows)


def stats(x: pd.DataFrame, days: int) -> dict:
    if not len(x):
        return {"n": 0, "per_day": 0.0, "bps": np.nan, "t": np.nan, "win": np.nan}
    r = x["r"]
    return {"n": len(r), "per_day": len(r) / days, "bps": r.mean() * 1e4,
            "t": r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 2 else np.nan, "win": float((r > 0).mean())}


def daily_portfolio(x: pd.DataFrame, a: str, b: str) -> pd.Series:
    """Доход портфеля по дням входа: каждая сделка — 2% капитала."""
    s = x.set_index("t")["r"] * POS_FRACTION
    idx = pd.date_range(a, b, freq="1D", tz="UTC", inclusive="left")
    return s.resample("1D").sum().reindex(idx, fill_value=0.0)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 260)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--symbols", required=True)
    a = ap.parse_args()
    r = collect(Path(a.root), a.symbols.split(","))
    sw = r[r["sweep"]]
    days = {p: (pd.Timestamp(b_) - pd.Timestamp(a_)).days for p, (a_, b_) in PER.items()}
    print(f"===== OI VARIANTS: частота oi_liq с фильтром выноса уровня (сделок всего {len(r)}) =====")
    rows = []
    for (k, thr), g in sw.groupby(["k", "thr"]):
        row = {"k": k, "thr": thr}
        for p, (a_, b_) in PER.items():
            st = stats(g[(g.t >= a_) & (g.t < b_)], days[p])
            row.update({f"{p}_{key}": v for key, v in st.items() if key != "n"})
        rows.append(row)
    t = pd.DataFrame(rows)
    print("\nС фильтром выноса, все монеты (сделок в день, б.п. на сделку, t, доля прибыльных):")
    print(t.round(2).to_string(index=False))
    base = t[(t["k"] == BASE[0]) & (t["thr"] == BASE[1])].iloc[0]
    cand = t[t["is_per_day"] >= 2 * base["is_per_day"]]
    if not len(cand):
        print("\nНи один вариант не даёт вдвое больше сделок на IS.")
        sys.exit(0)
    pick = cand.sort_values("is_t", ascending=False).iloc[0]
    k_, thr_ = int(pick["k"]), float(pick["thr"])
    print(f"\nВЫБРАН ПО IS: k={k_}, z={thr_} (база k={BASE[0]}, z={BASE[1]})")
    print("Экзамен на монетах вне подбора (ext54 + fresh), с фильтром выноса:")
    rows = []
    for name, (kk, tt) in (("база", BASE), ("выбранный", (k_, thr_))):
        g = sw[(sw["k"] == kk) & (sw["thr"] == tt) & sw["group"].isin(["ext54", "fresh"])]
        for p, (a_, b_) in PER.items():
            rows.append({"variant": name, "period": p, **stats(g[(g.t >= a_) & (g.t < b_)], days[p])})
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print("\nСделок в день (все монеты, с фильтром выноса) — доля дней:")
    rows = []
    for name, (kk, tt) in (("база", BASE), ("выбранный", (k_, thr_))):
        g = sw[(sw["k"] == kk) & (sw["thr"] == tt)]
        for p, (a_, b_) in PER.items():
            cnt = g[(g.t >= a_) & (g.t < b_)].set_index("t")["r"].resample("1D").size()
            cnt = cnt.reindex(pd.date_range(a_, b_, freq="1D", tz="UTC", inclusive="left"), fill_value=0)
            port = daily_portfolio(g[(g.t >= a_) & (g.t < b_)], a_, b_)
            m = metrics(port)
            rows.append({"variant": name, "period": p, "0": float((cnt == 0).mean()),
                         "1-2": float(cnt.between(1, 2).mean()), "3-9": float(cnt.between(3, 9).mean()),
                         "10+": float((cnt >= 10).mean()), "max": int(cnt.max()), "median": float(cnt.median()),
                         "sharpe": m["sharpe"], "ann_ret": m["ann_ret"], "max_dd": m["max_dd"]})
    print(pd.DataFrame(rows).round(2).to_string(index=False))
