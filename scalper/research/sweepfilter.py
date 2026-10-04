"""
Вынос уровня как фильтр для oi_liq: сильнее ли откуп ликвидаций, если каскад снёс ближайшую вершину/впадину
(там стоят стопы и ликвидации) и цена вернулась за неё.

Гипотеза (зафиксирована до запуска): для лонга после каскада вниз L — поддержка (последняя подтверждённая впадина
по методу LuxAlgo, n=10 на 1h), известная за k=12 часов до входа (до начала каскада). Группы:
    swept_back  min(low) за 12 ч ниже L, а close на входе снова выше L — вынос стопов завершён
    swept       ниже L пробили и не вернулись
    no_sweep    до L каскад не дошёл
Для шорта — зеркально по сопротивлению. Ожидание: swept_back лучше остальных. n=5 и n=20 — для устойчивости.
Сделки oi_liq как в research.spotfilter: выход через 4 ч, за вычетом 12 б.п. круга.

Экзамен (правило зафиксировано по 16 монетам Bybit после просмотра: swept_back почти не встречается, а разница
«каскад снёс уровень» против «не дошёл» устойчива): брать oi_liq только если sweep = swept или swept_back (n=10).
Проверка — монеты Binance, которых не было в подборе (ext54, fresh), только пока оборот за 30 дней >= $20 млн.

    python -m research.sweepfilter --root <data> [--symbols ...] [--exam]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30, group_of
from .data import UNIVERSE
from .levels import sr_levels
from .universe import EXTRA
from .wave2 import Data2, oi_price

P = {"k": 12, "thr": 2.5, "hold": 4, "mode": "liq", "sign": 1}
LOOKBACK = 12
PIVOTS = (10, 5, 20)
PER = {"is": ("2022-01-01", "2024-07-01"), "val": ("2024-07-01", "2025-07-01"), "ho": ("2025-07-01", "2026-10-01")}


def trades(root: Path, syms: list[str], adv: bool = False) -> pd.DataFrame:
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
        pos, _ = oi_price(h, **P)
        pos = np.asarray(pos, float)
        if adv:
            pos = np.where((adv30(root, s).reindex(h.index, method="ffill") >= ADV_MIN).to_numpy(), pos, 0.0)
        prev = np.concatenate([[0.0], pos[:-1]])
        hi, lo, c = h["high"].to_numpy(), h["low"].to_numpy(), h["close"].to_numpy()
        levels = {n: sr_levels(hi, lo, n) for n in PIVOTS}
        for e in np.flatnonzero((pos != 0) & (prev == 0)):
            if e + 4 >= len(c) or e < LOOKBACK:
                continue
            dirn = pos[e]
            row = {"symbol": s, "group": group_of(s), "t": h.index[e], "r": dirn * (c[e + 4] / c[e] - 1) - 12e-4}
            for n, (res, sup) in levels.items():
                lvl = sup[e - LOOKBACK] if dirn > 0 else res[e - LOOKBACK]
                if np.isnan(lvl):
                    row[f"g{n}"] = "no_level"
                    continue
                w = slice(e - LOOKBACK + 1, e + 1)
                swept = lo[w].min() < lvl if dirn > 0 else hi[w].max() > lvl
                back = c[e] > lvl if dirn > 0 else c[e] < lvl
                row[f"g{n}"] = "swept_back" if swept and back else "swept" if swept else "no_sweep"
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--symbols", default=",".join(UNIVERSE + EXTRA))
    ap.add_argument("--exam", action="store_true", help="фильтр ликвидности $20 млн и разбивка по группам монет")
    a = ap.parse_args()
    root = Path(a.root)
    syms = [s for s in a.symbols.split(",") if any((root / f"{s}-{tf}.parquet").exists() for tf in ("5m", "1h"))]
    r = trades(root, syms, adv=a.exam)
    print(f"===== SWEEP FILTER: oi_liq + вынос уровня ({len(syms)} монет, сделок {len(r)}) =====")
    for n in PIVOTS:
        print(f"\n[уровни n={n}]" + ("  <- основная гипотеза" if n == PIVOTS[0] else ""))
        rows = []
        for per, (a_, b_) in PER.items():
            x = r[(r.t >= a_) & (r.t < b_)]
            row = {"period": per, "all_n": len(x), "all_bps": x.r.mean() * 1e4}
            for g in ("swept_back", "swept", "no_sweep", "no_level"):
                y = x[x[f"g{n}"] == g]
                row[f"{g}_n"] = len(y)
                row[f"{g}_bps"] = y.r.mean() * 1e4 if len(y) else np.nan
            rows.append(row)
        print(pd.DataFrame(rows).round(1).to_string(index=False))
    if a.exam and len(r):
        r["sweep"] = np.where(r["g10"].isin(["swept", "swept_back"]), "sweep", np.where(r["g10"] == "no_sweep",
                                                                                        "no_sweep", "no_level"))
        print("\nЭКЗАМЕН (n=10): oi_liq только после выноса уровня против остальных, б.п. на сделку за вычетом 12 б.п.")
        rows = []
        for grp in ("core16", "ext54", "fresh", "new (ext54+fresh)"):
            g = r[r["group"].isin(["ext54", "fresh"])] if grp.startswith("new") else r[r["group"] == grp]
            for per, (a_, b_) in PER.items():
                x = g[(g.t >= a_) & (g.t < b_)]
                row = {"group": grp, "period": per, "all_n": len(x), "all_bps": x.r.mean() * 1e4}
                for k in ("sweep", "no_sweep"):
                    y = x[x["sweep"] == k]
                    row[f"{k}_n"], row[f"{k}_bps"] = len(y), y.r.mean() * 1e4 if len(y) else np.nan
                    row[f"{k}_win"] = float((y.r > 0).mean()) if len(y) else np.nan
                rows.append(row)
        print(pd.DataFrame(rows).round(2).to_string(index=False))
