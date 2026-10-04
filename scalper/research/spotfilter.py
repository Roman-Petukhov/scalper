"""
Спот как фильтр для oi_liq на 70 монетах Binance (OI Binance, спот Binance).
Правило зафиксировано до запуска (по результатам на 16 монетах Bybit): брать сигнал, только если поток агрессоров
спота за те же 12 часов (z к своей истории, со знаком сделки) > 0. Сравнение — по сделкам, по периодам.

    python -m research.spotfilter --root <data>
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .data import UNIVERSE
from .engine import _to_utc
from .universe import EXTRA
from .wave2 import Data2, oi_price

P = {"k": 12, "thr": 2.5, "hold": 4, "mode": "liq", "sign": 1}
PER = {"is": ("2022-01-01", "2024-07-01"), "val": ("2024-07-01", "2025-07-01"), "ho": ("2025-07-01", "2026-10-01")}


def trades(root: Path, syms: list[str]) -> pd.DataFrame:
    d = Data2(root, syms)
    rows = []
    for s in syms:
        sp_path = root / f"{s}-spot-1h.parquet"
        if not sp_path.exists():
            continue
        h = d.get(s, "1h", "full")
        sp = pd.read_parquet(sp_path, columns=["open_time", "volume", "taker_buy_volume"])
        sp.index = _to_utc(sp["open_time"])
        sp = sp[~sp.index.duplicated()].sort_index().reindex(h.index)
        v12 = sp["volume"].rolling(12).sum()
        tfi = (2 * sp["taker_buy_volume"].rolling(12).sum() - v12) / v12
        tz = (tfi - tfi.rolling(720, min_periods=240).mean()) / tfi.rolling(720, min_periods=240).std()
        pos, _ = oi_price(h, **P)
        pos = np.asarray(pos, float)
        prev = np.concatenate([[0.0], pos[:-1]])
        c = h["close"].to_numpy()
        for e in np.flatnonzero((pos != 0) & (prev == 0)):
            if e + 4 < len(c):
                rows.append({"symbol": s, "t": h.index[e], "spot_z": tz.iloc[e] * pos[e],
                             "r": pos[e] * (c[e + 4] / c[e] - 1) - 12e-4})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    r = trades(Path(a.root), UNIVERSE + EXTRA).dropna()
    r["group"] = np.where(r["symbol"].isin(UNIVERSE), "core16", "ext54")
    print(f"сделок oi_liq с данными спота: {len(r)}")
    for grp in ("core16", "ext54", "all"):
        g = r if grp == "all" else r[r["group"] == grp]
        print(f"\n[{grp}]")
        for per, (a_, b_) in PER.items():
            x = g[(g.t >= a_) & (g.t < b_)]
            sup, ag = x[x.spot_z > 0], x[x.spot_z <= 0]
            print(f"  {per}: все {len(x)} сд. {x.r.mean() * 1e4:+.0f} б.п. | спот за нас {len(sup)} сд. "
                  f"{sup.r.mean() * 1e4:+.0f} б.п., плюс {(sup.r > 0).mean():.0%} | спот против {len(ag)} сд. "
                  f"{ag.r.mean() * 1e4:+.0f} б.п., плюс {(ag.r > 0).mean():.0%}")
