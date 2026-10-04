"""
Выгрузка списков сделок для моделирования счёта (research.portfolio_sim2):
    coil_trades.csv       coil в базовой настройке (research.coil_check.BASE, обе стороны, стоп 1.5 ATR, тейк 5R):
                          время входа, монета, сторона, R, длительность (4h-бары), риск в долях цены (1.5 ATR / close)
    announce_trades.csv   события research.listings2 (класс, время, монета, есть ли на Bybit, результаты сделок, б.п.)

    python -m research.export_more collect --symbols ...   (по частям)
    python -m research.export_more report                  (сборка в $OUT, по умолчанию ../out)
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30
from .coil_check import BASE, coil_signals
from .shard import all_parts, mine, part_path
from .trend import COST, EXITS, MAX_HOLD, trade_machine
from .trend2 import bars4h
from .wave2 import Data2

STOP_K = 1.5


def coil_trades(root: Path, syms: list[str]) -> pd.DataFrame:
    data = Data2(root, syms)
    out = []
    for s in syms:
        try:
            h = data.get(s, "1h", "full")
            if h["oi"].notna().sum() < 24 * 60:
                continue
            d = bars4h(h)
            allowed = (adv30(root, s).reindex(d.index, method="ffill") >= ADV_MIN).to_numpy()
            up, dn = coil_signals(d, *BASE)
            L, S = up & allowed, dn & allowed
            a = {c: d[c].to_numpy(dtype="float64") for c in ("open", "high", "low", "close", "atr", "funding")}
            i, r, du = trade_machine(a["open"], a["high"], a["low"], a["close"], a["atr"], a["funding"], L, S,
                                     STOP_K, EXITS.index("tp5"), MAX_HOLD, COST)
        except Exception as e:
            print(f"  {s}: пропуск ({e})", flush=True)
            continue
        finally:
            data.cache.clear()
        if len(i):
            out.append(pd.DataFrame({"t": d.index[i], "symbol": s, "side": np.where(L[i], 1, -1), "R": r,
                                     "bars": du, "risk": STOP_K * a["atr"][i] / a["close"][i]}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        tr = coil_trades(Path(a.root).expanduser(), mine([x for x in a.symbols.split(",") if x]))
        if len(tr):
            tr.to_parquet(part_path("coil_trades"), index=False)
    else:
        out = Path(os.environ.get("OUT", "../out"))
        out.mkdir(parents=True, exist_ok=True)
        for name, fname in (("coil_trades", "coil_trades.csv"), ("listings2", "announce_trades.csv")):
            parts = all_parts(name)
            if not parts:
                print(f"{name}: частей нет")
                continue
            df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).sort_values("t")
            df.to_csv(out / fname, index=False)
            print(f"{fname}: {len(df)} строк")
