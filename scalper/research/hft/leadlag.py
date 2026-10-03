"""
Опережение Binance -> Bybit на событиях с миллисекундами: остаётся ли перевес после реальной задержки бота
и комиссий Bybit.

Вход (taker): лонг — по первой покупке агрессора на Bybit после tau+L (если её не было в течение секунды —
по ask стакана), шорт — по первой продаже агрессора (или bid). Выход через h секунд:
    taker: продажа по bid (лонг) / покупка по ask (шорт), комиссия 5.5 + 5.5 б.п.
    maker: выход лимитом по противоположной стороне (ask для лонга) — оптимистичная граница, 5.5 + 2 б.п.

    python -m research.hft.leadlag --root <папка с ev-*.parquet> --out <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .build import EXIT_S, LATENCIES_MS
from .edge import period_of

TAKER, MAKER = 5.5, 2.0
GAP_BUCKETS = (3, 5, 10, 20, 1e9)


def load(root: Path) -> pd.DataFrame:
    parts = []
    for p in sorted(root.glob("ev-*.parquet")):
        _, sym, day = p.stem.split("-", 2)
        e = pd.read_parquet(p)
        e["day"], e["period"] = day, period_of(day)
        parts.append(e)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def returns(e: pd.DataFrame, L: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Чистая доходность сделки в б.п.: (taker-выход, maker-выход)."""
    side = e["side"].to_numpy()
    entry = np.where(side > 0, e[f"tbuy_{L}"].fillna(e[f"ask_{L}"]), e[f"tsell_{L}"].fillna(e[f"bid_{L}"]))
    x_t = np.where(side > 0, e[f"xbid_{h}"], e[f"xask_{h}"])
    x_m = np.where(side > 0, e[f"xask_{h}"], e[f"xbid_{h}"])
    r_t = side * (x_t / entry - 1) * 1e4 - 2 * TAKER
    r_m = side * (x_m / entry - 1) * 1e4 - TAKER - MAKER
    return r_t, r_m


def table(e: pd.DataFrame) -> pd.DataFrame:
    rows = []
    days = e.groupby(["symbol", "period"])["day"].nunique()
    e = e.assign(bucket=pd.cut(e["gap_bps"].abs(), GAP_BUCKETS, right=False))
    for (sym, per, b), g in e.groupby(["symbol", "period", "bucket"], observed=True):
        for L in LATENCIES_MS:
            for h in EXIT_S:
                rt, rm = returns(g, L, h)
                rt, rm = rt[np.isfinite(rt)], rm[np.isfinite(rm)]
                if len(rt) < 20:
                    continue
                rows.append({"symbol": sym, "period": per, "gap": str(b), "L_ms": L, "h_s": h, "n": len(rt),
                             "per_day": len(rt) / days[(sym, per)], "net_taker": rt.mean(), "net_maker_exit": rm.mean(),
                             "t_taker": rt.mean() / (rt.std(ddof=1) / np.sqrt(len(rt))),
                             "gross": rt.mean() + 2 * TAKER})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    e = load(Path(a.root))
    print(f"событий: {len(e)}, монет: {e['symbol'].nunique()}, дней: {e['day'].nunique()}")
    t = table(e)
    t.to_csv(out / "leadlag.csv", index=False)
    # сводка: как перевес тает с задержкой (все монеты, все разрывы >= 5 б.п., выход через 5 с)
    s = t[(t["h_s"] == 5)].groupby(["period", "L_ms"]).apply(
        lambda g: pd.Series({"gross": np.average(g["gross"], weights=g["n"]),
                             "net_taker": np.average(g["net_taker"], weights=g["n"]),
                             "net_maker_exit": np.average(g["net_maker_exit"], weights=g["n"]), "n": g["n"].sum()}),
        include_groups=False)
    print("\nВсе события, выход через 5 с: перевес в б.п. в зависимости от задержки входа")
    print(s.round(2).to_string())
    for per in ("is", "val", "ho"):
        x = t[(t["period"] == per) & (t["L_ms"] == 100)].sort_values("net_maker_exit", ascending=False).head(20)
        print(f"\n--- {per}, задержка 100 мс: лучшие связки (монета × разрыв × горизонт) ---")
        print(x.round(2).to_string(index=False))
