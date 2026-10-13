"""Лонги бота 4h: агрессоры спота вместо агрессоров фьючерса.

Гипотеза: долгосрочный покупатель берёт спот, а не перп, поэтому лонговый пробой с покупателями видно на споте, даже
если на фьючерсе рыночных покупок не больше 55%. База — лонги research/longfilt.py (все фильтры бота, кроме агрессоров;
ретест, 3R, долив на +1.5R). Спот — свечи Binance той же монеты, 1h, сложенные в свечу 4h пробоя. Варианты (заданы
до запуска), только на лонгах, у которых есть спот:
    перп 55% (бот)       для сравнения
    спот 55 / 52.5 / 50% доля рыночных покупок на споте вместо перпа
    спот 55% или перп    добавка к боту: спот ≥ 55%, а перп < 55%
    спот сильнее перпа   доля покупок на споте выше, чем на перпе, на 3 п.п. и больше
    оба ≥ 52.5%          и спот, и перп
Принятие: R на сделку > 0 во всех периодах 2021 / IS / VAL / HO / 2026, где сделок ≥ 10, и (кроме добавки) больше,
чем у базы без фильтра, в каждом из них. Вариантов 7.

    python -m research.spotlong collect --symbols <1h-qualified>
    python -m research.spotlong report
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import longfilt
from .broad import adv30
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _cell
from .tline import tf_frame

VARIANTS = {
    "перп 55% (бот)": lambda z: z.buy >= 0.55,
    "без фильтра": lambda z: pd.Series(True, index=z.index),
    "спот 55%": lambda z: z.spot >= 0.55,
    "спот 52.5%": lambda z: z.spot >= 0.525,
    "спот 50%": lambda z: z.spot >= 0.50,
    "добавка: спот 55%, перп < 55%": lambda z: (z.spot >= 0.55) & (z.buy < 0.55),
    "спот сильнее перпа на 3 п.п.": lambda z: z.spot - z.buy >= 0.03,
    "оба ≥ 52.5%": lambda z: (z.spot >= 0.525) & (z.buy >= 0.525),
}


def spot_on_bars(root: Path, sym: str, index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Доля рыночных покупок на споте и доля спота в обороте (спот / (спот + перп)) в свечах 4h; NaN — нет спота."""
    p = root / f"{sym}-spot-1h.parquet"
    nan = np.full(len(index), np.nan)
    if not p.exists():
        return nan, nan
    sp = pd.read_parquet(p, columns=["open_time", "quote_volume", "taker_buy_volume", "volume"])
    sp.index = pd.to_datetime(sp["open_time"], unit="ms", utc=True)
    sp = sp[~sp.index.duplicated()].sort_index()[["quote_volume", "taker_buy_volume", "volume"]]
    sp = sp.resample("4h").sum(min_count=1).reindex(index)
    buy = (sp["taker_buy_volume"] / sp["volume"].replace(0, np.nan)).to_numpy(dtype="float64")
    return buy, sp["quote_volume"].to_numpy(dtype="float64")


def collect(root: Path, syms: list[str]) -> None:
    b = tf_frame(root, "BTCUSDT", "4h")
    btc = b["close"] if b is not None else None
    rows = []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, "4h")
            if d is None or len(d) < 500:
                continue
            liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
            r = longfilt.coin_rows(d, liq, s, btc)
            if not r:
                continue
            spot, sq = spot_on_bars(root, s, d.index)
            pos = d.index.get_indexer([x["t"] for x in r])
            pq = d["quote_volume"].to_numpy(dtype="float64")
            for x, i in zip(r, pos):
                x["spot"] = spot[i]
                x["spot_share"] = sq[i] / (sq[i] + pq[i]) if np.isfinite(sq[i]) and pq[i] > 0 else np.nan
            rows += r
        except Exception as e:
            print(f"  spotlong {s}: пропуск ({e})", flush=True)
    print(f"  spotlong: лонгов {len(rows)}, со спотом {sum(np.isfinite(x['spot']) for x in rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("spotlong"), index=False)


def report() -> None:
    parts = all_parts("spotlong")
    print("\n===== SPOTLONG: лонги бота 4h — агрессоры спота вместо агрессоров фьючерса; сделка как у бота =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df.dropna(subset=["R"])
    print(f"лонгов {len(df)}, со спотом {df.spot.notna().sum()} ({df.spot.notna().mean():.0%}); медиана доли спота в "
          f"обороте {df.spot_share.median():.0%}; доля покупок: перп медиана {df.buy.median():.1%}, спот "
          f"{df.spot.median():.1%}; корреляция {df[['buy', 'spot']].corr().iloc[0, 1]:.2f}")
    z0 = df[df.spot.notna()]
    per = {**PER_ALL, "2026": ("2026-01-01", "2026-10-01")}
    judged = [k for k in per if k != "2020"]
    base = z0
    for nm, f in VARIANTS.items():
        z = z0[f(z0)]
        ok = []
        for p in judged:
            a, b = per[p]
            m, mb = z[(z.t >= a) & (z.t < b)], base[(base.t >= a) & (base.t < b)]
            if len(m) >= 10:
                ok.append(m.R.mean() > 0 and (nm.startswith("добавка") or m.R.mean() > mb.R.mean()))
        tag = "" if nm in ("перп 55% (бот)", "без фильтра") else (
            "  → принято" if ok and all(ok) else f"  → не принято ({sum(ok)} из {len(ok)})")
        print(f"\n  {nm}: {len(z)} сделок, всего {z.R.sum():+.1f}R{tag}")
        print("    " + "  ".join(f"{p}: {_cell(z[(z.t >= a) & (z.t < b)])}" for p, (a, b) in per.items()))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x])
    else:
        report()
