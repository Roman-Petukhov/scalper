"""Лонги бота 4h без фильтра агрессоров, но с условием по OI: хватает ли шортов, чтобы их вынести.

База — лонги research/longfilt.py (пробой нисходящей линии, дневной тренд, закрытие в верхней половине, стоп за свингом,
ретест, 3R, долив на +1.5R) без агрессоров. OI, funding и позиции топ-трейдеров — метрики Binance (с 2022 года).
Варианты (заданы до запуска), всё — на закрытии свечи пробоя:
    oi_up         OI за 3 дня (18 свечей) вырос на 5% и больше
    шорты копятся OI за 3 дня вырос на 5%+, а цена за те же 3 дня до пробоя не выросла
    фандинг < 0   сумма funding за 3 дня отрицательная — шорты платят
    толпа в шорт  соотношение лонг / шорт топ-трейдеров ниже своей медианы за 30 дней
    вынос пошёл   OI на свече пробоя упал (шорты закрываются по рынку)
    копятся + фандинг   «шорты копятся» и «фандинг < 0» вместе
Принятие: R на сделку > 0 во всех периодах IS / VAL / HO / 2026, где сделок ≥ 10, и больше, чем у базы без фильтра,
в каждом из них. Вариантов 6.

    python -m research.oilong collect --symbols <1h-qualified>     (данные с метриками: prepq)
    python -m research.oilong report
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
from .wave2 import metrics_on_bars

N = 18
OI_UP = 0.05
VARIANTS = {
    "агрессоры ≥ 55% (бот)": lambda z: z.buy >= 0.55,
    "без фильтра": lambda z: pd.Series(True, index=z.index),
    "OI за 3 дня +5%": lambda z: z.oi_chg >= OI_UP,
    "шорты копятся (OI +5%, цена не выросла)": lambda z: (z.oi_chg >= OI_UP) & (z.ret_before <= 0),
    "фандинг за 3 дня < 0": lambda z: z.fund < 0,
    "топ-трейдеры в шорт сильнее обычного": lambda z: z.top_dev < 0,
    "вынос пошёл: OI на свече пробоя упал": lambda z: z.doi_bar < 0,
    "шорты копятся + фандинг < 0": lambda z: (z.oi_chg >= OI_UP) & (z.ret_before <= 0) & (z.fund < 0),
}


def oi_features(d: pd.DataFrame, m: pd.DataFrame) -> dict[str, np.ndarray]:
    oi = m["oi"].astype(float).replace(0, np.nan)
    top = np.log(m["top_lsr"].astype(float).where(m["top_lsr"] > 0))
    c = d["close"]
    return {"oi_chg": (oi / oi.shift(N) - 1).to_numpy(), "doi_bar": (oi / oi.shift(1) - 1).to_numpy(),
            "ret_before": (c.shift(1) / c.shift(1 + N) - 1).to_numpy(),
            "fund": d["funding"].fillna(0.0).rolling(N).sum().to_numpy(),
            "top_dev": (top - top.rolling(180, min_periods=60).median()).to_numpy()}


def collect(root: Path, syms: list[str]) -> None:
    rows = []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, "4h")
            if d is None or len(d) < 500:
                continue
            liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
            r = longfilt.coin_rows(d, liq, s, None)
            if not r:
                continue
            fx = oi_features(d, metrics_on_bars(s, root, d.index, 240))
            pos = d.index.get_indexer([x["t"] for x in r])
            for x, i in zip(r, pos):
                for k, v in fx.items():
                    x[k] = v[i]
            rows += r
        except Exception as e:
            print(f"  oilong {s}: пропуск ({e})", flush=True)
    print(f"  oilong: лонгов {len(rows)}, с OI {sum(np.isfinite(x['oi_chg']) for x in rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("oilong"), index=False)


def report() -> None:
    parts = all_parts("oilong")
    print("\n===== OILONG: лонги бота 4h без агрессоров, но с условием по OI / funding / позициям топ-трейдеров "
          "(хватит ли шортов на вынос); сделка как у бота =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df.dropna(subset=["R", "oi_chg"])
    per = {k: v for k, v in {**PER_ALL, "2026": ("2026-01-01", "2026-10-01")}.items() if k not in ("2020", "2021")}
    print(f"лонгов с OI {len(df)}; доли: " + ", ".join(f"{k} {f(df).mean():.0%}" for k, f in VARIANTS.items()
                                                      if k != "без фильтра"))
    for nm, f in VARIANTS.items():
        z = df[f(df)]
        ok = []
        for a, b in per.values():
            m, mb = z[(z.t >= a) & (z.t < b)], df[(df.t >= a) & (df.t < b)]
            if len(m) >= 10:
                ok.append(m.R.mean() > 0 and m.R.mean() > mb.R.mean())
        tag = "" if nm in ("агрессоры ≥ 55% (бот)", "без фильтра") else (
            "  → мало сделок" if not ok else "  → принято" if all(ok) else f"  → не принято ({sum(ok)} из {len(ok)})")
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
