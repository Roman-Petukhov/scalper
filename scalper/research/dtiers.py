"""Дневной тренд (research/dtrend.py) на монетах поменьше: группы по месту в рейтинге оборота за 30 дней.

Капитализации в данных нет — размер монеты меряется оборотом (adv30, только прошлое, от $20M в день). Правило тренда
то же (лонг, ансамбль Дончиана 5–360, размер по волатильности, без включателя); в каждой группе вес делится на число
мест в группе. Группы: 1–20 (основной вариант), 21–50, 51–100, 101–200 и 1–100 целиком; у мелких групп — также
издержки x2 (проскальзывание на тонкой ликвидности). Счёт с ботом — как в research/dtregime.py: бот 1% на сделку +
тренд с долей капитала k группы 1–20 (одна для всех групп).

    python -m research.dtrend collect --symbols <монеты 1h-qualified>
    python -m research.dtiers report
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from .broad import ADV_MIN
from .dtrend import _stats, portfolio
from .dtregime import _mstats
from .oos import PER_ALL
from .shard import all_parts
from .tline import _bot_base

TIERS = {"1–20": dict(top=20), "1–10": dict(top=10), "11–20": dict(skip=10, top=20), "11–50": dict(skip=10, top=50),
         "11–100": dict(skip=10, top=100), "11–100, издержки x2": dict(skip=10, top=100, cost_k=2.0),
         "21–50": dict(skip=20, top=50), "51–100": dict(skip=50, top=100), "1–100": dict(top=100)}


def report() -> None:
    parts, bparts = all_parts("dtrend"), all_parts("dtrend_bot")
    print("\n===== DTIERS: дневной тренд по группам монет (место по обороту за 30 дней) =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["day"] = pd.to_datetime(df["day"], utc=True).dt.tz_localize(None)
    close = df.pivot_table(index="day", columns="symbol", values="close")
    fund = df.pivot_table(index="day", columns="symbol", values="funding").reindex_like(close)
    adv = df.pivot_table(index="day", columns="symbol", values="adv").reindex_like(close)
    n_ok = (adv >= ADV_MIN).sum(axis=1)
    print("  монет с оборотом от $20M на дату, медиана по годам: " + ", ".join(
        f"{y} {int(v)}" for y, v in n_ok[n_ok.index >= "2020-01-01"].groupby(n_ok.index[n_ok.index >= "2020-01-01"].year).median().items()))
    res = {nm: portfolio(close, fund, adv, **kw) for nm, kw in TIERS.items()}
    for nm, r in res.items():
        print(f"\n  {nm}")
        for p, (a, b) in PER_ALL.items():
            print(f"    {p:>4}: {_stats(r[(r.index >= a) & (r.index < b)])}")
        y = (1 + r[r.index >= "2020-01-01"]).groupby(r.index[r.index >= "2020-01-01"].year).prod() - 1
        print("    по годам: " + ", ".join(f"{k} {v:+.0%}" for k, v in y.items()))
    if not bparts:
        return
    bt = pd.concat([pd.read_parquet(p) for p in bparts], ignore_index=True)
    bt["t"] = pd.to_datetime(bt["t"], utc=True)
    bt = _bot_base(bt[bt.adv.isna() | (bt.adv >= ADV_MIN)])
    bt = bt[bt.entry == "retest"]
    bot_m = bt.groupby(bt.t.dt.tz_localize(None).dt.to_period("M")).R3.sum() * 0.01
    tr_m = {nm: (1 + r).groupby(r.index.to_period("M")).prod() - 1 for nm, r in res.items()}
    idx = tr_m["1–20"].index[tr_m["1–20"].index >= pd.Period("2020-06", "M")]
    bot_m = bot_m.reindex(idx, fill_value=0.0)
    k = bot_m.std() / tr_m["1–20"].reindex(idx).std()
    print(f"\n  счёт по месяцам {idx[0]}–{idx[-1]}: бот 1% на сделку + тренд с долей капитала k = {k:.2f}")
    print(f"    бот один: {_mstats(bot_m)}")
    for nm, m in tr_m.items():
        m = m.reindex(idx).fillna(0.0)
        combo = bot_m + k * m
        print(f"    + тренд {nm}: {_mstats(combo)}; корреляция с ботом {bot_m.corr(m):+.2f}; тренд × k с 2022: "
              f"{k * m[idx >= pd.Period('2022-01', 'M')].mean() * 1200:+.0f}% в год")
        kt = bot_m.std() / m.std() if m.std() > 0 else 0.0
        print(f"      своя доля k = {kt:.2f} (волатильность как у бота): {_mstats(bot_m + kt * m)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 300)
    report()
