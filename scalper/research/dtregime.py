"""Дневной тренд (research/dtrend.py) только в бычьем рынке: включатель по режиму и счёт вместе с ботом 4h.

Повод: в бычьем рынке монеты дают +100% и больше, а бот почти только шортит и берёт 3R. Дневной тренд (лонг, ансамбль
каналов Дончиана 5–360 дней, топ-20 по обороту, размер по волатильности, без цели) в 2020–2021 давал Sharpe 1.4–1.7,
а с 2022 года ≈ 0.4 и 60–75% убыточных месяцев. Вопрос: включать его только в бычьем режиме.

Включатели заданы до запуска (известны на закрытии дня, позиции на следующий день; выключение закрывает позиции):
    всегда        без включателя (как в dtrend)
    широта > 60%  доля монет с оборотом от $20M, закрытие которых выше дневной EMA50 (от 20 монет)
    BTC > SMA200  закрытие BTC выше 200-дневной средней
    оба           оба условия сразу
Счёт вместе с ботом (месяцы): бот — ретест, 3R, риск 1% на сделку; тренд — доля капитала k, одна и та же для всех
включателей: k такое, что у тренда «всегда» волатильность месяца равна волатильности бота.

    python -m research.dtrend collect --symbols <монеты 1h-qualified>     (части dtrend и dtrend_bot)
    python -m research.dtregime report
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from .broad import ADV_MIN
from .dtrend import _stats, portfolio
from .oos import PER_ALL
from .shard import all_parts
from .tline import _bot_base

BREADTH_ON = 0.60
MIN_COINS = 20


def gates(close: pd.DataFrame, adv: pd.DataFrame) -> dict[str, pd.Series]:
    ema = close.ewm(span=50, adjust=False, min_periods=50).mean()
    ok = (adv >= ADV_MIN) & close.notna() & ema.notna()
    n = ok.sum(axis=1)
    breadth = ((close > ema) & ok).sum(axis=1) / n.where(n >= MIN_COINS)
    btc = close["BTCUSDT"] if "BTCUSDT" in close else pd.Series(np.nan, index=close.index)
    btc_up = btc > btc.rolling(200, min_periods=200).mean()
    wide = breadth > BREADTH_ON
    return {"всегда": pd.Series(True, index=close.index), "широта > 60%": wide, "BTC > SMA200": btc_up,
            "оба": wide & btc_up}


def _mstats(m: pd.Series) -> str:
    if len(m) < 6 or m.std() == 0:
        return "—"
    eq = (1 + m).cumprod()
    return (f"{m.mean() * 12:+.0%} год, Sharpe {m.mean() / m.std() * np.sqrt(12):.2f}, просадка "
            f"{(1 - eq / eq.cummax()).max():.0%}, худший мес. {m.min():+.0%}, убыт. мес. {(m < 0).mean():.0%}")


def report() -> None:
    parts, bparts = all_parts("dtrend"), all_parts("dtrend_bot")
    print("\n===== DTREGIME: дневной тренд (лонг, Дончиан 5–360, топ-20) только в бычьем режиме; включатели заданы "
          "до запуска =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["day"] = pd.to_datetime(df["day"], utc=True).dt.tz_localize(None)
    close = df.pivot_table(index="day", columns="symbol", values="close")
    fund = df.pivot_table(index="day", columns="symbol", values="funding").reindex_like(close)
    adv = df.pivot_table(index="day", columns="symbol", values="adv").reindex_like(close)
    g = gates(close, adv)
    res = {nm: portfolio(close, fund, adv, gate=gt) for nm, gt in g.items()}
    for nm, r in res.items():
        on = g[nm].reindex(r.index).fillna(False)
        print(f"\n  {nm}: включён {on[r.index >= '2020-01-01'].mean():.0%} дней; по периодам (доля дней включён)")
        for p, (a, b) in PER_ALL.items():
            sel = (r.index >= a) & (r.index < b)
            print(f"    {p:>4} ({on[sel].mean():.0%}): {_stats(r[sel])}")
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
    idx = tr_m["всегда"].index[tr_m["всегда"].index >= pd.Period("2020-06", "M")]
    bot_m = bot_m.reindex(idx, fill_value=0.0)
    k = bot_m.std() / tr_m["всегда"].reindex(idx).std()
    print(f"\n  счёт по месяцам {idx[0]}–{idx[-1]}: бот 1% на сделку + тренд с долей капитала k = {k:.2f} "
          "(одна для всех включателей)")
    print(f"    бот один: {_mstats(bot_m)}")
    for nm, m in tr_m.items():
        m = m.reindex(idx).fillna(0.0)
        combo = bot_m + k * m
        print(f"    + тренд «{nm}»: {_mstats(combo)}; корреляция с ботом {bot_m.corr(m):+.2f}")
        print("       по периодам, % в год (бот / тренд × k / вместе): " + "; ".join(
            f"{p} {bot_m[s].mean() * 1200:+.0f} / {k * m[s].mean() * 1200:+.0f} / {combo[s].mean() * 1200:+.0f}"
            for p, (a, b) in PER_ALL.items()
            if (s := (idx >= pd.Period(a[:7], "M")) & (idx < pd.Period(b[:7], "M"))).sum() >= 6))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 300)
    report()
