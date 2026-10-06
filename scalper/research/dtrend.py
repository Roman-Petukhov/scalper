"""Вторая стратегия: медленный тренд по каналам Дончиана на дневках (по Zarattini, Pagani, Barbon 2025, «Catching
Crypto Trends», SFI 25-80) — проверка до внедрения и связь с ботом 4h.

Правило задано заранее, не подбирается:
    модели      периоды N = 5, 10, 20, 30, 60, 90, 150, 250, 360 дней; модель N в лонге с закрытия, которое не ниже
                максимума закрытий за N прошлых дней, выходит на закрытии не выше середины канала (среднее максимума и
                минимума за N дней). Сигнал монеты — доля моделей в лонге (0…1). Вариант «лонг + шорт» — зеркально
    вселенная   каждый день — 20 монет с наибольшим средним оборотом за 30 прошлых дней (adv30, без заглядывания);
                вышла из топа — позиция закрывается
    размер      вес = сигнал × (40% / годовая волатильность монеты за 90 дней) / 20, общий вес не больше 2
    исполнение  вес решается на закрытии дня, держится следующий день; комиссия taker 0.055% с оборота, funding —
                лонг платит при положительной ставке
Метрики по периодам 2020, 2021, IS, VAL, HO: доходность и волатильность годовые, Sharpe, худшая просадка, доля
убыточных месяцев. Связь с ботом: корреляция месячных результатов с ботом 4h (ретест, 3R, риск 1% на сделку) и
портфель «бот + тренд» с равной волатильностью частей.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import tline
from .broad import ADV_MIN, adv30
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .tline import TAKER, _bot_base, coin_trades, market_context, tf_frame

LOOKBACKS = (5, 10, 20, 30, 60, 90, 150, 250, 360)
TOP = 20
COIN_VOL = 0.40
GROSS_MAX = 2.0
VOL_DAYS = 90


def collect(root: Path, syms: list[str]) -> None:
    ctx = market_context(root)
    tline.LINES, tline.SCALE_LINES = ("zz",), {}
    days, bot = [], []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, "1d")
            if d is None or len(d) < 30:
                continue
            a = adv30(root, s).reindex(d.index)
            days.append(pd.DataFrame({"symbol": s, "day": d.index, "close": d["close"].to_numpy(),
                                      "funding": d["funding"].to_numpy(), "adv": a.to_numpy()}))
            x = coin_trades(root, s, "4h", ctx)
            if len(x):
                bot.append(x[(x.line == "zz") & ~x.confirm][["symbol", "t", "side", "entry", "confirm", "aggr",
                                                               "with_trend", "close_loc", "adv", "R3"]])
        except Exception as e:
            print(f"  dtrend {s}: пропуск ({e})", flush=True)
    print(f"  dtrend: монет {len(days)}, с сделками бота {len(bot)}", flush=True)
    if days:
        pd.concat(days, ignore_index=True).to_parquet(part_path("dtrend"), index=False)
    if bot:
        pd.concat(bot, ignore_index=True).to_parquet(part_path("dtrend_bot"), index=False)


def _models(close: pd.DataFrame, n: int, short: bool) -> pd.DataFrame:
    """Состояние модели N по каждой монете: 1 / 0 / −1 (−1 только при short)."""
    hi = close.rolling(n, min_periods=n).max().shift(1)
    lo = close.rolling(n, min_periods=n).min().shift(1)
    mid = (hi + lo) / 2
    x = close.to_numpy()
    h, l_, m = hi.to_numpy(), lo.to_numpy(), mid.to_numpy()
    st = np.zeros_like(x)
    cur = np.zeros(x.shape[1])
    for i in range(len(x)):
        up, dn = x[i] >= h[i], x[i] <= l_[i]
        cur = np.where((cur > 0) & (x[i] <= m[i]), 0.0, cur)
        cur = np.where((cur < 0) & (x[i] >= m[i]), 0.0, cur)
        cur = np.where((cur == 0) & up, 1.0, cur)
        if short:
            cur = np.where((cur == 0) & dn, -1.0, cur)
        cur = np.where(np.isnan(x[i]), 0.0, cur)
        st[i] = cur
    return pd.DataFrame(st, index=close.index, columns=close.columns)


def portfolio(close: pd.DataFrame, fund: pd.DataFrame, adv: pd.DataFrame, lookbacks=LOOKBACKS, top: int = TOP,
              short: bool = False, cost_k: float = 1.0, only: list[str] | None = None) -> pd.Series:
    """Дневная доходность портфеля (с издержками)."""
    sig = sum(_models(close, n, short) for n in lookbacks) / len(lookbacks)
    ret = close.pct_change(fill_method=None)
    vol = ret.rolling(VOL_DAYS, min_periods=30).std() * np.sqrt(365)
    if only is not None:
        member = pd.DataFrame(False, index=close.index, columns=close.columns)
        member[[c for c in only if c in member.columns]] = True
    else:
        rk = adv.where(adv >= ADV_MIN).rank(axis=1, ascending=False, method="first")
        member = rk <= top
    k = len(only) if only is not None else top
    w = (sig * (COIN_VOL / vol) / k).where(member & close.notna(), 0.0).fillna(0.0)
    gross = w.abs().sum(axis=1)
    w = w.mul(np.minimum(1.0, GROSS_MAX / gross.replace(0, np.nan)).fillna(1.0), axis=0)
    held = w.shift(1).fillna(0.0)
    pnl = (held * ret.fillna(0.0)).sum(axis=1) - (held * fund.fillna(0.0)).sum(axis=1)
    cost = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1).shift(1).fillna(0.0) * TAKER * cost_k
    return pnl - cost


def _stats(r: pd.Series) -> str:
    if len(r) < 30 or r.std() == 0:
        return "—"
    ann, vol = r.mean() * 365, r.std() * np.sqrt(365)
    eq = (1 + r).cumprod()
    dd = (1 - eq / eq.cummax()).max()
    m = (1 + r).groupby(r.index.to_period("M")).prod() - 1
    return f"{ann:+.0%} год, вол {vol:.0%}, Sharpe {ann / vol:.2f}, просадка {dd:.0%}, убыт. мес. {(m < 0).mean():.0%}"


def report() -> None:
    parts, bparts = all_parts("dtrend"), all_parts("dtrend_bot")
    print("\n===== DTREND: тренд по каналам Дончиана на дневках, ансамбль 9 периодов, топ-20 по обороту, размер по "
          "волатильности; издержки — taker с оборота и funding =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["day"] = pd.to_datetime(df["day"], utc=True).dt.tz_localize(None)
    close = df.pivot_table(index="day", columns="symbol", values="close")
    fund = df.pivot_table(index="day", columns="symbol", values="funding").reindex_like(close)
    adv = df.pivot_table(index="day", columns="symbol", values="adv").reindex_like(close)
    print(f"дней {len(close)}, монет {close.shape[1]}, с {close.index[0]:%Y-%m-%d}")
    variants = {
        "лонг, ансамбль 9 периодов, топ-20 (основной)": dict(),
        "  издержки x2": dict(cost_k=2.0),
        "лонг + шорт": dict(short=True),
        "лонг, только быстрые 5–30": dict(lookbacks=(5, 10, 20, 30)),
        "лонг, только медленные 60–360": dict(lookbacks=(60, 90, 150, 250, 360)),
        "лонг, один период 20": dict(lookbacks=(20,)),
        "лонг, топ-10": dict(top=10),
        "лонг, топ-50": dict(top=50),
        "лонг, только BTC": dict(only=["BTCUSDT"]),
    }
    res = {nm: portfolio(close, fund, adv, **kw) for nm, kw in variants.items()}
    hold = close.pct_change(fill_method=None)["BTCUSDT"].fillna(0.0) if "BTCUSDT" in close else None
    for nm, r in [*res.items(), *([("для сравнения: держать BTC", hold)] if hold is not None else [])]:
        print(f"\n  {nm}")
        for p, (a, b) in PER_ALL.items():
            print(f"    {p:>4}: {_stats(r[(r.index >= a) & (r.index < b)])}")
        print(f"    всё: {_stats(r[r.index >= '2020-01-01'])}")

    if not bparts:
        return
    bt = pd.concat([pd.read_parquet(p) for p in bparts], ignore_index=True)
    bt["t"] = pd.to_datetime(bt["t"], utc=True)
    bt = _bot_base(bt[bt.adv.isna() | (bt.adv >= ADV_MIN)])
    bt = bt[bt.entry == "retest"]
    bot_m = bt.groupby(bt.t.dt.tz_localize(None).dt.to_period("M")).R3.sum() * 0.01      # риск 1% на сделку
    tr = res["лонг, ансамбль 9 периодов, топ-20 (основной)"]
    tr_m = (1 + tr).groupby(tr.index.to_period("M")).prod() - 1
    idx = tr_m.index[(tr_m.index >= pd.Period("2020-06", "M"))]
    bot_m = bot_m.reindex(idx, fill_value=0.0)
    tr_m = tr_m.reindex(idx)
    print(f"\n  связь с ботом 4h (месяцы {idx[0]}–{idx[-1]}): корреляция {bot_m.corr(tr_m):+.2f}; "
          f"в худшие 10 месяцев бота тренд в среднем {tr_m[bot_m.nsmallest(10).index].mean():+.1%}")
    scale = bot_m.std() / tr_m.std()
    combo = (bot_m + tr_m * scale) / 2

    def mstats(m: pd.Series) -> str:
        eq = (1 + m).cumprod()
        return (f"{m.mean() * 12:+.0%} год, Sharpe {m.mean() / m.std() * np.sqrt(12):.2f}, просадка "
                f"{(1 - eq / eq.cummax()).max():.0%}, убыт. мес. {(m < 0).mean():.0%}")
    print(f"    бот один (1% на сделку): {mstats(bot_m)}")
    print(f"    тренд один, той же волатильности: {mstats(tr_m * scale)}")
    print(f"    пополам, равная волатильность: {mstats(combo)}")
    for p, (a, b) in PER_ALL.items():
        sel = (idx >= pd.Period(a[:7], "M")) & (idx < pd.Period(b[:7], "M"))
        if sel.sum() >= 6:
            print(f"    {p:>4}: бот {bot_m[sel].mean() * 12:+.0%} / тренд {(tr_m * scale)[sel].mean() * 12:+.0%} / "
                  f"вместе {combo[sel].mean() * 12:+.0%} в год; Sharpe вместе "
                  f"{combo[sel].mean() / combo[sel].std() * np.sqrt(12):.2f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 300)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x])
    else:
        report()
