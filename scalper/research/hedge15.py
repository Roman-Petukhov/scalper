"""Хедж BTC для 15m: правило бота 15m без изменений (шорт от пологой линии, по рынку, топ-150 по обороту, цель 3R,
срок 200 свечей — research/targets15.py), к каждой сделке добавлена обратная позиция BTC на бету монеты (бета по
4h-закрытиям за 60 дней до сигнала, как в боте), результат хеджа — по цене BTC от входа до выхода, комиссии тейкера
на две стороны. R хеджа = P&L хеджа / риск сделки, как в research/hedged.py.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30
from .shard import all_parts, mine, part_path
from .tline import PER, TAKER, _bot_base, coin_trades, market_context, tf_frame
from .wide15 import MAX_SLOPE_ATR

TOP = 150
BAR = pd.Timedelta(minutes=15)
BETA_BARS, BETA_MIN_BARS, BETA_RANGE = 360, 120, (0.0, 3.0)      # как в domain/hedge.py
COLS = ["symbol", "t", "side", "entry", "aggr", "with_trend", "close_loc", "slope_atr", "risk_pct", "R3", "hold_bars"]


def _beta(coin4: pd.Series, btc4: pd.Series, t: pd.Timestamp) -> float:
    both = pd.concat([coin4, btc4], axis=1, join="inner").loc[:t].dropna()
    r = np.log(both).diff().dropna().iloc[-BETA_BARS:]
    if len(r) < BETA_MIN_BARS or not r.iloc[:, 1].var() > 0:
        return np.nan
    b = float(r.cov().iloc[0, 1] / r.iloc[:, 1].var())
    return min(max(b, BETA_RANGE[0]), BETA_RANGE[1])


def collect(root: Path, syms: list[str]) -> None:
    ctx = market_context(root)
    btc_d = tf_frame(root, "BTCUSDT", "15m")
    btc = btc_d["close"] if btc_d is not None else None
    btc4 = btc.resample("4h").last() if btc is not None else None
    trades, advs = [], []
    for s in mine(syms):
        try:
            a = adv30(root, s).dropna()
            advs.append(pd.DataFrame({"symbol": s, "day": a.index, "adv": a.to_numpy()}))
            x = coin_trades(root, s, "15m", ctx)
            if not len(x) or btc is None or s == "BTCUSDT":
                continue
            x = x[~x.confirm & (x.line == "zz") & (x.side == -1)][COLS].copy()
            if not len(x):
                continue
            d = tf_frame(root, s, "15m")
            coin4 = d["close"].resample("4h").last()
            b15 = btc.reindex(d.index).ffill()
            t_in = x["t"] + BAR
            t_out = x["t"] + (x["hold_bars"].astype(int) + 1) * BAR
            p_in, p_out = b15.reindex(t_in, method="ffill").to_numpy(), b15.reindex(t_out, method="ffill").to_numpy()
            x["beta"] = [_beta(coin4, btc4, t) for t in x["t"]]
            x["btc_ret"] = p_out / p_in - 1
            x["t_out"] = t_out.to_numpy()
            trades.append(x)
        except Exception as e:
            print(f"  hedge15 {s}: пропуск ({e})", flush=True)
    print(f"  hedge15: монет со сделками {len(trades)}", flush=True)
    if trades:
        pd.concat(trades, ignore_index=True).to_parquet(part_path("hedge15"), index=False)
    if advs:
        pd.concat(advs, ignore_index=True).to_parquet(part_path("hedge15_adv"), index=False)


def hedge_r(side: np.ndarray, beta: np.ndarray, btc_ret: np.ndarray, rp: np.ndarray) -> np.ndarray:
    """Результат хеджа в R сделки: BTC против стороны на бету × номинал, комиссии тейкера на две стороны."""
    return (-side * beta * btc_ret - 2 * TAKER * np.abs(beta)) / rp


def _dd(r: pd.Series, t_out: pd.Series) -> float:
    c = np.cumsum(r.to_numpy()[np.argsort(t_out.to_numpy(), kind="stable")])
    return float(np.max(np.maximum.accumulate(np.r_[0.0, c])[1:] - c)) if len(c) else 0.0


def _row(z: pd.DataFrame, col: str) -> str:
    m = z.groupby(z.t.dt.to_period("M"))[col].sum()
    t = z[col].mean() / (z[col].std() / np.sqrt(len(z))) if len(z) > 2 and z[col].std() > 0 else np.nan
    return (f"{z[col].mean():+.3f}R (t {t:.1f}) · в мес. {m.mean():+.1f} / σ {m.std():.1f} / худший {m.min():+.1f} / "
            f"просадка {_dd(z[col], z.t_out):.1f}")


def report() -> None:
    tp, ap = all_parts("hedge15"), all_parts("hedge15_adv")
    print(f"\n===== HEDGE15: правило бота 15m (пологие шорты, по рынку, топ-{TOP}, 3R) без хеджа и с хеджем BTC на бету =====")
    if not tp or not ap:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in tp], ignore_index=True)
    adv = pd.concat([pd.read_parquet(p) for p in ap], ignore_index=True)
    for c in ("t", "t_out"):
        df[c] = pd.to_datetime(df[c], utc=True)
    adv["day"] = pd.to_datetime(adv["day"], utc=True)
    adv["rank"] = adv.groupby("day")["adv"].rank(ascending=False, method="first")
    df["day"] = df.t.dt.floor("D")
    df = df.merge(adv[["symbol", "day", "adv", "rank"]], on=["symbol", "day"], how="left")
    df["per"] = ""
    for p, (a, b) in PER.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    base = _bot_base(df.assign(confirm=False))
    g = base[(base.entry == "market") & (base.adv >= ADV_MIN) & (base.slope_atr <= MAX_SLOPE_ATR) & (base["rank"] <= TOP)]
    g = g[g.beta.notna() & g.btc_ret.notna()].copy()
    g["hedge_R"] = hedge_r(g.side.to_numpy(float), g.beta.to_numpy(float), g.btc_ret.to_numpy(float), g.risk_pct.to_numpy(float))
    g["R3h"] = g.R3 + g.hedge_R
    print(f"  сделок {len(g)}, бета медиана {g.beta.median():.2f}, средний результат хеджа {g.hedge_R.mean():+.3f}R, "
          f"комиссии хеджа в среднем {(2 * TAKER * g.beta.abs() / g.risk_pct).mean():.3f}R на сделку, "
          f"стоп медиана {g.risk_pct.median() * 100:.2f}% цены")
    for p in PER:
        z = g[g.per == p]
        if len(z) < 10:
            continue
        print(f"  {p:4s} n={len(z):5d}  без хеджа {_row(z, 'R3')}\n{'':16s}с хеджем  {_row(z, 'R3h')}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 400)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x])
    else:
        report()
