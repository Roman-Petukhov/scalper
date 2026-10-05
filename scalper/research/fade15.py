"""Против агрессоров на 15m: вход против свечи, которую толкнул сильный перевес агрессивных покупок или продаж.

Основа — Kitron & Wengrowicz (2026, arXiv 2608.21888): на 15m крипта откатывается против прошлой свечи у 90% пар
Binance, сильнее всего после свечей, которые толкнули агрессоры, и тем сильнее, чем больше перевес; к 4h эффект
исчезает. Средний перевес в статье — ~1.3 б.п. на сделку при издержках 5 б.п., поэтому проверяются только сильные
случаи (крупная свеча и перевес агрессоров в верхних 20% / 10% за прошлые 30 дней этой монеты), где ход больше.

Сделка: сигнал на закрытии свечи t, вход против неё —
    рынок    по закрытию свечи t (taker), выход по рынку на закрытии свечи t+h
    лимит    лимитка на 0.25 ATR дальше закрытия (maker), живёт 2 свечи; выход по рынку на закрытии t+h от входа
Издержки: taker 5.5 б.п., maker 2 б.п. на сторону (как у бота на Bybit); funding не учитывается (часы, не дни).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .shard import all_parts, mine, part_path
from .smc import _atr
from .tline import MAKER, PER, TAKER, htf_trend, tf_frame

HOLDS = (1, 2, 4, 8)            # выход через 15 мин, 30 мин, 1 ч, 2 ч
LIMIT_ATR = 0.25                # лимитка дальше закрытия сигнальной свечи
LIMIT_BARS = 2
WINDOW = 30 * 96                # 30 дней 15m-свечей для порога перевеса (только прошлое)
MIN_MOVE_ATR = 0.5              # не меньше полу-ATR тела — иначе ход меньше издержек


def coin_fades(root: Path, sym: str) -> pd.DataFrame:
    d = tf_frame(root, sym, "15m")
    if d is None or len(d) < WINDOW + 500:
        return pd.DataFrame()
    o, h, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    v = d["volume"].replace(0, np.nan)
    imb = (2 * d["taker_buy_volume"] / v - 1)                                    # -1 … +1, плюс — покупатели
    q80 = imb.abs().shift(1).rolling(WINDOW, min_periods=WINDOW // 3).quantile(0.8).to_numpy()
    q90 = imb.abs().shift(1).rolling(WINDOW, min_periods=WINDOW // 3).quantile(0.9).to_numpy()
    imb = imb.to_numpy()
    a = _atr(d).to_numpy()
    move = (c - o) / np.where(a > 0, a, np.nan)
    trend4 = htf_trend(d, "15m", "4h")
    sig = (np.sign(move) == np.sign(imb)) & (np.abs(imb) >= q80) & (np.abs(move) >= MIN_MOVE_ATR)
    idx = np.flatnonzero(sig[: len(c) - max(HOLDS) - LIMIT_BARS - 1])
    rows = []
    for t in idx:
        side = -int(np.sign(move[t]))                                            # против свечи
        row = {"symbol": sym, "t": d.index[t], "side": side, "move_atr": abs(move[t]), "imb": abs(imb[t]),
               "top10": abs(imb[t]) >= q90[t], "with_trend4": trend4[t] == side, "atr_bp": a[t] / c[t] * 1e4}
        for k in HOLDS:
            row[f"mkt{k}"] = (side * (c[t + k] / c[t] - 1) - 2 * TAKER) * 1e4
        px = c[t] - side * LIMIT_ATR * a[t]
        fill = next((j for j in range(t + 1, t + 1 + LIMIT_BARS) if (lo[j] <= px if side > 0 else h[j] >= px)), -1)
        row["filled"] = fill >= 0
        for k in HOLDS:
            row[f"lim{k}"] = (side * (c[fill + k] / px - 1) - MAKER - TAKER) * 1e4 if fill >= 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def collect(root: Path, syms15: list[str]) -> None:
    parts = []
    for s in mine(syms15):
        try:
            x = coin_fades(root, s)
            if len(x):
                parts.append(x)
        except Exception as e:
            print(f"  fade {s}: пропуск ({e})", flush=True)
    print(f"  fade15: монет со сделками {len(parts)}", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path("fade15"), index=False)


def _cell(x: pd.Series, t: pd.Series) -> str:
    x = x.dropna()
    if len(x) < 30:
        return f"n={len(x)}"
    day = t.loc[x.index].dt.floor("D")
    g = (x - x.mean()).groupby(day.to_numpy()).sum()
    tstat = x.sum() / np.sqrt((g ** 2).sum()) if (g ** 2).sum() > 0 else np.nan
    months = max(1, t.loc[x.index].dt.to_period("M").nunique())
    return f"{x.mean():+.1f} б.п. (t {tstat:.1f}, {np.mean(x > 0):.0%}, {len(x) / months:.0f}/мес)"


def report() -> None:
    parts = all_parts("fade15")
    print("\n===== FADE15: против агрессоров на 15m (Kitron & Wengrowicz 2026); ячейка — средний результат сделки "
          "после издержек в б.п. от цены (t по дням, прибыльных, сделок в месяц на 70 монет) =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["per"] = ""
    for p, (a, b) in PER.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    print(f"сигналов {len(df):,}, монет {df.symbol.nunique()}; издержки: рынок {2 * TAKER * 1e4:.0f} б.п. туда-обратно, "
          f"лимит + рынок {(MAKER + TAKER) * 1e4:.1f} б.п.; лимиток исполнилось {df.filled.mean():.0%}")
    groups = [("перевес в верхних 20%, тело >= 0.5 ATR", df),
              ("  тело >= 1 ATR", df[df.move_atr >= 1.0]),
              ("  тело >= 2 ATR", df[df.move_atr >= 2.0]),
              ("перевес в верхних 10%, тело >= 1 ATR", df[df.top10 & (df.move_atr >= 1.0)]),
              ("  + по тренду 4h (откат против тренда)", df[df.top10 & (df.move_atr >= 1.0) & df.with_trend4]),
              ("тело >= 1 ATR, по тренду 4h", df[(df.move_atr >= 1.0) & df.with_trend4]),
              ("тело >= 1 ATR, против тренда 4h", df[(df.move_atr >= 1.0) & ~df.with_trend4])]
    for kind, nm in (("mkt", "вход по рынку"), ("lim", f"вход лимиткой {LIMIT_ATR} ATR дальше закрытия")):
        rows = []
        for gname, g in groups:
            for k in HOLDS:
                col = f"{kind}{k}"
                rows.append({"сигнал": gname, "выход": f"через {k * 15} мин",
                             **{p: _cell(g.loc[g.per == p, col], g.t) for p in PER}})
        print(f"\n  --- {nm} ---")
        print(pd.DataFrame(rows).to_string(index=False))
    g = df[df.move_atr >= 1.0]
    print("\n  по годам (тело >= 1 ATR, лимитка, выход через 1 ч): " + ", ".join(
        f"{y}: {x.mean():+.1f} ({len(x)})" for y, x in g.groupby(g.t.dt.year)["lim4"] if len(x.dropna())))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 300)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols15", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols15.split(",") if x])
    else:
        report()
