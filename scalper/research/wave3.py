"""
Волна 3: новые независимые гипотезы (тот же протокол IS -> VAL -> HOLDOUT, что и в волне 2).

    intraday_mom   доходность с начала суток UTC предсказывает остаток суток (моментум/реверсия)
    premium_rev    экстремальная премия perp к индексу возвращается к норме
    squeeze_break  пробой после сжатия диапазона
    oi_div         OI резко растёт при стоящей цене -> входим по направлению агрессоров

    python -m research.wave3 --root <data> --out <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import strategies as S
from .data import UNIVERSE
from .engine import _to_utc, state_machine
from .search import grid
from .wave2 import VAL_MIN_SHARPE, Data2, _price_z, _z, _zeros, gate, holdout, search, select


class Data3(Data2):
    def get(self, sym: str, tf: str, period: str) -> pd.DataFrame:
        key = (sym, tf, "p")
        if key not in self.cache:
            df = super().get(sym, tf, "full").copy()
            df["premium"] = premium_on_bars(sym, self.root, df.index, tf)
            self.cache[key] = df
        df = self.cache[key]
        if period == "full":
            return df
        from .wave2 import PERIODS
        a, b = PERIODS[period]
        return df[(df.index >= a) & (df.index < b)]


def premium_on_bars(symbol: str, root: Path, index: pd.DatetimeIndex, tf: str) -> np.ndarray:
    """Close премии perp к индексу на баре tf (известен на закрытии бара)."""
    p = Path(root) / f"{symbol}-premium-5m.parquet"
    if not p.exists():
        return np.full(len(index), np.nan)
    k = pd.read_parquet(p)
    k.index = _to_utc(k["open_time"])
    k = k[~k.index.duplicated()].sort_index()
    if tf != "5m":
        k = k[["open", "high", "low", "close"]].resample(
            {"15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h"}[tf], label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"})
    return k["close"].reindex(index).to_numpy(dtype="float64")


def intraday_mom(df, entry_h=16, thr=0.5, sign=1):
    """Решение на закрытии бара (entry_h-1):00 UTC по доходности с открытия суток, держим до 23:00 бара
    (выход на закрытии суток). sign=+1 моментум, -1 реверсия."""
    idx = df.index
    day = idx.floor("1D")
    day_open = df["open"].groupby(day).transform("first")
    r = np.log(df["close"] / day_open)
    z = r / (S._vol(df["close"], 24 * 30) * np.sqrt(entry_h))
    h = idx.hour.to_numpy()
    d = np.where(z > thr, sign, np.where(z < -thr, -sign, 0.0)).astype(float)
    d[h != entry_h - 1] = np.nan
    pos = pd.Series(d, index=idx).groupby(day).ffill().fillna(0.0).to_numpy().copy()
    pos[(h < entry_h - 1) | (h == 23)] = 0.0
    return pos, None


def premium_rev(df, win=96, thr=2.0, hold=16, sign=1):
    """sign=+1: премия аномально высокая -> шорт perp (ждём схождения), низкая -> лонг."""
    z = _z(df["premium"], win) * sign
    return state_machine(S._b(z < -thr), S._b(z > thr), S._b(z > 0), S._b(z < 0), hold), None


def squeeze_break(df, n=16, c=0.8, hold=32):
    hi = df["high"].rolling(n).max().shift(1)
    lo = df["low"].rolling(n).min().shift(1)
    bar_rng = (df["high"] - df["low"]).rolling(96 * 5, min_periods=96).mean()
    squeeze = ((hi - lo) / (bar_rng * np.sqrt(n))) < c
    cl = df["close"]
    return state_machine(S._b(squeeze & (cl > hi)), S._b(squeeze & (cl < lo)),
                         _zeros(len(df)), _zeros(len(df)), hold), None


def oi_div(df, k=4, thr=2.0, hold=12, sign=1, win=24 * 30):
    oi = df["oi"].replace(0, np.nan)
    doi = _z(np.log(oi / oi.shift(k)), win)
    flat = _price_z(df["close"], k).abs() < 0.5
    buy = df["taker_buy_volume"].rolling(k).sum()
    imb = (2 * buy - df["volume"].rolling(k).sum()) * sign
    gate_ = (doi > thr) & flat
    return state_machine(S._b(gate_ & (imb > 0)), S._b(gate_ & (imb < 0)),
                         _zeros(len(df)), _zeros(len(df)), hold), None


FAMILIES = {
    "intraday_mom": [("1h", intraday_mom, grid(entry_h=[8, 12, 16, 20, 22], thr=[0.0, 0.5, 1.0], sign=[1, -1]))],
    "premium_rev": [("15m", premium_rev, grid(win=[96, 672], thr=[2.0, 3.0], hold=[4, 16], sign=[1, -1])),
                    ("1h", premium_rev, grid(win=[24 * 7], thr=[2.0, 3.0], hold=[4, 12], sign=[1, -1]))],
    "squeeze_break": [("15m", squeeze_break, grid(n=[8, 16, 32], c=[0.6, 0.8, 1.0], hold=[8, 32]))],
    "oi_div": [("1h", oi_div, grid(k=[4, 12], thr=[2.0, 2.5], hold=[4, 12], sign=[1, -1]))],
}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 250)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--symbols", default=",".join(UNIVERSE))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Data3(Path(a.root), a.symbols.split(","))

    res = search(data, out, FAMILIES, panel=lambda: [], tag="wave3")
    print(f"\nВолна 3, всего конфигураций: {len(res)}")
    print(res.groupby("family")["sharpe"].describe()[["count", "50%", "max"]].round(2).to_string())
    fin = select(res)
    print("\nФИНАЛИСТЫ (IS):")
    print(fin[["family", "tf", "params", "sharpe", "ann_ret", "max_dd", "trades_per_day", "avg_gross",
               "bps_per_trade", "sym_pos_frac", "family_median_sharpe"]].round(3).to_string(index=False))
    val = gate(data, fin, out, FAMILIES, tag="wave3")
    print(f"\nVAL 2024-07..2025-06 (ворота: Sharpe>={VAL_MIN_SHARPE} при 6 б.п. и >0 при 10 б.п.):")
    print(val.round(3).to_string(index=False))
    passed = val[val["passed"]]
    if len(passed):
        print("\nHOLDOUT 2025-07..2026-09 (один прогон):")
        print(holdout(data, passed, out, FAMILIES, tag="wave3").round(3).to_string(index=False))
    else:
        print("\nНи один финалист не прошёл VAL — holdout не трогаем.")
