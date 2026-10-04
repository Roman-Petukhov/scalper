"""
Пробой «коробки» (боковика) на 15m — сетап с графика: узкий диапазон несколько часов, нижняя (верхняя) граница
протестирована несколько раз (там копятся стопы), затем первое закрытие за границей.

Коробка на баре t — бары [t-L, t-1]: верх = max(high), низ = min(low), высота h. Условия:
    сжатие   h / close <= comp * дневная волатильность (std 15m-доходностей за 30 дней * sqrt(96))
    касания  число отдельных подходов к границе (low <= низ + 0.1h; подходы разделены >= 1 ч) >= touches
    пробой   close[t] < низ - 0.1h (вниз) или close[t] > верх + 0.1h (вверх); close[t-1] — внутри коробки
Режимы: follow — по пробою (тейк tp*h, стоп sl*h от входа), fade — против пробоя (тейк tp*h, стоп sl*h).
Вход — рыночный по закрытию бара пробоя; одна позиция на монету; выход по барьеру или через max_hold.
Протокол IS -> VAL -> HOLDOUT, издержки 6 б.п./сторону (стресс 10).

    python -m research.boxbreak --root <data> --out <dir> [--symbols ...] [--show SYMBOL:YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from .breakout.lines import triple_barrier
from .data import UNIVERSE
from .engine import load
from .lux import PERIODS
from .search import Data

LOOKBACKS = (16, 32, 48)                 # 4 / 8 / 12 часов
COMPS = (0.5, 1.0, 1.5)
TOUCHES = (1, 2, 3)
MODES = ("follow", "fade")
EXITS = {"tp1_sl05_4h": (1.0, 0.5, 16), "tp2_sl1_12h": (2.0, 1.0, 48)}
TOL, BUF, GAP = 0.1, 0.1, 4
COST, STRESS = 6.0, 10.0
MIN_TRADES = 200


@njit(cache=True)
def _visits(x, lo, hi, level, upper, gap):
    """Сколько раз цена отдельно подходила к уровню: бары с x у уровня, разделённые не меньше чем gap баров."""
    cnt, last = 0, -10**9
    for j in range(lo, hi):
        near = x[j] >= level if upper else x[j] <= level
        if near:
            if j - last >= gap:
                cnt += 1
            last = j
    return cnt


@njit(cache=True)
def box_breaks(high, low, close, sigma, L, comp, min_touch):
    """Сигналы пробоя коробки: side (+1 вверх, -1 вниз, 0 нет) и высота коробки на баре сигнала."""
    m = len(close)
    side = np.zeros(m, np.int8)
    height = np.full(m, np.nan)
    for t in range(L, m):
        hi, lo = -np.inf, np.inf
        for j in range(t - L, t):
            hi = max(hi, high[j])
            lo = min(lo, low[j])
        h = hi - lo
        if not (h > 0) or np.isnan(sigma[t - 1]) or h / close[t - 1] > comp * sigma[t - 1]:
            continue
        if close[t] < lo - BUF * h:
            if _visits(low, t - L, t, lo + TOL * h, False, GAP) >= min_touch:
                side[t], height[t] = -1, h
        elif close[t] > hi + BUF * h:
            if _visits(high, t - L, t, hi - TOL * h, True, GAP) >= min_touch:
                side[t], height[t] = 1, h
    return side, height


def bars15(df5: pd.DataFrame) -> pd.DataFrame:
    df = df5.resample("15min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["close"])
    lr = np.log(df["close"]).diff()
    df["sigma"] = lr.rolling(96 * 30, min_periods=96 * 10).std() * np.sqrt(96)
    return df


def trades(df: pd.DataFrame, cache: dict, sym: str, L, comp, touches, mode, exit_name, cost) -> pd.DataFrame:
    key = (sym, L, comp, touches)
    if key not in cache:
        cache[key] = box_breaks(df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy(),
                                df["sigma"].to_numpy(), L, comp, touches)
    side, height = cache[key]
    idx = np.flatnonzero(side != 0)
    if not len(idx):
        return pd.DataFrame(columns=["t", "r"])
    tp, sl, hold = EXITS[exit_name]
    s = side[idx].astype(np.float64) * (1.0 if mode == "follow" else -1.0)
    ret, bars = triple_barrier(idx.astype(np.int64), s, df["open"].to_numpy(), df["high"].to_numpy(),
                               df["low"].to_numpy(), df["close"].to_numpy(), np.nan_to_num(height), tp, sl, hold, cost)
    keep, busy_until = [], -1                       # одна позиция на монету
    for k, t in enumerate(idx):
        if t > busy_until and np.isfinite(ret[k]):
            keep.append(k)
            busy_until = t + bars[k]
    keep = np.asarray(keep, dtype=np.int64)
    return pd.DataFrame({"t": df.index[idx[keep]], "r": ret[keep], "side": side[idx[keep]]})


def evaluate(frames: dict, cache: dict, cfg: dict, cost: float) -> dict:
    tr = pd.concat([trades(df, cache, s, cost=cost, **cfg).assign(symbol=s) for s, df in frames.items()])
    out = {}
    for per, (a, b) in PERIODS.items():
        x = tr[(tr["t"] >= a) & (tr["t"] < b)]["r"]
        days = (pd.Timestamp(b) - pd.Timestamp(a)).days
        out[f"{per}_n"] = len(x)
        out[f"{per}_bps"] = x.mean() * 1e4 if len(x) else np.nan
        out[f"{per}_t"] = x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 30 else np.nan
        out[f"{per}_per_day"] = len(x) / days
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 260)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--symbols", default=",".join(UNIVERSE))
    ap.add_argument("--show", default="SOLUSDT:2026-10-02")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Data(Path(a.root), a.symbols.split(","))
    frames = {s: bars15(data.get(s, "5m", "full")) for s in data.symbols}
    cache: dict = {}

    sym, day = a.show.split(":")
    if (Path(a.root) / f"{sym}-5m.parquet").exists():
        print(f"Пример с графика: {sym} {day} — пробои коробки (все L, comp, касания), время UTC:")
        df = bars15(load(sym, Path(a.root), "5m"))           # без обрезки по концу HOLDOUT
        rows = []
        for L, comp, tch in itertools.product(LOOKBACKS, COMPS, TOUCHES):
            side, h = box_breaks(df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy(),
                                 df["sigma"].to_numpy(), L, comp, tch)
            m = (df.index.strftime("%Y-%m-%d") == day) & (side != 0)
            for t in np.flatnonzero(m):
                rows.append({"L": L, "comp": comp, "touches": tch, "time_utc": df.index[t], "side": int(side[t]),
                             "close": df["close"].iloc[t], "box_h": round(float(h[t]), 3)})
        print(pd.DataFrame(rows).to_string(index=False) if rows else "  сигналов нет")

    rows = []
    for L, comp, tch, mode, ex in itertools.product(LOOKBACKS, COMPS, TOUCHES, MODES, EXITS):
        cfg = {"L": L, "comp": comp, "touches": tch, "mode": mode, "exit_name": ex}
        rows.append({**cfg, **evaluate(frames, cache, cfg, COST)})
    res = pd.DataFrame(rows)
    res.to_csv(out / "boxbreak.csv", index=False)
    print(f"\nКонфигураций {len(res)}. Средний результат сделки IS (б.п., 6 б.п./сторону): медиана "
          f"{res['is_bps'].median():.1f}, доля > 0: {(res['is_bps'] > 0).mean():.0%}")
    for col in ("L", "comp", "touches", "mode", "exit_name"):
        print(f"  {col}: " + ", ".join(f"{k}={v:+.1f}" for k, v in res.groupby(col)["is_bps"].median().items()))
    top = res[res["is_n"] >= MIN_TRADES].sort_values("is_t", ascending=False).head(10)
    print(f"\nТОП-10 по t-статистике IS (не меньше {MIN_TRADES} сделок), все периоды при 6 б.п.:")
    cols = ["L", "comp", "touches", "mode", "exit_name"] + [f"{p}_{k}" for p in ("is", "val", "ho")
                                                              for k in ("n", "bps", "t")]
    print(top[cols].round(2).to_string(index=False))
    print("\nВорота: VAL > 0 при 10 б.п. и t(VAL) > 1.5 -> HOLDOUT при 10 б.п.:")
    g = []
    for _, f in top.iterrows():
        cfg = {k: f[k] for k in ("L", "comp", "touches", "mode", "exit_name")}
        cfg["L"], cfg["touches"], cfg["comp"] = int(cfg["L"]), int(cfg["touches"]), float(cfg["comp"])
        st = evaluate(frames, cache, cfg, STRESS)
        ok = bool(st["val_bps"] > 0 and f["val_t"] > 1.5)
        g.append({**cfg, "is_bps": f["is_bps"], "val_bps_10": st["val_bps"], "val_t": f["val_t"], "passed": ok,
                  "ho_bps_10": st["ho_bps"] if ok else np.nan, "ho_t": f["ho_t"] if ok else np.nan,
                  "ho_n": f["ho_n"]})
    print(pd.DataFrame(g).round(2).to_string(index=False))
