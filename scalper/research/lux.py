"""
Индикатор «Trendlines with Breaks» (метод LuxAlgo, CC BY-NC-SA 4.0; здесь собственная реализация логики скрипта)
как торговая система: сигнал «B» вверх — лонг, «B» вниз — шорт/выход. Вход — по закрытию бара сигнала на 5m.

Сетка: length × mult × способ наклона (Atr/Stdev/Linreg) × ТФ сигнала (5m/15m) × фильтр старшего ТФ
(нет / состояние того же индикатора на 1h или 4h / наклон EMA50 на 1h или 4h) × выход (переворот / + стоп по
времени 4 ч). Протокол: IS 2022-01..2024-06 — отбор, VAL 2024-07..2025-06 — ворота, HOLDOUT 2025-07..2026-09.

    python -m research.lux --root <data> --out <dir> [--symbols ...]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from .data import UNIVERSE
from .engine import metrics, run
from .search import Data

PERIODS = {"is": ("2022-01-01", "2024-07-01"), "val": ("2024-07-01", "2025-07-01"),
           "ho": ("2025-07-01", "2026-10-01")}
LENGTHS = (7, 14, 21, 30, 50)
MULTS = (0.5, 1.0, 1.5, 2.0)
METHODS = ("atr", "stdev", "linreg")
SIG_TFS = ("5m", "15m")
FILTERS = ("none", "lux1h", "lux4h", "ema1h", "ema4h")
EXITS = ("sar", "sar_4h")
TF_RULE = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h"}
TF_MIN = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}
COST, STRESS = 6.0, 10.0
MIN_TRADES_PER_DAY = 0.5
VAL_MIN = 0.5


# ---------- индикатор ----------

def slope_series(df: pd.DataFrame, length: int, mult: float, method: str) -> np.ndarray:
    """Как в скрипте: Atr — ta.atr(length)/length*mult (RMA), Stdev — ta.stdev(close,length)/length*mult
    (смещённая оценка), Linreg — |cov(close, bar_index)| / var(bar_index) / 2 * mult."""
    c = df["close"]
    if method == "atr":
        pc = c.shift(1)
        tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
        s = tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean() / length * mult
    elif method == "stdev":
        s = c.rolling(length).std(ddof=0) / length * mult
    else:
        n = pd.Series(np.arange(len(c), dtype="float64"), index=c.index)
        cov = (c * n).rolling(length).mean() - c.rolling(length).mean() * n.rolling(length).mean()
        var_n = (length ** 2 - 1) / 12.0                       # дисперсия length подряд идущих индексов
        s = cov.abs() / var_n / 2 * mult
    return s.to_numpy(dtype="float64")


@njit(cache=True)
def _pivot(x, i, n, high):
    if i - n < 0 or i + n >= len(x):
        return False
    v = x[i]
    for k in range(i - n, i + n + 1):
        if k == i:
            continue
        if high:
            if (k < i and x[k] > v) or (k > i and x[k] >= v):
                return False
        else:
            if (k < i and x[k] < v) or (k > i and x[k] <= v):
                return False
    return True


@njit(cache=True)
def lux_signals(high, low, close, slope, length):
    """Сигналы «Upper Break» (upos растёт) и «Lower Break» (dnos растёт) — по логике скрипта.
    До первой вершины/впадины линии нет (в скрипте upper=0 даёт ложные ранние сигналы)."""
    m = len(close)
    up = np.zeros(m, np.bool_)
    dn = np.zeros(m, np.bool_)
    upper, lower = np.nan, np.nan
    s_ph, s_pl = 0.0, 0.0
    upos, dnos = 0, 0
    for t in range(m):
        c = t - length
        ph = _pivot(high, c, length, True)
        pl = _pivot(low, c, length, False)
        if ph:
            s_ph = slope[t]
            upper = high[c]
        elif not np.isnan(upper):
            upper -= s_ph
        if pl:
            s_pl = slope[t]
            lower = low[c]
        elif not np.isnan(lower):
            lower += s_pl
        prev_u, prev_d = upos, dnos
        if ph:
            upos = 0
        elif not np.isnan(upper) and close[t] > upper - s_ph * length:
            upos = 1
        if pl:
            dnos = 0
        elif not np.isnan(lower) and close[t] < lower + s_pl * length:
            dnos = 1
        up[t] = upos > prev_u
        dn[t] = dnos > prev_d
    return up, dn


@njit(cache=True)
def positions(up, dn, htf, use_filter, max_hold):
    """Сигнал вверх -> лонг (если фильтр разрешает), вниз -> шорт; против фильтра — только выход.
    Если старший ТФ развернулся против позиции — выход. max_hold>0 — стоп по времени (в барах)."""
    m = len(up)
    pos = np.zeros(m)
    cur = 0.0
    held = 0
    for t in range(m):
        if cur != 0.0:
            held += 1
            if max_hold > 0 and held >= max_hold:
                cur = 0.0
            elif use_filter and htf[t] != 0 and np.sign(htf[t]) != np.sign(cur):
                cur = 0.0
        if up[t] and not dn[t]:
            if (not use_filter) or htf[t] > 0:
                if cur <= 0.0:
                    cur, held = 1.0, 0
            elif cur < 0.0:
                cur = 0.0
        elif dn[t] and not up[t]:
            if (not use_filter) or htf[t] < 0:
                if cur >= 0.0:
                    cur, held = -1.0, 0
            elif cur > 0.0:
                cur = 0.0
        pos[t] = cur
    return pos


# ---------- таймфреймы ----------

def resample(df5: pd.DataFrame, tf: str) -> pd.DataFrame:
    if tf == "5m":
        return df5
    return df5.resample(TF_RULE[tf], label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()


def to_5m(values: pd.Series, tf: str, idx5: pd.DatetimeIndex) -> np.ndarray:
    """Значение бара старшего ТФ доступно с момента его закрытия: сопоставляем по времени закрытия 5m-бара."""
    if tf == "5m":
        return values.reindex(idx5).to_numpy()
    closes = pd.Series(values.to_numpy(), index=values.index + pd.Timedelta(minutes=TF_MIN[tf]))
    close5 = idx5 + pd.Timedelta(minutes=5)
    return closes.reindex(close5, method="ffill").to_numpy()


def htf_direction(df5: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "none":
        return np.zeros(len(df5))
    tf = kind[-2:]
    h = resample(df5, tf)
    if kind.startswith("lux"):
        up, dn = lux_signals(h["high"].to_numpy(), h["low"].to_numpy(), h["close"].to_numpy(),
                             slope_series(h, 14, 1.0, "atr"), 14)
        state = positions(up, dn, np.zeros(len(h)), False, 0)          # переворотное состояние индикатора
        s = pd.Series(state, index=h.index)
    else:
        ema = h["close"].ewm(span=50, adjust=False).mean()
        s = np.sign(ema.diff()).fillna(0.0)
    return np.nan_to_num(to_5m(s, tf, df5.index))


# ---------- прогон ----------

def run_symbol(df5: pd.DataFrame, cache: dict, sym: str, length, mult, method, sig_tf, filt, exit_mode, cost):
    key = (sym, length, mult, method, sig_tf)
    if key not in cache:
        d = resample(df5, sig_tf)
        up, dn = lux_signals(d["high"].to_numpy(), d["low"].to_numpy(), d["close"].to_numpy(),
                             slope_series(d, length, mult, method), length)
        cache[key] = (np.nan_to_num(to_5m(pd.Series(up, index=d.index).astype(float), sig_tf, df5.index)) > 0.5,
                      np.nan_to_num(to_5m(pd.Series(dn, index=d.index).astype(float), sig_tf, df5.index)) > 0.5)
    up, dn = cache[key]
    if sig_tf != "5m":
        # сигнал старшего ТФ должен сработать один раз — на 5m-баре, где закрылся его бар
        up = up & ~np.concatenate([[False], up[:-1]])
        dn = dn & ~np.concatenate([[False], dn[:-1]])
    fkey = (sym, "f", filt)
    if fkey not in cache:
        cache[fkey] = htf_direction(df5, filt)
    pos = positions(up, dn, cache[fkey], filt != "none", 48 if exit_mode == "sar_4h" else 0)
    return run(df5, pos, cost, 5, None)


def cut(s: pd.Series, per: str) -> pd.Series:
    a, b = PERIODS[per]
    return s[(s.index >= a) & (s.index < b)]


def evaluate(dfs: dict, cache: dict, cfg: dict, per: str, cost: float) -> dict:
    pnls, trades = [], 0
    for sym, df5 in dfs.items():
        r = run_symbol(df5, cache, sym, cost=cost, **cfg)
        p, ps = cut(r.pnl, per), cut(r.pos, per)
        pnls.append(p)
        prev = ps.shift(1).fillna(0.0)
        trades += int(((ps != 0) & (ps != prev)).sum())
    port = pd.concat(pnls, axis=1).fillna(0.0).mean(axis=1)
    m = metrics(port)
    return {"sharpe": m["sharpe"], "ann_ret": m["ann_ret"], "max_dd": m["max_dd"], "trades": trades,
            "trades_per_day": trades / max(m["days"], 1),
            "bps_per_trade": float(port.sum() * len(dfs) / max(trades, 1) * 1e4)}


def grid():
    for length, mult, method, sig_tf, filt, ex in itertools.product(LENGTHS, MULTS, METHODS, SIG_TFS, FILTERS, EXITS):
        yield {"length": length, "mult": mult, "method": method, "sig_tf": sig_tf, "filt": filt, "exit_mode": ex}


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
    data = Data(Path(a.root), a.symbols.split(","))
    full = {s: data.get(s, "5m", "full") for s in data.symbols}
    is_dfs = {s: cut(df, "is") for s, df in full.items()}
    cache_is, cache_full = {}, {}

    default = {"length": 14, "mult": 1.0, "method": "atr", "sig_tf": "5m", "filt": "none", "exit_mode": "sar"}
    print("ИНДИКАТОР КАК ЕСТЬ (length 14, Slope 1, Atr, сигнал 5m, без фильтра, переворот):")
    for per in PERIODS:
        for cost in (COST, STRESS):
            r = evaluate(full, cache_full, default, per, cost)
            print(f"  {per:>4} {cost:g} б.п.: Sharpe {r['sharpe']:+.2f}, в год {r['ann_ret']:+.1%}, "
                  f"сделок/день {r['trades_per_day']:.1f}, {r['bps_per_trade']:+.1f} б.п./сделку")

    rows, t0 = [], time.time()
    for i, cfg in enumerate(grid(), 1):
        r = evaluate(is_dfs, cache_is, cfg, "is", COST)
        rows.append({**cfg, **r})
        if i % 100 == 0:
            print(f"  перебор: {i} конфигураций ({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "lux_is.csv", index=False)
    print(f"\nВсего конфигураций: {len(res)}; Sharpe IS: медиана {res['sharpe'].median():.2f}, "
          f"доля > 0: {(res['sharpe'] > 0).mean():.0%}, максимум {res['sharpe'].max():.2f}")
    print("\nЧто влияет (медиана Sharpe IS по каждому параметру):")
    for col in ("length", "mult", "method", "sig_tf", "filt", "exit_mode"):
        print(f"  {col}: " + ", ".join(f"{k}={v:+.2f}" for k, v in res.groupby(col)["sharpe"].median().items()))
    top = res[res["trades_per_day"] >= MIN_TRADES_PER_DAY].sort_values("sharpe", ascending=False).head(10)
    print("\nТОП-10 по IS (не менее 0.5 сделки в день на портфель):")
    print(top.round(3).to_string(index=False))
    gate = []
    for _, f in top.iterrows():
        cfg = {k: f[k] for k in ("length", "mult", "method", "sig_tf", "filt", "exit_mode")}
        cfg["length"] = int(cfg["length"])
        v6 = evaluate(full, cache_full, cfg, "val", COST)
        v10 = evaluate(full, cache_full, cfg, "val", STRESS)
        row = {**cfg, "is": f["sharpe"], "val_6": v6["sharpe"], "val_10": v10["sharpe"],
               "val_bps": v6["bps_per_trade"], "val_tpd": v6["trades_per_day"]}
        row["passed"] = bool(v6["sharpe"] >= VAL_MIN and v10["sharpe"] > 0)
        if row["passed"]:
            h6 = evaluate(full, cache_full, cfg, "ho", COST)
            h10 = evaluate(full, cache_full, cfg, "ho", STRESS)
            row.update({"ho_6": h6["sharpe"], "ho_10": h10["sharpe"], "ho_ret": h6["ann_ret"], "ho_dd": h6["max_dd"]})
        gate.append(row)
    g = pd.DataFrame(gate)
    g.to_csv(out / "lux_gates.csv", index=False)
    print("\nВОРОТА VAL (Sharpe >= 0.5 при 6 б.п. и > 0 при 10 б.п.) -> HOLDOUT:")
    print(g.round(3).to_string(index=False))
