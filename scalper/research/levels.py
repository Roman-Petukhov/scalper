"""
Горизонтальные уровни по методу LuxAlgo «Support and Resistance Levels with Breaks» (собственная реализация логики):
    сопротивление = fixnan(pivothigh(n, n)[1]), поддержка = fixnan(pivotlow(n, n)[1]) — последняя подтверждённая
    вершина/впадина, известная со следующего бара после подтверждения;
    osc = 100 * (EMA5(объём) - EMA10(объём)) / EMA10(объём);
    «B» вверх: close пересёк сопротивление снизу, нет длинной нижней тени (open-low <= close-open), osc > порога;
    «Bull Wick»: то же пересечение с длинной нижней тенью (объём не проверяется). Вниз — зеркально.
Режимы торговли (все направления — от закрытия бара сигнала, вход по закрытию):
    break_B    лонг по «B» вверх, шорт по «B» вниз (метки скрипта с объёмом)
    break_all  все метки пробоя (B + Wick)
    fade3/6    ложный пробой: после пересечения уровня (osc > порога) close вернулся за уровень в течение 3/6 баров
               -> позиция против пробоя
    bounce     отбой: тень проколола уровень, закрытие осталось по эту сторону, прошлый бар тоже был по эту сторону,
               osc > порога -> по отбою
Выход: переворот по противоположному сигналу (sar) или + стоп по времени 1 ч / 4 ч. Фильтр: нет / наклон EMA50 на 4h.
Протокол как в research.lux: IS 2022-01..2024-06 — отбор, VAL — ворота, HOLDOUT — экзамен. Издержки 6 б.п./сторону.

    python -m research.levels --root <data> --out <dir> [--symbols ...]
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from .data import UNIVERSE
from .engine import metrics, run
from .lux import COST, MIN_TRADES_PER_DAY, PERIODS, STRESS, TF_RULE, VAL_MIN, _pivot, cut, htf_direction, positions, \
    to_5m
from .search import Data

PIVOTS = (5, 10, 15, 20, 30)
VOL_THRESH = (0.0, 20.0, 50.0)
SIG_TFS = ("5m", "15m", "1h")
MODES = ("break_B", "break_all", "fade3", "fade6", "bounce")
FILTERS = ("none", "ema4h")
EXITS = {"sar": 0, "t1h": 12, "t4h": 48}               # стоп по времени в 5m-барах
DEFAULT = {"n": 15, "thr": 20.0, "sig_tf": "5m", "mode": "break_B", "filt": "none", "exit_mode": "sar"}


@njit(cache=True)
def sr_levels(high, low, n):
    """Уровни скрипта: значение на баре t — последняя вершина/впадина, подтверждённая не позже бара t-1."""
    m = len(high)
    res = np.full(m, np.nan)
    sup = np.full(m, np.nan)
    cur_r, cur_s = np.nan, np.nan
    for t in range(1, m):
        c = t - 1 - n
        if c >= 0:
            if _pivot(high, c, n, True):
                cur_r = high[c]
            if _pivot(low, c, n, False):
                cur_s = low[c]
        res[t] = cur_r
        sup[t] = cur_s
    return res, sup


@njit(cache=True)
def sr_signals(o, h, lo, c, osc, res, sup, thr, mode_id):
    """mode_id: 0 break_B, 1 break_all, 2 fade3, 3 fade6, 4 bounce. Возвращает (лонг, шорт)."""
    m = len(c)
    up = np.zeros(m, np.bool_)
    dn = np.zeros(m, np.bool_)
    k_fade = 3 if mode_id == 2 else 6
    last_up_t, last_up_lvl = -10**9, np.nan
    last_dn_t, last_dn_lvl = -10**9, np.nan
    for t in range(1, m):
        if np.isnan(res[t]) or np.isnan(res[t - 1]) or np.isnan(sup[t]) or np.isnan(sup[t - 1]):
            continue
        x_up = c[t] > res[t] and c[t - 1] <= res[t - 1]
        x_dn = c[t] < sup[t] and c[t - 1] >= sup[t - 1]
        vol_ok = osc[t] > thr
        wick_up = o[t] - lo[t] > c[t] - o[t]
        wick_dn = o[t] - c[t] < h[t] - o[t]
        if mode_id == 0:
            up[t] = x_up and not wick_up and vol_ok
            dn[t] = x_dn and not wick_dn and vol_ok
        elif mode_id == 1:
            up[t] = x_up and (wick_up or vol_ok)
            dn[t] = x_dn and (wick_dn or vol_ok)
        elif mode_id <= 3:
            # ложный пробой: вернулись за уровень пробоя в течение k баров -> против пробоя
            if t - last_up_t <= k_fade and c[t] < last_up_lvl:
                dn[t] = True
                last_up_t = -10**9
            if t - last_dn_t <= k_fade and c[t] > last_dn_lvl:
                up[t] = True
                last_dn_t = -10**9
            if x_up and vol_ok:
                last_up_t, last_up_lvl = t, res[t]
            if x_dn and vol_ok:
                last_dn_t, last_dn_lvl = t, sup[t]
        else:
            up[t] = lo[t] < sup[t] and c[t] > sup[t] and c[t - 1] > sup[t - 1] and vol_ok
            dn[t] = h[t] > res[t] and c[t] < res[t] and c[t - 1] < res[t - 1] and vol_ok
    return up, dn


def resample_ohlcv(df5: pd.DataFrame, tf: str) -> pd.DataFrame:
    if tf == "5m":
        return df5
    return df5.resample(TF_RULE[tf], label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


def volume_osc(v: pd.Series) -> np.ndarray:
    short, long = v.ewm(span=5, adjust=False).mean(), v.ewm(span=10, adjust=False).mean()
    return (100 * (short - long) / long.replace(0.0, np.nan)).to_numpy(dtype="float64")


def run_symbol(df5: pd.DataFrame, cache: dict, sym: str, n, thr, sig_tf, mode, filt, exit_mode, cost):
    key = (sym, n, thr, sig_tf, mode)
    if key not in cache:
        d = resample_ohlcv(df5, sig_tf)
        lk = (sym, "lvl", n, sig_tf)
        if lk not in cache:
            cache[lk] = sr_levels(d["high"].to_numpy(), d["low"].to_numpy(), n) + (volume_osc(d["volume"]),)
        res, sup, osc = cache[lk]
        up, dn = sr_signals(d["open"].to_numpy(), d["high"].to_numpy(), d["low"].to_numpy(), d["close"].to_numpy(),
                            np.nan_to_num(osc, nan=-1e9), res, sup, float(thr), MODES.index(mode))
        u = np.nan_to_num(to_5m(pd.Series(up, index=d.index).astype(float), sig_tf, df5.index)) > 0.5
        v = np.nan_to_num(to_5m(pd.Series(dn, index=d.index).astype(float), sig_tf, df5.index)) > 0.5
        if sig_tf != "5m":                                  # сигнал старшего ТФ — один раз, на баре его закрытия
            u = u & ~np.concatenate([[False], u[:-1]])
            v = v & ~np.concatenate([[False], v[:-1]])
        cache[key] = (u, v)
    up, dn = cache[key]
    fkey = (sym, "f", filt)
    if fkey not in cache:
        cache[fkey] = htf_direction(df5, filt)
    pos = positions(up, dn, cache[fkey], filt != "none", EXITS[exit_mode])
    return run(df5, pos, cost, 5, None)


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
    for n, thr, tf, mode, filt, ex in itertools.product(PIVOTS, VOL_THRESH, SIG_TFS, MODES, FILTERS, EXITS):
        yield {"n": n, "thr": thr, "sig_tf": tf, "mode": mode, "filt": filt, "exit_mode": ex}


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

    for tf in ("5m", "15m"):
        cfg = {**DEFAULT, "sig_tf": tf}
        print(f"ИНДИКАТОР КАК ЕСТЬ (pivot 15/15, объём > 20%, метки B, сигнал {tf}, переворот):")
        for per in PERIODS:
            for cost in (COST, STRESS):
                r = evaluate(full, cache_full, cfg, per, cost)
                print(f"  {per:>4} {cost:g} б.п.: Sharpe {r['sharpe']:+.2f}, в год {r['ann_ret']:+.1%}, "
                      f"сделок/день {r['trades_per_day']:.1f}, {r['bps_per_trade']:+.1f} б.п./сделку")

    rows, t0 = [], time.time()
    for i, cfg in enumerate(grid(), 1):
        rows.append({**cfg, **evaluate(is_dfs, cache_is, cfg, "is", COST)})
        if i % 150 == 0:
            print(f"  перебор: {i} конфигураций ({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "levels_is.csv", index=False)
    print(f"\nВсего конфигураций: {len(res)}; Sharpe IS: медиана {res['sharpe'].median():.2f}, "
          f"доля > 0: {(res['sharpe'] > 0).mean():.0%}, максимум {res['sharpe'].max():.2f}")
    print("\nЧто влияет (медиана Sharpe IS по каждому параметру):")
    for col in ("n", "thr", "sig_tf", "mode", "filt", "exit_mode"):
        print(f"  {col}: " + ", ".join(f"{k}={v:+.2f}" for k, v in res.groupby(col)["sharpe"].median().items()))
    print("\nЛучший Sharpe IS по режиму:")
    print(res.loc[res.groupby("mode")["sharpe"].idxmax()].round(3).to_string(index=False))
    top = res[res["trades_per_day"] >= MIN_TRADES_PER_DAY].sort_values("sharpe", ascending=False).head(10)
    print("\nТОП-10 по IS (не менее 0.5 сделки в день на портфель):")
    print(top.round(3).to_string(index=False))
    gate = []
    for _, f in top.iterrows():
        cfg = {k: f[k] for k in ("n", "thr", "sig_tf", "mode", "filt", "exit_mode")}
        cfg["n"], cfg["thr"] = int(cfg["n"]), float(cfg["thr"])
        v6 = evaluate(full, cache_full, cfg, "val", COST)
        v10 = evaluate(full, cache_full, cfg, "val", STRESS)
        row = {**cfg, "is": f["sharpe"], "val_6": v6["sharpe"], "val_10": v10["sharpe"],
               "val_bps": v6["bps_per_trade"], "val_tpd": v6["trades_per_day"]}
        row["passed"] = bool(v6["sharpe"] >= VAL_MIN and v10["sharpe"] > 0)
        if row["passed"]:
            h6 = evaluate(full, cache_full, cfg, "ho", COST)
            h10 = evaluate(full, cache_full, cfg, "ho", STRESS)
            row.update({"ho_6": h6["sharpe"], "ho_10": h10["sharpe"], "ho_ret": h6["ann_ret"], "ho_dd": h6["max_dd"],
                        "ho_bps": h6["bps_per_trade"]})
        gate.append(row)
    g = pd.DataFrame(gate)
    g.to_csv(out / "levels_gates.csv", index=False)
    print("\nВОРОТА VAL (Sharpe >= 0.5 при 6 б.п. и > 0 при 10 б.п.) -> HOLDOUT:")
    print(g.round(3).to_string(index=False))
