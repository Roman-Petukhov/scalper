"""
Уровни по зонам ликвидности (профиль объёма), а не по теням.

Объём каждого бара относится к его средневзвешенной цене VWAP = оборот / объём — туда, где сделки реально шли
(экстремумы теней в построение не входят). Профиль — скользящее окно 1/3/7 дней в лог-ценовых ячейках шириной
10/25 б.п. Раз в час (по барам до текущего) зоны пересчитываются: зона — подряд идущие ячейки в пределах ±15% от
цены, объём которых не меньше mult × среднего объёма непустых ячеек этого диапазона.
Для каждого бара по закрытию предыдущего: зона, в которой цена (zc), ближайшая зона выше (za) и ниже (zb).
Режимы (вход по закрытию бара сигнала):
    bounce   отбой от края: low зашёл в зону ниже, а close остался выше её верхнего края -> лонг; зеркально шорт
    break    закрытие за дальним краем зоны (выход из текущей зоны или пролёт зоны выше/ниже целиком) -> по пробою
    fade3/6  ложный пробой: после break в течение 3/6 баров close вернулся за пробитый край -> против пробоя
Фильтр объёма: off или osc = 100*(EMA5-EMA10)/EMA10 объёма > 20 (как в скрипте LuxAlgo). Выход и фильтр 4h — как
в research.levels. Протокол IS -> VAL -> HOLDOUT, издержки 6 б.п./сторону (стресс 10).

    python -m research.vpzones --root <data> --out <dir> [--symbols ...]
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
from .levels import EXITS, MIN_TRADES_PER_DAY, PERIODS, COST, STRESS, cut, evaluate, gate, volume_osc
from .lux import TF_MIN, TF_RULE, htf_direction, positions, to_5m
from .engine import run
from .search import Data

WINDOWS_MIN = {"1d": 1440, "3d": 4320, "7d": 10080}
BIN_BPS = (10.0, 25.0)
HVN_MULT = (1.5, 2.5)
SIG_TFS = ("5m", "15m", "1h")
MODES = ("bounce", "break", "fade3", "fade6")
VOL = {"off": -1e9, "20": 20.0}
FILTERS = ("none", "ema4h")
RECALC_MIN = 60
RANGE_PCT = 0.15
MAX_ZONES = 128


@njit(cache=True)
def vp_zones(vwap, vol, close, w, bin_bps, mult, recalc, range_pct):
    """Зоны профиля объёма; на баре t используются только бары < t. Возвращает края зон выше/ниже/вокруг c[t-1]."""
    m = len(close)
    lb = np.log1p(bin_bps / 1e4)
    idx = np.empty(m, np.int64)
    base = 1 << 60
    for t in range(m):
        k = int(np.floor(np.log(vwap[t]) / lb))
        idx[t] = k
        if k < base:
            base = k
    top = 0
    for t in range(m):
        idx[t] -= base
        if idx[t] > top:
            top = idx[t]
    prof = np.zeros(top + 1)
    half = int(np.ceil(np.log1p(range_pct) / lb))
    z_lo = np.empty(MAX_ZONES)
    z_hi = np.empty(MAX_ZONES)
    nz = 0
    out = np.full((6, m), np.nan)            # za_lo, za_hi, zb_lo, zb_hi, zc_lo, zc_hi
    for t in range(1, m):
        prof[idx[t - 1]] += vol[t - 1]
        if t - 1 - w >= 0:
            prof[idx[t - 1 - w]] -= vol[t - 1 - w]
        if t < w:
            continue
        if t % recalc == 0 or nz == 0:
            cur = int(np.floor(np.log(close[t - 1]) / lb)) - base
            a, b = max(cur - half, 0), min(cur + half, top)
            s, cnt = 0.0, 0
            for i in range(a, b + 1):
                if prof[i] > 1e-12:
                    s += prof[i]
                    cnt += 1
            nz = 0
            if cnt > 0:
                thr = mult * s / cnt
                i = a
                while i <= b and nz < MAX_ZONES:
                    if prof[i] >= thr:
                        j = i
                        while j + 1 <= b and prof[j + 1] >= thr:
                            j += 1
                        z_lo[nz] = np.exp((i + base) * lb)
                        z_hi[nz] = np.exp((j + 1 + base) * lb)
                        nz += 1
                        i = j + 1
                    else:
                        i += 1
        p = close[t - 1]
        best_a, best_b = np.inf, -np.inf
        for q in range(nz):
            if z_lo[q] <= p <= z_hi[q]:
                out[4, t], out[5, t] = z_lo[q], z_hi[q]
            elif z_lo[q] > p and z_lo[q] < best_a:
                best_a = z_lo[q]
                out[0, t], out[1, t] = z_lo[q], z_hi[q]
            elif z_hi[q] < p and z_hi[q] > best_b:
                best_b = z_hi[q]
                out[2, t], out[3, t] = z_lo[q], z_hi[q]
    return out


@njit(cache=True)
def zone_signals(h, lo, c, osc, z, thr, mode_id):
    """mode_id: 0 bounce, 1 break, 2 fade3, 3 fade6. Возвращает (лонг, шорт)."""
    m = len(c)
    up = np.zeros(m, np.bool_)
    dn = np.zeros(m, np.bool_)
    k_fade = 3 if mode_id == 2 else 6
    last_up_t, last_up_lvl = -10**9, np.nan
    last_dn_t, last_dn_lvl = -10**9, np.nan
    for t in range(1, m):
        za_lo, za_hi, zb_lo, zb_hi, zc_lo, zc_hi = z[0, t], z[1, t], z[2, t], z[3, t], z[4, t], z[5, t]
        vol_ok = osc[t] > thr
        if mode_id == 0:
            up[t] = (not np.isnan(zb_hi)) and lo[t] <= zb_hi and c[t] > zb_hi and vol_ok
            dn[t] = (not np.isnan(za_lo)) and h[t] >= za_lo and c[t] < za_lo and vol_ok
            continue
        lvl_up = np.nan
        if not np.isnan(zc_hi) and c[t] > zc_hi:
            lvl_up = zc_hi
        if not np.isnan(za_hi) and c[t] > za_hi:
            lvl_up = za_hi
        lvl_dn = np.nan
        if not np.isnan(zc_lo) and c[t] < zc_lo:
            lvl_dn = zc_lo
        if not np.isnan(zb_lo) and c[t] < zb_lo:
            lvl_dn = zb_lo
        b_up = (not np.isnan(lvl_up)) and vol_ok
        b_dn = (not np.isnan(lvl_dn)) and vol_ok
        if mode_id == 1:
            up[t], dn[t] = b_up, b_dn
            continue
        if t - last_up_t <= k_fade and c[t] < last_up_lvl:
            dn[t] = True
            last_up_t = -10**9
        if t - last_dn_t <= k_fade and c[t] > last_dn_lvl:
            up[t] = True
            last_dn_t = -10**9
        if b_up:
            last_up_t, last_up_lvl = t, lvl_up
        if b_dn:
            last_dn_t, last_dn_lvl = t, lvl_dn
    return up, dn


def resample_vp(df5: pd.DataFrame, tf: str) -> pd.DataFrame:
    if tf == "5m":
        d = df5
    else:
        d = df5.resample(TF_RULE[tf], label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
             "quote_volume": "sum"}).dropna(subset=["close"])
    vwap = (d["quote_volume"] / d["volume"]).where(d["volume"] > 0)
    vwap = vwap.where((vwap >= d["low"]) & (vwap <= d["high"]), d["close"])   # защита от битых строк
    return d.assign(vwap=vwap.fillna(d["close"]))


def run_symbol(df5: pd.DataFrame, cache: dict, sym: str, win, bin_bps, mult, sig_tf, mode, vol, filt, exit_mode,
               cost):
    key = (sym, win, bin_bps, mult, sig_tf, mode, vol)
    if key not in cache:
        zk = (sym, "z", win, bin_bps, mult, sig_tf)
        if zk not in cache:
            d = resample_vp(df5, sig_tf)
            w = WINDOWS_MIN[win] // TF_MIN[sig_tf]
            rc = max(RECALC_MIN // TF_MIN[sig_tf], 1)
            z = vp_zones(d["vwap"].to_numpy(dtype="float64"), d["volume"].to_numpy(dtype="float64"),
                         d["close"].to_numpy(dtype="float64"), w, float(bin_bps), float(mult), rc, RANGE_PCT)
            cache[zk] = (d, z, np.nan_to_num(volume_osc(d["volume"]), nan=-1e9))
        d, z, osc = cache[zk]
        up, dn = zone_signals(d["high"].to_numpy(), d["low"].to_numpy(), d["close"].to_numpy(), osc, z,
                              VOL[vol], MODES.index(mode))
        u = np.nan_to_num(to_5m(pd.Series(up, index=d.index).astype(float), sig_tf, df5.index)) > 0.5
        v = np.nan_to_num(to_5m(pd.Series(dn, index=d.index).astype(float), sig_tf, df5.index)) > 0.5
        if sig_tf != "5m":
            u = u & ~np.concatenate([[False], u[:-1]])
            v = v & ~np.concatenate([[False], v[:-1]])
        cache[key] = (u, v)
    up, dn = cache[key]
    fkey = (sym, "f", filt)
    if fkey not in cache:
        cache[fkey] = htf_direction(df5, filt)
    pos = positions(up, dn, cache[fkey], filt != "none", EXITS[exit_mode])
    return run(df5, pos, cost, 5, None)


KEYS = ["win", "bin_bps", "mult", "sig_tf", "mode", "vol", "filt", "exit_mode"]


def grid():
    for vals in itertools.product(WINDOWS_MIN, BIN_BPS, HVN_MULT, SIG_TFS, MODES, VOL, FILTERS, EXITS):
        yield dict(zip(KEYS, vals))


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

    rows, t0 = [], time.time()
    for i, cfg in enumerate(grid(), 1):
        rows.append({**cfg, **evaluate(is_dfs, cache_is, cfg, "is", COST, run_symbol)})
        if i % 150 == 0:
            print(f"  перебор: {i} конфигураций ({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "vpzones_is.csv", index=False)
    print(f"\nВсего конфигураций: {len(res)}; Sharpe IS: медиана {res['sharpe'].median():.2f}, "
          f"доля > 0: {(res['sharpe'] > 0).mean():.0%}, максимум {res['sharpe'].max():.2f}")
    print("\nЧто влияет (медиана Sharpe IS по каждому параметру):")
    for col in KEYS:
        print(f"  {col}: " + ", ".join(f"{k}={v:+.2f}" for k, v in res.groupby(col)["sharpe"].median().items()))
    print("\nЛучший Sharpe IS по режиму:")
    print(res.loc[res.groupby("mode")["sharpe"].idxmax()].round(3).to_string(index=False))
    top = res[res["trades_per_day"] >= MIN_TRADES_PER_DAY].sort_values("sharpe", ascending=False).head(10)
    print("\nТОП-10 по IS (не менее 0.5 сделки в день на портфель):")
    print(top.round(3).to_string(index=False))
    g = gate(top, KEYS, full, cache_full, run_symbol, {"bin_bps": float, "mult": float})
    g.to_csv(out / "vpzones_gates.csv", index=False)
    print("\nВОРОТА VAL (Sharpe >= 0.5 при 6 б.п. и > 0 при 10 б.п.) -> HOLDOUT:")
    print(g.round(3).to_string(index=False))
