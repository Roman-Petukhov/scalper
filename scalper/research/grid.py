"""
Сетка лимитных ордеров: (A) как самостоятельная стратегия вокруг «справедливой» цены, (B) как способ входа
в oi_liq (лесенка лимитов во время каскада вместо одного рыночного входа).

A. Адаптивная сетка (5m, Bybit, 16 монет). Центр C — EMA за сутки / VWAP за сутки / POC профиля объёма за сутки,
шаг S = m * ATR (часовой эквивалент), N уровней на сторону, каждая ступень — 1/N капитала. Пока позиция пустая,
сетка перецентрируется на текущий центр; с позицией C и S заморожены. Лонговая ступень k (покупка по C - k*S)
закрывается тейком на шаг выше (C - (k-1)*S), шортовая — зеркально. Стоп: цена ушла за
C -/+ (N+1)*S — закрываем всё рыночно (даже если позиция неполная), перецентрирование. Фильтр «только боковик»: новые ступени открываются,
лишь пока коэффициент эффективности движения за сутки < 0.3.
Исполнение консервативное: лимит исполнен, только если цена прошла сквозь уровень; тейк ступени не исполняется
в баре её открытия; внутри бара сначала исполняются ордера, увеличивающие позицию, затем стоп, затем тейки;
из пустой позиции в баре, задевшем обе стороны, исполняется только сторона, против которой закрылся бар.
Комиссии: maker 2 б.п. на лимиты, стоп — taker 5.5 б.п. Funding учитывается.

B. Лесенка для oi_liq (те же параметры сигнала, выход через 4 ч рыночно): K лимитов по цене сигнала
и ниже (для лонга) с шагом s * ATR1h, каждый 1/K капитала, живут до выхода. Сравнение с рыночным входом.

    python -m research.grid --root <bybit data> --out <dir>
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from .data import UNIVERSE
from .engine import metrics
from .lux import PERIODS, VAL_MIN, cut
from .search import Data
from .wave2 import Data2, oi_price

MAKER, TAKER = 2.0, 5.5
DAY = 288                                            # 5m-баров в сутках
ANCHORS = ("ema1d", "vwap1d", "poc1d")
STEPS = (0.5, 1.0, 2.0)
LEVELS = (3, 5, 8)
FILTERS = ("none", "range")
ER_MAX = 0.3


# ---------- A: самостоятельная сетка ----------

@njit(cache=True)
def grid_machine(open_, high, low, close, funding, anchor, step, active, n_lv, maker_bps, taker_bps):
    """anchor/step/active — значения на закрытии бара t-1 используются для ордеров бара t.
    Возвращает доходность по барам (на капитал), позицию (доля капитала), число исполнений, тейков, стопов."""
    m = len(close)
    ret = np.zeros(m)
    q_end = np.zeros(m)
    mk, tk = maker_bps * 1e-4, taker_bps * 1e-4
    k = 0                                            # ступеней в позиции: >0 лонг, <0 шорт
    c0, s0 = np.nan, np.nan
    opened = np.full(n_lv + 1, -1)                   # бар открытия каждой ступени (1..n_lv)
    fills, tps, stops = 0, 0, 0
    u = 1.0 / n_lv
    for t in range(1, m):
        q_prev = k * u
        ret[t] = q_prev * (close[t] / close[t - 1] - 1.0) - q_prev * funding[t]
        if k == 0:
            if np.isnan(anchor[t - 1]) or np.isnan(step[t - 1]) or step[t - 1] <= 0:
                continue
            c0, s0 = anchor[t - 1], step[t - 1]
            if not active[t - 1]:
                continue
            buy_px, sell_px = c0 - s0, c0 + s0
            can_buy = buy_px < close[t - 1] and low[t] < buy_px
            can_sell = sell_px > close[t - 1] and high[t] > sell_px
            if can_buy and can_sell:                 # обе стороны задеты: худшая — против закрытия бара
                if close[t] - low[t] < high[t] - close[t]:
                    can_sell = False
                else:
                    can_buy = False
            if can_buy:
                k = 1
                opened[1] = t
                ret[t] += u * (close[t] / buy_px - 1.0) - u * mk
                fills += 1
            elif can_sell:
                k = -1
                opened[1] = t
                ret[t] += -u * (close[t] / sell_px - 1.0) - u * mk
                fills += 1
            if k == 0:
                continue
        side = 1.0 if k > 0 else -1.0
        n = abs(k)
        # 1) наращивание позиции (худший порядок)
        while n < n_lv and active[t - 1]:
            px = c0 - side * (n + 1) * s0
            hit = low[t] < px if side > 0 else high[t] > px
            if not hit:
                break
            n += 1
            opened[n] = t
            ret[t] += side * u * (close[t] / px - 1.0) - u * mk
            fills += 1
        # 2) стоп: цена ушла за последний уровень (и при неполной позиции, если фильтр запретил добор)
        stop_px = c0 - side * (n_lv + 1) * s0
        if (side > 0 and low[t] <= stop_px) or (side < 0 and high[t] >= stop_px):
            xp = min(stop_px, open_[t]) if side > 0 else max(stop_px, open_[t])
            ret[t] += -side * n * u * (close[t] / xp - 1.0) - n * u * tk
            stops += 1
            k = 0
            continue
        # 3) тейки ступеней, открытых в прошлых барах (сверху вниз по глубине)
        while n > 0 and opened[n] < t:
            px = c0 - side * (n - 1) * s0
            hit = high[t] > px if side > 0 else low[t] < px
            if not hit:
                break
            ret[t] += -side * u * (close[t] / px - 1.0) - u * mk
            tps += 1
            n -= 1
        k = int(side) * n
        q_end[t] = k * u
    return ret, q_end, fills, tps, stops


@njit(cache=True)
def rolling_poc(vwap, vol, w, bin_bps, recalc):
    """POC (ячейка с наибольшим объёмом) профиля за w прошлых баров; объём бара — по его VWAP."""
    m = len(vwap)
    lb = np.log1p(bin_bps / 1e4)
    idx = np.empty(m, np.int64)
    base = 1 << 60
    for t in range(m):
        idx[t] = int(np.floor(np.log(vwap[t]) / lb))
        base = min(base, idx[t])
    top = 0
    for t in range(m):
        idx[t] -= base
        top = max(top, idx[t])
    prof = np.zeros(top + 1)
    out = np.full(m, np.nan)
    cur = np.nan
    for t in range(m):
        prof[idx[t]] += vol[t]
        if t - w >= 0:
            prof[idx[t - w]] -= vol[t - w]
        if t + 1 < w:
            continue
        if t % recalc == 0 or np.isnan(cur):
            lo, hi = top, 0
            for j in range(max(t - w + 1, 0), t + 1):
                lo = min(lo, idx[j])
                hi = max(hi, idx[j])
            best, bi = -1.0, lo
            for i in range(lo, hi + 1):
                if prof[i] > best:
                    best, bi = prof[i], i
            cur = np.exp((bi + base + 0.5) * lb)
        out[t] = cur
    return out


def inputs(df: pd.DataFrame, anchor: str, cache: dict, sym: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = (sym, "base")
    if key not in cache:
        c = df["close"]
        pc = c.shift(1)
        tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
        atr1h = tr.ewm(alpha=1.0 / DAY, adjust=False, min_periods=DAY).mean() * np.sqrt(12.0)
        er = (c - c.shift(DAY)).abs() / c.diff().abs().rolling(DAY).sum()
        vwap_bar = (df["quote_volume"] / df["volume"]).where(df["volume"] > 0)
        vwap_bar = vwap_bar.where((vwap_bar >= df["low"]) & (vwap_bar <= df["high"]), c).fillna(c)
        cache[key] = {
            "atr": atr1h.to_numpy(), "range": (er < ER_MAX).to_numpy(),
            "ema1d": c.ewm(span=DAY, adjust=False, min_periods=DAY).mean().to_numpy(),
            "vwap1d": (df["quote_volume"].rolling(DAY).sum() / df["volume"].rolling(DAY).sum()).to_numpy(),
            "poc1d": rolling_poc(vwap_bar.to_numpy(dtype="float64"), df["volume"].to_numpy(dtype="float64"),
                                 DAY, 10.0, 12),
        }
    b = cache[key]
    return b[anchor], b["atr"], b["range"]


def run_grid(df: pd.DataFrame, cache: dict, sym: str, anchor, m, n, filt) -> tuple[pd.Series, pd.Series, tuple]:
    a, atr, rng = inputs(df, anchor, cache, sym)
    active = rng if filt == "range" else np.ones(len(df), np.bool_)
    ret, q, fills, tps, stops = grid_machine(df["open"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(),
                                             df["close"].to_numpy(), df["funding"].to_numpy(), a, m * atr,
                                             np.asarray(active, np.bool_), n, MAKER, TAKER)
    return pd.Series(ret, index=df.index), pd.Series(q, index=df.index), (fills, tps, stops)


def evaluate_grid(dfs: dict, cache: dict, cfg: dict, per: str) -> dict:
    pnls, f_all, t_all, s_all = [], 0, 0, 0
    for sym, df in dfs.items():
        r, q, (f, tp, st) = run_grid(df, cache, sym, **cfg)
        pnls.append(cut(r, per))
        f_all, t_all, s_all = f_all + f, t_all + tp, s_all + st
    port = pd.concat(pnls, axis=1).fillna(0.0).mean(axis=1)
    mt = metrics(port)
    return {"sharpe": mt["sharpe"], "ann_ret": mt["ann_ret"], "max_dd": mt["max_dd"], "fills": f_all,
            "take_profits": t_all, "stops": s_all, "fills_per_day": f_all / max(mt["days"], 1) / len(dfs)}


# ---------- B: лесенка для oi_liq ----------

P_OI = {"k": 12, "thr": 2.5, "hold": 4, "mode": "liq", "sign": 1}
LADDERS = [(1, 0.0)] + [(k, s) for k in (3, 5) for s in (0.25, 0.5, 1.0)]
HOLD_5M = 48


def ladder_trades(root: Path, syms: list[str]) -> pd.DataFrame:
    """Для каждого сигнала oi_liq: доходность на капитал для рыночного входа и для каждой лесенки (вход —
    лимиты по цене сигнала и ниже/выше, выход рыночно через 4 ч после сигнала)."""
    d1, d5 = Data2(root, syms), Data(root, syms)
    rows = []
    for s in syms:
        h = d1.get(s, "1h", "full")
        f5 = d5.get(s, "5m", "full")
        pos, _ = oi_price(h, **P_OI)
        pos = np.asarray(pos, float)
        prev = np.concatenate([[0.0], pos[:-1]])
        c1 = h["close"].to_numpy()
        tr = pd.concat([h["high"] - h["low"], (h["high"] - h["close"].shift()).abs(),
                        (h["low"] - h["close"].shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
        lo5, hi5, cl5 = f5["low"].to_numpy(), f5["high"].to_numpy(), f5["close"].to_numpy()
        for e in np.flatnonzero((pos != 0) & (prev == 0)):
            t_sig = h.index[e] + pd.Timedelta(hours=1)               # закрытие часового бара сигнала
            i0 = f5.index.searchsorted(t_sig)
            if i0 + HOLD_5M > len(cl5):
                continue
            d, p0, a = pos[e], c1[e], atr[e]
            w = slice(i0, i0 + HOLD_5M)
            exit_px = cl5[i0 + HOLD_5M - 1]
            row = {"symbol": s, "t": h.index[e], "dir": d}
            for kk, sp in LADDERS:
                if kk == 1 and sp == 0.0:                            # рыночный вход по цене сигнала
                    r = d * (exit_px / p0 - 1) - 2 * TAKER * 1e-4
                    row["market"], row["market_fill"] = r, 1.0
                    continue
                r, nf = 0.0, 0
                for i in range(kk):
                    lvl = p0 - d * i * sp * a
                    hit = lo5[w] < lvl if d > 0 else hi5[w] > lvl
                    if hit.any():
                        nf += 1
                        r += (d * (exit_px / lvl - 1) - (MAKER + TAKER) * 1e-4) / kk
                row[f"L{kk}x{sp}"], row[f"L{kk}x{sp}_fill"] = r, nf / kk
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 250)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--symbols", default=",".join(UNIVERSE))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    syms = a.symbols.split(",")

    print("===== B: лесенка лимитов для oi_liq (выход через 4 ч) =====")
    lt = ladder_trades(Path(a.root), syms)
    lt.to_csv(out / "grid_ladder.csv", index=False)
    cols = ["market"] + [f"L{k}x{s}" for k, s in LADDERS if k > 1]
    rows = []
    for per, (p_a, p_b) in PERIODS.items():
        x = lt[(lt.t >= p_a) & (lt.t < p_b)]
        for c in cols:
            rows.append({"period": per, "variant": c, "signals": len(x), "bps_per_signal": x[c].mean() * 1e4,
                         "fill_share": x[f"{c}_fill"].mean(), "win": float((x[c] > 0).mean())})
    print(pd.DataFrame(rows).pivot_table(index="variant", columns="period",
                                         values=["bps_per_signal", "fill_share"]).round(2).to_string())

    print("\n===== A: адаптивная сетка (5m, 16 монет) =====")
    data = Data(Path(a.root), syms)
    full = {s: data.get(s, "5m", "full") for s in syms}
    is_dfs = {s: cut(df, "is") for s, df in full.items()}
    cache_is, cache_full = {}, {}
    rows, t0 = [], time.time()
    for anchor, m, n, filt in itertools.product(ANCHORS, STEPS, LEVELS, FILTERS):
        cfg = {"anchor": anchor, "m": m, "n": n, "filt": filt}
        rows.append({**cfg, **evaluate_grid(is_dfs, cache_is, cfg, "is")})
    res = pd.DataFrame(rows)
    res.to_csv(out / "grid_is.csv", index=False)
    print(f"конфигураций {len(res)} ({time.time() - t0:.0f}s); Sharpe IS: медиана {res['sharpe'].median():.2f}, "
          f"доля > 0: {(res['sharpe'] > 0).mean():.0%}, максимум {res['sharpe'].max():.2f}")
    for col in ("anchor", "m", "n", "filt"):
        print(f"  {col}: " + ", ".join(f"{k}={v:+.2f}" for k, v in res.groupby(col)["sharpe"].median().items()))
    top = res.sort_values("sharpe", ascending=False).head(8)
    print("\nТОП-8 по IS:")
    print(top.round(3).to_string(index=False))
    gate = []
    for _, f in top.iterrows():
        cfg = {"anchor": f["anchor"], "m": float(f["m"]), "n": int(f["n"]), "filt": f["filt"]}
        v = evaluate_grid(full, cache_full, cfg, "val")
        row = {**cfg, "is": f["sharpe"], "val": v["sharpe"], "val_ret": v["ann_ret"], "val_dd": v["max_dd"],
               "passed": bool(v["sharpe"] >= VAL_MIN)}
        if row["passed"]:
            h = evaluate_grid(full, cache_full, cfg, "ho")
            row.update({"ho": h["sharpe"], "ho_ret": h["ann_ret"], "ho_dd": h["max_dd"], "ho_stops": h["stops"]})
        gate.append(row)
    print("\nВОРОТА VAL (Sharpe >= 0.5) -> HOLDOUT:")
    print(pd.DataFrame(gate).round(3).to_string(index=False))
