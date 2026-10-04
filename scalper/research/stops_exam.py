"""
Стоп-лоссы для oi_liq (с фильтром выноса уровня): рыночный вход и лесенка L5x0.5, выход через 4 ч либо по стопу.

Стоп ставится от цены сигнала на расстоянии X * ATR(14, 1h) за самым глубоким уровнем входа (для рыночного входа —
за ценой входа): X = нет / 1 / 1.5 / 2 / 3 / 5. Срабатывает рыночно (taker), при гэпе — по open бара.
Путь цены — 5m-бары (Bybit) или 1h-бары (Binance, консервативно: в баре, где исполнился лимит и достигнут стоп,
считаем и исполнение, и стоп). Выход по таймеру — рыночно по close.
Отчёт: б.п. на сигнал, доля прибыльных, худшие 1% сделок, худшая сделка, просадка дневного портфеля (2% на сигнал).

    python -m research.stops_exam --root <data> --symbols ... [--tf 5m|1h] [--exam]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from .broad import ADV_MIN, POS_FRACTION, adv30, group_of
from .engine import metrics
from .grid import MAKER, P_OI, TAKER
from .levels import sr_levels
from .search import Data
from .sweepfilter import LOOKBACK, PER
from .wave2 import Data2, oi_price

STOPS = (np.inf, 1.0, 1.5, 2.0, 3.0, 5.0)
LADDER = (5, 0.5)
HOLD_H = 4


@njit(cache=True)
def path_trade(d, a, levels, pre_filled, open_, high, low, close, stop_x, entry_fee, exit_fee):
    """Доходность на капитал (доли) за сигнал: уровни входа levels, каждый 1/len капитала; первые pre_filled
    исполнены сразу (рыночный вход), остальные — лимиты (исполнение только сквозь уровень). Стоп — за самым
    глубоким уровнем на stop_x * a, рыночный; иначе выход по close последнего бара.
    Возвращает (доходность, исполненная доля, сработал ли стоп)."""
    k = len(levels)
    deepest = levels[k - 1]
    stop = deepest - d * stop_x * a
    filled = np.zeros(k, np.bool_)
    filled[:pre_filled] = True
    n = len(close)
    for j in range(n):
        for i in range(k):
            if not filled[i]:
                hit = low[j] < levels[i] if d > 0 else high[j] > levels[i]
                if hit:
                    filled[i] = True
        nf = filled.sum()
        if nf > 0 and np.isfinite(stop):
            st = low[j] <= stop if d > 0 else high[j] >= stop
            if st:
                xp = min(stop, open_[j]) if d > 0 else max(stop, open_[j])
                r = 0.0
                for i in range(k):
                    if filled[i]:
                        r += (d * (xp / levels[i] - 1.0) - entry_fee - exit_fee) / k
                return r, nf / k, True
    r = 0.0
    nf = filled.sum()
    for i in range(k):
        if filled[i]:
            r += (d * (close[n - 1] / levels[i] - 1.0) - entry_fee - exit_fee) / k
    return r, nf / k, False


def collect(root: Path, syms: list[str], tf: str, exam: bool) -> pd.DataFrame:
    d1 = Data2(root, syms)
    dp = Data(root, syms) if tf == "5m" else None
    rows = []
    mk, tk = MAKER * 1e-4, TAKER * 1e-4
    for s in syms:
        try:
            h = d1.get(s, "1h", "full")
        except Exception as e:
            print(f"  {s}: пропуск ({e})", flush=True)
            continue
        if h["oi"].notna().sum() < 24 * 60:
            continue
        pos, _ = oi_price(h, **P_OI)
        pos = np.asarray(pos, float)
        if exam:
            pos = np.where((adv30(root, s).reindex(h.index, method="ffill") >= ADV_MIN).to_numpy(), pos, 0.0)
        prev = np.concatenate([[0.0], pos[:-1]])
        hi, lo, c, op = (h[x].to_numpy() for x in ("high", "low", "close", "open"))
        tr = np.maximum.reduce([hi - lo, np.abs(hi - np.roll(c, 1)), np.abs(lo - np.roll(c, 1))])
        atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
        res, sup = sr_levels(hi, lo, 10)
        p5 = dp.get(s, "5m", "full") if dp is not None else None
        for e in np.flatnonzero((pos != 0) & (prev == 0)):
            if e + HOLD_H >= len(c) or e < LOOKBACK:
                continue
            dirn, p0, a = pos[e], c[e], atr[e]
            lvl = sup[e - LOOKBACK] if dirn > 0 else res[e - LOOKBACK]
            w = slice(e - LOOKBACK + 1, e + 1)
            if np.isnan(lvl) or not ((lo[w].min() < lvl) if dirn > 0 else (hi[w].max() > lvl)):
                continue                                          # только сделки после выноса уровня
            if p5 is not None:
                t0 = h.index[e] + pd.Timedelta(hours=1)
                i0 = p5.index.searchsorted(t0)
                seg = p5.iloc[i0:i0 + HOLD_H * 12]
                if len(seg) < HOLD_H * 12:
                    continue
                po, ph, pl, pc = (seg[x].to_numpy() for x in ("open", "high", "low", "close"))
            else:
                ws = slice(e + 1, e + HOLD_H + 1)
                po, ph, pl, pc = op[ws], hi[ws], lo[ws], c[ws]
            row = {"symbol": s, "group": group_of(s), "t": h.index[e]}
            mkt = np.array([p0])
            lad = np.array([p0 - dirn * i * LADDER[1] * a for i in range(LADDER[0])])
            for sx in STOPS:
                tag = "none" if not np.isfinite(sx) else f"{sx:g}"
                # рыночный вход по p0 (taker), путь — со следующего бара
                r, _, st = path_trade(dirn, a, mkt, 1, po, ph, pl, pc, sx, tk, tk)
                row[f"mkt_{tag}"], row[f"mkt_{tag}_stop"] = r, st
                r, f, st = path_trade(dirn, a, lad, 0, po, ph, pl, pc, sx, mk, tk)
                row[f"lad_{tag}"], row[f"lad_{tag}_stop"], row["lad_fill"] = r, st, f
            rows.append(row)
    return pd.DataFrame(rows)


def report(r: pd.DataFrame, groups: list[str] | None) -> None:
    if groups:
        r = r[r["group"].isin(groups)]
    rows = []
    for kind in ("mkt", "lad"):
        for sx in STOPS:
            tag = "none" if not np.isfinite(sx) else f"{sx:g}"
            col = f"{kind}_{tag}"
            for per, (a, b) in PER.items():
                x = r[(r.t >= a) & (r.t < b)]
                if not len(x):
                    continue
                port = x.set_index("t")[col].mul(POS_FRACTION).resample("1D").sum()
                port = port.reindex(pd.date_range(a, b, freq="1D", tz="UTC", inclusive="left"), fill_value=0.0)
                m = metrics(port)
                rows.append({"entry": kind, "stop_atr": tag, "period": per, "n": len(x),
                             "bps": x[col].mean() * 1e4, "win": float((x[col] > 0).mean()),
                             "stopped": float(x[f"{col}_stop"].mean()),
                             "p1_bps": x[col].quantile(0.01) * 1e4, "worst_bps": x[col].min() * 1e4,
                             "sharpe": m["sharpe"], "max_dd": m["max_dd"]})
    t = pd.DataFrame(rows)
    for kind in ("mkt", "lad"):
        print(f"\n{'Рыночный вход' if kind == 'mkt' else 'Лесенка 5 x 0.5 ATR'}:")
        print(t[t["entry"] == kind].drop(columns="entry").round(3).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 260)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--tf", default="1h", choices=["1h", "5m"])
    ap.add_argument("--exam", action="store_true", help="фильтр ликвидности и монеты вне подбора отдельно")
    a = ap.parse_args()
    r = collect(Path(a.root), a.symbols.split(","), a.tf, a.exam)
    print(f"===== STOPS: oi_liq с выносом уровня, стопы от {a.tf}-пути (сигналов {len(r)}) =====")
    print("bps — на сигнал (2% капитала), p1/worst — 1% худших и худшая сделка, max_dd — просадка дневного портфеля")
    report(r, None)
    if a.exam:
        print("\n----- Монеты вне подбора (ext54 + fresh) -----")
        report(r, ["ext54", "fresh"])
