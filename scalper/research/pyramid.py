"""Принципы Ливермора на сделках бота 4h: пирамидинг (доливка в прибыльную позицию) и стоп по времени.

Сделки — правило бота (research/noline.py: пробой линии зигзага 3 ATR, оборот от $20M, агрессоры ≥ 55%, закрытие в своей
половине свечи, дневной тренд, стоп за свингом PIV5 ± 0.1 ATR, лимитка на линии 12 свечей, цель 3R, до 60 свечей).
Правила заданы до запуска; R — в риске первоначальной позиции:
    контроль    как бот
    B0          стоп в безубыток после +1R, без доливки (отделить эффект безубытка от доливки)
    P1          на +1R долить 0.5 позиции (стоп-ордер, taker), стоп всей позиции — в точку первого входа
    P2          то же на +1.5R
    P3          на +1R долить 1.0 позиции, стоп в точку входа (риск после доливки = исходному 1R)
    T1 / T2     стоп по времени: если за 6 / 12 свечей (1 / 2 дня) цена не дошла до +0.5R — выход по закрытию
    T3          за 24 свечи (4 дня) не дошла до +1R — выход
Порядок внутри свечи: стоп, затем доливка, затем цель (худший случай для нас); новый стоп — со следующей свечи.
Издержки: вход лимиткой maker, доливка taker, стоп и выход по времени taker, цель maker; funding на весь объём.
Принятие: R на сделку лучше контроля во всех шести периодах (2020, 2021, IS, VAL, HO, 2026) и не хуже без 5% лучших.

    python -m research.pyramid collect --symbols <1h-qualified>
    python -m research.pyramid report
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30
from .noline import triggers
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import (HOLD, MAKER, PIV, RETEST_BARS, STOP_BUF, TAKER, htf_trend, last_confirmed, pivots, retest_fill,
                    tf_frame)

TF = "4h"
TARGET_R = 3.0


@dataclass(frozen=True)
class Rule:
    add_at: float = 0.0          # доливка при +add_at R (0 — нет)
    add_size: float = 0.0        # доля исходной позиции
    be_at: float = 0.0           # стоп в точку входа при +be_at R (0 — нет)
    time_bars: int = 0           # стоп по времени: проверка через столько свечей (0 — нет)
    time_min_r: float = 0.0      # ... если лучший ход меньше стольких R — выход


VARIANTS: dict[str, Rule] = {
    "контроль (бот)": Rule(),
    "B0 безубыток на +1R": Rule(be_at=1.0),
    "P1 долив 0.5 на +1R": Rule(add_at=1.0, add_size=0.5, be_at=1.0),
    "P2 долив 0.5 на +1.5R": Rule(add_at=1.5, add_size=0.5, be_at=1.5),
    "P3 долив 1.0 на +1R": Rule(add_at=1.0, add_size=1.0, be_at=1.0),
    "T1 6 свечей < +0.5R": Rule(time_bars=6, time_min_r=0.5),
    "T2 12 свечей < +0.5R": Rule(time_bars=12, time_min_r=0.5),
    "T3 24 свечи < +1R": Rule(time_bars=24, time_min_r=1.0),
}


def simulate(o, h, lo, c, fund, e, side, entry, stop, rule: Rule, max_hold: int = HOLD[TF],
             entry_fee: float = MAKER) -> float:
    """Позиция открыта на баре e по цене entry (бар e уже проверен на стоп). Результат в R исходной позиции."""
    n = len(c)
    risk = side * (entry - stop)
    if not risk > 0:
        return np.nan
    tp = entry + side * TARGET_R * risk
    sp = stop
    add_px, added, size2 = np.nan, False, 0.0
    best, paid = 0.0, 0.0

    def close_all(px: float, exit_fee: float) -> float:
        r = (side * (px - entry) - (entry_fee + exit_fee) * entry) / risk
        if added:
            r += size2 * (side * (px - add_px) - (TAKER + exit_fee) * add_px) / risk
        return r - side * paid * entry / risk

    for j in range(e + 1, n):
        paid += fund[j] * (1.0 + size2)
        if (side > 0 and lo[j] <= sp) or (side < 0 and h[j] >= sp):
            ex = min(sp, o[j]) if side > 0 else max(sp, o[j])
            return close_all(ex, TAKER)
        new_sp = sp
        if rule.add_at > 0 and not added:
            lvl = entry + side * rule.add_at * risk
            if (side > 0 and h[j] >= lvl) or (side < 0 and lo[j] <= lvl):
                add_px = max(lvl, o[j]) if side > 0 else min(lvl, o[j])
                added, size2 = True, rule.add_size
        if rule.be_at > 0:
            lvl = entry + side * rule.be_at * risk
            if (side > 0 and h[j] >= lvl) or (side < 0 and lo[j] <= lvl):
                new_sp = entry if side * (entry - sp) > 0 else sp
        if (side > 0 and h[j] >= tp) or (side < 0 and lo[j] <= tp):
            return close_all(tp, MAKER)
        best = max(best, side * ((h[j] if side > 0 else lo[j]) - entry) / risk)
        if rule.time_bars and j - e == rule.time_bars and best < rule.time_min_r:
            return close_all(c[j], TAKER)
        if j - e >= max_hold or j == n - 1:
            return close_all(c[j], TAKER)
        sp = new_sp
    return np.nan


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str) -> list[dict]:
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64")
    a = _atr(d).to_numpy()
    trend = htf_trend(d, TF)
    buy = (d["taker_buy_volume"] / d["volume"].replace(0, np.nan)).to_numpy()
    rng = hi - lo
    loc = (c - lo) / np.where(rng > 0, rng, np.nan)
    sw_lo, sw_hi = last_confirmed(pivots(lo, PIV, False), len(c)), last_confirmed(pivots(hi, PIV, True), len(c))
    rows = []
    for _, t, side, level in triggers(d, a, lines_only=True):
        if t >= len(c) - 1 or not (a[t] > 0) or not (liq[t] >= ADV_MIN) or trend[t] != side:
            continue
        aggr = buy[t] if side > 0 else 1 - buy[t]
        cl = loc[t] if side > 0 else 1 - loc[t]
        if not (aggr >= 0.55 and cl >= 0.5):
            continue
        sw = sw_lo[t] if side > 0 else sw_hi[t]
        if sw < 0:
            continue
        stop = (lo[sw] - STOP_BUF * a[t]) if side > 0 else (hi[sw] + STOP_BUF * a[t])
        risk = side * (level - stop)
        if not (0.3 * a[t] <= risk <= 4.0 * a[t]):
            continue
        fill = retest_fill(hi, lo, t, side, level, TARGET_R * risk, RETEST_BARS)
        if fill < 0:
            continue
        row = {"symbol": sym, "t": d.index[t], "side": side}
        if (side > 0 and lo[fill] <= stop) or (side < 0 and hi[fill] >= stop):
            loss = (side * (stop - level) - (MAKER + TAKER) * level) / risk
            row |= {nm: loss for nm in VARIANTS}
        else:
            row |= {nm: simulate(o, hi, lo, c, f, fill, side, level, stop, rule) for nm, rule in VARIANTS.items()}
        rows.append(row)
    return rows


def collect(root: Path, syms: list[str]) -> None:
    rows = []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, TF)
            if d is None or len(d) < 500:
                continue
            liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
            rows += coin_rows(d, liq, s)
        except Exception as e:
            print(f"  pyramid {s}: пропуск ({e})", flush=True)
    print(f"  pyramid: сделок {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("pyramid"), index=False)


def _dd(r: pd.Series) -> float:
    eq = r.cumsum()
    return float((eq.cummax() - eq).max()) if len(eq) else np.nan


def report() -> None:
    parts = all_parts("pyramid")
    print("\n===== PYRAMID: пирамидинг и стоп по времени (Ливермор) на сделках бота 4h; R — в риске исходной позиции =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df.sort_values("t")
    per = {p: (df.t >= a) & (df.t < b) for p, (a, b) in {**PER_ALL, "2026": ("2026-01-01", "2026-10-01")}.items()}
    base = {p: df.loc[m, "контроль (бот)"].mean() for p, m in per.items()}
    months = df.t.dt.to_period("M").nunique()
    print(f"сделок {len(df)}, шортов {(df.side < 0).mean():.0%}")
    for nm in VARIANTS:
        z = df.assign(R=df[nm])
        cut = z.R.quantile(0.95)
        better = all(m.sum() >= 5 and z.R[m].mean() > base[p] for p, m in per.items())
        trim_ok = z.R[z.R < cut].mean() >= df["контроль (бот)"][df["контроль (бот)"] < df["контроль (бот)"]
                                                                  .quantile(0.95)].mean()
        print(f"\n  {nm}: {z.R.sum() / months:+.2f}R в месяц, просадка {_dd(z.R):.1f}R, без 5% лучших "
              f"{z.R[z.R < cut].mean():+.3f}R" + ("" if nm.startswith("контроль") else
                                                  f" → {'принято' if better and trim_ok else 'не принято'}"))
        print("    " + "  ".join(f"{p}: {_cell(z[m])}" for p, m in per.items()))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x])
    else:
        report()
