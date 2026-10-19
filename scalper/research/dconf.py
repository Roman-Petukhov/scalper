"""Сделка бота 4h, совпавшая с пробоем дневной трендовой в ту же сторону: держать дольше 3R и доливать по пути?

Сделки — правило бота 4h (research/pyramid.py: линия зигзага 3 ATR, фильтры, ретест, стоп за свингом). Метка «совпала
с дневкой»: линия зигзага 3 ATR на дневках (те же правила построения, research/tline.py zz_lines) пробита закрытием дня
в сторону сделки, и это закрытие известно не раньше чем за 5 (вариант — 10) дней до закрытия свечи сигнала 4h.
Вторая метка (шире): «монета в дневном тренде» — ансамбль каналов Дончиана research/dtrend.py (9 периодов 5–360 дней,
лонг и шорт) на последнем закрытом дне: не меньше половины моделей в сторону сделки.
Выходы, заданные до запуска (R — в риске исходной позиции; стоп проверяется раньше всего в свече):
    бот          цель 3R, долив 0.5 на +1.5R со стопом в точку входа (как в боте), до 60 свечей
    5R           то же, цель 5R, до 180 свечей (30 дней)
    лестница     без цели, до 180 свечей; доливки по 0.5 на +1.5R / +3R / +4.5R, после каждой стоп всей позиции —
                 на предыдущую ступень (вход / +1.5R / +3R); после +4.5R стоп подтягивается на 3 ATR (ATR 4h на входе)
                 за лучшим закрытием
    лестница без доливок  те же стопы, без добавления объёма (отделить эффект доливок от длинного удержания)
Сравнение: те же выходы на сделках без совпадения — если «лестница» лучше только на совпавших, дело в дневке.
Принятие: лестница или 5R лучше бота на совпавших сделках в каждом периоде, где их ≥ 10, и в сумме по годам.

    python -m research.dconf collect --symbols <1h-qualified>
    python -m research.dconf report
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
from .dtrend import LOOKBACKS, _models
from .smc import _atr, _cell
from .tline import (MAKER, PIV, RETEST_BARS, STOP_BUF, TAKER, htf_trend, last_confirmed, pivots, retest_fill,
                    tf_frame, zz_lines)

TF = "4h"
WINDOWS = (5, 10)                # дней: дневной пробой не раньше стольких дней до сигнала


@dataclass(frozen=True)
class Exit:
    target_r: float | None              # None — без цели
    adds: tuple[float, ...]             # ступени доливок, R
    add_size: float                     # объём каждой доливки (доля исходного)
    max_hold: int
    trail_atr: float = 0.0              # после последней ступени — стоп на столько ATR за лучшим закрытием (0 — нет)
    ladder_stops: bool = True           # после ступени k стоп — на ступень k−1 (вход для первой)


EXITS: dict[str, Exit] = {
    "бот (3R, долив на +1.5R)": Exit(3.0, (1.5,), 0.5, 60),
    "5R, долив на +1.5R": Exit(5.0, (1.5,), 0.5, 180),
    "лестница": Exit(None, (1.5, 3.0, 4.5), 0.5, 180, 3.0),
    "лестница без доливок": Exit(None, (1.5, 3.0, 4.5), 0.0, 180, 3.0),
}


def simulate(o, h, lo, c, fund, e, side, entry, stop, atr, ex: Exit, entry_fee: float = MAKER) -> float:
    """Позиция открыта на баре e по цене entry. Результат в R исходной позиции (комиссии и funding включены)."""
    return simulate_bar(o, h, lo, c, fund, e, side, entry, stop, atr, ex, entry_fee)[0]


def simulate_bar(o, h, lo, c, fund, e, side, entry, stop, atr, ex: Exit,
                 entry_fee: float = MAKER, adds_out: list | None = None) -> tuple[float, int]:
    """То же, что simulate, плюс бар выхода (−1 — позиция не закрылась или не открылась).
    adds_out — сюда пишутся ступени доливки (бар, цена исполнения), даже при add_size = 0 (ступень пройдена)."""
    n = len(c)
    risk = side * (entry - stop)
    if not risk > 0:
        return np.nan, -1
    units = [(entry, 1.0, entry_fee)]                    # (цена, объём, комиссия входа)
    sp, step, best, paid = stop, 0, c[e], 0.0
    tp = entry + side * ex.target_r * risk if ex.target_r else None

    def close_all(px: float, fee: float) -> float:
        r = sum(q * (side * (px - p) - (f + fee) * p) for p, q, f in units) / risk
        return r - side * paid * entry / risk

    for j in range(e + 1, n):
        paid += fund[j] * sum(q for _, q, _ in units)
        if (side > 0 and lo[j] <= sp) or (side < 0 and h[j] >= sp):
            return close_all(min(sp, o[j]) if side > 0 else max(sp, o[j]), TAKER), j
        new_sp = sp
        while step < len(ex.adds):
            lvl = entry + side * ex.adds[step] * risk
            if not ((side > 0 and h[j] >= lvl) or (side < 0 and lo[j] <= lvl)):
                break
            px = max(lvl, o[j]) if side > 0 else min(lvl, o[j])
            if adds_out is not None:
                adds_out.append((j, px))
            if ex.add_size > 0:
                units.append((px, ex.add_size, TAKER))
            if ex.ladder_stops:
                new_sp = entry + side * (ex.adds[step - 1] if step > 0 else 0.0) * risk
            step += 1
        if tp is not None and ((side > 0 and h[j] >= tp) or (side < 0 and lo[j] <= tp)):
            return close_all(tp, MAKER), j
        if side * (c[j] - best) > 0:
            best = c[j]
        if ex.trail_atr and step == len(ex.adds):
            new_sp = max(new_sp, best - ex.trail_atr * atr) if side > 0 else min(new_sp, best + ex.trail_atr * atr)
        if j - e >= ex.max_hold or j == n - 1:
            return close_all(c[j], TAKER), j
        if side * (new_sp - sp) > 0:
            sp = new_sp
    return np.nan, -1


def daily_breaks(d1: pd.DataFrame) -> list[tuple[pd.Timestamp, int]]:
    """(когда пробой дневной линии стал известен — закрытие дня, сторона)."""
    out = []
    for r in zz_lines(d1):
        if r["t"] > 0:
            out.append((d1.index[r["t"]] + pd.Timedelta(days=1), int(r["side"])))
    return out


def coin_rows(d: pd.DataFrame, d1: pd.DataFrame, liq: np.ndarray, sym: str) -> list[dict]:
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64")
    a = _atr(d).to_numpy()
    trend = htf_trend(d, TF)
    buy = (d["taker_buy_volume"] / d["volume"].replace(0, np.nan)).to_numpy()
    rng = hi - lo
    loc = (c - lo) / np.where(rng > 0, rng, np.nan)
    sw_lo, sw_hi = last_confirmed(pivots(lo, PIV, False), len(c)), last_confirmed(pivots(hi, PIV, True), len(c))
    brk = daily_breaks(d1) if d1 is not None and len(d1) > 100 else []
    dt = (sum(_models(d1[["close"]], n, True) for n in LOOKBACKS)["close"] / len(LOOKBACKS)
          if d1 is not None and len(d1) > 30 else pd.Series(dtype=float))
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
        fill = retest_fill(hi, lo, t, side, level, 3.0 * risk, RETEST_BARS)
        if fill < 0:
            continue
        known = d.index[t] + pd.Timedelta(hours=4)
        row = {"symbol": sym, "t": d.index[t], "side": side}
        for w in WINDOWS:
            row[f"conf{w}"] = any(sd == side and known - pd.Timedelta(days=w) <= kt <= known for kt, sd in brk)
        day = known.floor("D") - pd.Timedelta(days=1)                # последний закрытый день
        row["dtrend"] = bool(day in dt.index and side * dt.loc[day] >= 0.5)
        if (side > 0 and lo[fill] <= stop) or (side < 0 and hi[fill] >= stop):
            loss = (side * (stop - level) - (MAKER + TAKER) * level) / risk
            row |= {nm: loss for nm in EXITS}
        else:
            row |= {nm: simulate(o, hi, lo, c, f, fill, side, level, stop, a[t], ex) for nm, ex in EXITS.items()}
        rows.append(row)
    return rows


def collect(root: Path, syms: list[str]) -> None:
    rows = []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, TF)
            if d is None or len(d) < 500:
                continue
            d1 = tf_frame(root, s, "1d")
            liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
            rows += coin_rows(d, d1, liq, s)
        except Exception as e:
            print(f"  dconf {s}: пропуск ({e})", flush=True)
    print(f"  dconf: сделок {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("dconf"), index=False)


def report() -> None:
    parts = all_parts("dconf")
    print("\n===== DCONF: сделки бота 4h, совпавшие с пробоем дневной трендовой в ту же сторону — держать дольше 3R "
          "и доливать по пути? R — в риске исходной позиции =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    periods = {**PER_ALL, "2026": ("2026-01-01", "2026-10-01")}
    print(f"сделок {len(df)}, шортов {(df.side < 0).mean():.0%}; совпали с дневкой за 5 дней: {df.conf5.sum()} "
          f"({df.conf5.mean():.0%}), за 10 дней: {df.conf10.sum()} ({df.conf10.mean():.0%})")
    print(f"в дневном тренде (Дончиан, ≥ половины моделей): {df.dtrend.sum()} ({df.dtrend.mean():.0%})")
    base = list(EXITS)[0]
    labels = [(f"conf{w}", f"совпали с дневкой ({w} дней)", f"не совпали ({w} дней)") for w in WINDOWS]
    labels.append(("dtrend", "в дневном тренде (Дончиан)", "не в дневном тренде"))
    for col, yes, no in labels:
        for flag, nm in ((True, yes), (False, no)):
            z0 = df[df[col] == flag]
            print(f"\n  --- {nm}: {len(z0)} сделок, шортов {(z0.side < 0).mean():.0%} ---")
            for ex in EXITS:
                z = z0.assign(R=z0[ex])
                per = {p: (z.t >= a) & (z.t < b) for p, (a, b) in periods.items()}
                ok = all(z.R[m].mean() > z0[base][m].mean() for m in per.values() if m.sum() >= 10)
                yy = z.groupby(z.t.dt.year).R.sum() - z0.groupby(z0.t.dt.year)[base].sum()
                tail = "" if ex == base else (f"  → по периодам {'лучше везде' if ok else 'не везде'}; разница по "
                                              "годам, R: " + ", ".join(f"{k} {v:+.1f}" for k, v in yy.items()))
                print(f"    {ex}: всего {z.R.sum():+.1f}R, макс. {z.R.max():+.1f}R{tail}")
                print("      " + "  ".join(f"{p}: {_cell(z[m])}" for p, m in per.items()))


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
