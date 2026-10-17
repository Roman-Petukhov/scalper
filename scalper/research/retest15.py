"""15m (правило бота: шорт от пологой линии, топ-150): вход по рынку против ретеста и сколько ждать ретест.

Сигнал — как в боте (trader/domain/strategy.py): пробой вниз линии зигзага 3 ATR на закрытии 15m, агрессоры ≥ 55%,
тренд 1h, закрытие в нижней половине свечи, наклон линии ≤ 0.025 ATR за свечу, стоп за свингом PIV5 + 0.1 ATR
(0.3–4 ATR), 3R, срок 200 свечей, оборот от $20M, место в рейтинге оборота на день сигнала ≤ 150.
Входы (заданы до запуска): по рынку на закрытии; лимитка на линии, ждём 6 / 12 (сейчас в research и на 4h) / 24 / 48
свечей (1.5 / 3 / 6 / 12 часов). R на исполненную сделку и на сигнал (неисполненная — 0), доля исполнения, R в месяц.
Решение: ретест лучше рынка, если R в месяц выше во всех периодах IS / VAL / HO / 2026; окно — плато, а не пик.

    python -m research.retest15 collect --symbols <15m-wide>
    python -m research.retest15 report
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30
from .dconf import Exit, simulate
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import (MAKER, PIV, STOP_BUF, TAKER, htf_trend, last_confirmed, pivots, retest_fill, tf_frame,
                    zz_lines)

TF = "15m"
WINDOWS = (6, 12, 24, 48)
MAX_SLOPE = 0.025
TOP = 150
EXIT = Exit(3.0, (), 0.0, 200)
PER = {"is": ("2022-01-01", "2024-07-01"), "val": ("2024-07-01", "2025-07-01"), "ho": ("2025-07-01", "2026-01-01"),
       "2026": ("2026-01-01", "2026-10-01")}


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str) -> list[dict]:
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64")
    a = _atr(d).to_numpy()
    trend = htf_trend(d, TF)
    buy = (d["taker_buy_volume"] / d["volume"].replace(0, np.nan)).to_numpy()
    rng = hi - lo
    sw_hi = last_confirmed(pivots(hi, PIV, True), len(c))
    rows = []
    for r in zz_lines(d):
        t, side = int(r["t"]), int(r["side"])
        if side > 0 or t <= 0 or t >= len(c) - 1 or not (a[t] > 0) or not (liq[t] >= ADV_MIN):
            continue
        if abs(float(r["slope"])) / a[t] > MAX_SLOPE or trend[t] != -1 or not (1 - buy[t] >= 0.55):
            continue
        if not (rng[t] > 0 and (hi[t] - c[t]) / rng[t] >= 0.5) or sw_hi[t] < 0:
            continue
        level, stop = float(r["line_t"]), hi[sw_hi[t]] + STOP_BUF * a[t]
        row = {"symbol": sym, "t": d.index[t]}
        risk = stop - c[t]
        row["market"] = (simulate(o, hi, lo, c, f, t, -1, c[t], stop, a[t], EXIT, TAKER)
                         if 0.3 * a[t] <= risk <= 4 * a[t] else np.nan)
        rr = stop - level
        for w in WINDOWS:
            key = f"retest{w}"
            if not (0.3 * a[t] <= rr <= 4 * a[t]):
                row[key] = np.nan
                continue
            fill = retest_fill(hi, lo, t, -1, level, 3.0 * rr, w)
            if fill < 0:
                row[key] = np.inf                                         # лимитка не исполнилась
                continue
            if hi[fill] >= stop:
                row[key] = (level - stop - (MAKER + TAKER) * level) / rr
            else:
                row[key] = simulate(o, hi, lo, c, f, fill, -1, level, stop, a[t], EXIT, MAKER)
        rows.append(row)
    return rows


def collect(root: Path, syms: list[str]) -> None:
    rows, advs = [], []
    for s in mine(syms):
        try:
            a = adv30(root, s).dropna()
            advs.append(pd.DataFrame({"symbol": s, "day": a.index, "adv": a.to_numpy()}))
            d = tf_frame(root, s, TF)
            if d is None or len(d) < 1000:
                continue
            liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
            rows += coin_rows(d, liq, s)
        except Exception as e:
            print(f"  retest15 {s}: пропуск ({e})", flush=True)
    print(f"  retest15: сигналов {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("retest15"), index=False)
    if advs:
        pd.concat(advs, ignore_index=True).to_parquet(part_path("retest15_adv"), index=False)


def report() -> None:
    tp, ap = all_parts("retest15"), all_parts("retest15_adv")
    print("\n===== RETEST15: 15m, правило бота (пологие шорты, топ-150) — рынок против ретеста 6 / 12 / 24 / 48 свечей =====")
    if not tp or not ap:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in tp], ignore_index=True)
    adv = pd.concat([pd.read_parquet(p) for p in ap], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    adv["day"] = pd.to_datetime(adv["day"], utc=True)
    adv["rank"] = adv.groupby("day")["adv"].rank(ascending=False, method="first")
    df["day"] = df.t.dt.floor("D")
    df = df.merge(adv[["symbol", "day", "rank"]], on=["symbol", "day"], how="left")
    df = df[df["rank"] <= TOP]
    print(f"сигналов (топ-{TOP}) {len(df)}")
    entries = ["market"] + [f"retest{w}" for w in WINDOWS]
    per_month = {}
    for e in entries:
        x = df[["t", e]].rename(columns={e: "R"})
        x = x[x.R.notna()]
        filled = x[np.isfinite(x.R)]
        per_sig = x.assign(R=x.R.where(np.isfinite(x.R), 0.0))
        nm = "по рынку" if e == "market" else f"ретест {e[6:]} свечей"
        print(f"\n  {nm}: исполнено {len(filled) / max(len(x), 1):.0%}")
        cells = []
        per_month[e] = {}
        for p, (a, b) in PER.items():
            f_ = filled[(filled.t >= a) & (filled.t < b)]
            s_ = per_sig[(per_sig.t >= a) & (per_sig.t < b)]
            months = (pd.Timestamp(b) - pd.Timestamp(a)).days / 30.44
            per_month[e][p] = s_.R.sum() / months
            cells.append(f"{p}: {_cell(f_)}, {per_month[e][p]:+.1f}R/мес")
        print("    " + "  ".join(cells))
    print("\n  R в месяц (на сигнал, неисполненная = 0) — ретест лучше рынка во всех периодах?")
    for e in entries[1:]:
        ok = [per_month[e][p] > per_month["market"][p] for p in PER]
        print(f"    {e}: {'да' if all(ok) else f'нет ({sum(ok)} из {len(ok)})'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("mode", choices=["collect", "report"])
    ap_.add_argument("--root", default="~/bn")
    ap_.add_argument("--symbols", default="")
    a_ = ap_.parse_args()
    if a_.mode == "collect":
        collect(Path(a_.root).expanduser(), [x for x in a_.symbols.split(",") if x])
    else:
        report()
