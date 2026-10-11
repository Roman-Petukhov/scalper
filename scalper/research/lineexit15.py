"""15m: ранний выход по рынку, если свеча закрылась обратно за линией пробоя.

Сделки — правило бота 15m (как research/rebreak.py, группа first: шорт от пологой линии, по рынку, топ-150, фильтры
бота, стоп за свингом, цель 3R, удержание 200 свечей). Вариант выхода: на закрытии свечи j (fill < j <= fill + window),
если закрытие за линией (для шорта — выше линии) дальше buf ATR, позиция закрывается по этому закрытию (тейкер).
Стоп и цель проверяются раньше, как в two_targets. Сетка: buf 0 / 0.25 / 0.5 ATR × окно 4 / 12 / 200 свечей.
Повод: на 15m три четверти сигналов бота возвращаются за линию за 3 часа и в среднем убыточны (docs/knowledge.md);
на 4h такой выход проверяли — хуже (docs/research_report.md).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from .broad import ADV_MIN, adv30
from .rebreak import TOP15, events
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import (HOLD, MAKER, PER, PIV, STOP_BUF, TAKER, _years, htf_trend, last_confirmed, pivots, tf_frame,
                    two_targets, zz_lines)
from .wide15 import MAX_SLOPE_ATR

TF = "15m"
BUFS = (0.0, 0.25, 0.5)
WINDOWS = (4, 12, 200)


@njit(cache=True)
def line_exit(o, h, lo, c, fund, e, side, entry, stop, k, max_hold, entry_fee, line0, i0, slope, buf, window):
    """Вся позиция на k·R; выход по закрытию, если свеча закрылась за линией дальше buf (в цене) не позже window
    свечей после входа. Линия на баре j = line0 + slope·(j − i0). Стоп раньше тейка. Результат в R, бар выхода,
    1 — вышли по линии."""
    n = len(c)
    risk = side * (entry - stop)
    if not (risk > 0):
        return np.nan, e, 0
    tp = entry + side * k * risk
    paid = 0.0
    for j in range(e + 1, n):
        paid += fund[j]
        if (side > 0 and lo[j] <= stop) or (side < 0 and h[j] >= stop):
            ex = min(stop, o[j]) if side > 0 else max(stop, o[j])
            return (side * (ex - entry) - (entry_fee + TAKER) * entry) / risk - side * paid * entry / risk, j, 0
        if (side > 0 and h[j] >= tp) or (side < 0 and lo[j] <= tp):
            return (side * (tp - entry) - (entry_fee + MAKER) * entry) / risk - side * paid * entry / risk, j, 0
        back = j - e <= window and side * (c[j] - (line0 + slope * (j - i0))) < -buf
        if back or j - e >= max_hold or j == n - 1:
            return (side * (c[j] - entry) - (entry_fee + TAKER) * entry) / risk - side * paid * entry / risk, j, int(back)
    return np.nan, n - 1, 0


def coin_rows(d: pd.DataFrame, sym: str) -> pd.DataFrame:
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    a = _atr(d).to_numpy()
    trend = htf_trend(d, TF)
    buy = (d["taker_buy_volume"] / d["volume"].replace(0, np.nan)).to_numpy()
    rng = hi - lo
    loc = (c - lo) / np.where(rng > 0, rng, np.nan)
    sw_hi = last_confirmed(pivots(hi, PIV, True), len(c))
    rows = []
    for ev in events(c, zz_lines(d)):
        t, side = ev["t"], ev["side"]
        if ev["kind"] != "first" or side > 0 or t >= len(c) - 1 or not (a[t] > 0) or trend[t] != side:
            continue
        aggr, cl = 1 - buy[t], 1 - loc[t]
        slope_atr = abs(ev["slope"]) / a[t]
        if not (aggr >= 0.55 and cl >= 0.5 and slope_atr <= MAX_SLOPE_ATR) or sw_hi[t] < 0:
            continue
        stop = hi[sw_hi[t]] + STOP_BUF * a[t]
        px = c[t]
        risk = stop - px
        if not (0.3 * a[t] <= risk <= 4.0 * a[t]):
            continue
        r3, ex = two_targets(o, hi, lo, c, f, t, side, px, stop, 3.0, 3.0, False, HOLD[TF], TAKER)
        row = {"symbol": sym, "t": d.index[t], "R3": r3, "fail_lag": ev["fail_lag"], "hit_target": bool(r3 > 2.5)}
        for b in BUFS:
            for w in WINDOWS:
                r, _, by_line = line_exit(o, hi, lo, c, f, t, side, px, stop, 3.0, HOLD[TF], TAKER, ev["level"], t,
                                          ev["slope"], b * a[t], w)
                row[f"R_b{b:g}_w{w}"], row[f"x_b{b:g}_w{w}"] = r, bool(by_line)
        rows.append(row)
    return pd.DataFrame(rows)


def collect(root: Path, syms: list[str]) -> None:
    parts, advs = [], []
    for s in mine(syms):
        try:
            a = adv30(root, s).dropna()
            advs.append(pd.DataFrame({"symbol": s, "day": a.index, "adv": a.to_numpy()}))
            d = tf_frame(root, s, TF)
            if d is None or len(d) < 500:
                continue
            x = coin_rows(d, s)
            if len(x):
                parts.append(x)
        except Exception as e:
            print(f"  lineexit15 {s}: пропуск ({e})", flush=True)
    print(f"  lineexit15: монет со сделками {len(parts)}", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path("lineexit15"), index=False)
    if advs:
        pd.concat(advs, ignore_index=True).to_parquet(part_path("lineexit15_adv"), index=False)


def report() -> None:
    tp, ap = all_parts("lineexit15"), all_parts("lineexit15_adv")
    if not tp or not ap:
        print("lineexit15: частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in tp], ignore_index=True)
    adv = pd.concat([pd.read_parquet(p) for p in ap], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    adv["day"] = pd.to_datetime(adv["day"], utc=True)
    adv["rank"] = adv.groupby("day")["adv"].rank(ascending=False, method="first")
    df["day"] = df.t.dt.floor("D")
    df = df.merge(adv[["symbol", "day", "adv", "rank"]], on=["symbol", "day"], how="left")
    df = df[(df.adv >= ADV_MIN) & (df["rank"] <= TOP15)].reset_index(drop=True)
    df["per"] = ""
    for p, (a, b) in PER.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    cols = list(PER) + ["2026"]

    def cells(col: str, z: pd.DataFrame) -> dict:
        zz = z.assign(R=z[col])
        return {**{p: _cell(zz[zz.per == p]) for p in PER}, "2026": _cell(zz[zz.t.dt.year == 2026])}

    print(f"\n===== LINEEXIT15: правило бота 15m ({len(df)} сделок), ранний выход по закрытию обратно за линией; "
          "R на сделку, 3R =====")
    rows = [{"вариант": "контроль: стоп / цель / 200 свечей", "вышли по линии": "—", "из них дошли бы до цели": "—",
             **cells("R3", df)}]
    for b in BUFS:
        for w in WINDOWS:
            x = df[f"x_b{b:g}_w{w}"]
            rows.append({"вариант": f"за линией > {b:g} ATR, окно {w} свечей ({w * 15 / 60:g} ч)",
                         "вышли по линии": f"{x.mean():.0%}",
                         "из них дошли бы до цели": f"{df.loc[x, 'hit_target'].mean():.0%}" if x.any() else "—",
                         **cells(f"R_b{b:g}_w{w}", df)})
    print(pd.DataFrame(rows)[["вариант", "вышли по линии", "из них дошли бы до цели"] + cols].to_string(index=False))
    best = max(((b, w) for b in BUFS for w in WINDOWS),
               key=lambda bw: df.loc[df.per == "is", f"R_b{bw[0]:g}_w{bw[1]}"].mean())
    col = f"R_b{best[0]:g}_w{best[1]}"
    d = df[col] - df["R3"]
    g = d.groupby(df.t.dt.floor("D")).sum()
    print(f"  лучший по IS: буфер {best[0]:g} ATR, окно {best[1]} — разница с контролем на сделку {d.mean():+.3f}R, "
          f"t по дням {d.sum() / np.sqrt((g ** 2).sum()):.1f}; по периодам "
          + ", ".join(f"{p} {d[df.per == p].mean():+.3f}" for p in PER) + f", 2026 {d[df.t.dt.year == 2026].mean():+.3f}")
    print(f"  контроль по годам: {_years(df, 'R3')}")
    print(f"  лучший по IS по годам: {_years(df, col)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 400)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    syms = [x for x in a.symbols.split(",") if x]
    {"collect": lambda: collect(Path(a.root).expanduser(), syms), "report": report}[a.mode]()
