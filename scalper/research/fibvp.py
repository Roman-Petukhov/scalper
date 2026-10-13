"""Фибоначчи и профиль объёма на сигналах бота 4h — закрыть оставшийся теханализ.

Сигналы — правило бота (research/noline.py: пробой линии зигзага 3 ATR, оборот от $20M, агрессоры ≥ 55%, закрытие в
своей половине свечи, дневной тренд, стоп за свингом PIV5 ± 0.1 ATR). Контроль — бот: лимитка на линии 12 свечей, 3R.
Правила заданы до запуска:

A. Фибоначчи. Движение — от свинга стопа до закрытия свечи пробоя (импульс, который пробил линию).
    вход        лимитка на откате 0.382 / 0.5 / 0.618 движения (вместо линии), 12 свечей, стоп и 3R — как у бота
    цель        вход на линии, тейк на проекции 1.618 движения от точки входа (вместо 3R; R цели — от 1 до 10)
B. Профиль объёма за 30 прошлых дней (1h-свечи, объём на типичную цену (H+L+C)/3, 50 корзин по цене): POC — корзина с
   наибольшим объёмом, зона стоимости — 70% объёма вокруг POC (VAL…VAH). Метки сигнала бота:
    V1 «пробил уровень профиля»  свеча пробоя пересекла POC, VAH или VAL (между прошлым закрытием и своим)
    V2 «вышел из зоны стоимости» закрытие за зоной в сторону сделки (лонг — выше VAH, шорт — ниже VAL)
Принятие: вариант или «метка да» лучше бота / «метки нет» во всех шести периодах (2020, 2021, IS, VAL, HO, 2026)
при доле метки от 20%. R на сигнал: неисполненная лимитка — 0R (сделки нет), как в research/entrycandle.py.

    python -m research.fibvp collect --symbols <1h-qualified>
    python -m research.fibvp report
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30
from .noline import triggers
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import (HOLD, MAKER, PIV, RETEST_BARS, STOP_BUF, TAKER, htf_trend, last_confirmed, pivots, retest_fill,
                    tf_frame, two_targets)

TF = "4h"
FIBS = (0.382, 0.5, 0.618)
EXT = 1.618
VP_DAYS = 30
VP_BINS = 50
VA_SHARE = 0.70
VARIANTS = ("line", *(f"fib{f}" for f in FIBS), "ext1.618", "market")


def _trade(o, hi, lo, c, f, t, side, level, stop, k, entry_bars=RETEST_BARS) -> tuple[float, int]:
    """Лимитка на level после свечи t; результат в R (NaN — не исполнилась) и бар исполнения."""
    if not side * (level - stop) > 0:
        return np.nan, -1
    risk = side * (level - stop)
    fill = retest_fill(hi, lo, t, side, level, k * risk, entry_bars)
    if fill < 0:
        return np.nan, -1
    if (side > 0 and lo[fill] <= stop) or (side < 0 and hi[fill] >= stop):
        return (side * (stop - level) - (MAKER + TAKER) * level) / risk, fill
    r, _ = two_targets(o, hi, lo, c, f, fill, side, level, stop, k, k, False, HOLD[TF], MAKER)
    return r, fill


def value_area(price: np.ndarray, vol: np.ndarray) -> tuple[float, float, float]:
    """POC, VAL, VAH профиля объёма."""
    ok = np.isfinite(price) & np.isfinite(vol) & (vol > 0)
    price, vol = price[ok], vol[ok]
    if len(price) < 24 or price.max() <= price.min():
        return np.nan, np.nan, np.nan
    edges = np.linspace(price.min(), price.max(), VP_BINS + 1)
    b = np.clip(np.searchsorted(edges, price, side="right") - 1, 0, VP_BINS - 1)
    h = np.bincount(b, weights=vol, minlength=VP_BINS)
    mid = (edges[:-1] + edges[1:]) / 2
    p = int(np.argmax(h))
    lo_i, hi_i, acc, tot = p, p, h[p], h.sum()
    while acc < VA_SHARE * tot:
        dn = h[lo_i - 1] if lo_i > 0 else -1.0
        up = h[hi_i + 1] if hi_i < VP_BINS - 1 else -1.0
        if up >= dn:
            hi_i += 1
            acc += up
        else:
            lo_i -= 1
            acc += dn
    return float(mid[p]), float(edges[lo_i]), float(edges[hi_i + 1])


def coin_rows(d: pd.DataFrame, h1: pd.DataFrame, liq: np.ndarray, sym: str) -> list[dict]:
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64")
    a = _atr(d).to_numpy()
    trend = htf_trend(d, TF)
    buy = (d["taker_buy_volume"] / d["volume"].replace(0, np.nan)).to_numpy()
    rng = hi - lo
    loc = (c - lo) / np.where(rng > 0, rng, np.nan)
    sw_lo, sw_hi = last_confirmed(pivots(lo, PIV, False), len(c)), last_confirmed(pivots(hi, PIV, True), len(c))
    tp = ((h1["high"] + h1["low"] + h1["close"]) / 3).to_numpy()
    hv, ht = h1["volume"].to_numpy(dtype="float64"), h1.index
    rows = []
    for name, t, side, level in triggers(d, a, lines_only=True):
        if t < 1 or t >= len(c) - 1 or not (a[t] > 0) or not (liq[t] >= ADV_MIN) or trend[t] != side:
            continue
        aggr = buy[t] if side > 0 else 1 - buy[t]
        cl = loc[t] if side > 0 else 1 - loc[t]
        if not (aggr >= 0.55 and cl >= 0.5):
            continue
        sw = sw_lo[t] if side > 0 else sw_hi[t]
        if sw < 0:
            continue
        stop = (lo[sw] - STOP_BUF * a[t]) if side > 0 else (hi[sw] + STOP_BUF * a[t])
        risk_line = side * (level - stop)
        if not (0.3 * a[t] <= risk_line <= 4.0 * a[t]):
            continue
        swing = lo[sw] if side > 0 else hi[sw]
        move = c[t] - swing                                    # знак = сторона сделки
        out = {"symbol": sym, "t": d.index[t], "side": side}
        out["line"], _ = _trade(o, hi, lo, c, f, t, side, level, stop, 3.0)
        for fb in FIBS:
            lv = c[t] - fb * move
            out[f"fib{fb}"] = _trade(o, hi, lo, c, f, t, side, lv, stop, 3.0)[0] if side * move > 0 else np.nan
        k_ext = float(np.clip(EXT * side * move / risk_line, 1.0, 10.0)) if side * move > 0 else 3.0
        out["ext1.618"], _ = _trade(o, hi, lo, c, f, t, side, level, stop, k_ext)
        out["ext_k"] = k_ext
        rm = side * (c[t] - stop)
        out["market"] = (two_targets(o, hi, lo, c, f, t, side, c[t], stop, 3.0, 3.0, False, HOLD[TF], TAKER)[0]
                         if rm > 0 else np.nan)
        end = d.index[t] + pd.Timedelta(hours=4)               # профиль — только по закрытым часам до конца свечи
        sel = (ht >= end - pd.Timedelta(days=VP_DAYS)) & (ht < end - pd.Timedelta(hours=4))
        poc, val, vah = value_area(tp[sel], hv[sel])
        a_, b_ = sorted((c[t - 1], c[t]))
        out["v1"] = bool(any(a_ < x <= b_ for x in (poc, val, vah) if np.isfinite(x)))
        out["v2"] = bool((side > 0 and c[t] > vah) or (side < 0 and c[t] < val)) if np.isfinite(vah) else False
        out["vp_ok"] = bool(np.isfinite(poc))
        rows.append(out)
    return rows


def collect(root: Path, syms: list[str]) -> None:
    rows = []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, TF)
            h1 = tf_frame(root, s, "1h")
            if d is None or h1 is None or len(d) < 500:
                continue
            liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
            rows += coin_rows(d, h1, liq, s)
        except Exception as e:
            print(f"  fibvp {s}: пропуск ({e})", flush=True)
    print(f"  fibvp: сигналов {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("fibvp"), index=False)


def _sig(z: pd.DataFrame, col: str) -> pd.DataFrame:
    """R на сигнал: неисполненная лимитка — 0R."""
    return z.assign(R=z[col].fillna(0.0))


def report() -> None:
    parts = all_parts("fibvp")
    print("\n===== FIBVP: Фибоначчи (вход на откате, цель 1.618) и профиль объёма (POC / зона стоимости) на сигналах "
          "бота 4h; R на сигнал, неисполненная лимитка — 0R =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    per = {p: (df.t >= a) & (df.t < b) for p, (a, b) in {**PER_ALL, "2026": ("2026-01-01", "2026-10-01")}.items()}
    print(f"сигналов {len(df)}; с профилем объёма {df.vp_ok.mean():.0%}")
    print("\n  A. Способ входа и цель (у каждого варианта — доля исполненных)")
    base = {p: _sig(df[m], "line").R.mean() for p, m in per.items()}
    for col in VARIANTS:
        z = _sig(df, col)
        better = all(m.sum() >= 10 and _sig(df[m], col).R.mean() > base[p] for p, m in per.items())
        print(f"    {col:>9} (исполн. {df[col].notna().mean():.0%}): " + "  ".join(
            f"{p}: {_cell(z[m])}" for p, m in per.items()) + ("" if col == "line" else
                                                              f"  → {'лучше бота во всех' if better else 'не во всех'}"))
    print(f"    R цели 1.618: медиана {df.ext_k.median():.1f}R (p10 {df.ext_k.quantile(0.1):.1f}, p90 "
          f"{df.ext_k.quantile(0.9):.1f})")
    print("\n  B. Метки профиля объёма (вход как у бота, на линии)")
    v = df[df.vp_ok]
    for lab, nm in (("v1", "V1 пробил POC / VAH / VAL"), ("v2", "V2 закрытие за зоной стоимости")):
        diffs = {}
        for p, m in per.items():
            y, n_ = _sig(v[m[df.vp_ok] & v[lab]], "line"), _sig(v[m[df.vp_ok] & ~v[lab]], "line")
            diffs[p] = y.R.mean() - n_.R.mean() if len(y) >= 5 and len(n_) >= 5 else np.nan
        ok = all(np.isfinite(x) and x > 0 for x in diffs.values()) and v[lab].mean() >= 0.2
        print(f"    {nm}: доля «да» {v[lab].mean():.0%}; «да − нет», R: " + ", ".join(
            f"{p} {x:+.2f}" for p, x in diffs.items()) + f" → {'принято' if ok else 'не принято'}")
        for flag, word in ((True, "да"), (False, "нет")):
            z = _sig(v[v[lab] == flag], "line")
            print(f"      {word:>3}: " + "  ".join(f"{p}: {_cell(z[m[df.vp_ok][v[lab] == flag]])}" for p, m in per.items()))


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
