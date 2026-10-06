"""Боковики на 4h: лимитка от границы коробки (лонг у низа, шорт у верха), цель — середина или противоположная граница.

Определение флэта задано заранее (по Bulkowski, «прямоугольник»: минимум 5 касаний — 3 одной границы и 2 другой), без
подбора на данных: окно N = 40 свечей (≈ 7 суток), высота коробки 3–8 ATR, смещение закрытий от начала к концу окна
не больше 1.5 ATR, касание — экстремум в пределах 0.25 ATR от границы, на закрытии последней свечи цена внутри коробки.
Лимитка на 0.1 ATR внутри от границы, живёт 6 свечей; стоп за границей на 1 ATR; срок сделки 30 свечей; оборот от
$20M; один сигнал на сторону не чаще раза в 12 свечей. Издержки и funding как у бота (research/tline.py).
Срезы: цель (середина / противоположная граница), сторона, совпадение стороны с дневным трендом.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import MAKER, TAKER, htf_trend, retest_fill, tf_frame, two_targets

N = 40
W_MIN, W_MAX = 3.0, 8.0
DRIFT = 1.5
TOUCH = 0.25
IN = 0.1
STOP_ATR = 1.0
VALID = 6
HOLD = 30
GAP = 12


def _touches(x: np.ndarray, edge: float, a: float, top: bool) -> int:
    """Число отдельных касаний границы (подряд идущие свечи — одно касание)."""
    hit = (x >= edge - TOUCH * a) if top else (x <= edge + TOUCH * a)
    return int(np.sum(hit & ~np.r_[False, hit[:-1]]))


def range_signals(hi: np.ndarray, lo: np.ndarray, c: np.ndarray, a: np.ndarray) -> list[tuple[int, int, float, float, float]]:
    """(бар сигнала, сторона, уровень лимитки, стоп, противоположная граница). Видна только история до бара включительно."""
    out, last = [], {1: -10 ** 9, -1: -10 ** 9}
    for t in range(N, len(c) - 1):
        at = a[t]
        if not at > 0:
            continue
        w_hi, w_lo = hi[t - N + 1:t + 1], lo[t - N + 1:t + 1]
        top, bot = float(w_hi.max()), float(w_lo.min())
        if not (W_MIN * at <= top - bot <= W_MAX * at) or abs(c[t] - c[t - N + 1]) > DRIFT * at or not (bot < c[t] < top):
            continue
        n_top, n_bot = _touches(w_hi, top, at, True), _touches(w_lo, bot, at, False)
        if not ((n_top >= 3 and n_bot >= 2) or (n_top >= 2 and n_bot >= 3)):
            continue
        for side, level, stop, far in ((1, bot + IN * at, bot - STOP_ATR * at, top - IN * at),
                                       (-1, top - IN * at, top + STOP_ATR * at, bot + IN * at)):
            if t - last[side] >= GAP:
                out.append((t, side, level, stop, far))
                last[side] = t
    return out


def coin_trades(d: pd.DataFrame, liq: np.ndarray, sym: str) -> pd.DataFrame:
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    a = _atr(d).to_numpy()
    trend = htf_trend(d, "4h")
    rows = []
    for t, side, level, stop, far in range_signals(hi, lo, c, a):
        if not liq[t] >= ADV_MIN:
            continue
        risk = side * (level - stop)
        mid = (level + far) / 2
        fill = retest_fill(hi, lo, t, side, level, abs(mid - level), VALID)
        if fill < 0:
            continue
        stop_in_fill = (side > 0 and lo[fill] <= stop) or (side < 0 and hi[fill] >= stop)
        for tgt, price in (("mid", mid), ("opp", far)):
            k = side * (price - level) / risk
            if k <= 0:
                continue
            if stop_in_fill:
                r = (side * (stop - level) - (MAKER + TAKER) * level) / risk
            else:
                r, _ = two_targets(o, hi, lo, c, f, fill, side, level, stop, k, k, False, HOLD, MAKER)
            rows.append({"symbol": sym, "t": d.index[t], "side": side, "tgt": tgt, "R": r, "k": k,
                         "with_trend": bool(trend[t] == side), "risk_pct": risk / level})
    return pd.DataFrame(rows)


def collect(root: Path, syms: list[str]) -> None:
    parts = []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, "4h")
            if d is None or len(d) < 500:
                continue
            liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
            x = coin_trades(d, liq, s)
            if len(x):
                parts.append(x)
        except Exception as e:
            print(f"  range4 {s}: пропуск ({e})", flush=True)
    print(f"  range4: монет со сделками {len(parts)}", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path("range4"), index=False)


def _months(p: str) -> float:
    a, b = PER_ALL[p]
    return (pd.Timestamp(b) - pd.Timestamp(a)).days / 30.4


def report() -> None:
    parts = all_parts("range4")
    print("\n===== RANGE4: боковики 4h (коробка 40 свечей, 3+2 касания, лимитка у границы), ячейка — R на сделку (t по дням, "
          "прибыльных, сделок в месяц) =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["per"] = ""
    for p, (a, b) in PER_ALL.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    print(f"  сделок {len(df) // 2}, шортов {(df.side < 0).mean():.0%}, стоп медиана {df.risk_pct.median() * 100:.1f}% цены, "
          f"цель в R: mid {df[df.tgt == 'mid'].k.median():.2f} / opp {df[df.tgt == 'opp'].k.median():.2f}")
    cuts = {"все": lambda z: z, "лонг": lambda z: z[z.side > 0], "шорт": lambda z: z[z.side < 0],
            "по тренду": lambda z: z[z.with_trend], "против тренда": lambda z: z[~z.with_trend]}
    for tgt, nm in (("mid", "цель — середина"), ("opp", "цель — противоположная граница")):
        rows = []
        for cut, fn in cuts.items():
            z = fn(df[df.tgt == tgt])
            rows.append({"срез": cut, **{p: _cell(z[z.per == p]) for p in PER_ALL},
                         "R/мес " + "/".join(PER_ALL): " / ".join(f"{z[z.per == p].R.sum() / _months(p):+.1f}" for p in PER_ALL)})
        print(f"\n  --- {nm} ---")
        print(pd.DataFrame(rows).to_string(index=False))
    z = df[df.tgt == "mid"]
    print("  по годам (середина, все): " + ", ".join(f"{y}: {g.R.mean():+.2f} ({len(g)})" for y, g in z.groupby(z.t.dt.year)))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 400)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x])
    else:
        report()
