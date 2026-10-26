"""Выход по отрыву от EMA на сделках бота — «Climax Extension» Келла (документ «стратегии чемпионов», раздел 4.1): когда
цена оторвалась от 10 EMA далеко, часть прибыли фиксируется, остаток выходит при нарушении средней. Идея владельца: «если
цена ушла далеко от EMA, закрывать сделку».

Сделки — правило бота (research/noline.py: линии, агрессоры >= 55%, закрытие >= 0.5, тренд старшего ТФ): 4h ретест (обе
стороны), 15m по рынку (шорты, наклон <= 0.025 — как в живых настройках). Контроль — выход бота: вся позиция на 3R
(R3 из noline). Варианты, заданы до запуска:
  EMA        — 10 / 20 / 50 периодов того же ТФ;
  k          — 2 / 3 / 4 / 5 ATR (ATR сделки): закрытие дальше k ATR от EMA в сторону сделки;
  доля       — 0.5 (половина, остаток идёт к стопу / 3R / сроку) или 1.0 (вся позиция) на закрытии такой свечи.
Стоп и цель 3R проверяются раньше в той же свече (как two_targets). Сделка, закрывшаяся в свече входа, не меняется.
Принятие (на срез ТФ × сторона, парные сделки): средняя разность «вариант − контроль» > 0 в каждом периоде 2020 / 2021 /
IS / VAL / HO, где сделок >= 20, t по дням разности >= 3 (вариантов много — Harvey, Liu, Zhu). Решают 4h (обе стороны) и
15m шорты.

    python -m research.emaexit collect --symbols <монеты>      # EXIT_TFS=4h или 15m
    python -m research.emaexit report
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import noline
from .broad import adv30
from .oos import PER_ALL
from .ppr import day_t
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import HOLD, MAKER, TAKER, tf_frame, zz_lines

TFS = tuple(os.environ.get("EXIT_TFS", "4h,15m").split(","))
ENTRY = {"4h": "retest", "15m": "market"}
MAX_SLOPE_15M = 0.025
EMAS = (10, 20, 50)
KS = (2.0, 3.0, 4.0, 5.0)
FRACS = (0.5, 1.0)
TP_R = 3.0


def ext_exit(o, hi, lo, c, f, e: int, side: int, entry: float, stop: float, hold: int, fee: float,
             ema: np.ndarray, atr: np.ndarray, k: float, frac: float) -> float:
    """R сделки (вход на закрытии бара e) с выходом на 3R и «отрывом» от EMA: на закрытии бара, где
    side·(close − EMA) >= k·ATR, закрывается доля frac позиции. Стоп раньше тейка, тейк раньше отрыва (один бар)."""
    risk = side * (entry - stop)
    if not risk > 0:
        return np.nan
    tp = entry + side * TP_R * risk
    left, r, paid, fired = 1.0, 0.0, 0.0, False
    n = len(c)
    for j in range(e + 1, n):
        paid += f[j] * left
        if (side > 0 and lo[j] <= stop) or (side < 0 and hi[j] >= stop):
            ex = min(stop, o[j]) if side > 0 else max(stop, o[j])
            r += left * (side * (ex - entry) - (fee + TAKER) * entry) / risk
            return r - side * paid * entry / risk
        if (side > 0 and hi[j] >= tp) or (side < 0 and lo[j] <= tp):
            r += left * (side * (tp - entry) - (fee + MAKER) * entry) / risk
            return r - side * paid * entry / risk
        if not fired and np.isfinite(ema[j]) and atr[j] > 0 and side * (c[j] - ema[j]) >= k * atr[j]:
            fired = True                                      # доля закрывается один раз: остаток идёт дальше
            q = left if frac >= 1.0 else min(left, frac)
            r += q * (side * (c[j] - entry) - (fee + TAKER) * entry) / risk
            left -= q
            if left <= 1e-12:
                return r - side * paid * entry / risk
        if j - e >= hold or j == n - 1:
            r += left * (side * (c[j] - entry) - (fee + TAKER) * entry) / risk
            return r - side * paid * entry / risk
    return np.nan


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str) -> list[dict]:
    noline.TF = tf
    x = noline.coin_trades(d, liq, sym, lines_only=True)
    if not len(x):
        return []
    x = x[x.entry == ENTRY[tf]]
    if tf == "15m":
        x = x[x.side < 0]
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].fillna(0.0).to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    atr = _atr(d).to_numpy()
    emas = {n: d["close"].ewm(span=n, adjust=False, min_periods=n).mean().to_numpy() for n in EMAS}
    recs = {(int(r["t"]), int(r["side"])): r for r in zz_lines(d) if r["t"] > 0}
    fee = MAKER if ENTRY[tf] == "retest" else TAKER
    rows = []
    for r in x.itertuples(index=False):
        t, side = d.index.get_loc(r.t), int(r.side)
        rec = recs.get((t, side))
        if rec is None or not (atr[t] > 0) or not np.isfinite(r.R3):
            continue
        if tf == "15m" and abs(float(rec["slope"])) / atr[t] > MAX_SLOPE_15M:
            continue
        fill, ex = int(r.fill_i), int(r.exit_i)
        row = {"symbol": sym, "tf": tf, "t": r.t, "side": side, "R": float(r.R3)}
        immediate = ex == fill and ENTRY[tf] == "retest"        # стоп в свече входа — выход уже посчитан
        for n, k, fr in itertools.product(EMAS, KS, FRACS):
            row[f"e{n}_k{k:g}_f{fr:g}"] = float(r.R3) if immediate else ext_exit(
                o, hi, lo, c, f, fill, side, float(r.px), float(r.stop), HOLD[tf], fee, emas[n], atr, k, fr)
        rows.append(row)
    return rows


def collect(root: Path, syms: list[str]) -> None:
    for tf in TFS:
        rows = []
        for s in mine(syms):
            try:
                if tf != "4h" and not (root / f"{s}-{tf}.parquet").exists():
                    continue
                d = tf_frame(root, s, tf)
                if d is None or len(d) < 500:
                    continue
                rows += coin_rows(d, adv30(root, s).reindex(d.index.floor("D")).to_numpy(), s, tf)
            except Exception as e:
                print(f"  emaexit {s} {tf}: пропуск ({e})", flush=True)
        print(f"  emaexit {tf}: сделок {len(rows)}", flush=True)
        if rows:
            pd.DataFrame(rows).to_parquet(part_path(f"emaexit_{tf}"), index=False)


def paired_verdict(z: pd.DataFrame, col: str) -> tuple[bool, str]:
    """Вариант против контроля на тех же сделках: разность R по периодам и t по дням."""
    d = z.assign(D=z[col] - z["R"]).dropna(subset=["D"])
    cells, ok = [], []
    for p, (a, b) in PER_ALL.items():
        m = d[(d.t >= a) & (d.t < b)]
        if m.empty:
            continue
        cells.append(f"{p} {m.D.mean():+.3f}R ({len(m)})")
        if len(m) >= 20:
            ok.append(m.D.mean() > 0)
    t = day_t(d, "D") if len(d) > 10 else float("nan")
    good = bool(ok) and all(ok) and np.isfinite(t) and t >= 3
    return good, f"{'  '.join(cells)}  всё время {d.D.mean():+.3f}R (t {t:.1f}); вариант {d[col].mean():+.3f}R"


def report() -> None:
    print("\n===== EMAEXIT: выход по отрыву от EMA (Climax Extension Келла) вместо 3R на сделках бота (разность на тех "
          "же сделках, t по дням) =====")
    passed: list[str] = []
    for tf in TFS:
        parts = all_parts(f"emaexit_{tf}")
        if not parts:
            print(f"  emaexit {tf}: нет данных")
            continue
        x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
        x = x.dropna(subset=["R"])
        slices = (("обе стороны", x),) if tf == "4h" else (("шорты", x),)
        for label, z in slices:
            print(f"\n  --- {tf}, {label}: сделок {len(z)}, контроль 3R: {_cell(z)} ---")
            for n, k, fr in itertools.product(EMAS, KS, FRACS):
                col = f"e{n}_k{k:g}_f{fr:g}"
                ok, txt = paired_verdict(z, col)
                print(f"    EMA{n}, k {k:g} ATR, доля {fr:g}: {txt} → {'принято' if ok else 'нет'}")
                if ok:
                    passed.append(f"{tf} {label}: EMA{n} k {k:g} доля {fr:g}")
    print("\n  ИТОГ: " + ("прошли — " + "; ".join(passed) if passed else
                         "выход по отрыву от EMA не лучше выхода бота на 3R ни в одном варианте"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [s for s in a.symbols.split(",") if s])
    else:
        report()
