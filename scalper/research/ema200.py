"""EMA200 вместо / вместе с EMA50 в фильтре тренда бота и «чистая линия» — две проверки по просьбе владельца.

Сделки — правило бота без фильтра тренда (он навешивается здесь): 4h ретест (обе стороны), 15m по рынку (наклон
<= 0.04, как в живых настройках; шорты и обе стороны). Тренд бота: закрытие последней свечи старшего ТФ (дневки для 4h,
часа для 15m) против её EMA50. Варианты, заданы до запуска:
  ema200        — закрытие старшего ТФ против EMA200 вместо EMA50;
  ema50+200     — обе по старшему ТФ совпадают со стороной;
  ema50+d200    — тренд бота и закрытие дневки против дневной EMA200 (для 4h это то же, что ema50+200 — пропуск).
Принятие: ema200 — R в месяц не ниже, чем у бота (EMA50), в каждом периоде 2020 / 2021 / IS / VAL / HO и t по дням
>= 3; надстройки (ema50+200, ema50+d200) — сделки, которые они убирают из сделок бота, убыточны в каждом периоде, где их
>= 20, t по дням <= −3, и R в месяц с фильтром не ниже.

«Чистая линия»: самое далёкое закрытие за линией между опорой A и свечой пробоя (в ATR свечи пробоя; > 0 — линию уже
протыкали закрытием, хотя пробой засчитан позже: вторая опора подтверждается зигзагом не сразу). Пороги 0 / 0.1 / 0.3
ATR: сделки с проколом больше порога режутся; принятие — как у надстроек.

    python -m research.ema200 collect --symbols <монеты>      # E200_TFS=4h или 15m
    python -m research.ema200 report
"""
from __future__ import annotations

import argparse
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
from .spec import cut_verdicts, months
from .tline import HTF, _htf_bars, _to_bars, tf_frame, zz_lines

TFS = tuple(os.environ.get("E200_TFS", "4h,15m").split(","))
ENTRY = {"4h": "retest", "15m": "market"}
MAX_SLOPE = 0.04
PIERCE = (0.0, 0.1, 0.3)


def htf_sign(d: pd.DataFrame, tf: str, span: int, rule: str | None = None) -> np.ndarray:
    """+1 / −1: закрытие последней закрытой свечи старшего ТФ против её EMA(span); NaN, пока истории меньше span."""
    rule = rule or HTF[tf]
    agg = _htf_bars(d, rule)
    e = agg["close"].ewm(span=span, adjust=False).mean()
    st = np.sign(agg["close"] - e).where(np.arange(len(agg)) >= span)
    return _to_bars(st, d, rule, tf)


def pierce(c: np.ndarray, rec: dict, t: int, atr: float) -> float:
    """Самое далёкое закрытие за линией (в сторону пробоя) между опорой A и свечой пробоя, в ATR; 0 — не было."""
    a, side, slope = int(rec["a"]), int(rec["side"]), float(rec["slope"])
    if rec.get("log"):
        return float("nan")
    k = np.arange(a + 1, t)
    if not len(k):
        return 0.0
    line = c[a] + slope * (k - a)
    return float(max(0.0, np.max(side * (c[a + 1:t] - line))) / atr)


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str) -> list[dict]:
    noline.TF = tf
    saved = noline.htf_trend
    parts = []
    try:
        for s in (1, -1):                                  # сделки бота без фильтра тренда: обе стороны «по тренду»
            noline.htf_trend = (lambda d_, tf_, s=s: np.full(len(d_), float(s)))
            x = noline.coin_trades(d, liq, sym, lines_only=True)
            if len(x):
                parts.append(x[(x.side == s) & (x.entry == ENTRY[tf])])
    finally:
        noline.htf_trend = saved
    if not parts:
        return []
    x = pd.concat(parts, ignore_index=True)
    c = d["close"].to_numpy(dtype="float64")
    atr = _atr(d).to_numpy()
    recs = {(int(r["t"]), int(r["side"])): r for r in zz_lines(d) if r["t"] > 0}
    tr = {"ema50": htf_sign(d, tf, 50), "ema200": htf_sign(d, tf, 200), "d200": htf_sign(d, tf, 200, "1D")}
    rows = []
    for r in x.itertuples(index=False):
        t, side = d.index.get_loc(r.t), int(r.side)
        rec = recs.get((t, side))
        if rec is None or not (atr[t] > 0) or not np.isfinite(r.R3):
            continue
        if tf != "4h" and abs(float(rec["slope"])) / atr[t] > MAX_SLOPE:
            continue
        rows.append({"symbol": sym, "tf": tf, "t": r.t, "side": side, "R": float(r.R3),
                     **{k: float(v[t] * side) if np.isfinite(v[t]) else np.nan for k, v in tr.items()},
                     "pierce": pierce(c, rec, t, atr[t])})
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
                print(f"  ema200 {s} {tf}: пропуск ({e})", flush=True)
        print(f"  ema200 {tf}: сделок {len(rows)}", flush=True)
        if rows:
            pd.DataFrame(rows).to_parquet(part_path(f"ema200_{tf}"), index=False)


def replace_verdict(v: pd.DataFrame, base: pd.DataFrame) -> tuple[bool, str]:
    cells, ok = [], []
    for p, (a, b) in PER_ALL.items():
        mv, mb = v[(v.t >= a) & (v.t < b)], base[(base.t >= a) & (base.t < b)]
        if mv.empty and mb.empty:
            continue
        rv, rb = mv.R.sum() / months(a, b), mb.R.sum() / months(a, b)
        cells.append(f"{p}: {_cell(mv) if len(mv) else '—'}, R в месяц {rv:+.1f} | бот {rb:+.1f}")
        ok.append(rv >= rb)
    t = day_t(v, "R") if len(v) > 10 else float("nan")
    return bool(ok) and all(ok) and t >= 3, "\n".join(f"      {c}" for c in cells) + \
        f"\n      всё время: {_cell(v)} | бот {_cell(base)}"


def report() -> None:
    print("\n===== EMA200: EMA200 в фильтре тренда бота и «чистая линия» (t по дням, прибыльных, сделок в месяц) =====")
    passed = []
    for tf in TFS:
        parts = all_parts(f"ema200_{tf}")
        if not parts:
            print(f"  ema200 {tf}: нет данных")
            continue
        x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
        x = x.dropna(subset=["R", "ema50", "ema200"])
        for label, sub in ((("обе стороны", x),) if tf == "4h" else (("шорты", x[x.side < 0]), ("обе стороны", x))):
            primary = tf == "4h" or label == "шорты"
            base = sub[sub.ema50 > 0]
            print(f"\n  --- {tf}, вход {ENTRY[tf]}, {label}: сделок бота (EMA50) {len(base)}, против тренда "
                  f"{_cell(sub[sub.ema50 < 0])} ---")
            ok, txt = replace_verdict(sub[sub.ema200 > 0], base)
            print(f"    EMA200 вместо EMA50: {'принято' if ok and primary else 'нет'}\n{txt}")
            if ok and primary:
                passed.append(f"{tf} {label} EMA200")
            adds = (("ema50+200", base.assign(f=(base.ema200 > 0).astype(float))),)
            if tf != "4h":
                adds += (("ema50+d200", base.dropna(subset=["d200"]).assign(f=lambda q: (q.d200 > 0).astype(float))),)
            for nm, z in adds:
                for _, (ok, txt) in cut_verdicts(z, "f", (0.5,)).items():
                    print(f"    {nm} (режем сделки бота против EMA200): {'принято' if ok and primary else 'нет'}\n{txt}")
                    if ok and primary:
                        passed.append(f"{tf} {label} {nm}")
            z = base.dropna(subset=["pierce"]).assign(clean=lambda q: -q.pierce)
            print(f"    прокол линии закрытием до пробоя: есть у {(z.pierce > 0).mean():.0%} сделок, "
                  f"медиана среди них {z.pierce[z.pierce > 0].median():.2f} ATR")
            for T in PIERCE:
                for _, (ok, txt) in cut_verdicts(z, "clean", (-T - 1e-12,)).items():
                    print(f"    чистая линия, режем прокол > {T:g} ATR: {'принято' if ok and primary else 'нет'}\n{txt}")
                    if ok and primary:
                        passed.append(f"{tf} {label} прокол>{T:g}")
    print("\n  ИТОГ: " + ("прошли — " + ", ".join(passed) if passed else "ни EMA200, ни «чистая линия» не приняты"))


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
