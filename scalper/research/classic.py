"""Классические индикаторы на 4h со стандартными параметрами (без подбора): самостоятельные системы и фильтры к боту.

Индикаторы (известны на закрытии свечи): RSI 14 (Уайлдер), медленный Stochastic 14/3/3, CCI 20, MACD 12/26/9,
Ichimoku 9/26/52 (облако сдвинуто на 26 вперёд, то есть на свече t — облако, посчитанное на t − 26), дивергенция RSI:
цена обновила экстремум окна [t − 30, t − 5], а RSI ниже своего значения в том экстремуме (медвежья; бычья —
зеркально).

Системы (вход по закрытию сигнальной свечи, стоп 1.5 ATR, цель 3R, до 60 свечей, тейкер, funding; одна позиция на
монету; оборот от $20M; вариант «+ тренд» — только по дневному тренду бота, EMA50 дневок):
    RSI     лонг — RSI пересёк 30 снизу вверх, шорт — 70 сверху вниз
    STOCH   лонг — %K пересёк %D вверх ниже 20, шорт — вниз выше 80
    CCI     лонг — CCI пересёк −100 вверх, шорт — +100 вниз
    MACD    лонг / шорт — MACD пересёк сигнальную вверх / вниз
    DIV     лонг — бычья дивергенция RSI, шорт — медвежья
    ICHI    лонг — закрытие вышло над облаком при Tenkan > Kijun, шорт — зеркально
Фильтры к сигналам бота (первые пробои, агрессоры >= 55%, закрытие в своей половине, тренд, оборот; ретест и рынок),
зафиксированы до запуска — принимается фильтр, который лучше дополнения во всех шести периодах:
    F1 RSI не перекуплен по стороне (< 70)      F2 Stochastic не перекуплен (< 80)     F3 CCI по стороне < 100
    F4 гистограмма MACD по стороне > 0          F5 нет дивергенции RSI против сделки   F6 цена за облаком по стороне

    python -m research.classic collect --symbols <монеты 1h-qualified>
    python -m research.classic report
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .broad import ADV_MIN, adv30
from .oos import PER_ALL
from .rebreak import coin_rows
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import HOLD, TAKER, htf_trend, tf_frame, two_targets

TF = "4h"
SYSTEMS = ("RSI", "STOCH", "CCI", "MACD", "DIV", "ICHI")
STOP_ATR, TARGET = 1.5, 3.0
DIV_FAR, DIV_NEAR = 30, 5


def indicators(d: pd.DataFrame) -> pd.DataFrame:
    h, lo, c = d["high"], d["low"], d["close"]
    up, dn = c.diff().clip(lower=0), (-c.diff()).clip(lower=0)
    rs = up.ewm(alpha=1 / 14, adjust=False).mean() / dn.ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + rs)
    raw = 100 * (c - lo.rolling(14).min()) / (h.rolling(14).max() - lo.rolling(14).min()).replace(0, np.nan)
    k = raw.rolling(3).mean()
    tp = (h + lo + c) / 3
    md = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - tp.rolling(20).mean()) / (0.015 * md.replace(0, np.nan))
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    tenkan = (h.rolling(9).max() + lo.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + lo.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((h.rolling(52).max() + lo.rolling(52).min()) / 2).shift(26)
    out = pd.DataFrame({"rsi": rsi, "k": k, "d": k.rolling(3).mean(), "cci": cci, "macd": macd, "macd_sig": sig,
                        "tenkan": tenkan, "kijun": kijun, "cloud_hi": np.maximum(span_a, span_b),
                        "cloud_lo": np.minimum(span_a, span_b)}, index=d.index)
    bull, bear = divergences(h.to_numpy(dtype="float64"), lo.to_numpy(dtype="float64"), rsi.to_numpy(dtype="float64"))
    out["div_bull"], out["div_bear"] = bull, bear
    return out


def divergences(h: np.ndarray, lo: np.ndarray, rsi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Медвежья: high[t] > max high окна [t − 30, t − 5], а RSI[t] < RSI в баре этого максимума. Бычья — зеркально."""
    n = len(h)
    bull, bear = np.zeros(n, bool), np.zeros(n, bool)
    w = DIV_FAR - DIV_NEAR + 1
    if n <= DIV_FAR:
        return bull, bear
    hw, lw, rw = (sliding_window_view(x, w)[: n - DIV_FAR] for x in (h, lo, rsi))
    t = np.arange(DIV_FAR, n)
    ih, il = np.nanargmax(np.where(np.isnan(hw), -np.inf, hw), axis=1), np.nanargmin(np.where(np.isnan(lw), np.inf, lw), axis=1)
    rows = np.arange(len(t))
    bear[t] = (h[t] > hw[rows, ih]) & (rsi[t] < rw[rows, ih])
    bull[t] = (lo[t] < lw[rows, il]) & (rsi[t] > rw[rows, il])
    return bull, bear


def signals(x: pd.DataFrame, c: pd.Series) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    p = x.shift(1)
    cross_up = lambda a, b, pa, pb: (pa < pb) & (a >= b)       # noqa: E731
    out = {
        "RSI": (cross_up(x.rsi, 30, p.rsi, 30), cross_up(-x.rsi, -70, -p.rsi, -70)),
        "STOCH": (cross_up(x.k, x.d, p.k, p.d) & (x.k < 20), cross_up(-x.k, -x.d, -p.k, -p.d) & (x.k > 80)),
        "CCI": (cross_up(x.cci, -100, p.cci, -100), cross_up(-x.cci, -100, -p.cci, -100)),
        "MACD": (cross_up(x.macd, x.macd_sig, p.macd, p.macd_sig), cross_up(-x.macd, -x.macd_sig, -p.macd, -p.macd_sig)),
        "DIV": (x.div_bull, x.div_bear),
        "ICHI": (cross_up(c, x.cloud_hi, c.shift(1), p.cloud_hi) & (x.tenkan > x.kijun),
                 cross_up(-c, -x.cloud_lo, -c.shift(1), -p.cloud_lo) & (x.tenkan < x.kijun)),
    }
    return {k: (np.asarray(a.fillna(False) if hasattr(a, "fillna") else a, bool),
                np.asarray(b.fillna(False) if hasattr(b, "fillna") else b, bool)) for k, (a, b) in out.items()}


def system_rows(d: pd.DataFrame, x: pd.DataFrame, liq: np.ndarray, sym: str) -> pd.DataFrame:
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    a = _atr(d).to_numpy()
    trend = htf_trend(d, TF)
    rows = []
    for name, (L, S) in signals(x, d["close"]).items():
        for with_trend in (False, True):
            busy = -1
            for t in np.flatnonzero(L | S):
                if t <= busy or t >= len(c) - 1 or not (a[t] > 0) or not (liq[t] >= ADV_MIN):
                    continue
                side = 1 if L[t] else -1
                if with_trend and trend[t] != side:
                    continue
                r, ex = two_targets(o, hi, lo, c, f, t, side, c[t], c[t] - side * STOP_ATR * a[t], TARGET, TARGET,
                                    False, HOLD[TF], TAKER)
                busy = int(ex)
                rows.append((name, with_trend, sym, d.index[t], side, r))
    return pd.DataFrame(rows, columns=["system", "trend", "symbol", "t", "side", "R"])


def filter_rows(d: pd.DataFrame, x: pd.DataFrame, liq: np.ndarray, sym: str) -> pd.DataFrame:
    ev = coin_rows(d, liq, sym, TF, ("market", "retest"))
    if not len(ev):
        return ev
    ev = ev[(ev.kind == "first") & ev.with_trend & (ev.adv >= ADV_MIN) & (ev.close_loc >= 0.5) & (ev.aggr >= 0.55)]
    if not len(ev):
        return ev
    xi = x.reindex(pd.DatetimeIndex(ev["t"]))
    s = ev["side"].to_numpy()
    c = d["close"].reindex(pd.DatetimeIndex(ev["t"])).to_numpy()
    long_ = s > 0
    return pd.DataFrame({
        "symbol": sym, "t": ev["t"].to_numpy(), "side": s, "entry": ev["entry"].to_numpy(), "R3": ev["R3"].to_numpy(),
        "rsi_a": np.where(long_, xi.rsi, 100 - xi.rsi), "stoch_a": np.where(long_, xi.k, 100 - xi.k),
        "cci_a": s * xi.cci.to_numpy(), "macd_a": s * (xi.macd - xi.macd_sig).to_numpy(),
        "div_against": np.where(long_, xi.div_bear, xi.div_bull),
        "ichi_a": np.where(long_, c > xi.cloud_hi, c < xi.cloud_lo)})


def collect(root: Path, syms: list[str]) -> None:
    sy, fl = [], []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, TF)
            if d is None or len(d) < 500:
                continue
            liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
            x = indicators(d)
            a, b = system_rows(d, x, liq, s), filter_rows(d, x, liq, s)
            if len(a):
                sy.append(a)
            if len(b):
                fl.append(b)
        except Exception as e:
            print(f"  classic {s}: пропуск ({e})", flush=True)
    print(f"  classic: монет с системными сделками {len(sy)}, с сигналами бота {len(fl)}", flush=True)
    if sy:
        pd.concat(sy, ignore_index=True).to_parquet(part_path("classic_sys"), index=False)
    if fl:
        pd.concat(fl, ignore_index=True).to_parquet(part_path("classic_flt"), index=False)


# ---------- отчёт ----------

COLS = list(PER_ALL) + ["2026"]


def _per(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["per"] = ""
    for p, (a, b) in PER_ALL.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    return df


def _row(name: str, g: pd.DataFrame, col: str) -> dict:
    g = g.assign(R=g[col])
    return {"группа": name, **{p: _cell(g[g.per == p]) for p in PER_ALL}, "2026": _cell(g[g.t.dt.year == 2026])}


def _means(g: pd.DataFrame, col: str) -> list[float]:
    return [g.loc[g.per == p, col].mean() for p in PER_ALL] + [g.loc[g.t.dt.year == 2026, col].mean()]


RULES = (("F1 RSI по стороне < 70", lambda z: z.rsi_a < 70, "rsi_a"),
         ("F2 Stochastic по стороне < 80", lambda z: z.stoch_a < 80, "stoch_a"),
         ("F3 CCI по стороне < 100", lambda z: z.cci_a < 100, "cci_a"),
         ("F4 MACD-гистограмма по стороне > 0", lambda z: z.macd_a > 0, "macd_a"),
         ("F5 нет дивергенции RSI против", lambda z: ~z.div_against.astype(bool), "div_against"),
         ("F6 цена за облаком Ichimoku по стороне", lambda z: z.ichi_a.astype(bool), "ichi_a"))


def report() -> None:
    sp, fp = all_parts("classic_sys"), all_parts("classic_flt")
    if not sp or not fp:
        print("classic: частей нет")
        return
    sy = _per(pd.concat([pd.read_parquet(p) for p in sp], ignore_index=True))
    print("\n===== CLASSIC 4h: индикаторы как системы (стандартные параметры; вход по закрытию, стоп 1.5 ATR, цель 3R, "
          "до 60 свечей, оборот от $20M); R на сделку =====")
    rows = []
    for name in SYSTEMS:
        for tr in (False, True):
            g = sy[(sy.system == name) & (sy.trend == tr)]
            rows.append(_row(f"{name}{' + тренд' if tr else ''}", g, "R"))
            for side, nm in ((1, "лонг"), (-1, "шорт")):
                rows.append(_row(f"   {nm}", g[g.side == side], "R"))
    print(pd.DataFrame(rows)[["группа"] + COLS].to_string(index=False))

    fl = _per(pd.concat([pd.read_parquet(p) for p in fp], ignore_index=True))
    print("\n===== CLASSIC 4h: индикаторы как фильтры к сигналам бота (правила зафиксированы до запуска); R на сделку, "
          "3R =====")
    for entry in ("retest", "market"):
        z = fl[fl.entry == entry]
        print(f"\n  --- вход: {'ретест' if entry == 'retest' else 'по рынку'} ---")
        rows = [_row("сигналы бота", z, "R3")]
        verdicts = []
        for name, rule, col in RULES:
            valid = z[col].notna()
            yes, no = z[valid & rule(z)], z[valid & ~rule(z)]
            rows += [_row(f"{name}: да", yes, "R3"), _row(f"   нет", no, "R3")]
            diff = [a - b for a, b in zip(_means(yes, "R3"), _means(no, "R3"))]
            ok = all(np.isfinite(v) and v > 0 for v in diff)
            verdicts.append(f"  {name}: доля «да» {len(yes) / max(valid.sum(), 1):.0%}; «да − нет» "
                            + ", ".join(f"{p} {v:+.2f}" for p, v in zip(COLS, diff))
                            + f" → {'лучше во всех периодах' if ok else 'не во всех периодах'}")
        print(pd.DataFrame(rows)[["группа"] + COLS].to_string(index=False))
        print("\n".join(verdicts))


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
