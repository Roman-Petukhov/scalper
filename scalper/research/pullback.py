"""Пробой линии коррекции по тренду (идея из ручной разметки DOT 15m): в восходящем тренде цена откатывает от вершины,
по вершинам отката ведётся нисходящая линия; лонг — закрытие выше неё. В нисходящем тренде зеркально: линия по впадинам
отскока, шорт — закрытие ниже. Бот так не торгует: его линии — по крупным разворотам (зигзаг 3 ATR) вдоль всего
движения, а пробой — против этого движения.

Правило (задано заранее, не подбирается):
    вершина A   закрытие — максимум за SWING свечей; откат подтверждён, когда закрытие ушло ниже на DEPTH ATR
    линия       из A по касательной к вершинам отката (фрактал 2 по закрытиям, известны через 2 свечи), самая пологая,
                над которой нет ни одной из этих вершин; наклон вниз
    пробой      первое закрытие выше линии, не раньше MIN_BARS свечей после A и не позже MAX_BARS; если закрытие
                обновило A раньше пробоя — откат закончился без сигнала
    вход        по рынку на закрытии пробоя или лимиткой на линии (ретест, 12 свечей)
    стоп        за минимумом отката (тени) ∓ 0.1 ATR, 0.3–4 ATR от входа
    выход       2R, 3R или «к вершине A»; срок — как у бота (15m 200, 1h 120, 4h 60 свечей); комиссии, funding
Срезы: по тренду старшего ТФ (как у бота: EMA50 старшего ТФ) / по тренду двух старших ТФ / против тренда (контроль) /
плюс фильтр агрессоров >= 55%. Оборот от $20M; для 15m — ещё топ-150 по обороту на день сигнала.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import (HOLD, HTF2, MAKER, PER, RETEST_BARS, TAKER, _per_month, _years, htf_trend, retest_fill, tf_frame,
                    two_targets)

SWING = 40
DEPTH = 1.5
MIN_BARS = 6
MAX_BARS = 60
FR = 2
STOP_BUF = 0.1


def coin_pullbacks(d: pd.DataFrame, tf: str, liq: np.ndarray, sym: str) -> pd.DataFrame:
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    a = _atr(d).to_numpy()
    tr1, tr2 = htf_trend(d, tf), htf_trend(d, tf, HTF2[tf])
    buy = (d["taker_buy_volume"] / d["volume"].replace(0, np.nan)).to_numpy()
    n = len(c)
    roll_max = pd.Series(c).rolling(SWING, min_periods=SWING).max().to_numpy()
    roll_min = pd.Series(c).rolling(SWING, min_periods=SWING).min().to_numpy()
    rows, seen = [], set()
    for s in (1, -1):                                         # s = +1: восходящий тренд, откат вниз, лонг
        ext = roll_max if s > 0 else roll_min
        for A in range(SWING, n - 2):
            if c[A] != ext[A] or not (a[A] > 0):
                continue
            conf = -1
            for j in range(A + 1, min(A + MAX_BARS, n - 1)):
                if s * (c[j] - c[A]) > 0:
                    break
                if s * (c[A] - c[j]) >= DEPTH * a[A]:
                    conf = j
                    break
            if conf < 0:
                continue
            best = None
            for t in range(conf + 1, min(A + MAX_BARS, n - 1)):
                if s * (c[t] - c[A]) > 0:
                    break                                     # обновили вершину — откат закончился без линии
                # вершины отката (фрактал FR по закрытиям), известные к закрытию t−1
                k = t - 1 - FR
                if k > A + 1 and all(s * (c[k] - c[k + i]) >= 0 and s * (c[k] - c[k - i]) >= 0 for i in range(1, FR + 1)):
                    sl = (c[k] - c[A]) / (k - A)
                    if best is None or s * sl > s * best[0]:
                        best = (sl, k)
                if best is None or not (s * best[0] < 0) or t - A < MIN_BARS:
                    continue
                sl, b = best
                lt, lp = c[A] + sl * (t - A), c[A] + sl * (t - 1 - A)
                if s * (c[t] - lt) > 0 and s * (c[t - 1] - lp) <= 0:
                    if (t, s) in seen or not (liq[t] >= ADV_MIN) or not (a[t] > 0):
                        break
                    seen.add((t, s))
                    pull_ext = lo[A: t + 1].min() if s > 0 else hi[A: t + 1].max()
                    stop = pull_ext - s * STOP_BUF * a[t]
                    row = {"symbol": sym, "tf": tf, "t": d.index[t], "side": s, "trend1": tr1[t] == s,
                           "trend2": tr2[t] == s, "aggr": buy[t] if s > 0 else 1 - buy[t],
                           "depth_atr": s * (c[A] - (c[A: t].min() if s > 0 else c[A: t].max())) / a[t],
                           "bars": t - A, "slope_atr": abs(sl) / a[t]}
                    for entry in ("market", "retest"):
                        if entry == "market":
                            fill, px, fee = t, c[t], TAKER
                        else:
                            px, fee = lt + sl, MAKER                       # линия на следующей свече
                            if not (s * (px - stop) > 0):
                                continue
                            fill = retest_fill(hi, lo, t, s, px, 3 * s * (px - stop), RETEST_BARS)
                            if fill < 0:
                                continue
                        risk = s * (px - stop)
                        if not (0.3 * a[t] <= risk <= 4.0 * a[t]):
                            continue
                        out = dict(row, entry=entry, risk_pct=risk / px)
                        stop_in_fill = entry == "retest" and s * (stop - (lo[fill] if s > 0 else hi[fill])) >= 0
                        k_a = min(max(s * (c[A] - px) / risk, 0.5), 6.0)
                        for nm, kk in (("R2", 2.0), ("R3", 3.0), ("RA", k_a)):
                            if stop_in_fill:
                                out[nm] = -1.0 - (fee + TAKER) * px / risk
                            else:
                                out[nm] = two_targets(o, hi, lo, c, f, fill, s, px, stop, kk, kk, False, HOLD[tf], fee)[0]
                        rows.append(out)
                    break
    return pd.DataFrame(rows)


def collect(root: Path, syms_h: list[str], syms15: list[str]) -> None:
    parts, advs = [], []
    for tf, syms in (("4h", syms_h), ("1h", syms_h), ("15m", syms15)):
        for s in mine(syms):
            try:
                d = tf_frame(root, s, tf)
                if d is None or len(d) < 500:
                    continue
                al = adv30(root, s)
                if tf == "15m":
                    advs.append(pd.DataFrame({"symbol": s, "day": al.dropna().index, "adv": al.dropna().to_numpy()}))
                x = coin_pullbacks(d, tf, al.reindex(d.index.floor("D")).to_numpy(), s)
                if len(x):
                    parts.append(x)
            except Exception as e:
                print(f"  pullback {tf} {s}: пропуск ({e})", flush=True)
    print(f"  pullback: частей {len(parts)}", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path("pullback"), index=False)
    if advs:
        pd.concat(advs, ignore_index=True).to_parquet(part_path("pullback_adv"), index=False)


def report() -> None:
    parts, aparts = all_parts("pullback"), all_parts("pullback_adv")
    print("\n===== PULLBACK: пробой линии коррекции по тренду (лонг после отката в восходящем тренде, шорт — зеркально); "
          "ячейка — R на сделку (t по дням, прибыльных, сделок в месяц) =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["per"] = ""
    for p, (a, b) in PER.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    if aparts:
        adv = pd.concat([pd.read_parquet(p) for p in aparts], ignore_index=True)
        adv["day"] = pd.to_datetime(adv["day"], utc=True)
        adv["rank"] = adv.groupby("day")["adv"].rank(ascending=False, method="first")
        df["day"] = df.t.dt.floor("D")
        df = df.merge(adv[["symbol", "day", "rank"]], on=["symbol", "day"], how="left")
    for tf in ("15m", "1h", "4h"):
        g0 = df[df.tf == tf]
        if tf == "15m" and "rank" in g0:
            g0 = g0[g0["rank"] <= 150]
        if not len(g0):
            continue
        for entry in ("market", "retest"):
            g = g0[g0.entry == entry]
            items = [("по тренду старшего ТФ", g[g.trend1]), ("  лонги", g[g.trend1 & (g.side > 0)]),
                     ("  шорты", g[g.trend1 & (g.side < 0)]), ("по тренду двух старших ТФ", g[g.trend1 & g.trend2]),
                     ("по тренду + агрессоры >= 55%", g[g.trend1 & (g.aggr >= 0.55)]),
                     ("против тренда (контроль)", g[~g.trend1])]
            for col, nm in (("R3", "3R"), ("R2", "2R"), ("RA", "к вершине A")):
                rows = [{"срез": s, **{p: _cell(z.assign(R=z[col])[z.per == p]) for p in PER},
                         "R/мес IS / VAL / HO": _per_month(z, col)} for s, z in items]
                print(f"\n  --- {tf}, {'рынок' if entry == 'market' else 'ретест'}, выход {nm} ---")
                print(pd.DataFrame(rows).to_string(index=False))
            z = g[g.trend1]
            print(f"  {tf} {entry}, по тренду, 3R, по годам: {_years(z)}; издержки x2: " + ", ".join(
                f"{p}: {(q.R3 - 2 * TAKER / q.risk_pct).mean():+.3f}" for p, q in z.groupby('per') if p))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 400)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--symbols15", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x],
                [x for x in a.symbols15.split(",") if x])
    else:
        report()
