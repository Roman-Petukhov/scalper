"""Горячие монеты в устойчивом росте: лонг на откате и удержание, пока тренд жив (4h).

Правила заданы до запуска:
    горячая     на последнем закрытом дне: оборот от $20M, рост за 30 дней в верхних 10% ликвидных монет этого дня,
                закрытие выше дневной EMA50, рост ровный — самый большой дневной рост за 30 дней не больше половины
                всего 30-дневного (лог-доходности), то есть не одна свеча-памп
    в тренде    оборот от $20M и закрытие дня выше EMA50 (то же без «горячей» части) — для сравнения
Входы (по рынку на закрытии свечи 4h; на монете одна позиция за раз):
    ema20       откат: минимум свечи коснулся EMA20 4h, закрытие выше неё, прошлое закрытие тоже выше;
                стоп — минимум 6 свечей (сутки) − 0.1 ATR
    линия       пробой вверх линии коррекции (зигзаг 3 ATR, research/noline.py); стоп за свингом PIV5 − 0.1 ATR
    максимум    закрытие выше максимума закрытий 120 свечей (20 дней) — вход Дончиана; стоп как у «линии»
    риск 0.3–4 ATR, иначе сигнал пропускается
Выходы: «середина» — закрытие 4h не выше середины канала закрытий 120 свечей (держим, пока тренд жив), до 360 свечей;
        «3R» — цель 3R, до 60 свечей. Стоп проверяется первым. Taker на входе и выходе по стопу / середине, maker на цели,
        funding включён. R — в риске позиции.
Принятие: для пары вход × выход горячие лучше «в тренде, не горячих» и R > 0 во всех периодах 2021 / IS / VAL / HO /
2026, где сделок ≥ 10. Пар 6.

    python -m research.hotpull collect --symbols <1h-qualified>
    python -m research.hotpull report
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
from .tline import MAKER, PIV, STOP_BUF, TAKER, last_confirmed, pivots, tf_frame

TF = "4h"
CH = 120                 # свечей 4h в канале (20 дней)
HOT_Q = 0.90
ENTRIES = ("ema20", "линия", "максимум")
EXITS = {"середина": (None, 360), "3R": (3.0, 60)}


def daily_table(d1: pd.DataFrame, adv: pd.Series, sym: str) -> pd.DataFrame:
    """Признаки дня D (известны на закрытии D): рост за 30 дней, ровность, выше EMA50, оборот."""
    c = d1["close"]
    lr = np.log(c).diff()
    r30 = np.log(c / c.shift(30))
    smooth = lr.rolling(30).max() <= 0.5 * r30
    return pd.DataFrame({"symbol": sym, "day": d1.index, "r30": r30.to_numpy(), "smooth": smooth.to_numpy(),
                         "above": (c > c.ewm(span=50, adjust=False).mean()).to_numpy(),
                         "adv": adv.reindex(d1.index).to_numpy()})


def ride(o, h, lo, c, f, mid, e, entry, stop, target_r, max_hold) -> tuple[float, int]:
    """Лонг по рынку на закрытии бара e. Результат в R и бар выхода."""
    risk = entry - stop
    tp = entry + target_r * risk if target_r else None
    paid = 0.0
    for j in range(e + 1, len(c)):
        paid += f[j]
        if lo[j] <= stop:
            px, fee = min(stop, o[j]), TAKER
        elif tp is not None and h[j] >= tp:
            px, fee = tp, MAKER
        elif tp is None and c[j] <= mid[j]:
            px, fee = c[j], TAKER
        elif j - e >= max_hold or j == len(c) - 1:
            px, fee = c[j], TAKER
        else:
            continue
        return (px - entry - (TAKER + fee) * entry - paid * entry) / risk, j
    return np.nan, len(c)


def entry_signals(d: pd.DataFrame, a: np.ndarray) -> dict[str, list[tuple[int, float]]]:
    """Бар сигнала и стоп для каждого входа (лонг)."""
    hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("high", "low", "close"))
    ema = d["close"].ewm(span=20, adjust=False).mean().to_numpy()
    low6 = d["low"].rolling(6).min().to_numpy()
    sw = last_confirmed(pivots(lo, PIV, False), len(c))
    prev_max = d["close"].shift(1).rolling(CH).max().to_numpy()
    out: dict[str, list[tuple[int, float]]] = {k: [] for k in ENTRIES}
    for t in range(1, len(c)):
        if lo[t] <= ema[t] < c[t] and c[t - 1] > ema[t - 1]:
            out["ema20"].append((t, low6[t] - STOP_BUF * a[t]))
        if c[t] > prev_max[t] and not c[t - 1] > prev_max[t - 1] and sw[t] >= 0:
            out["максимум"].append((t, lo[sw[t]] - STOP_BUF * a[t]))
    for _, t, side, _ in triggers(d, a, lines_only=True):
        if side > 0 and sw[t] >= 0:
            out["линия"].append((t, lo[sw[t]] - STOP_BUF * a[t]))
    return out


def coin_rows(d: pd.DataFrame, sym: str) -> list[dict]:
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64")
    a = _atr(d).to_numpy()
    ch = d["close"].shift(1).rolling(CH)
    mid = ((ch.max() + ch.min()) / 2).to_numpy()
    rows = []
    for name, sigs in entry_signals(d, a).items():
        for ex, (k, hold) in EXITS.items():
            free = 0
            for t, stop in sorted(sigs):
                if t < free or t >= len(c) - 1 or not (a[t] > 0) or not np.isfinite(stop):
                    continue
                risk = c[t] - stop
                if not (0.3 * a[t] <= risk <= 4.0 * a[t]):
                    continue
                r, j = ride(o, hi, lo, c, f, mid, t, c[t], stop, k, hold)
                free = j + 1
                known = d.index[t] + pd.Timedelta(hours=4)
                rows.append({"symbol": sym, "t": d.index[t], "day": known.floor("D") - pd.Timedelta(days=1),
                             "entry": name, "exit": ex, "R": r, "bars": j - t})
    return rows


def collect(root: Path, syms: list[str]) -> None:
    rows, days = [], []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, TF)
            d1 = tf_frame(root, s, "1d")
            if d is None or d1 is None or len(d) < 500:
                continue
            days.append(daily_table(d1, adv30(root, s), s))
            rows += coin_rows(d, s)
        except Exception as e:
            print(f"  hotpull {s}: пропуск ({e})", flush=True)
    print(f"  hotpull: сделок {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("hotpull"), index=False)
        pd.concat(days, ignore_index=True).to_parquet(part_path("hotpull_days"), index=False)


def label(tr: pd.DataFrame, days: pd.DataFrame) -> pd.DataFrame:
    """Метки сделок по дню перед сигналом: trend (оборот + выше EMA50) и hot (плюс верхние 10% роста и ровность)."""
    liq = days[days.adv >= ADV_MIN].copy()
    liq["q"] = liq.groupby("day").r30.transform(lambda x: x.quantile(HOT_Q))
    liq["trend"] = liq.above.astype(bool)
    liq["hot"] = liq.trend & (liq.r30 >= liq.q) & liq.smooth.astype(bool) & (liq.r30 > 0)
    out = tr.merge(liq[["symbol", "day", "trend", "hot"]], on=["symbol", "day"], how="left")
    out[["trend", "hot"]] = out[["trend", "hot"]].fillna(False).astype(bool)
    return out


def report() -> None:
    parts, dparts = all_parts("hotpull"), all_parts("hotpull_days")
    print("\n===== HOTPULL: горячие монеты (верхние 10% роста за 30 дней, ровный рост, выше EMA50) — лонг на откате "
          "и удержание по середине канала 20 дней; 4h, R в риске позиции =====")
    if not parts or not dparts:
        print("частей нет")
        return
    tr = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    days = pd.concat([pd.read_parquet(p) for p in dparts], ignore_index=True)
    for x in (tr, days):
        x["day"] = pd.to_datetime(x["day"], utc=True)
    tr["t"] = pd.to_datetime(tr["t"], utc=True)
    df = label(tr.dropna(subset=["R"]), days)
    per = {**PER_ALL, "2026": ("2026-01-01", "2026-10-01")}
    judged = {k: v for k, v in per.items() if k != "2020"}
    print(f"сделок {len(df)}; в тренде {df.trend.mean():.0%}, горячих {df.hot.mean():.0%}")
    for en in ENTRIES:
        for ex in EXITS:
            z = df[(df.entry == en) & (df.exit == ex)]
            hot, cold = z[z.hot], z[z.trend & ~z.hot]
            ok = []
            for a, b in judged.values():
                mh, mc = (hot.t >= a) & (hot.t < b), (cold.t >= a) & (cold.t < b)
                if mh.sum() >= 10:
                    ok.append(hot.R[mh].mean() > 0 and hot.R[mh].mean() > cold.R[mc].mean())
            verdict = "принято" if ok and all(ok) else ("мало сделок" if not ok else "не принято")
            print(f"\n  вход «{en}», выход «{ex}» → {verdict}")
            for nm, g in (("горячие", hot), ("в тренде, не горячие", cold), ("вне тренда", z[~z.trend])):
                print(f"    {nm}: {len(g)} сделок, всего {g.R.sum():+.1f}R, макс. {g.R.max():+.1f}R, медиана "
                      f"удержания {g.bars.median() / 6:.1f} дн.")
                print("      " + "  ".join(f"{p}: {_cell(g[(g.t >= a) & (g.t < b)])}" for p, (a, b) in per.items()))


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
