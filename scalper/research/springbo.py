"""Лонг после выноса минимума: ложный пробой вниз (сбор стопов, «спринг» Вайкоффа), затем пробой нисходящей линии.

Правила (заданы до запуска), 4h:
    вынос        минимум свечи ниже последнего подтверждённого свинг-лоу (PIV5), а закрытие этой или одной из двух
                 следующих свечей — снова выше него; вынос известен на закрытии свечи возврата; дно выноса — минимум
                 от свечи пробоя до возврата
    сигнал       пробой вверх нисходящей линии зигзага 3 ATR (линии бота) не позже 30 свечей (5 дней) после возврата
    стоп         под дном выноса − 0.1 ATR (риск 0.3–4 ATR)
    вход         по рынку на закрытии свечи пробоя / лимиткой на линии (ретест, 12 свечей, как у бота)
    выход        как у бота: 3R, долив 0.5 на +1.5R со стопом в точку входа (research/dconf.py)
Сравнение — лонговые пробои линий без выноса за 5 дней (стоп как у бота — за свингом PIV5).
Фильтры (везде — закрытие в верхней половине свечи): только оборот от $20M / + дневной тренд / + тренд и агрессоры
≥ 55% (как у бота).
Принятие варианта (вход × фильтр, 6 штук): после выноса R > 0 во всех периодах 2021 / IS / VAL / HO / 2026, где
сделок ≥ 10, и больше, чем без выноса, в каждом из них.

    python -m research.springbo collect --symbols <1h-qualified>
    python -m research.springbo report
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30
from .dconf import EXITS, simulate
from .noline import triggers
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import MAKER, PIV, RETEST_BARS, STOP_BUF, TAKER, htf_trend, last_confirmed, pivots, retest_fill, tf_frame

TF = "4h"
RECLAIM, AFTER = 3, 30
BOT_EXIT = EXITS["бот (3R, долив на +1.5R)"]
FILTERS = ("оборот", "тренд", "тренд + агрессоры")
ENTRIES = ("рынок", "ретест")


def sweeps(lo: np.ndarray, c: np.ndarray, sw: np.ndarray) -> list[tuple[int, float]]:
    """(бар возврата, дно выноса) для каждого выноса последнего подтверждённого свинг-лоу."""
    out, used = [], -1
    for s in range(1, len(c)):
        p = sw[s - 1]
        if p < 0 or p == used or not lo[s] < lo[p]:
            continue
        level = lo[p]
        for r in range(s, min(len(c), s + RECLAIM)):
            if c[r] > level:
                out.append((r, float(lo[s: r + 1].min())))
                used = p
                break
    return out


def _trade(o, hi, lo, c, f, t, level, stop, a, entry) -> float:
    if entry == "рынок":
        px, fill, fee = c[t], t, TAKER
    else:
        px, fee = level, MAKER
        fill = retest_fill(hi, lo, t, 1, level, 3.0 * (level - stop), RETEST_BARS)
        if fill < 0:
            return np.nan
        if lo[fill] <= stop:
            return (stop - level - (MAKER + TAKER) * level) / (level - stop)
    if not px - stop > 0:
        return np.nan
    return simulate(o, hi, lo, c, f, fill, 1, px, stop, a, BOT_EXIT, fee)


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str) -> list[dict]:
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64")
    a = _atr(d).to_numpy()
    trend = htf_trend(d, TF)
    buy = (d["taker_buy_volume"] / d["volume"].replace(0, np.nan)).to_numpy()
    rng = hi - lo
    loc = (c - lo) / np.where(rng > 0, rng, np.nan)
    sw = last_confirmed(pivots(lo, PIV, False), len(c))
    sp = sweeps(lo, c, sw)
    sp_bar = np.array([r for r, _ in sp], dtype=np.int64)
    rows = []
    for _, t, side, level in triggers(d, a, lines_only=True):
        if side < 0 or t >= len(c) - 1 or not (a[t] > 0) or not (liq[t] >= ADV_MIN) or not (loc[t] >= 0.5):
            continue
        k = np.searchsorted(sp_bar, t, side="right") - 1          # последний вынос, известный к закрытию t
        spring = k >= 0 and t - sp_bar[k] <= AFTER
        if spring:
            stop = sp[k][1] - STOP_BUF * a[t]
        elif sw[t] >= 0:
            stop = lo[sw[t]] - STOP_BUF * a[t]
        else:
            continue
        risk = level - stop
        if not (0.3 * a[t] <= risk <= 4.0 * a[t]):
            continue
        row = {"symbol": sym, "t": d.index[t], "spring": bool(spring), "trend": bool(trend[t] > 0),
               "aggr": bool(buy[t] >= 0.55)}
        for en in ENTRIES:
            row[en] = _trade(o, hi, lo, c, f, t, level, stop, a[t], en)
        rows.append(row)
    return rows


def collect(root: Path, syms: list[str]) -> None:
    rows = []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, TF)
            if d is None or len(d) < 500:
                continue
            liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
            rows += coin_rows(d, liq, s)
        except Exception as e:
            print(f"  springbo {s}: пропуск ({e})", flush=True)
    print(f"  springbo: лонговых пробоев {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("springbo"), index=False)


def report() -> None:
    parts = all_parts("springbo")
    print("\n===== SPRINGBO: лонг после выноса минимума и пробоя нисходящей линии (4h); стоп под дном выноса, выход как "
          "у бота =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    per = {**PER_ALL, "2026": ("2026-01-01", "2026-10-01")}
    judged = [k for k in per if k != "2020"]
    print(f"лонговых пробоев {len(df)}; после выноса {df.spring.mean():.0%}")
    masks = {"оборот": pd.Series(True, index=df.index), "тренд": df.trend, "тренд + агрессоры": df.trend & df.aggr}
    for fl in FILTERS:
        for en in ENTRIES:
            z = df[masks[fl]].assign(R=lambda x: x[en])
            y, n_ = z[z.spring], z[~z.spring]
            ok = []
            for p in judged:
                a, b = per[p]
                my, mn = y[(y.t >= a) & (y.t < b)].R.dropna(), n_[(n_.t >= a) & (n_.t < b)].R.dropna()
                if len(my) >= 10:
                    ok.append(my.mean() > 0 and my.mean() > mn.mean())
            verdict = "принято" if ok and all(ok) else f"не принято ({sum(ok)} из {len(ok)})"
            print(f"\n  {fl}, вход {en} → {verdict}")
            for nm, g in (("после выноса", y), ("без выноса", n_)):
                print(f"    {nm}: {g.R.notna().sum()} сделок, всего {g.R.sum():+.1f}R")
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
