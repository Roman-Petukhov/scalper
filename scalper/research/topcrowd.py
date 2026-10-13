"""Крупные правы? Позиции топ-трейдеров Binance относительно своей нормы как фильтр стороны — проверка на новых выборках.

Гипотеза из research/oilong.py: когда топ-трейдеры в шорт сильнее своей нормы (соотношение лонг / шорт ниже медианы за
30 дней), лонговые пробои 4h хуже (IS −0.39R, 2026 −0.46R). На тех же 790 лонгах повторять бессмысленно — там она
найдена. Проверка (задана до запуска) — на выборках, которые в поиске не участвовали:
    H-B  шорты бота 4h со всеми фильтрами бота (агрессоры ≥ 55%, тренд, закрытие, стоп, ретест): «крупные в шорт
         сильнее нормы» лучше, чем «крупные в лонг сильнее нормы», во всех периодах IS / VAL / HO / 2026
    H-C  лонговые пробои линий 1h и 2h без фильтра агрессоров (остальные фильтры бота): «крупные в лонг сильнее нормы»
         лучше, чем «в шорт», во всех периодах — на каждом ТФ
Гипотеза принята, если проходят обе. Для справки — 4h лонги (та же выборка, где найдено) и шорты 1h / 2h.
Сделка как у бота: ретест линии, 3R, долив 0.5 на +1.5R, срок по ТФ (research/tline.py HOLD). Данные — с метриками (2022+).

    python -m research.topcrowd collect --symbols <1h-qualified>
    python -m research.topcrowd report
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
from .noline import triggers
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import (BAR_MIN, HOLD, MAKER, PIV, RETEST_BARS, STOP_BUF, TAKER, htf_trend, last_confirmed, pivots,
                    retest_fill, tf_frame)
from .wave2 import metrics_on_bars

TFS = ("4h", "2h", "1h")


def top_dev(top: pd.Series, tf: str) -> np.ndarray:
    """log(лонг / шорт топ-трейдеров) минус медиана за 30 прошлых дней: < 0 — крупные в шорт сильнее обычного."""
    x = np.log(top.astype(float).where(top > 0))
    n = 30 * 1440 // BAR_MIN[tf]
    return (x - x.rolling(n, min_periods=n // 3).median()).to_numpy()


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str, dev: np.ndarray) -> list[dict]:
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64")
    a = _atr(d).to_numpy()
    trend = htf_trend(d, tf)
    buy = (d["taker_buy_volume"] / d["volume"].replace(0, np.nan)).to_numpy()
    rng = hi - lo
    loc = (c - lo) / np.where(rng > 0, rng, np.nan)
    sw_lo, sw_hi = last_confirmed(pivots(lo, PIV, False), len(c)), last_confirmed(pivots(hi, PIV, True), len(c))
    ex = Exit(3.0, (1.5,), 0.5, HOLD[tf])
    rows = []
    for _, t, side, level in triggers(d, a, lines_only=True):
        if t >= len(c) - 1 or not (a[t] > 0) or not (liq[t] >= ADV_MIN) or trend[t] != side:
            continue
        if not np.isfinite(dev[t]):
            continue
        cl = loc[t] if side > 0 else 1 - loc[t]
        sw = sw_lo[t] if side > 0 else sw_hi[t]
        if not (cl >= 0.5) or sw < 0:
            continue
        stop = (lo[sw] - STOP_BUF * a[t]) if side > 0 else (hi[sw] + STOP_BUF * a[t])
        risk = side * (level - stop)
        if not (0.3 * a[t] <= risk <= 4.0 * a[t]):
            continue
        fill = retest_fill(hi, lo, t, side, level, 3.0 * risk, RETEST_BARS)
        if fill < 0:
            continue
        if (side > 0 and lo[fill] <= stop) or (side < 0 and hi[fill] >= stop):
            r = (side * (stop - level) - (MAKER + TAKER) * level) / risk
        else:
            r = simulate(o, hi, lo, c, f, fill, side, level, stop, a[t], ex)
        aggr = buy[t] if side > 0 else 1 - buy[t]
        rows.append({"symbol": sym, "tf": tf, "t": d.index[t], "side": side, "aggr": bool(aggr >= 0.55),
                     "dev": dev[t], "R": r})
    return rows


def collect(root: Path, syms: list[str]) -> None:
    rows = []
    for s in mine(syms):
        try:
            for tf in TFS:
                d = tf_frame(root, s, tf)
                if d is None or len(d) < 500:
                    continue
                m = metrics_on_bars(s, root, d.index, BAR_MIN[tf])
                liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
                rows += coin_rows(d, liq, s, tf, top_dev(m["top_lsr"], tf))
        except Exception as e:
            print(f"  topcrowd {s}: пропуск ({e})", flush=True)
    print(f"  topcrowd: сделок {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("topcrowd"), index=False)


def split(z: pd.DataFrame, per: dict, good_long: bool) -> tuple[list[bool], list[str]]:
    """good_long — «крупные в лонг» должны быть лучше (для лонгов); иначе «крупные в шорт» (для шортов)."""
    ok, cells = [], []
    for p, (a, b) in per.items():
        m = z[(z.t >= a) & (z.t < b)]
        up, dn = m[m.dev >= 0], m[m.dev < 0]
        good, bad = (up, dn) if good_long else (dn, up)
        if len(good) >= 10 and len(bad) >= 10:
            ok.append(good.R.mean() > bad.R.mean())
        cells.append(f"{p}: крупные в лонг {_cell(up)} | в шорт {_cell(dn)}")
    return ok, cells


def report() -> None:
    parts = all_parts("topcrowd")
    print("\n===== TOPCROWD: позиции топ-трейдеров Binance (относительно своей нормы за 30 дней) как фильтр — проверка "
          "гипотезы «крупные правы» на новых выборках =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df.dropna(subset=["R", "dev"])
    per = {k: v for k, v in {**PER_ALL, "2026": ("2026-01-01", "2026-10-01")}.items() if k not in ("2020", "2021")}
    tests = [("H-B шорты бота 4h (с агрессорами)", df[(df.tf == "4h") & (df.side < 0) & df.aggr], False, True),
             ("H-C лонги 2h без агрессоров", df[(df.tf == "2h") & (df.side > 0)], True, True),
             ("H-C лонги 1h без агрессоров", df[(df.tf == "1h") & (df.side > 0)], True, True),
             ("справка: лонги 4h без агрессоров (где найдено)", df[(df.tf == "4h") & (df.side > 0)], True, False),
             ("справка: шорты 2h с агрессорами", df[(df.tf == "2h") & (df.side < 0) & df.aggr], False, False),
             ("справка: шорты 1h с агрессорами", df[(df.tf == "1h") & (df.side < 0) & df.aggr], False, False)]
    passed = []
    for nm, z, gl, judged in tests:
        ok, cells = split(z, per, gl)
        res = "мало сделок" if not ok else ("да" if all(ok) else f"нет ({sum(ok)} из {len(ok)})")
        if judged:
            passed.append(bool(ok) and all(ok))
        print(f"\n  {nm}: {len(z)} сделок, «крупные в лонг сильнее нормы» {(z.dev >= 0).mean():.0%}; "
              f"нужная сторона лучше во всех периодах — {res}")
        for line in cells:
            print("    " + line)
    print(f"\n  → гипотеза «крупные правы»: {'принята' if all(passed) else 'не принята'} "
          f"({sum(passed)} из {len(passed)} проверок)")


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
