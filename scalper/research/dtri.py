"""Нисходящий треугольник 4h (шорт) — кандидат из research/patterns.py — на новых данных и против сигналов бота.

1. 2020–2021 (RESEARCH_START=2020-01): на этих годах фигуры не проверялись — правило из patterns.py без изменений.
2. Пересечение с ботом: сигнал бота 4h (правило бота, ретест) по той же монете в ту же сторону в пределах ±OVERLAP
   свечей от пробоя треугольника. Если треугольник в основном повторяет бота — второй тип сигнала ничего не даст.
3. Бот + треугольники, которых нет у бота: R в месяц вместе.
Для сравнения те же таблицы для остальных фигур 4h.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import tline
from .broad import ADV_MIN, adv30
from .oos import PER_ALL
from .patterns import coin_patterns
from .shard import all_parts, mine, part_path
from .smc import _cell
from .tline import TAKER, _bot_base, _years, coin_trades, market_context, tf_frame

OVERLAP = 3
TRI = "нисходящий треугольник"


def collect(root: Path, syms: list[str]) -> None:
    ctx = market_context(root)
    tline.LINES, tline.SCALE_LINES = ("zz",), {}
    pats, bots = [], []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, "4h")
            if d is None or len(d) < 500:
                continue
            liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
            x = coin_patterns(d, "4h", liq, s)
            if len(x):
                pats.append(x)
            b = coin_trades(root, s, "4h", ctx)
            if len(b):
                bots.append(b[(b.line == "zz") & ~b.confirm][["symbol", "t", "side", "entry", "confirm", "aggr",
                                                               "with_trend", "close_loc", "adv", "R3"]])
        except Exception as e:
            print(f"  dtri {s}: пропуск ({e})", flush=True)
    print(f"  dtri: монет с фигурами {len(pats)}, с ботом {len(bots)}", flush=True)
    if pats:
        pd.concat(pats, ignore_index=True).to_parquet(part_path("dtri"), index=False)
    if bots:
        pd.concat(bots, ignore_index=True).to_parquet(part_path("dtri_bot"), index=False)


def _per(df: pd.DataFrame) -> pd.DataFrame:
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["per"] = ""
    for p, (a, b) in PER_ALL.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    return df


def _pm(z: pd.DataFrame, col: str = "R3") -> str:
    return " / ".join(f"{z[z.per == p][col].sum() / ((pd.Timestamp(b) - pd.Timestamp(a)).days / 30.4):+.1f}"
                      for p, (a, b) in PER_ALL.items())


def _table(items: list[tuple[str, pd.DataFrame]], col: str = "R3") -> None:
    rows = [{"вариант": nm, **{p: _cell(z[z.per == p].assign(R=z[col])) for p in PER_ALL},
             "R/мес " + " / ".join(PER_ALL): _pm(z, col)} for nm, z in items]
    print(pd.DataFrame(rows).to_string(index=False))


def report() -> None:
    pp, bp = all_parts("dtri"), all_parts("dtri_bot")
    print("\n===== DTRI: нисходящий треугольник 4h (шорт) на 2020–2021 и пересечение с сигналами бота; ячейка — R на "
          "сделку (t по дням, прибыльных, сделок в месяц), всё на 3R =====")
    if not pp or not bp:
        print("частей нет")
        return
    pat = _per(pd.concat([pd.read_parquet(p) for p in pp], ignore_index=True))
    bot = _per(pd.concat([pd.read_parquet(p) for p in bp], ignore_index=True))
    bot = _bot_base(bot[bot.adv.isna() | (bot.adv >= ADV_MIN)])
    bot = bot[bot.entry == "retest"]
    win = OVERLAP * pd.Timedelta(hours=4).value
    near = []
    keyed = {k: np.sort(np.array([x.value for x in g.t])) for k, g in bot.groupby(["symbol", "side"])}
    for sym, sd, t in zip(pat.symbol, pat.side, pat.t):
        arr = keyed.get((sym, sd))
        k = np.searchsorted(arr, t.value - win) if arr is not None else 0
        near.append(bool(arr is not None and k < len(arr) and arr[k] <= t.value + win))
    pat["with_bot"] = near

    tri = pat[pat.pattern == TRI]
    print("\n  --- 1. нисходящий треугольник, правило без изменений ---")
    _table([("все", tri), ("с фильтром бота", tri[tri.bot]), ("бот (ретест) для сравнения", bot)])
    print(f"  по годам: {_years(tri)}")
    print("  издержки x2: " + ", ".join(f"{p}: {(z.R3 - 2 * TAKER / z.risk_pct).mean():+.3f}"
                                     for p, z in tri.groupby('per') if p))
    for col in ("R2", "Rmm"):
        print(f"  выход {col}: " + ", ".join(f"{p}: {z[col].mean():+.3f} ({len(z)})" for p, z in tri.groupby('per') if p))

    print(f"\n  --- 2. пересечение с ботом (сигнал бота по той же монете и стороне в пределах ±{OVERLAP} свечей) ---")
    print(f"  доля треугольников, совпавших с ботом: {tri.with_bot.mean():.0%}")
    _table([("совпали с ботом", tri[tri.with_bot]), ("нет у бота", tri[~tri.with_bot])])

    print("\n  --- 3. бот + треугольники, которых нет у бота (R в месяц, просадка R) ---")
    extra = tri[~tri.with_bot][["t", "per", "R3"]]
    both = pd.concat([bot[["t", "per", "R3"]], extra], ignore_index=True)
    for nm, z in (("бот", bot), ("бот + новые треугольники", both)):
        dd = []
        for p in PER_ALL:
            eq = z[z.per == p].sort_values("t").R3.cumsum().to_numpy()
            dd.append(f"{(np.maximum.accumulate(np.r_[0, eq]) - np.r_[0, eq]).max():.1f}" if len(eq) else "—")
        print(f"  {nm}: R/мес {_pm(z)}; просадка R {' / '.join(dd)}")
    print(f"  корреляция месячных R бота и новых треугольников: "
          f"{bot.groupby(bot.t.dt.tz_convert(None).dt.to_period('M')).R3.sum().corr(extra.groupby(extra.t.dt.tz_convert(None).dt.to_period('M')).R3.sum()):+.2f}")

    print("\n  --- для сравнения: остальные фигуры 4h на 2020–2021 и далее ---")
    _table([(nm, pat[pat.pattern == nm]) for nm in sorted(pat.pattern.unique()) if nm != TRI])


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
