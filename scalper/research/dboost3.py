"""Топ-100 для тренда «сильнейшие 7 + долив + приток»: другие каналы Дончиана для монет поменьше.

Мысль (research/dtiers.py): рост монет за пределами топ-20 — короткие пампы, медленные модели ансамбля (60–360 дней)
входят поздно и выходят после отката к середине канала. Варианты (заданы до запуска, без перебора чисел):
    A  топ-100, только быстрые каналы 5 / 10 / 20 / 30 дней
    B  топ-100: места 1–20 по обороту — полный ансамбль 5…360, места 21–100 — только быстрые 5–30
    C  топ-20, только быстрые 5–30 (контроль: помогают ли быстрые каналы сами по себе)
База — правило бумажного счёта бота: топ-20, полный ансамбль, 7 сильнейших, долив, включатель «приток стейблкоинов».
Везде тот же включатель, размер, издержки. Принятие (research/dboost.py): Sharpe выше базы во всех периодах
IS / VAL / HO / 2026 и просадка с 2022 при равной волатильности не больше ×1.5 базы. Вариантов 3.

    python -m research.dboost3 collect --symbols <1h-qualified>      (+ python -m research.extreg collect)
    python -m research.dboost3 report
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import dtrend
from .broad import ADV_MIN
from .dboost import JUDGED, PER, SINCE, _bot_months, _combo, _dd, _sharpe, judge
from .dboost2 import book, ensemble, inflow_gate, returns
from .shard import all_parts

FAST = (5, 10, 20, 30)


def split_signal(close: pd.DataFrame, adv: pd.DataFrame, top_full: int = 20) -> pd.DataFrame:
    """Места 1–top_full по обороту на дату — полный ансамбль, остальные — быстрый."""
    rk = adv.where(adv >= ADV_MIN).rank(axis=1, ascending=False, method="first")
    return ensemble(close).where(rk <= top_full, ensemble(close, FAST))


def report() -> None:
    parts, bparts, eparts = all_parts("dtrend"), all_parts("dtrend_bot"), all_parts("extreg")
    print("\n===== DBOOST3: топ-100 с быстрыми каналами для монет поменьше; база — бумажный счёт бота (топ-20, приток); "
          "ячейка — доходность в год / Sharpe =====")
    if not parts or not eparts:
        print(f"частей нет (монеты {len(parts)}, стейблкоины {len(eparts)})")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["day"] = pd.to_datetime(df["day"], utc=True).dt.tz_localize(None)
    close = df.pivot_table(index="day", columns="symbol", values="close")
    fund = df.pivot_table(index="day", columns="symbol", values="funding").reindex_like(close)
    adv = df.pivot_table(index="day", columns="symbol", values="adv").reindex_like(close)
    ext = pd.concat([pd.read_parquet(p) for p in eparts], ignore_index=True)
    ext["day"] = pd.to_datetime(ext["day"], utc=True).dt.tz_localize(None)
    gate = inflow_gate(ext.groupby("day")["stable"].last().dropna(), close.index).astype(float)
    print(f"дней {len(close)}, монет {close.shape[1]}; приток включён {gate[gate.index >= SINCE].mean():.0%} дней с 2022")

    def run(w: pd.DataFrame) -> pd.Series:
        return returns(w.mul(gate, axis=0), close, fund)

    res = {"база: топ-20, полный ансамбль, приток": run(book(close, adv)),
           "A: топ-100, только быстрые 5–30": run(book(close, adv, top=100, sig=ensemble(close, FAST))),
           "B: топ-100, 1–20 полный, 21–100 быстрые": run(book(close, adv, top=100, sig=split_signal(close, adv))),
           "C: топ-20, только быстрые 5–30": run(book(close, adv, sig=ensemble(close, FAST))),
           "для справки: топ-100, полный ансамбль": run(book(close, adv, top=100))}
    base = res["база: топ-20, полный ансамбль, приток"]
    bot_m = _bot_months(bparts) if bparts else None
    for nm, r in res.items():
        x = r[r.index >= SINCE]
        head = (f"\n  {nm}: с 2022 {x.mean() * 365:+.0%} в год, вол {x.std() * np.sqrt(365):.0%}, Sharpe {_sharpe(x):.2f}, "
                f"просадка {_dd(x):.0%}")
        if nm[0] in "ABC":
            ok, better, ratio = judge(r, base)
            head += (f"  → {'принято' if ok else 'не принято'} (Sharpe выше базы в {sum(better)} из {len(JUDGED)}, "
                     f"просадка при равной вол. ×{ratio:.2f} базы)")
        print(head)
        print("    " + "  ".join(f"{p} {r[(r.index >= a) & (r.index < b)].mean() * 365:+.0%} / "
                                 f"{_sharpe(r[(r.index >= a) & (r.index < b)]):.2f}" for p, (a, b) in PER.items()))
        if bot_m is not None:
            print("    " + _combo(bot_m, r))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        dtrend.collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x])
    else:
        report()
