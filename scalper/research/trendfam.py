"""Почему Дончиан: три семейства трендовых правил в одинаковых условиях (research/dtrend.py — монеты, топ-20 по обороту,
размер по волатильности, издержки taker и funding, лонг).

Правила заданы заранее; в каждом — ансамбль тех же 9 периодов N = 5…360 дней, сигнал монеты — доля моделей в лонге:
    Дончиан     вход на закрытии не ниже максимума N прошлых дней, выход на закрытии не выше середины канала (dtrend)
    средняя     в лонге, пока закрытие выше простой средней за N дней
    моментум    в лонге, пока закрытие выше закрытия N дней назад (time-series momentum)
    все три     среднее трёх сигналов
Отчёт: доходность / волатильность / Sharpe / просадка / убыточные месяцы по периодам 2020, 2021, IS, VAL, HO и в целом,
оборот (сумма |изменений веса| в год), корреляция дневных результатов семейств; то же при издержках x2.
Принятие нет — это сравнение: если другое семейство лучше Дончиана во всех периодах, менять кандидата.

    python -m research.trendfam collect --symbols <1h-qualified>
    python -m research.trendfam report
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import adv30
from .dtrend import LOOKBACKS, _models, _stats, portfolio
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .tline import TAKER, tf_frame


def sma_signal(close: pd.DataFrame, lookbacks=LOOKBACKS) -> pd.DataFrame:
    s = sum((close > close.rolling(n, min_periods=n).mean()).astype(float) for n in lookbacks) / len(lookbacks)
    return s.where(close.notna(), 0.0)


def mom_signal(close: pd.DataFrame, lookbacks=LOOKBACKS) -> pd.DataFrame:
    s = sum((close > close.shift(n)).astype(float) for n in lookbacks) / len(lookbacks)
    return s.where(close.notna(), 0.0)


def don_signal(close: pd.DataFrame, lookbacks=LOOKBACKS) -> pd.DataFrame:
    return sum(_models(close, n, False) for n in lookbacks) / len(lookbacks)


def signals(close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {"Дончиан": don_signal(close), "средняя": sma_signal(close), "моментум": mom_signal(close)}
    out["все три"] = (out["Дончиан"] + out["средняя"] + out["моментум"]) / 3
    return out


def collect(root: Path, syms: list[str]) -> None:
    days = []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, "1d")
            if d is None or len(d) < 30:
                continue
            a = adv30(root, s).reindex(d.index)
            days.append(pd.DataFrame({"symbol": s, "day": d.index, "close": d["close"].to_numpy(),
                                      "funding": d["funding"].to_numpy(), "adv": a.to_numpy()}))
        except Exception as e:
            print(f"  trendfam {s}: пропуск ({e})", flush=True)
    print(f"  trendfam: монет {len(days)}", flush=True)
    if days:
        pd.concat(days, ignore_index=True).to_parquet(part_path("trendfam"), index=False)


def report() -> None:
    parts = all_parts("trendfam")
    print("\n===== TRENDFAM: Дончиан против средней и моментума — ансамбль 9 периодов, лонг, топ-20, размер по "
          "волатильности, издержки taker и funding =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["day"] = pd.to_datetime(df["day"], utc=True).dt.tz_localize(None)
    close = df.pivot_table(index="day", columns="symbol", values="close")
    fund = df.pivot_table(index="day", columns="symbol", values="funding").reindex_like(close)
    adv = df.pivot_table(index="day", columns="symbol", values="adv").reindex_like(close)
    print(f"дней {len(close)}, монет {close.shape[1]}, с {close.index[0]:%Y-%m-%d}")
    sig = signals(close)
    res = {}
    for nm, s in sig.items():
        res[nm] = portfolio(close, fund, adv, sig=s)
        gross = portfolio(close, fund, adv, sig=s, cost_k=0.0)
        x2 = portfolio(close, fund, adv, sig=s, cost_k=2.0)
        turn = (gross - res[nm])[gross.index >= "2020-01-01"].mean() * 365 / TAKER
        print(f"\n  {nm} (оборот {turn:.0f} капиталов в год)")
        for p, (a, b) in PER_ALL.items():
            print(f"    {p:>4}: {_stats(res[nm][(res[nm].index >= a) & (res[nm].index < b)])}")
        print(f"     всё: {_stats(res[nm][res[nm].index >= '2020-01-01'])}")
        print(f"     издержки x2, всё: {_stats(x2[x2.index >= '2020-01-01'])}")
    r = pd.DataFrame(res)
    r = r[r.index >= "2020-01-01"]
    print("\n  корреляция дневных результатов: " + ", ".join(
        f"{a} / {b} {r[a].corr(r[b]):.2f}" for i, a in enumerate(r) for b in list(r)[i + 1:]))
    print("  Sharpe по периодам: " + "; ".join(
        f"{p} " + " / ".join(f"{(lambda z: z.mean() / z.std() * np.sqrt(365))(r.loc[(r.index >= a) & (r.index < b), c]):.2f}"
                             for c in r) for p, (a, b) in PER_ALL.items()) + f"  (порядок: {', '.join(r)})")


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
