"""15m, шорт от пологой линии (стратегия бота) за пределами 70 монет прошлой проверки.

Прошлый прогон считал 15m только на фиксированном списке core16 + ext54. Здесь — все 725 монет из
symbols_qualified.json, а ликвидность берётся на момент сигнала: место монеты в рейтинге среднего дневного оборота
за 30 прошлых дней (adv30, без заглядывания вперёд) среди всех монет набора в этот день. Так видно, держится ли
плюс на монетах 71–200 и не было ли в прошлом результате эффекта «сегодняшних лидеров».

Правило — как у бота: пробой линии по значимым точкам, агрессоры >= 55%, тренд старшего ТФ, закрытие в верхней
половине свечи, шорт, наклон линии <= 0.025 ATR на свечу, вход по рынку, всё на 3R, оборот от $20M в день.
Издержки x2 / x3 — ещё 1 / 2 комиссии taker туда-обратно (запас на спред и проскальзывание у менее ликвидных монет).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

from .broad import ADV_MIN, adv30
from .shard import all_parts, mine, part_path
from .smc import _cell
from .universe import EXTRA
from .data import UNIVERSE
from .tline import PER, TAKER, _bot_base, _per_month, _years, coin_trades, market_context

MAX_SLOPE_ATR = 0.025           # порог бота (нижняя треть наклона по IS на 70 монетах)
BUCKETS = ((1, 70), (71, 130), (131, 200), (201, 10_000))
TOP_N = (70, 100, 150, 200, 300)
COLS = ["symbol", "t", "side", "entry", "aggr", "with_trend", "close_loc", "slope_atr", "risk_pct", "R3"]


def collect(root: Path, syms: list[str]) -> None:
    ctx = market_context(root)
    trades, advs = [], []
    for s in mine(syms):
        try:
            a = adv30(root, s).dropna()
            advs.append(pd.DataFrame({"symbol": s, "day": a.index, "adv": a.to_numpy()}))
            x = coin_trades(root, s, "15m", ctx)
            if len(x):
                x = x[~x.confirm & (x.line == "zz") & (x.side == -1)]
                trades.append(x[COLS])
        except Exception as e:
            print(f"  wide15 {s}: пропуск ({e})", flush=True)
    print(f"  wide15: монет со сделками {len(trades)}, с оборотом {len(advs)}", flush=True)
    if trades:
        pd.concat(trades, ignore_index=True).to_parquet(part_path("wide15"), index=False)
    if advs:
        pd.concat(advs, ignore_index=True).to_parquet(part_path("wide15_adv"), index=False)


def _rows(items: list[tuple[str, pd.DataFrame]]) -> None:
    rows = []
    for nm, z in items:
        rows.append({"вариант": nm, **{p: _cell(z[z.per == p].assign(R=z["R3"])) for p in PER},
                     "R/мес IS / VAL / HO": _per_month(z, "R3")})
    print(pd.DataFrame(rows).to_string(index=False))


def _costs(z: pd.DataFrame, k: int) -> pd.DataFrame:
    """Ещё k комиссий taker туда-обратно, в R."""
    return z.assign(R3=z["R3"] - k * 2 * TAKER / z["risk_pct"])


def report() -> None:
    tp, ap = all_parts("wide15"), all_parts("wide15_adv")
    print("\n===== WIDE15: 15m, шорт от пологой линии (правило бота) на 725 монетах; место в рейтинге оборота — на день "
          "сигнала; ячейка — R на сделку (t по дням, прибыльных, сделок в месяц); R/мес — сумма R за месяц =====")
    if not tp or not ap:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in tp], ignore_index=True)
    adv = pd.concat([pd.read_parquet(p) for p in ap], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    adv["day"] = pd.to_datetime(adv["day"], utc=True)
    adv["rank"] = adv.groupby("day")["adv"].rank(ascending=False, method="first")
    df["day"] = df.t.dt.floor("D")
    df = df.merge(adv[["symbol", "day", "adv", "rank"]], on=["symbol", "day"], how="left")
    df["per"] = ""
    for p, (a, b) in PER.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    base = _bot_base(df.assign(confirm=False))
    base = base[(base.entry == "market") & (base.adv >= ADV_MIN)]
    gentle = base[base.slope_atr <= MAX_SLOPE_ATR]
    per_day = adv[adv.adv >= ADV_MIN].groupby("day").size()
    print(f"монет с данными {adv.symbol.nunique()}, со сделками {df.symbol.nunique()}; монет с оборотом от $20M в день: "
          f"медиана {per_day.median():.0f}, по годам " + ", ".join(
              f"{y}: {v:.0f}" for y, v in per_day.groupby(per_day.index.year).median().items()))
    print(f"порог наклона по IS на всех монетах (нижняя треть): {base[base.per == 'is'].slope_atr.quantile(1 / 3):.4f} "
          f"ATR/свечу; бот — {MAX_SLOPE_ATR}")

    print("\n  --- группы по месту в рейтинге оборота на день сигнала ---")
    items = []
    for lo, hi in BUCKETS:
        z = gentle[(gentle["rank"] >= lo) & (gentle["rank"] <= hi)]
        nm = f"места {lo}–{hi}" if hi < 10_000 else f"места {lo}+"
        items += [(nm, z), ("  издержки x2", _costs(z, 1)), ("  издержки x3", _costs(z, 2))]
    _rows(items)

    print("\n  --- что даст настройка «Монет: самые ликвидные» = N (все пологие шорты с местом <= N) ---")
    items = []
    for n in TOP_N:
        z = gentle[gentle["rank"] <= n]
        items += [(f"топ-{n}", z), ("  издержки x2", _costs(z, 1))]
    items += [("все с оборотом от $20M", gentle), ("  издержки x2", _costs(gentle, 1))]
    _rows(items)

    core = set(UNIVERSE) | set(EXTRA)
    print("\n  --- список прошлой проверки (core16 + ext54) против топ-70 на день сигнала ---")
    _rows([("список core16 + ext54 (как в прошлом прогоне)", gentle[gentle.symbol.isin(core)]),
           ("топ-70 на день сигнала", gentle[gentle["rank"] <= 70]),
           ("  из них не в списке", gentle[(gentle["rank"] <= 70) & ~gentle.symbol.isin(core)])])

    print("\n  --- для сравнения: все шорты 15m без фильтра наклона ---")
    _rows([(f"топ-{n}", base[base["rank"] <= n]) for n in (70, 200)] + [("все", base)])

    print("\n  по годам:")
    for nm, z in (("топ-70", gentle[gentle["rank"] <= 70]), ("места 71–200",
                                                            gentle[(gentle["rank"] > 70) & (gentle["rank"] <= 200)]),
                  ("места 201+", gentle[gentle["rank"] > 200])):
        print(f"  {nm}: {_years(z)}")
    q = pd.qcut(gentle["rank"], 5, labels=False, duplicates="drop")
    tab = gentle.assign(q=q).groupby(["q", "per"])["R3"].mean().unstack().reindex(columns=list(PER))
    edges = gentle["rank"].quantile([0, .2, .4, .6, .8, 1]).round().astype(int).tolist()
    print(f"\n  средний R по квинтилям места в рейтинге (границы мест {edges}; 0 — самые ликвидные):")
    print("  " + tab.round(3).to_string().replace("\n", "\n  "))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 300)
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("mode", choices=["collect", "report"])
    ap_.add_argument("--root", default="~/bn")
    ap_.add_argument("--symbols", default="")
    a_ = ap_.parse_args()
    if a_.mode == "collect":
        collect(Path(a_.root).expanduser(), [x for x in a_.symbols.split(",") if x])
    else:
        report()
