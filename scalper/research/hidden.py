"""Скрытый пробой линии бота 4h. Линия по значимым точкам становится известна, когда точку B подтвердил зигзаг
(разворот 3 ATR), — к этому моменту цена могла уже закрыться за линией между B и подтверждением, вернуться и пробить
«ещё раз» (пример RVNUSDT 06.10.2026: закрытия 02–04.10 под линией, сигнал — на повторном пробое). Трейдер такую
линию считает уже пробитой. Срезы: нет закрытий за линией после B / есть / глубже 0.25 ATR / последнее — за 1–12 свечей
до пробоя. Правило бота без изменений, всё на 3R, периоды 2020–2021 (RESEARCH_START=2020-01), IS, VAL, HO.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

from . import tline
from .broad import ADV_MIN
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _cell
from .tline import TAKER, _bot_base, _years, coin_trades, market_context

COLS = ["symbol", "t", "side", "line", "entry", "confirm", "aggr", "with_trend", "close_loc", "adv", "risk_pct", "R3",
        "hid_n", "hid_atr", "hid_last"]


def collect(root: Path, syms: list[str]) -> None:
    ctx = market_context(root)
    tline.LINES, tline.SCALE_LINES = ("zz",), {}
    parts = []
    for s in mine(syms):
        try:
            x = coin_trades(root, s, "4h", ctx)
            if len(x):
                parts.append(x[(x.line == "zz") & ~x.confirm][COLS].copy())
        except Exception as e:
            print(f"  hidden {s}: пропуск ({e})", flush=True)
    print(f"  hidden: монет со сделками {len(parts)}", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path("hidden"), index=False)


def _per_month(z: pd.DataFrame) -> str:
    return " / ".join(f"{z[z.per == p].R3.sum() / ((pd.Timestamp(b) - pd.Timestamp(a)).days / 30.4):+.1f}"
                      for p, (a, b) in PER_ALL.items())


def report() -> None:
    parts = all_parts("hidden")
    print("\n===== HIDDEN: 4h, правило бота — закрытия за линией между точкой B и пробоем (линия уже была пробита до "
          "того, как стала известна); ячейка — R на сделку (t по дням, прибыльных, сделок в месяц), всё на 3R =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["per"] = ""
    for p, (a, b) in PER_ALL.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    base = _bot_base(df[df.adv.isna() | (df.adv >= ADV_MIN)])
    for entry in ("retest", "market"):
        g = base[base.entry == entry]
        print(f"\n  --- {'ретест (бот)' if entry == 'retest' else 'рынок'}: доля сделок со скрытым пробоем "
              f"{(g.hid_n > 0).mean():.0%} ---")
        items = [("все (бот сейчас)", g), ("чистая линия: после B закрытий за линией нет", g[g.hid_n == 0]),
                 ("скрытый пробой: были закрытия за линией", g[g.hid_n > 0]),
                 ("  глубже 0.25 ATR", g[g.hid_atr > 0.25]), ("  не глубже 0.25 ATR", g[(g.hid_n > 0) & (g.hid_atr <= 0.25)]),
                 ("  последнее за 1–12 свечей до пробоя", g[(g.hid_last >= 1) & (g.hid_last <= 12)]),
                 ("  3+ закрытий за линией", g[g.hid_n >= 3])]
        rows = [{"вариант": nm, **{p: _cell(z[z.per == p].assign(R=z.R3)) for p in PER_ALL},
                 "R/мес " + " / ".join(PER_ALL): _per_month(z)} for nm, z in items]
        print(pd.DataFrame(rows).to_string(index=False))
        for nm, z in items[1:3]:
            print(f"  {nm}: по годам {_years(z)}; издержки x2: " + ", ".join(
                f"{p}: {(q.R3 - 2 * TAKER / q.risk_pct).mean():+.3f}" for p, q in z.groupby('per') if p))


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
