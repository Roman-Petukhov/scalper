"""Куда деваются пробои: воронка сигналов 4h и линии помельче.

1. Воронка (линии бота — зигзаг 3 ATR): все пробои линий → оборот от $20M → агрессоры >= 55% → тренд старшего ТФ →
   закрытие в верхней половине свечи → стоп 0.3–4 ATR → лимитка ретеста исполнилась. В месяц на весь рынок, по периодам.
2. Линии помельче: зигзаг 2 и 1.5 ATR вместо 3 (больше вершин — больше линий, ближе к тому, как линии рисует трейдер).
   Сколько пробоев и сигналов и какой R на сделку с тем же правилом бота. Всё на 3R.
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
from .shard import all_parts, mine, part_path
from .smc import _cell
from .tline import _years, coin_trades, market_context, tf_frame, zz_lines

KS = {"zz": 3.0, "zz2": 2.0, "zz1_5": 1.5}
COLS = ["symbol", "t", "side", "line", "entry", "confirm", "aggr", "with_trend", "close_loc", "adv", "risk_pct", "R3"]


def collect(root: Path, syms: list[str]) -> None:
    ctx = market_context(root)
    tline.LINES, tline.SCALE_LINES = ("zz",), {"zz2": 2.0, "zz1_5": 1.5}
    raw, trades = [], []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, "4h")
            if d is None or len(d) < 500:
                continue
            liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
            for name, k in KS.items():
                br = [(r["t"], r["side"]) for r in zz_lines(d, k=k) if r["t"] > 0]
                if br:
                    raw.append(pd.DataFrame({"symbol": s, "line": name, "t": d.index[[t for t, _ in br]],
                                             "side": [sd for _, sd in br], "adv": liq[[t for t, _ in br]]}))
            x = coin_trades(root, s, "4h", ctx)
            if len(x):
                trades.append(x[~x.confirm][COLS])
        except Exception as e:
            print(f"  funnel {s}: пропуск ({e})", flush=True)
    print(f"  funnel: монет {len(raw)}, со сделками {len(trades)}", flush=True)
    if raw:
        pd.concat(raw, ignore_index=True).to_parquet(part_path("funnel_raw"), index=False)
    if trades:
        pd.concat(trades, ignore_index=True).to_parquet(part_path("funnel"), index=False)


def _per(df: pd.DataFrame) -> pd.DataFrame:
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["per"] = ""
    for p, (a, b) in PER_ALL.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    return df


def _months(p: str) -> float:
    a, b = PER_ALL[p]
    return (pd.Timestamp(b) - pd.Timestamp(a)).days / 30.4


def report() -> None:
    rp, tp = all_parts("funnel_raw"), all_parts("funnel")
    print("\n===== FUNNEL: 4h — сколько пробоев линий случается и где они отсеиваются; линии помельче (зигзаг 2 и 1.5 "
          "ATR) с тем же правилом бота =====")
    if not rp or not tp:
        print("частей нет")
        return
    raw = _per(pd.concat([pd.read_parquet(p) for p in rp], ignore_index=True))
    tr = _per(pd.concat([pd.read_parquet(p) for p in tp], ignore_index=True))
    for name, k in KS.items():
        r0 = raw[raw.line == name]
        m = tr[(tr.line == name) & (tr.entry == "market")]
        rt = tr[(tr.line == name) & (tr.entry == "retest")]
        liq_m = m[m.adv >= ADV_MIN]
        stages = [
            ("все пробои линий", r0),
            ("оборот от $20M", r0[r0.adv >= ADV_MIN]),
            ("+ стоп 0.3–4 ATR", liq_m),
            ("+ агрессоры >= 55%", liq_m[liq_m.aggr >= 0.55]),
            ("+ тренд старшего ТФ", liq_m[(liq_m.aggr >= 0.55) & liq_m.with_trend]),
            ("+ закрытие в верхней половине (сигнал бота)",
             liq_m[(liq_m.aggr >= 0.55) & liq_m.with_trend & (liq_m.close_loc >= 0.5)]),
            ("+ ретест исполнился (сделка)", rt[(rt.adv >= ADV_MIN) & (rt.aggr >= 0.55) & rt.with_trend & (rt.close_loc >= 0.5)]),
        ]
        print(f"\n  --- линии по зигзагу {k:g} ATR{' (бот)' if name == 'zz' else ''}: в месяц на весь рынок ---")
        rows = [{"этап": nm, **{p: f"{(z.per == p).sum() / _months(p):.0f}" for p in PER_ALL}} for nm, z in stages]
        print(pd.DataFrame(rows).to_string(index=False))
    print("\n  --- R на сделку с правилом бота (ретест, 3R): линии по зигзагу 3 / 2 / 1.5 ATR ---")
    rows = []
    for name, k in KS.items():
        for entry in ("retest", "market"):
            z = tr[(tr.line == name) & (tr.entry == entry) & (tr.adv >= ADV_MIN) & (tr.aggr >= 0.55) & tr.with_trend
                   & (tr.close_loc >= 0.5)]
            rows.append({"линии": f"зигзаг {k:g} ATR", "вход": "ретест" if entry == "retest" else "рынок",
                         **{p: _cell(z.assign(R=z.R3)[z.per == p]) for p in PER_ALL},
                         "R/мес " + " / ".join(PER_ALL): " / ".join(f"{z[z.per == p].R3.sum() / _months(p):+.1f}"
                                                                    for p in PER_ALL)})
    print(pd.DataFrame(rows).to_string(index=False))
    zz2 = tr[(tr.line == "zz2") & (tr.entry == "retest") & (tr.adv >= ADV_MIN) & (tr.aggr >= 0.55) & tr.with_trend
             & (tr.close_loc >= 0.5)]
    print(f"  зигзаг 2 ATR, ретест, по годам: {_years(zz2)}")


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
