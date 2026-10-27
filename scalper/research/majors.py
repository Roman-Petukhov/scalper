"""Правило бота на крупных монетах отдельно — вопрос владельца «а на биткоине как лонги отрабатывают?». Бот считает
по сотням монет, а BTC / ETH в его сделках не разбирались. Литература: у BTC вне выборки правила теханализа не работают
(Hudson, Urquhart 2019), на BTC / ETH 2022–23 прибыльных правил нет (Li и др. 2023). Но BTC в 2022–2025 рос, а альты
падали к доллару — асимметрия лонгов у мейджоров может быть слабее.

Сделки — пробои линий бота (зигзаг 3 ATR) с фильтрами бота кроме агрессоров (агрессоры записываются): 15m по рынку
(наклон <= 0.025 ATR), 1h / 4h / 1d — ретест; 3R, срок бота. Набор задан до запуска: BTC, ETH и десятка крупных
(BTC ETH BNB XRP SOL ADA DOGE LTC LINK TRX — выбраны по сегодняшнему весу, это взгляд назад, не отбор по прибыли).
Принятие (только десятка, лонги, на каждом ТФ, без фильтра агрессоров и с >= 55%): R на сделку > 0 в каждом периоде
2020 / 2021 / IS / VAL / HO, где сделок >= 20 (таких >= 3), t по дням >= 3. BTC и ETH по одной — для справки: сделок
мало, вывода по ним нет.

    python -m research.majors collect --symbols BTCUSDT,ETHUSDT,...
    python -m research.majors report
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import noline
from .broad import adv30
from .oos import PER_ALL
from .ppr import day_t
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .spec import months
from .tline import tf_frame, zz_lines

MAJORS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT", "LTCUSDT", "LINKUSDT", "TRXUSDT")
TFS = tuple(os.environ.get("MJ_TFS", "15m,1h,4h,1d").split(","))
ENTRY = {"15m": "market", "1h": "retest", "4h": "retest", "1d": "retest"}
MAX_SLOPE_15M = 0.025
FILTERS = {"без фильтра агрессоров": 0.0, "агрессоры >= 55% (бот)": 0.55}
GROUPS = {"десятка": MAJORS, "BTC": ("BTCUSDT",), "ETH": ("ETHUSDT",)}


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str) -> list[dict]:
    noline.TF = tf
    x = noline.coin_trades(d, liq, sym, lines_only=True, min_aggr=0.0)
    if not len(x):
        return []
    x = x[x.entry == ENTRY[tf]]
    atr = _atr(d).to_numpy()
    recs = {(int(r["t"]), int(r["side"])): r for r in zz_lines(d) if r["t"] > 0} if tf == "15m" else {}
    rows = []
    for r in x.itertuples(index=False):
        if not np.isfinite(r.R3):
            continue
        if tf == "15m":
            t = d.index.get_loc(r.t)
            rec = recs.get((t, int(r.side)))
            if rec is None or not (atr[t] > 0) or abs(float(rec["slope"])) / atr[t] > MAX_SLOPE_15M:
                continue
        rows.append({"symbol": sym, "tf": tf, "t": r.t, "side": int(r.side), "aggr": float(r.aggr), "R": float(r.R3)})
    return rows


def collect(root: Path, syms: list[str]) -> None:
    for tf in TFS:
        rows = []
        for s in mine(syms):
            try:
                if tf == "15m" and not (root / f"{s}-15m.parquet").exists():
                    continue
                d = tf_frame(root, s, tf)
                if d is None or len(d) < 300:
                    continue
                rows += coin_rows(d, adv30(root, s).reindex(d.index.floor("D")).to_numpy(), s, tf)
            except Exception as e:
                print(f"  majors {s} {tf}: пропуск ({e})", flush=True)
        print(f"  majors {tf}: сделок {len(rows)}", flush=True)
        if rows:
            pd.DataFrame(rows).to_parquet(part_path(f"majors_{tf}"), index=False)


def verdict(v: pd.DataFrame) -> tuple[bool, str]:
    cells, pos = [], []
    for p, (a, b) in PER_ALL.items():
        m = v[(v.t >= a) & (v.t < b)]
        if m.empty:
            continue
        cells.append(f"{p} {m.R.mean():+.2f}R ({len(m)}, {m.R.sum() / months(a, b):+.1f}/мес)")
        if len(m) >= 20:
            pos.append(m.R.mean() > 0)
    t = day_t(v, "R") if len(v) > 10 else float("nan")
    return len(pos) >= 3 and all(pos) and np.isfinite(t) and t >= 3, \
        f"{'  '.join(cells)}\n          всё время {_cell(v)}, сумма {v.R.sum():+.1f}R"


def report() -> None:
    print("\n===== MAJORS: правило бота на BTC, ETH и десятке крупных (R на сделку, сделок, R в месяц) =====")
    passed = []
    for tf in TFS:
        parts = all_parts(f"majors_{tf}")
        if not parts:
            print(f"  majors {tf}: нет данных")
            continue
        x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
        print(f"\n  ===== {tf}, вход {ENTRY[tf]}: сделок {len(x)}, монет {x.symbol.nunique()} =====")
        for gname, syms in GROUPS.items():
            g = x[x.symbol.isin(syms)]
            for fname, thr in FILTERS.items():
                for label, side in (("лонги", 1), ("шорты", -1)):
                    z = g[(g.side == side) & (g.aggr >= thr)]
                    if len(z) == 0:
                        print(f"    {gname}, {fname}, {label}: сделок нет")
                        continue
                    ok, txt = verdict(z)
                    ok = ok and gname == "десятка" and side > 0
                    tag = (" → принято" if ok else " → нет") if gname == "десятка" and side > 0 else ""
                    print(f"    {gname}, {fname}, {label}{tag}\n          {txt}")
                    if ok:
                        passed.append(f"{tf} {fname}")
    print("\n  ИТОГ (лонги десятки): " + ("прошли — " + "; ".join(passed) if passed else
                                      "лонги на крупных монетах не прошли ни на одном ТФ"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default=",".join(MAJORS))
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [s for s in a.symbols.split(",") if s])
    else:
        report()
