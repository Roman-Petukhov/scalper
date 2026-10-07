"""Ранний выход, когда поток «выдохся»: правило бота 4h без изменений (линия, ретест, стоп, 3R, 60 свечей), меняется
только выход. Пока сделка в плюсе, на закрытии каждой свечи проверяется признак затухания; сработал — выход по
закрытию (тейкер), не дожидаясь тейка. Стоп и тейк внутри свечи проверяются раньше, как в research/tline.py.

Признаки (заданы заранее, по IS не подбираются; «наша сторона» — покупатели для лонга, продавцы для шорта):
  aggr2  — доля агрессоров нашей стороны < 50% две свечи подряд;
  aggr45 — доля агрессоров нашей стороны < 45% в одной свече;
  vol    — средний объём двух последних свечей < 50% среднего за 20 свечей до сигнала;
  div    — дивергенция: новое лучшее закрытие сделки, а дельта агрессоров за 3 свечи против нас.
Порог плюса: «> 0» (любой плюс) и «>= 1R». Свеча исполнения ретеста не проверяется (вход внутри неё).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from .broad import ADV_MIN, adv30
from .noline import TF, coin_trades
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _cell
from .targets4 import _dd, _months
from .tline import HOLD, MAKER, PER, TAKER, _bot_base, _years, market_context, tf_frame
from .tline import coin_trades as tline_trades
from .wide15 import MAX_SLOPE_ATR

TOP15 = 150

VOL_REF = 20
MODES = {"aggr2": 1, "aggr45": 2, "vol": 3, "div": 4}
GATES = {"p0": 0.0, "p1": 1.0}
VARIANTS = {f"{m}_{g}": (code, gate) for m, code in MODES.items() for g, gate in GATES.items()}
NAMES = {"aggr2": "агрессоры < 50% две свечи", "aggr45": "агрессоры < 45% одна свеча",
         "vol": "объём < 50% среднего", "div": "новый экстремум, дельта против"}


@njit(cache=True)
def exhaust_exit(o, h, lo, c, fund, own, vol, fill, side, entry, stop, k, max_hold, entry_fee, mode, gate, vref):
    """Позиция открыта в баре fill по entry; цель k·R, стоп раньше тейка. mode 0 — без раннего выхода (контроль),
    1–4 — признаки из описания модуля; ранний выход только при плюсе > gate·R (gate 0 — любой плюс).
    Возвращает (R, бар выхода, 1 — ранний выход)."""
    n = len(c)
    risk = side * (entry - stop)
    if not (risk > 0):
        return np.nan, fill, 0
    tp = entry + side * k * risk
    paid = 0.0
    best = c[fill]
    run = 0
    for j in range(fill + 1, n):
        paid += fund[j]
        if (side > 0 and lo[j] <= stop) or (side < 0 and h[j] >= stop):
            ex = min(stop, o[j]) if side > 0 else max(stop, o[j])
            return (side * (ex - entry) - (entry_fee + TAKER) * entry) / risk - side * paid * entry / risk, j, 0
        if (side > 0 and h[j] >= tp) or (side < 0 and lo[j] <= tp):
            return (side * (tp - entry) - (entry_fee + MAKER) * entry) / risk - side * paid * entry / risk, j, 0
        sh = own[j]
        flag = False
        if mode == 1:
            run = run + 1 if sh < 0.5 else 0
            flag = run >= 2
        elif mode == 2:
            flag = sh < 0.45
        elif mode == 3:
            flag = j - fill >= 2 and 0.5 * (vol[j] + vol[j - 1]) < 0.5 * vref
        elif mode == 4:
            d3 = 0.0
            for q in range(max(fill + 1, j - 2), j + 1):
                if own[q] == own[q]:
                    d3 += (2.0 * own[q] - 1.0) * vol[q]
            flag = side * (c[j] - best) > 0 and d3 < 0
        if side * (c[j] - best) > 0:
            best = c[j]
        gain = side * (c[j] - entry)
        early = mode > 0 and flag and gain > gate * risk
        if early or j - fill >= max_hold or j == n - 1:
            r = (side * (c[j] - entry) - (entry_fee + TAKER) * entry) / risk - side * paid * entry / risk
            return r, j, 1 if early else 0
    return np.nan, n - 1, 0


def trade_exits(d: pd.DataFrame, x: pd.DataFrame, tf: str = TF, fee: float = MAKER, keep: tuple[str, ...] = ()
                ) -> pd.DataFrame:
    """Для сделок x (fill_i, side, px, stop, t, risk_pct, R3) — R контроля и всех вариантов; keep — колонки x,
    которые нужно перенести (фильтры отчёта). fee — комиссия входа (ретест — maker, по рынку — taker)."""
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    vol = d["volume"].to_numpy(dtype="float64")
    buy = (d["taker_buy_volume"] / d["volume"].replace(0, np.nan)).to_numpy(dtype="float64")
    rows = []
    for r in x.itertuples(index=False):
        side, fill, px, stop = int(r.side), int(r.fill_i), float(r.px), float(r.stop)
        t = d.index.get_loc(r.t)
        own = buy if side > 0 else 1.0 - buy
        vref = float(np.nanmean(vol[max(0, t - VOL_REF):t])) if t > 0 else np.nan
        risk = side * (px - stop)
        stop_in_fill = (side > 0 and lo[fill] <= stop) or (side < 0 and hi[fill] >= stop)
        out = {"symbol": r.symbol, "t": r.t, "side": side, "risk_pct": r.risk_pct, "R3": r.R3,
               **{k: getattr(r, k) for k in keep}}
        loss = (side * (stop - px) - (fee + TAKER) * px) / risk
        for name, (code, gate) in {"ctl": (0, 0.0), **VARIANTS}.items():
            if stop_in_fill and fee == MAKER:          # по рынку вход на закрытии fill — стоп в нём уже не сработает
                out[f"R_{name}"], out[f"E_{name}"] = loss, 0
                continue
            rr, _, early = exhaust_exit(o, hi, lo, c, f, own, vol, fill, side, px, stop, 3.0, HOLD[tf], fee,
                                        code, gate, vref)
            out[f"R_{name}"], out[f"E_{name}"] = rr, early
        rows.append(out)
    return pd.DataFrame(rows)


def collect(root: Path, syms: list[str]) -> None:
    parts = []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, TF)
            if d is None or len(d) < 500:
                continue
            liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
            x = coin_trades(d, liq, s)
            if not len(x):
                continue
            x = x[(x.trig == "line") & (x.entry == "retest")]
            if len(x):
                parts.append(trade_exits(d, x))
        except Exception as e:
            print(f"  exhaust {s}: пропуск ({e})", flush=True)
    print(f"  exhaust: монет со сделками {len(parts)}", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path("exhaust"), index=False)


KEEP15 = ("entry", "aggr", "with_trend", "close_loc", "slope_atr")


def collect15(root: Path, syms: list[str]) -> None:
    """Правило бота 15m (research/hedge15.py): шорт от пологой линии зигзага, по рынку, 3R, 200 свечей."""
    ctx = market_context(root)
    parts, advs = [], []
    for s in mine(syms):
        try:
            a = adv30(root, s).dropna()
            advs.append(pd.DataFrame({"symbol": s, "day": a.index, "adv": a.to_numpy()}))
            x = tline_trades(root, s, "15m", ctx)
            if not len(x):
                continue
            x = x[~x.confirm & (x.line == "zz") & (x.side == -1) & (x.entry == "market")].copy()
            if not len(x):
                continue
            d = tf_frame(root, s, "15m")
            x["fill_i"] = d.index.get_indexer(x["t"])
            x = x[x.fill_i >= 0]
            x["px"] = d["close"].to_numpy()[x.fill_i.to_numpy()]
            x["stop"] = x.px * (1 - x.side * x.risk_pct)
            if len(x):
                parts.append(trade_exits(d, x, "15m", TAKER, KEEP15))
        except Exception as e:
            print(f"  exhaust15 {s}: пропуск ({e})", flush=True)
    print(f"  exhaust15: монет со сделками {len(parts)}", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path("exhaust15"), index=False)
    if advs:
        pd.concat(advs, ignore_index=True).to_parquet(part_path("exhaust15_adv"), index=False)


def report15() -> None:
    tp, ap = all_parts("exhaust15"), all_parts("exhaust15_adv")
    print(f"\n===== EXHAUST15: правило бота 15m (пологие шорты, по рынку, топ-{TOP15}, 3R), ранний выход при затухании "
          "потока; ячейка — R на сделку (t по дням, прибыльных, сделок в месяц) =====")
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
    base = _bot_base(df.assign(confirm=False))
    g = base[(base.adv >= ADV_MIN) & (base.slope_atr <= MAX_SLOPE_ATR) & (base["rank"] <= TOP15)].copy()
    tables(g, PER)


def report() -> None:
    parts = all_parts("exhaust")
    print("\n===== EXHAUST: правило бота 4h (линия, ретест, 3R), ранний выход при затухании потока; ячейка — R на "
          "сделку (t по дням, прибыльных, сделок в месяц) =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    tables(df, PER_ALL)


def tables(df: pd.DataFrame, periods: dict[str, tuple[str, str]]) -> None:
    df = df.copy()
    df["per"] = ""
    for p, (a, b) in periods.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    print(f"  сделок {len(df)}; контроль совпадает с ботом: "
          f"{np.isclose(df.R_ctl, df.R3, equal_nan=True).mean():.1%}")
    cols = [("ctl", "контроль: только стоп / 3R / 60 свечей")] + [
        (v, f"{NAMES[v.rsplit('_', 1)[0]]}, плюс {'> 0' if v.endswith('p0') else '>= 1R'}") for v in VARIANTS]
    rows = [{"выход": nm, **{p: _cell(df.assign(R=df[f"R_{v}"])[df.per == p]) for p in PER_ALL}} for v, nm in cols]
    print("\n  --- R на сделку ---")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n  --- как на счёте (R в месяц / худшая просадка R / убыточных месяцев) ---")
    rows = []
    for v, nm in cols:
        row = {"выход": nm}
        for p, (a, b) in periods.items():
            z = df[df.per == p]
            m = _months(z, f"R_{v}", a, b)
            row[p] = f"{m.mean():+.2f} / {_dd(z, f'R_{v}'):.1f} / {(m < 0).mean():.0%}" if len(z) else "—"
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n  --- сделки с ранним выходом: доля, R у них с ранним выходом против контроля, чем закончились бы ---")
    for v, nm in cols[1:]:
        e = df[df[f"E_{v}"] == 1]
        if not len(e):
            print(f"  {nm}: ранних выходов нет")
            continue
        print(f"  {nm}: {len(e) / len(df):.0%} сделок; R {e[f'R_{v}'].mean():+.2f} против {e.R_ctl.mean():+.2f}; "
              f"без выхода дошли бы до тейка {(e.R_ctl > 2.5).mean():.0%}, до стопа {(e.R_ctl < -0.8).mean():.0%}")
    for v, nm in cols:
        print(f"  {nm}: по годам {_years(df, f'R_{v}')}; издержки x2: " + ", ".join(
            f"{p}: {(z[f'R_{v}'] - 2 * TAKER / z.risk_pct).mean():+.3f}" for p, z in df.groupby("per") if p))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 400)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report", "collect15", "report15"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    syms = [x for x in a.symbols.split(",") if x]
    {"collect": lambda: collect(Path(a.root).expanduser(), syms), "report": report,
     "collect15": lambda: collect15(Path(a.root).expanduser(), syms), "report15": report15}[a.mode]()
