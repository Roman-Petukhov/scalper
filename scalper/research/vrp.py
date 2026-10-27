"""Продажа волатильности опционами BTC / ETH — «заработок в боковике» с ограниченным риском (вопрос владельца про
сетку; премия за волатильность у BTC положительна в среднем и выше в спокойном режиме — Almeida и др., arXiv 2410.15195).

Раз в неделю, в пятницу 08:00 UTC (недельная экспирация Deribit), продаём опционы на 7 дней и держим до экспирации:
    кондор   продажа пута и колла на ±z·σ·√T, покупка крыльев на ±(z + 1)·σ·√T — убыток ограничен
    стрэнгл  только продажа пута и колла на ±z·σ·√T — без крыльев (для сравнения, хвост открыт)
Цена — Блэк-Шоулз, ставка 0, σ — DVOL Deribit (подразумеваемая волатильность на 30 дней) последнего дня до входа.
Консервативно, задано до запуска: продаём по σ × 0.9 (недельная волатильность обычно ниже 30-дневной, и улыбка),
покупаем крылья по σ × 1.1; комиссия на ногу min(0.03% базового, 12.5% цены опциона), расчёт по цене спота через 7 дней
(часовая свеча Binance 08:00). Варианты: BTC / ETH × кондор / стрэнгл × z 1.0 / 1.5 × фильтр — всегда / DVOL не ниже
медианы 90 дней / DVOL ≥ 1.2 × реализованной волатильности за 30 дней = 24.
Результат: кондор — в R (на максимальный убыток), стрэнгл — в % базового. Принятие: средний результат > 0 в каждом
периоде IS / VAL / HO (сделок >= 20), t >= 3 за всё время и плюс при комиссиях и разнице σ × 2.

    python -m research.vrp collect            (DVOL с Deribit, цены — 1h Binance BTCUSDT / ETHUSDT)
    python -m research.vrp report
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .extreg import _get, parse_dvol
from .oos import PER_ALL
from .shard import all_parts, part_path, shard_id
from .tline import tf_frame

URL = ("https://www.deribit.com/api/v2/public/get_volatility_index_data?currency={c}&start_timestamp={a}"
       "&end_timestamp={b}&resolution=43200")
START = pd.Timestamp("2021-04-01", tz="UTC")
ASSETS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
T = 7 / 365
SELL_X, BUY_X = 0.9, 1.1
FEE_UND, FEE_CAP = 0.0003, 0.125
ZS = (1.0, 1.5)
WING = 1.0
STRUCTS = ("кондор", "стрэнгл")
FILTERS = ("всегда", "DVOL ≥ медианы 90 дней", "DVOL ≥ 1.2 × реализ.")
VARIANTS = list(itertools.product(ASSETS, STRUCTS, ZS, FILTERS))


def _ncdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs(s: float, k: float, sigma: float, t: float, call: bool) -> float:
    if sigma <= 0 or t <= 0:
        return max(0.0, (s - k) if call else (k - s))
    d1 = (math.log(s / k) + 0.5 * sigma * sigma * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return s * _ncdf(d1) - k * _ncdf(d2) if call else k * _ncdf(-d2) - s * _ncdf(-d1)


def fee(s: float, price: float, mult: float) -> float:
    return mult * min(FEE_UND * s, FEE_CAP * price)


def week_trade(s0: float, s1: float, iv: float, z: float, struct: str, cost_x: float = 1.0) -> tuple[float, float]:
    """(прибыль в $ на 1 базовый, максимальный убыток в $ — inf у стрэнгла). cost_x — множитель комиссий и разницы σ."""
    w = iv * math.sqrt(T)
    sell_sig, buy_sig = iv * (1 - (1 - SELL_X) * cost_x), iv * (1 + (BUY_X - 1) * cost_x)
    kp, kc = s0 * math.exp(-z * w), s0 * math.exp(z * w)
    legs = [(kp, False, -1, sell_sig), (kc, True, -1, sell_sig)]
    if struct == "кондор":
        legs += [(s0 * math.exp(-(z + WING) * w), False, +1, buy_sig), (s0 * math.exp((z + WING) * w), True, +1, buy_sig)]
    credit, payoff = 0.0, 0.0                                   # премия за вычетом комиссий; выплаты на экспирации
    for k, call, q, sg in legs:
        p = bs(s0, k, sg, T, call)
        credit += -q * p - fee(s0, p, cost_x)
        payoff += q * max(0.0, (s1 - k) if call else (k - s1))
    if struct != "кондор":
        return credit + payoff, math.inf
    width = max(legs[3][0] - kc, kp - legs[2][0])
    return credit + payoff, width - credit


def fetch(currency: str) -> pd.Series:
    pages, end = [], int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    start = int(START.timestamp() * 1000)
    for _ in range(200):
        p = _get(URL.format(c=currency, a=start, b=end))
        if not isinstance(p, dict) or "result" not in p:
            break
        pages.append(p)
        cont = p["result"].get("continuation")
        if not cont or not p["result"].get("data") or int(cont) <= start:
            break
        end = int(cont)
    return parse_dvol(pages)


def asset_rows(asset: str, dvol: pd.Series, h1: pd.DataFrame) -> list[dict]:
    c = h1["close"]
    r = np.log(c).diff()
    rv30 = (r.rolling(24 * 30, min_periods=24 * 20).std() * math.sqrt(24 * 365))
    med90 = dvol.rolling(90, min_periods=60).median()
    rows = []
    fridays = pd.date_range(START, c.index.max() - pd.Timedelta(days=7), freq="W-FRI") + pd.Timedelta(hours=8)
    for t0 in fridays:
        t1 = t0 + pd.Timedelta(days=7)
        if t0 not in c.index or t1 not in c.index:
            continue
        day = (t0 - pd.Timedelta(days=1)).normalize()
        if day not in dvol.index or not np.isfinite(dvol[day]):
            continue
        iv = dvol[day] / 100
        s0, s1 = float(c[t0 - pd.Timedelta(hours=1)]), float(c[t1 - pd.Timedelta(hours=1)])   # закрытие часа до 08:00
        rv = float(rv30.get(t0 - pd.Timedelta(hours=1), np.nan))
        flags = {"всегда": True, "DVOL ≥ медианы 90 дней": bool(dvol[day] >= med90.get(day, np.inf)),
                 "DVOL ≥ 1.2 × реализ.": bool(np.isfinite(rv) and iv >= 1.2 * rv)}
        for struct, z in itertools.product(STRUCTS, ZS):
            pnl, ml = week_trade(s0, s1, iv, z, struct)
            pnl2, _ = week_trade(s0, s1, iv, z, struct, cost_x=2.0)
            res = (pnl / ml, pnl2 / ml) if struct == "кондор" else (pnl / s0 * 100, pnl2 / s0 * 100)
            rows.append({"asset": asset, "t": t0, "struct": struct, "z": z, "iv": iv, "move": s1 / s0 - 1,
                         "res": res[0], "res2": res[1], **{f"f{i}": flags[f] for i, f in enumerate(FILTERS)}})
    return rows


def collect(root: Path) -> None:
    if shard_id()[0] != 0:
        return
    rows = []
    for a, sym in ASSETS.items():
        dv = fetch(a)
        h1 = tf_frame(root, sym, "1h")
        print(f"  vrp {a}: DVOL {len(dv)} дней ({dv.index.min()} … {dv.index.max()}), часов {0 if h1 is None else len(h1)}",
              flush=True)
        if len(dv) and h1 is not None:
            rows += asset_rows(a, dv, h1)
    print(f"  vrp: сделок {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("vrp"), index=False)


def report() -> None:
    print("\n===== VRP: продажа недельной волатильности BTC / ETH (кондор — в R на макс. убыток, стрэнгл — в % базового) =====")
    parts = all_parts("vrp")
    if not parts:
        print("  vrp: нет данных")
        return
    x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
    passed = []
    for asset, struct, z, filt in VARIANTS:
        g = x[(x.asset == asset) & (x.struct == struct) & (x.z == z) & x[f"f{FILTERS.index(filt)}"]]
        if len(g) < 20:
            print(f"    {asset} {struct} z {z:g} {filt}: сделок {len(g)}")
            continue
        unit = "R" if struct == "кондор" else "%"
        cells, ok = [], []
        for p, (a, b) in PER_ALL.items():
            m = g[(g.t >= a) & (g.t < b)]
            if len(m):
                cells.append(f"{p} {m.res.mean():+.3f}{unit} ({len(m)})")
            if len(m) >= 20 and p in ("is", "val", "ho"):
                ok.append(m.res.mean() > 0)
        t = g.res.mean() / g.res.std() * math.sqrt(len(g)) if g.res.std() > 0 else float("nan")
        eq = g.res.cumsum()
        good = len(ok) == 3 and all(ok) and t >= 3 and g.res2.mean() > 0
        print(f"    {asset} {struct} z {z:g} {filt}: {'принято' if good else 'нет'}\n      {'  '.join(cells)}\n"
              f"      всё время {g.res.mean():+.3f}{unit} на неделю (t {t:.1f}, прибыльных {(g.res > 0).mean():.0%}, "
              f"худшая {g.res.min():+.2f}{unit}, просадка {float((eq.cummax() - eq).max()):.1f}{unit}); x2: "
              f"{g.res2.mean():+.3f}{unit}")
        if good:
            passed.append(f"{asset} {struct} z {z:g} {filt}")
    print("\n  ИТОГ: " + ("прошли — " + "; ".join(passed) if passed else "продажа волатильности не прошла ни в одном варианте"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser())
    else:
        report()
