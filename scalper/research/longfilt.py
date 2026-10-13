"""Лонги бота 4h: чем заменить фильтр агрессоров ≥ 55%, который пропускает лишь 5% лонговых пробоев (research/sidefunnel.py).

База — лонговые пробои линий зигзага 3 ATR со всеми фильтрами бота, кроме агрессоров: оборот от $20M, дневной тренд
вверх, закрытие в верхней половине свечи, стоп за свингом 0.3–4 ATR, лимитка на линии исполнилась; сделка как у бота
(3R, долив 0.5 на +1.5R — research/dconf.py). Варианты фильтра (заданы до запуска):
    aggr50 / aggr52.5    доля рыночных покупок в свече ≥ 50 / 52.5% (порог только для лонгов)
    rel3 / rel5          доля покупок выше своей медианы за 30 дней (180 свечей до пробоя) на 3 / 5 п.п.
    vol2 / vol3          объём свечи ≥ 2 / 3 медианы 30 прошлых свечей
    rng1.5               свеча пробоя ≥ 1.5 ATR
    delta20              дельта агрессоров за 20 свечей до пробоя > 0 (накопление до пробоя)
    rs7 / rs30           монета сильнее BTC за 7 / 30 дней
    top25                закрытие в верхней четверти свечи
Вторая волна (гипотеза «киты покупают лимитками»: цена растёт, а рыночные продажи перевешивают — кто-то поглощает их
пассивными заявками):
    acc20 / acc60        за 20 / 60 свечей до пробоя цена выросла, а дельта агрессоров (покупки − продажи) / объём < 0
    acc60s               то же за 60 свечей, рост не меньше 2 ATR и дельта не выше −2% объёма (явное поглощение)
Каждый — «вместо 55%» и «добавка к 55%» (сделки, где вариант да, а 55% нет).
Принятие: R на сделку > 0 во всех периодах 2021 / IS / VAL / HO / 2026, где сделок ≥ 10, и (для «вместо») больше, чем
у базы без фильтра, в каждом из них. Вариантов 15 × 2 — одиночный «проход» допускает случайность.

    python -m research.longfilt collect --symbols <1h-qualified>
    python -m research.longfilt report
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30
from .dconf import EXITS, simulate
from .noline import triggers
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import MAKER, PIV, RETEST_BARS, STOP_BUF, TAKER, htf_trend, last_confirmed, pivots, retest_fill, tf_frame

TF = "4h"
BOT_EXIT = EXITS["бот (3R, долив на +1.5R)"]
REL_BARS, VOL_BARS, DELTA_BARS, ACC_LONG = 180, 30, 20, 60
VARIANTS: dict[str, str] = {
    "aggr55": "агрессоры ≥ 55% (бот)",
    "none": "без фильтра",
    "aggr50": "агрессоры ≥ 50%",
    "aggr52.5": "агрессоры ≥ 52.5%",
    "rel3": "покупки выше своей нормы на 3 п.п.",
    "rel5": "покупки выше своей нормы на 5 п.п.",
    "vol2": "объём ≥ 2 медиан",
    "vol3": "объём ≥ 3 медиан",
    "rng1.5": "свеча ≥ 1.5 ATR",
    "delta20": "дельта за 20 свечей до пробоя > 0",
    "rs7": "сильнее BTC за 7 дней",
    "rs30": "сильнее BTC за 30 дней",
    "top25": "закрытие в верхней четверти",
    "acc20": "поглощение 20 свечей: цена вверх, дельта < 0",
    "acc60": "поглощение 60 свечей: цена вверх, дельта < 0",
    "acc60s": "поглощение 60 свечей: рост ≥ 2 ATR, дельта ≤ −2%",
}


def features(d: pd.DataFrame, btc: pd.Series | None) -> dict[str, np.ndarray]:
    """Признаки на закрытии каждой свечи (только прошлое и сама свеча)."""
    c, v = d["close"], d["volume"]
    buy = d["taker_buy_volume"] / v.replace(0, np.nan)
    norm = buy.shift(1).rolling(REL_BARS, min_periods=REL_BARS // 2).median()
    vmed = v.shift(1).rolling(VOL_BARS, min_periods=VOL_BARS // 2).median()
    delta = ((2 * d["taker_buy_volume"] - v).shift(1).rolling(DELTA_BARS).sum()
             / v.shift(1).rolling(DELTA_BARS).sum().replace(0, np.nan))
    flow = (2 * d["taker_buy_volume"] - v).shift(1)
    delta60 = flow.rolling(ACC_LONG).sum() / v.shift(1).rolling(ACC_LONG).sum().replace(0, np.nan)
    out = {"buy": buy.to_numpy(), "rel": (buy - norm).to_numpy(), "vol_x": (v / vmed.replace(0, np.nan)).to_numpy(),
           "delta20": delta.to_numpy(), "delta60": delta60.to_numpy(),
           "ret20": (c.shift(1) - c.shift(1 + DELTA_BARS)).to_numpy(), "ret60": (c.shift(1) - c.shift(1 + ACC_LONG)).to_numpy()}
    bc = btc.reindex(d.index, method="ffill") if btc is not None else pd.Series(np.nan, index=d.index)
    for days in (7, 30):
        n = days * 6
        out[f"rs{days}"] = ((c / c.shift(n)) - (bc / bc.shift(n))).to_numpy()
    return out


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, btc: pd.Series | None) -> list[dict]:
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64")
    a = _atr(d).to_numpy()
    trend = htf_trend(d, TF)
    rng = hi - lo
    loc = (c - lo) / np.where(rng > 0, rng, np.nan)
    sw_lo = last_confirmed(pivots(lo, PIV, False), len(c))
    fx = features(d, btc)
    rows = []
    for _, t, side, level in triggers(d, a, lines_only=True):
        if side < 0 or t >= len(c) - 1 or not (a[t] > 0) or not (liq[t] >= ADV_MIN) or trend[t] != 1:
            continue
        if not (loc[t] >= 0.5) or sw_lo[t] < 0:
            continue
        stop = lo[sw_lo[t]] - STOP_BUF * a[t]
        risk = level - stop
        if not (0.3 * a[t] <= risk <= 4.0 * a[t]):
            continue
        fill = retest_fill(hi, lo, t, 1, level, 3.0 * risk, RETEST_BARS)
        if fill < 0:
            continue
        r = ((stop - level - (MAKER + TAKER) * level) / risk if lo[fill] <= stop
             else simulate(o, hi, lo, c, f, fill, 1, level, stop, a[t], BOT_EXIT))
        rows.append({"symbol": sym, "t": d.index[t], "R": r, "buy": fx["buy"][t], "rel": fx["rel"][t],
                     "vol_x": fx["vol_x"][t], "rng_atr": rng[t] / a[t], "delta20": fx["delta20"][t],
                     "rs7": fx["rs7"][t], "rs30": fx["rs30"][t], "loc": loc[t], "delta60": fx["delta60"][t],
                     "ret20_atr": fx["ret20"][t] / a[t], "ret60_atr": fx["ret60"][t] / a[t]})
    return rows


def masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {"aggr55": df.buy >= 0.55, "none": pd.Series(True, index=df.index), "aggr50": df.buy >= 0.50,
            "aggr52.5": df.buy >= 0.525, "rel3": df.rel >= 0.03, "rel5": df.rel >= 0.05, "vol2": df.vol_x >= 2,
            "vol3": df.vol_x >= 3, "rng1.5": df.rng_atr >= 1.5, "delta20": df.delta20 > 0, "rs7": df.rs7 > 0,
            "rs30": df.rs30 > 0, "top25": df["loc"] >= 0.75,
            "acc20": (df.ret20_atr > 0) & (df.delta20 < 0), "acc60": (df.ret60_atr > 0) & (df.delta60 < 0),
            "acc60s": (df.ret60_atr >= 2) & (df.delta60 <= -0.02)}


def collect(root: Path, syms: list[str]) -> None:
    b = tf_frame(root, "BTCUSDT", TF)
    btc = b["close"] if b is not None else None
    rows = []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, TF)
            if d is None or len(d) < 500:
                continue
            liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
            rows += coin_rows(d, liq, s, btc)
        except Exception as e:
            print(f"  longfilt {s}: пропуск ({e})", flush=True)
    print(f"  longfilt: лонгов {len(rows)}; BTC {'есть' if btc is not None else 'нет'}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("longfilt"), index=False)


def _verdict(z: pd.DataFrame, base: pd.DataFrame | None, per: dict) -> str:
    ok, cmp = [], []
    for a, b in per.values():
        m = (z.t >= a) & (z.t < b)
        if m.sum() < 10:
            continue
        ok.append(z.R[m].mean() > 0)
        if base is not None:
            mb = (base.t >= a) & (base.t < b)
            cmp.append(z.R[m].mean() > base.R[mb].mean())
    if not ok:
        return "мало сделок"
    good = all(ok) and all(cmp)
    return "принято" if good else "не принято"


def report() -> None:
    parts = all_parts("longfilt")
    print("\n===== LONGFILT: лонги бота 4h — чем заменить фильтр агрессоров ≥ 55%; база — все остальные фильтры бота, "
          "сделка как у бота (ретест, 3R, долив на +1.5R) =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    per = {**PER_ALL, "2026": ("2026-01-01", "2026-10-01")}
    judged = {k: v for k, v in per.items() if k != "2020"}
    mk = masks(df)
    base = df
    print(f"лонгов в базе {len(df)}; доля каждого фильтра: " + ", ".join(f"{k} {m.mean():.0%}" for k, m in mk.items()))
    for mode in ("вместо 55%", "добавка к 55% (вариант да, 55% нет)"):
        print(f"\n  {mode}")
        for k, nm in VARIANTS.items():
            if mode.startswith("добавка") and k in ("aggr55", "none"):
                continue
            m = mk[k] if mode == "вместо 55%" else (mk[k] & ~mk["aggr55"])
            z = df[m]
            tag = "" if k in ("aggr55", "none") else "  → " + _verdict(z, base if mode == "вместо 55%" else None, judged)
            print(f"    {nm}: {len(z)} сделок, всего {z.R.sum():+.1f}R{tag}")
            print("      " + "  ".join(f"{p}: {_cell(z[(z.t >= a) & (z.t < b)])}" for p, (a, b) in per.items()))


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
