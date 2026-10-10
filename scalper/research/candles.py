"""Свечные модели на 4h: подтверждение ретеста бота, поглощение на свече пробоя и поглощение как отдельная стратегия.

Литература (docs/knowledge.md): как самостоятельные сигналы свечи после издержек не зарабатывают (Marshall, Young,
Rose 2006; Duvinage, Mazza, Petitjean 2013), но «могут дополнять другую систему». Cohen (2021) на одном Bitcoin
находит поглощение рабочим. Здесь три проверки:

  1. Ретест бота (линия, ретест, 3R — research/noline.py): какой свечой закрылся бар, где исполнилась лимитка.
     against — закрылась обратно за линию (против сделки); pin — длинная тень к линии (>= 50% диапазона), закрытие в
     нашей половине; engulf — поглощение в сторону сделки; neutral — остальное. Правило «закрылась против — выход по
     закрытию этой свечи» считается отдельно.
  2. Свеча пробоя бота — поглощение в сторону сделки или нет.
  3. Поглощение отдельно: лонг по бычьему (шорт по медвежьему) поглощению на закрытии, стоп за минимумом двух свечей
     ± 0.1 ATR (0.3–4 ATR), цель 2R / 3R, 60 свечей, тейкер на входе, оборот от $20M. Контекст: любой; разворот
     (10 свечей до модели шли против сделки); по дневному тренду (как у бота). Сделки одной монеты могут
     пересекаться — R в месяц здесь не исполнимый, смотреть R на сделку.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30
from .noline import TF, coin_trades
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import HOLD, MAKER, STOP_BUF, TAKER, _years, htf_trend, tf_frame, two_targets

REV_BARS = 10
KINDS = {"against": "закрылась обратно за линию", "pin": "тень к линии (пин-бар / молот)",
         "engulf": "поглощение в сторону сделки", "neutral": "обычная свеча"}


def engulfing(o: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Бычье / медвежье поглощение на баре t: тело t противоположного цвета и полностью накрывает тело t−1."""
    po, pc = np.r_[np.nan, o[:-1]], np.r_[np.nan, c[:-1]]
    bull = (pc < po) & (c > o) & (o <= pc) & (c >= po) & (c - o > po - pc)
    bear = (pc > po) & (c < o) & (o >= pc) & (c <= po) & (o - c > pc - po)
    return bull, bear


def retest_kind(o: float, h: float, lo: float, c: float, side: int, level: float, engulf_with: bool) -> str:
    """Чем закрылась свеча ретеста относительно линии и стороны сделки."""
    if side * (c - level) < 0:
        return "against"
    rng = h - lo
    if rng > 0:
        wick = (min(o, c) - lo) if side > 0 else (h - max(o, c))
        loc = (c - lo) / rng if side > 0 else (h - c) / rng
        if wick >= 0.5 * rng and loc >= 0.5:
            return "pin"
    return "engulf" if engulf_with else "neutral"


def bot_rows(d: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    o, h, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    bull, bear = engulfing(o, c)
    rows = []
    for r in x.itertuples(index=False):
        side, f, px, stop = int(r.side), int(r.fill_i), float(r.px), float(r.stop)
        t = d.index.get_loc(r.t)
        risk = side * (px - stop)
        stop_in_fill = (side > 0 and lo[f] <= stop) or (side < 0 and h[f] >= stop)
        kind = retest_kind(o[f], h[f], lo[f], c[f], side, px, bool(bull[f] if side > 0 else bear[f]))
        exit_against = r.R3
        if kind == "against" and not stop_in_fill:
            exit_against = (side * (c[f] - px) - (MAKER + TAKER) * px) / risk
        rows.append({"symbol": r.symbol, "t": r.t, "side": side, "risk_pct": r.risk_pct, "R3": r.R3, "kind": kind,
                     "stop_in_fill": stop_in_fill, "R_cut": exit_against,
                     "brk_engulf": bool(bull[t] if side > 0 else bear[t])})
    return pd.DataFrame(rows)


def engulf_trades(d: pd.DataFrame, liq: np.ndarray, sym: str) -> pd.DataFrame:
    o, h, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    a = _atr(d).to_numpy()
    trend = htf_trend(d, TF)
    bull, bear = engulfing(o, c)
    rows = []
    for side, hit in ((1, bull), (-1, bear)):
        for t in np.flatnonzero(hit):
            if t < REV_BARS + 1 or t >= len(c) - 1 or not (a[t] > 0) or not (liq[t] >= ADV_MIN):
                continue
            stop = (min(lo[t], lo[t - 1]) - STOP_BUF * a[t]) if side > 0 else (max(h[t], h[t - 1]) + STOP_BUF * a[t])
            risk = side * (c[t] - stop)
            if not (0.3 * a[t] <= risk <= 4.0 * a[t]):
                continue
            r2 = two_targets(o, h, lo, c, f, t, side, c[t], stop, 2.0, 2.0, False, HOLD[TF], TAKER)[0]
            r3 = two_targets(o, h, lo, c, f, t, side, c[t], stop, 3.0, 3.0, False, HOLD[TF], TAKER)[0]
            rows.append({"symbol": sym, "t": d.index[t], "side": side, "risk_pct": risk / c[t], "R2": r2, "R3": r3,
                         "reversal": side * (c[t - 1] - c[t - 1 - REV_BARS]) < 0, "with_trend": trend[t] == side})
    return pd.DataFrame(rows)


def collect(root: Path, syms: list[str]) -> None:
    bots, engs = [], []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, TF)
            if d is None or len(d) < 500:
                continue
            liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
            x = coin_trades(d, liq, s)
            if len(x):
                x = x[(x.trig == "line") & (x.entry == "retest")]
                if len(x):
                    bots.append(bot_rows(d, x))
            e = engulf_trades(d, liq, s)
            if len(e):
                engs.append(e)
        except Exception as e:
            print(f"  candles {s}: пропуск ({e})", flush=True)
    print(f"  candles: монет со сделками бота {len(bots)}, с поглощениями {len(engs)}", flush=True)
    if bots:
        pd.concat(bots, ignore_index=True).to_parquet(part_path("candles_bot"), index=False)
    if engs:
        pd.concat(engs, ignore_index=True).to_parquet(part_path("candles_eng"), index=False)


def _load(name: str) -> pd.DataFrame | None:
    parts = all_parts(name)
    if not parts:
        return None
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["per"] = ""
    for p, (a, b) in PER_ALL.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    return df


def _row(name: str, z: pd.DataFrame, col: str) -> dict:
    return {"группа": name, "доля": f"{len(z)}", **{p: _cell(z.assign(R=z[col])[z.per == p]) for p in PER_ALL}}


def report() -> None:
    print("\n===== CANDLES: свечные модели на 4h; ячейка — R на сделку (t по дням, прибыльных, сделок в месяц) =====")
    bot = _load("candles_bot")
    if bot is not None:
        print(f"\n  --- 1. ретест бота ({len(bot)} сделок): свеча, в которой исполнилась лимитка ---")
        rows = [_row("все (бот)", bot, "R3")] + [_row(nm, bot[bot.kind == k], "R3") for k, nm in KINDS.items()]
        print(pd.DataFrame(rows).to_string(index=False))
        ag = bot[bot.kind == "against"]
        print(f"  «против»: {len(ag) / len(bot):.0%} сделок, из них стоп в той же свече {ag.stop_in_fill.mean():.0%}; "
              f"по годам {_years(ag)}")
        print("\n  правило «свеча ретеста закрылась против — выход по её закрытию» (все сделки бота):")
        print(pd.DataFrame([_row("бот", bot, "R3"), _row("с правилом", bot, "R_cut")]).to_string(index=False))
        print(f"  с правилом по годам: {_years(bot, 'R_cut')}")
        print("\n  --- 2. свеча пробоя бота — поглощение в сторону сделки ---")
        print(pd.DataFrame([_row("поглощение", bot[bot.brk_engulf], "R3"),
                            _row("нет", bot[~bot.brk_engulf], "R3")]).to_string(index=False))
    eng = _load("candles_eng")
    if eng is not None:
        print(f"\n  --- 3. поглощение отдельно, 4h, оборот от $20M ({len(eng)} моделей; R до издержек x2 в скобках) ---")
        for col in ("R2", "R3"):
            rows = []
            for side, sn in ((1, "лонг (бычье)"), (-1, "шорт (медвежье)")):
                g = eng[eng.side == side]
                for nm, z in (("любой контекст", g), ("после хода против (разворот)", g[g.reversal]),
                              ("по дневному тренду", g[g.with_trend]),
                              ("разворот + по тренду", g[g.reversal & g.with_trend])):
                    rows.append(_row(f"{sn}: {nm}", z, col))
            print(f"\n  цель {col[1:]}R:")
            print(pd.DataFrame(rows).to_string(index=False))
        for side in (1, -1):
            g = eng[(eng.side == side) & eng.with_trend]
            print(f"  {'лонг' if side > 0 else 'шорт'} по тренду, 3R: по годам {_years(g)}; издержки x2: " + ", ".join(
                f"{p}: {(z.R3 - 2 * TAKER / z.risk_pct).mean():+.3f}" for p, z in g.groupby("per") if p))


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
