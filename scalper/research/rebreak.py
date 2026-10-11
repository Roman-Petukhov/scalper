"""Повторный пробой той же линии после ложного: пробой → закрытие обратно за линией → снова закрытие за ней.

Линии бота (zz_lines, зигзаг 3 ATR). Для каждой линии:
  first   — первый пробой (сигнал бота);
  failed  — первый пробой, после которого цена закрылась обратно за линией не позже FAIL_BARS свечей;
  rebreak — после такого возврата снова закрытие за той же (продолженной) линией не позже REBREAK_BARS свечей.
Фильтры, стоп, цель 3R и удержание — как у бота: 4h — оборот от $20M, агрессоры >= 55%, закрытие в своей половине
свечи, дневной тренд, стоп за свингом PIV=5 ± 0.1 ATR (0.3–4 ATR), вход по рынку и ретест (лимитка на линии 12 свечей);
15m — то же, но только шорты, линия не круче MAX_SLOPE_ATR, топ-150 по обороту, вход по рынку.
Фильтры хранятся колонками и применяются в отчёте: так же видно, что даёт повторный пробой без фильтров свечи.
«Бот бы вошёл» — повторный пробой, когда сделка первого пробоя (если она была) уже закрыта: две позиции по монете
бот не держит.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import (HOLD, MAKER, PER, PIV, RETEST_BARS, STOP_BUF, TAKER, _years, htf_trend, last_confirmed, pivots,
                    retest_fill, tf_frame, two_targets, zz_lines)
from .wide15 import MAX_SLOPE_ATR

FAIL_BARS = 12          # возврат за линию не позже стольких свечей после пробоя — пробой ложный
REBREAK_BARS = 48       # повторное закрытие за линией не позже стольких свечей после возврата
TOP15 = 150


def events(c: np.ndarray, lines: list[dict]) -> list[dict]:
    """Первые пробои и повторные пробои после возврата за линию. Линия продолжается по своему наклону."""
    n, out = len(c), []
    for r in lines:
        t, side = r["t"], r["side"]
        if t <= 0:
            continue
        a, sl = r["a"], r["slope"]

        def line(j: int) -> float:
            return c[a] + sl * (j - a)

        first = {"kind": "first", "t": t, "side": side, "level": line(t), "slope": sl, "first_t": t, "fail_lag": -1}
        out.append(first)
        f = next((j for j in range(t + 1, min(t + FAIL_BARS + 1, n)) if side * (c[j] - line(j)) < 0), -1)
        if f < 0:
            continue
        first["fail_lag"] = f - t
        rb = next((j for j in range(f + 1, min(f + REBREAK_BARS + 1, n)) if side * (c[j] - line(j)) > 0), -1)
        if rb > 0:
            out.append({"kind": "rebreak", "t": rb, "side": side, "level": line(rb), "slope": sl, "first_t": t,
                        "fail_lag": f - t})
    return out


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str, entries: tuple[str, ...]) -> pd.DataFrame:
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    a = _atr(d).to_numpy()
    trend = htf_trend(d, tf)
    buy = (d["taker_buy_volume"] / d["volume"].replace(0, np.nan)).to_numpy()
    rng = hi - lo
    loc = (c - lo) / np.where(rng > 0, rng, np.nan)
    sw_lo, sw_hi = last_confirmed(pivots(lo, PIV, False), len(c)), last_confirmed(pivots(hi, PIV, True), len(c))
    rows = []
    for ev in events(c, zz_lines(d)):
        t, side, level = ev["t"], ev["side"], ev["level"]
        if t >= len(c) - 1 or not (a[t] > 0):
            continue
        sw = sw_lo[t] if side > 0 else sw_hi[t]
        if sw < 0:
            continue
        stop = (lo[sw] - STOP_BUF * a[t]) if side > 0 else (hi[sw] + STOP_BUF * a[t])
        base = {"symbol": sym, "kind": ev["kind"], "t": d.index[t], "side": side, "first_t": d.index[ev["first_t"]],
                "fail_lag": ev["fail_lag"], "adv": liq[t], "with_trend": bool(trend[t] == side),
                "aggr": buy[t] if side > 0 else 1 - buy[t], "close_loc": loc[t] if side > 0 else 1 - loc[t],
                "slope_atr": abs(ev["slope"]) / a[t]}
        for entry in entries:
            if entry == "market":
                fill, px, fee = t, c[t], TAKER
            else:
                if not (side * (level - stop) > 0):
                    continue
                fill = retest_fill(hi, lo, t, side, level, 3 * side * (level - stop), RETEST_BARS)
                if fill < 0:
                    continue
                px, fee = level, MAKER
            risk = side * (px - stop)
            if not (0.3 * a[t] <= risk <= 4.0 * a[t]):
                continue
            if entry == "retest" and ((side > 0 and lo[fill] <= stop) or (side < 0 and hi[fill] >= stop)):
                r3, ex = (side * (stop - px) - (fee + TAKER) * px) / risk, fill
            else:
                r3, ex = two_targets(o, hi, lo, c, f, fill, side, px, stop, 3.0, 3.0, False, HOLD[tf], fee)
            rows.append(base | {"entry": entry, "R3": r3, "exit_t": d.index[min(int(ex), len(c) - 1)]})
    return pd.DataFrame(rows)


def collect(root: Path, syms: list[str], tf: str) -> None:
    name = "rebreak" if tf == "4h" else "rebreak15"
    entries = ("market", "retest") if tf == "4h" else ("market",)
    parts, advs = [], []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, tf)
            if d is None or len(d) < 500:
                continue
            adv = adv30(root, s)
            if tf != "4h":
                a = adv.dropna()
                advs.append(pd.DataFrame({"symbol": s, "day": a.index, "adv": a.to_numpy()}))
            x = coin_rows(d, adv.reindex(d.index.floor("D")).to_numpy(), s, tf, entries)
            if len(x):
                parts.append(x)
        except Exception as e:
            print(f"  {name} {s}: пропуск ({e})", flush=True)
    print(f"  {name}: монет с событиями {len(parts)}", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path(name), index=False)
    if advs:
        pd.concat(advs, ignore_index=True).to_parquet(part_path(name + "_adv"), index=False)


def blocked(df: pd.DataFrame) -> pd.Series:
    """Повторный пробой, когда сделка первого пробоя той же линии (та же сторона и вход) ещё открыта."""
    first = df[df.kind == "first"][["symbol", "first_t", "side", "entry", "exit_t"]].rename(columns={"exit_t": "first_exit"})
    m = df.merge(first, on=["symbol", "first_t", "side", "entry"], how="left")
    hit = (m["kind"] == "rebreak") & m["first_exit"].notna() & (m["first_exit"] > m["t"])
    return pd.Series(hit.to_numpy(), index=df.index)


def tables(df: pd.DataFrame, periods: dict[str, tuple[str, str]], entries: tuple[str, ...], title: str) -> None:
    df = df.copy()
    df["per"] = ""
    for p, (a, b) in periods.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    cols = list(periods) + ["2026"]
    candle = (df.aggr >= 0.55) & (df.close_loc >= 0.5)
    print(f"\n===== REBREAK {title}: повторный пробой той же линии после ложного (возврат за линию <= {FAIL_BARS} "
          f"свечей, повтор <= {REBREAK_BARS} свечей после возврата); R на сделку, 3R =====")
    for entry in entries:
        z = df[df.entry == entry]
        f, rb = z[z.kind == "first"], z[z.kind == "rebreak"]
        fc, rbc = f[candle.loc[f.index]], rb[candle.loc[rb.index]]
        groups = [
            ("первый пробой (бот)", fc),
            (f"  из них ложные (возврат <= {FAIL_BARS} свечей)", fc[fc.fail_lag >= 0]),
            ("повторный пробой, фильтры бота", rbc),
            ("  возврат был быстрым (<= 4 свечей)", rbc[rbc.fail_lag <= 4]),
            ("  первый пробой прошёл фильтры бота (был сигнал)", rbc[rbc.set_index(["symbol", "first_t", "side"]).index.isin(
                fc.set_index(["symbol", "first_t", "side"]).index)]),
            ("  бот бы вошёл (сделка первого пробоя закрыта)", rbc[~blocked(z).loc[rbc.index]]),
            ("повторный пробой без фильтров свечи (только тренд)", rb),
        ]
        rows = []
        for nm, g in groups:
            g = g.assign(R=g.R3)
            rows.append({"группа": nm, **{p: _cell(g[g.per == p]) for p in periods}, "2026": _cell(g[g.t.dt.year == 2026])})
        print(f"\n  --- вход: {'по рынку' if entry == 'market' else 'ретест'} ---")
        print(pd.DataFrame(rows)[["группа"] + cols].to_string(index=False))
        print(f"  повторный, фильтры бота, по годам: {_years(rbc)}")
        print(f"  первый пробой (бот) по годам: {_years(fc)}")


def report() -> None:
    parts = all_parts("rebreak")
    if not parts:
        print("rebreak: частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    for k in ("t", "first_t", "exit_t"):
        df[k] = pd.to_datetime(df[k], utc=True)
    df = df[(df.adv >= ADV_MIN) & df.with_trend].reset_index(drop=True)
    tables(df, PER_ALL, ("retest", "market"), "4h (правило бота)")


def report15() -> None:
    tp, ap = all_parts("rebreak15"), all_parts("rebreak15_adv")
    if not tp or not ap:
        print("rebreak15: частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in tp], ignore_index=True)
    adv = pd.concat([pd.read_parquet(p) for p in ap], ignore_index=True)
    for k in ("t", "first_t", "exit_t"):
        df[k] = pd.to_datetime(df[k], utc=True)
    adv["day"] = pd.to_datetime(adv["day"], utc=True)
    adv["rank"] = adv.groupby("day")["adv"].rank(ascending=False, method="first")
    df["day"] = df.t.dt.floor("D")
    df = df.drop(columns="adv").merge(adv[["symbol", "day", "adv", "rank"]], on=["symbol", "day"], how="left")
    df = df[(df.side < 0) & df.with_trend & (df.adv >= ADV_MIN) & (df.slope_atr <= MAX_SLOPE_ATR) & (df["rank"] <= TOP15)]
    tables(df.reset_index(drop=True), PER, ("market",), f"15m (правило бота: пологие шорты, топ-{TOP15})")


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
    root = Path(a.root).expanduser()
    {"collect": lambda: collect(root, syms, "4h"), "report": report,
     "collect15": lambda: collect(root, syms, "15m"), "report15": report15}[a.mode]()
