"""Плацебо для бота 4h: выбирает ли пробой линии момент или прибыль — просто дрейф «мусорные альты падают».

Сделки бота — правило research/noline.py (линия, все фильтры, ретест, 3R, удержание 60 свечей). Для каждой сделки
PLACEBO_K раз берётся случайная свеча той же монеты в окне ±WINDOW свечей (±15 дней) от исполнения, с оборотом от
$20M; вход по закрытию этой свечи, та же сторона, тот же стоп в процентах цены, та же цель 3R, те же издержки
(вход мейкер, как у ретеста) и funding:
  any   — любая свеча окна: монета + месяц + сторона + размер стопа, без момента;
  trend — только свечи, где дневной тренд бота в сторону сделки: то же плюс фильтр тренда.
Если бот не лучше плацебо — прибыль в выборе монет и периода (дрейф), а не в пробое линии.

Фон без сигнала (base): каждая 6-я свеча (раз в сутки) каждой ликвидной монеты, шорт и лонг, стоп 2 ATR, 3R —
сколько дал бы «шорт любого альта в любой день» и тот же шорт только при дневном тренде вниз.

Критерии заданы до запуска (R на сделку, периоды IS / VAL / HO):
  момент подтверждён — бот выше плацебо trend в каждом из IS, VAL, HO и по всем периодам с 2020 доля розыгрышей
    плацебо со средним не ниже бота < 1%;
  дрейф объясняет — плацебо trend по всем периодам даёт не меньше половины среднего R бота.
"""
from __future__ import annotations

import argparse
import sys
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30
from .noline import TF, coin_trades
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import HOLD, MAKER, htf_trend, tf_frame, two_targets

PLACEBO_K = 200
WINDOW = 90                 # свечей 4h = 15 дней
BASE_STEP = 6               # фон: одна свеча в сутки
BASE_STOP_ATR = 2.0
KINDS = ("any", "trend")


def placebo_bars(ok: np.ndarray, fill: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """k случайных свечей (с повторами) из окна ±WINDOW вокруг fill, где ok; пусто — нет подходящих."""
    lo, hi = max(0, fill - WINDOW), min(len(ok) - 2, fill + WINDOW)
    if hi < lo:
        return np.empty(0, dtype="int64")
    pool = lo + np.flatnonzero(ok[lo:hi + 1])
    if not len(pool):
        return np.empty(0, dtype="int64")
    return rng.choice(pool, size=k, replace=True)


def trade_r(o, hi, lo, c, f, e: int, side: int, risk_pct: float) -> float:
    """R входа по закрытию свечи e со стопом risk_pct цены и целью 3R (как у бота)."""
    px = c[e]
    stop = px * (1 - side * risk_pct)
    r, _ = two_targets(o, hi, lo, c, f, e, side, px, stop, 3.0, 3.0, False, HOLD[TF], MAKER)
    return float(r)


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, k: int = PLACEBO_K
              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(сделки бота, плацебо, фон) одной монеты."""
    o, hi, lo, c = (d[x].to_numpy(dtype="float64") for x in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    a = _atr(d).to_numpy()
    trend = htf_trend(d, TF)
    liquid = (liq >= ADV_MIN) & (a > 0)
    rng = np.random.default_rng(zlib.crc32(sym.encode()))
    x = coin_trades(d, liq, sym, lines_only=True)
    bots, plac = [], []
    if len(x):
        x = x[(x.trig == "line") & (x.entry == "retest")].reset_index(drop=True)
    for i, r in enumerate(x.itertuples()):
        tid = f"{sym}#{i}"
        bots.append({"tid": tid, "symbol": sym, "t": r.t, "side": int(r.side), "R": float(r.R3),
                     "risk_pct": float(r.risk_pct)})
        for kind in KINDS:
            ok = liquid & (trend == r.side) if kind == "trend" else liquid
            for j, e in enumerate(placebo_bars(ok, int(r.fill_i), k, rng)):
                plac.append({"tid": tid, "kind": kind, "k": j,
                             "R": trade_r(o, hi, lo, c, f, int(e), int(r.side), float(r.risk_pct))})
    base = []
    for e in range(int(rng.integers(BASE_STEP)), len(c) - 1, BASE_STEP):
        if not liquid[e]:
            continue
        rp = BASE_STOP_ATR * a[e] / c[e]
        for side in (1, -1):
            base.append({"symbol": sym, "t": d.index[e], "side": side, "with_trend": bool(trend[e] == side),
                         "R": trade_r(o, hi, lo, c, f, e, side, rp)})
    return pd.DataFrame(bots), pd.DataFrame(plac), pd.DataFrame(base)


def collect(root: Path, syms: list[str]) -> None:
    out: dict[str, list[pd.DataFrame]] = {"bot": [], "plc": [], "base": []}
    for s in mine(syms):
        try:
            d = tf_frame(root, s, TF)
            if d is None or len(d) < 500:
                continue
            liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
            for name, z in zip(("bot", "plc", "base"), coin_rows(d, liq, s)):
                if len(z):
                    out[name].append(z)
        except Exception as e:
            print(f"  placebo {s}: пропуск ({e})", flush=True)
    print(f"  placebo: монет со сделками {len(out['bot'])}", flush=True)
    for name, zs in out.items():
        if zs:
            pd.concat(zs, ignore_index=True).to_parquet(part_path(f"placebo_{name}"), index=False)


def period_of(t: pd.Series) -> pd.Series:
    per = pd.Series("", index=t.index)
    for p, (a, b) in PER_ALL.items():
        per[(t >= a) & (t < b)] = p
    return per


def draw_means(bot: pd.DataFrame, plc: pd.DataFrame, kind: str) -> np.ndarray:
    """Среднее R каждого розыгрыша плацебо по сделкам bot (в розыгрыше — по одной случайной сделке на сделку бота)."""
    z = plc[(plc.kind == kind) & plc.tid.isin(bot.tid)].dropna(subset=["R"])
    if not len(z):
        return np.empty(0)
    return z.groupby("k")["R"].mean().to_numpy()


def compare(bot: pd.DataFrame, plc: pd.DataFrame) -> dict[str, str]:
    """Ячейки таблицы: бот, плацебо any / trend (среднее розыгрышей), разница и доля розыгрышей не хуже бота."""
    b = bot.R.dropna()
    row = {"бот": f"{b.mean():+.3f} ({len(b)})" if len(b) else "—"}
    for kind in KINDS:
        m = draw_means(bot, plc, kind)
        if not len(m) or not len(b):
            row[kind] = "—"
            continue
        row[kind] = f"{m.mean():+.3f}"
        row[f"бот−{kind}"] = f"{b.mean() - m.mean():+.3f}"
        row[f"p {kind}"] = f"{np.mean(m >= b.mean()):.3f}"
    return row


def verdict(bot: pd.DataFrame, plc: pd.DataFrame) -> list[str]:
    out = []
    above = []
    for p in list(PER_ALL)[-3:]:
        z = bot[bot.per == p]
        m = draw_means(z, plc, "trend")
        above.append(bool(len(m) and z.R.mean() > m.mean()))
    allp = bot[bot.per != ""]
    m = draw_means(allp, plc, "trend")
    pval = float(np.mean(m >= allp.R.mean())) if len(m) else np.nan
    share = float(m.mean() / allp.R.mean()) if len(m) and allp.R.mean() > 0 else np.nan
    timing = all(above) and pval < 0.01
    out.append(f"  момент: бот выше плацебо trend в IS / VAL / HO — {' / '.join('да' if a else 'нет' for a in above)}; "
               f"p по всем периодам {pval:.3f} → {'ПОДТВЕРЖДЁН' if timing else 'не подтверждён'}")
    out.append(f"  дрейф: плацебо trend / бот = {share:.0%} → "
               f"{'объясняет заметную часть (>= 50%)' if share >= 0.5 else 'объясняет меньше половины'}")
    return out


def report() -> None:
    print("\n===== PLACEBO: бот 4h против случайного входа (та же монета, ±15 дней, сторона, стоп в %, 3R, издержки); "
          f"{PLACEBO_K} розыгрышей; R на сделку =====")
    parts = {n: all_parts(f"placebo_{n}") for n in ("bot", "plc", "base")}
    if not parts["bot"]:
        print("частей нет")
        return
    bot, plc = (pd.concat([pd.read_parquet(p) for p in parts[n]], ignore_index=True) for n in ("bot", "plc"))
    bot["t"] = pd.to_datetime(bot["t"], utc=True)
    bot["per"] = period_of(bot["t"])
    print("  any — любая ликвидная свеча окна; trend — только при дневном тренде в сторону сделки; "
          "p — доля розыгрышей плацебо со средним не ниже бота")
    for name, sel in (("все сделки", bot), ("шорты", bot[bot.side < 0]), ("лонги", bot[bot.side > 0])):
        rows = [{"период": p, **compare(sel[sel.per == p], plc)} for p in PER_ALL]
        rows.append({"период": "все", **compare(sel[sel.per != ""], plc)})
        print(f"\n  --- {name} ---")
        print(pd.DataFrame(rows).to_string(index=False))
    print("\n  Критерии (заданы до запуска):")
    for line in verdict(bot, plc):
        print(line)
    if parts["base"]:
        base = pd.concat([pd.read_parquet(p) for p in parts["base"]], ignore_index=True)
        base["t"] = pd.to_datetime(base["t"], utc=True)
        base["per"] = period_of(base["t"])
        print(f"\n  --- фон без сигнала: раз в сутки каждая ликвидная монета, стоп {BASE_STOP_ATR:g} ATR, 3R; "
              "R на сделку (t по дням, прибыльных, сделок в месяц) ---")
        rows = []
        for nm, z in (("шорт всех", base[base.side < 0]), ("шорт, тренд вниз", base[(base.side < 0) & base.with_trend]),
                      ("лонг всех", base[base.side > 0]), ("лонг, тренд вверх", base[(base.side > 0) & base.with_trend])):
            rows.append({"фон": nm, **{p: _cell(z[z.per == p]) for p in PER_ALL}})
        print(pd.DataFrame(rows).to_string(index=False))


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
