"""Общий счёт «бот 4h + coil»: пересекаются ли сделки, коррелируют ли просадки, какой риск давать каждой стратегии.

Сделки (одни и те же монеты 1h-qualified, 2022-01 … 2026-09 — с этого времени есть OI для coil):
    bot   правило бота 4h: линия, ретест, 3R, фильтры бота (research/noline.py, как в research/margsim.py)
    coil  базовый coil (research/coil_check.py BASE): сжатие волатильности 3 дня + набор OI, вход на закрытии за
          диапазоном только против толпы по funding, стоп 1.5 ATR, тейк 5R, до 60 свечей; лонги и шорты
Счёт проигрывается по сделкам в порядке входа (вход — закрытие свечи сигнала / исполнения ретеста), капитал — по
закрытым сделкам: размер — риск % капитала до стопа (номинал не больше плечо × капитал), общий лимит позиций,
одна позиция на монету (как в боте: режим One-Way), дневной стоп = 4 × больший риск. Без хеджа BTC.
Что печатается: сделки и R по годам у каждой стратегии; пересечения (та же монета в одно время, та же / встречная
сторона); корреляция R по месяцам и по дням выхода; счёт для нескольких распределений риска.

    python -m research.joint collect --symbols <монеты 1h-qualified>
    python -m research.joint report
"""
from __future__ import annotations

import argparse
import heapq
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30
from .coil_check import BASE, coil_signals
from .margsim import max_dd, monthly
from .noline import TF, coin_trades
from .shard import all_parts, mine, part_path
from .tline import tf_frame
from .trend import COST, EXITS, MAX_HOLD, PER, trade_machine
from .trend2 import bars4h
from .wave2 import Data2

BAR = pd.Timedelta(hours=4)
START = pd.Timestamp("2022-01-01", tz="UTC")
COIL_STOP = 1.5
E0 = 1000.0
LEV = 10
MIN_NOTIONAL = 5.0
COLS = ["strategy", "symbol", "t_in", "t_out", "side", "R", "risk_pct"]


def bot_trades(root: Path, s: str) -> pd.DataFrame:
    d = tf_frame(root, s, TF)
    if d is None or len(d) < 500:
        return pd.DataFrame(columns=COLS)
    x = coin_trades(d, adv30(root, s).reindex(d.index.floor("D")).to_numpy(), s)
    x = x[(x.trig == "line") & (x.entry == "retest")] if len(x) else x
    if not len(x):
        return pd.DataFrame(columns=COLS)
    return pd.DataFrame({"strategy": "bot", "symbol": s, "t_in": d.index[x["fill_i"].to_numpy()] + BAR,
                         "t_out": d.index[x["exit_i"].to_numpy()] + BAR, "side": x["side"].to_numpy(),
                         "R": x["R3"].to_numpy(), "risk_pct": x["risk_pct"].to_numpy()})


def coil_trades(data: Data2, root: Path, s: str) -> pd.DataFrame:
    h = data.get(s, "1h", "full")
    if h["oi"].notna().sum() < 24 * 60:
        return pd.DataFrame(columns=COLS)
    d = bars4h(h)
    allowed = (adv30(root, s).reindex(d.index, method="ffill") >= ADV_MIN).to_numpy()
    a = {c: d[c].to_numpy(dtype="float64") for c in ("open", "high", "low", "close", "atr", "funding")}
    up, dn = coil_signals(d, *BASE)
    none = np.zeros(len(d), np.bool_)
    out = []
    for side, L, S in ((1, up & allowed, none), (-1, none, dn & allowed)):
        i, r, dur = trade_machine(a["open"], a["high"], a["low"], a["close"], a["atr"], a["funding"], L, S, COIL_STOP,
                                  EXITS.index("tp5"), MAX_HOLD, COST)
        if len(i):
            out.append(pd.DataFrame({"strategy": "coil", "symbol": s, "t_in": d.index[i] + BAR,
                                     "t_out": d.index[np.minimum(i + dur, len(d) - 1)] + BAR, "side": side, "R": r,
                                     "risk_pct": COIL_STOP * a["atr"][i] / a["close"][i]}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=COLS)


def collect(root: Path, syms: list[str]) -> None:
    data = Data2(root, syms)
    parts = []
    for s in mine(syms):
        for fn in (lambda: bot_trades(root, s), lambda: coil_trades(data, root, s)):
            try:
                x = fn()
                if len(x):
                    parts.append(x)
            except Exception as e:
                print(f"  joint {s}: пропуск ({e})", flush=True)
        data.cache.clear()
    if parts:
        df = pd.concat(parts, ignore_index=True)
        print(f"  joint: сделок бота {(df.strategy == 'bot').sum()}, coil {(df.strategy == 'coil').sum()}", flush=True)
        df.to_parquet(part_path("joint"), index=False)


# ---------- счёт ----------

@dataclass(frozen=True)
class Mix:
    name: str
    risk: dict[str, float]          # стратегия → риск % капитала на сделку (нет в словаре — не торгуем)
    max_pos: int = 12


MIXES = (
    Mix("бот 1%", {"bot": 1.0}),
    Mix("coil 1%", {"coil": 1.0}),
    Mix("бот 1% + coil 1%", {"bot": 1.0, "coil": 1.0}),
    Mix("бот 1% + coil 0.5%", {"bot": 1.0, "coil": 0.5}),
    Mix("бот 0.5% + coil 0.5%", {"bot": 0.5, "coil": 0.5}),
    Mix("бот 1% + coil 1%, до 6 позиций", {"bot": 1.0, "coil": 1.0}, max_pos=6),
    Mix("бот 2%", {"bot": 2.0}),
    Mix("бот 2% + coil 1%", {"bot": 2.0, "coil": 1.0}),
)


@dataclass
class Book:
    eq: pd.Series
    opened: dict[str, int]
    rej_pos: int = 0
    rej_sym: int = 0
    rej_day: int = 0


def simulate(df: pd.DataFrame, mix: Mix) -> Book:
    d = df[df.strategy.isin(list(mix.risk))].sort_values(["t_in", "strategy"], kind="stable")
    day_loss = 4 * max(mix.risk.values())
    eq, heap, seq, busy = E0, [], 0, {}
    times, vals = [], []
    book = Book(pd.Series(dtype="float64"), {k: 0 for k in mix.risk})
    day, day_start = None, E0

    def close_until(t: pd.Timestamp | None) -> None:
        nonlocal eq
        while heap and (t is None or heap[0][0] <= t):
            t_out, _, pnl, sym = heapq.heappop(heap)
            eq += pnl
            busy[sym] -= 1
            times.append(t_out)
            vals.append(eq)

    for st, sym, t_in, t_out, r, rp in zip(d.strategy, d.symbol, d.t_in, d.t_out, d.R, d.risk_pct):
        close_until(t_in)
        if t_in.floor("D") != day:
            day, day_start = t_in.floor("D"), eq
        if len(heap) >= mix.max_pos:
            book.rej_pos += 1
            continue
        if busy.get(sym, 0):
            book.rej_sym += 1
            continue
        if eq <= day_start * (1 - day_loss / 100):
            book.rej_day += 1
            continue
        if not (rp > 0) or not np.isfinite(r):
            continue
        n = min(eq * mix.risk[st] / 100 / rp, eq * LEV)
        if n < MIN_NOTIONAL:
            continue
        seq += 1
        heapq.heappush(heap, (t_out, seq, n * rp * r, sym))
        busy[sym] = busy.get(sym, 0) + 1
        book.opened[st] += 1
    close_until(None)
    book.eq = pd.Series(vals, index=pd.DatetimeIndex(times), dtype="float64").sort_index()
    return book


def _summary(b: Book) -> dict[str, object]:
    mo = monthly(b.eq, E0)
    row: dict[str, object] = {
        "сделок": " / ".join(f"{k} {v}" for k, v in b.opened.items()),
        "ср. мес. %": f"{mo.mean():+.1f}" if len(mo) else "-",
        "σ мес. %": f"{mo.std():.1f}" if len(mo) else "-",
        "Sharpe": f"{mo.mean() / mo.std() * np.sqrt(12):.2f}" if len(mo) > 2 and mo.std() > 0 else "-",
        "худший мес. %": f"{mo.min():.0f}" if len(mo) else "-",
        "просадка %": f"{max_dd(b.eq, E0):.0f}",
        "мес. в минусе": f"{(mo < 0).mean():.0%}" if len(mo) else "-",
    }
    for p, (a, z) in PER.items():
        sub = mo[(mo.index >= a) & (mo.index < z)]
        row[p] = f"{sub.mean():+.1f}" if len(sub) else "-"
    row |= {"отказ: лимит / монета / дн. стоп": f"{b.rej_pos} / {b.rej_sym} / {b.rej_day}"}
    return row


def overlaps(df: pd.DataFrame) -> str:
    bot, coil = df[df.strategy == "bot"], df[df.strategy == "coil"]
    m = coil.reset_index().merge(bot, on="symbol", suffixes=("", "_b"))
    m = m[(m.t_in < m.t_out_b) & (m.t_in_b < m.t_out)]
    same = m[m.side == m.side_b]["index"].nunique()
    opp = m[m.side != m.side_b]["index"].nunique()
    return (f"сделок coil, открытых одновременно со сделкой бота по той же монете: {same + opp} из {len(coil)} "
            f"({(same + opp) / max(len(coil), 1):.1%}); в ту же сторону {same}, во встречную {opp}")


def report() -> None:
    parts = all_parts("joint")
    print("\n===== JOINT: общий счёт «бот 4h (линия, ретест, 3R) + coil» на тех же монетах, 2022-01 … 2026-09; "
          "риск % капитала на сделку, лимит позиций общий, одна позиция на монету, без хеджа =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    for c in ("t_in", "t_out"):
        df[c] = pd.to_datetime(df[c], utc=True)
    df = df[(df.t_in >= START) & df.R.notna()].reset_index(drop=True)
    for st, g in df.groupby("strategy"):
        y = g.groupby(g.t_in.dt.year)["R"].agg(["size", "mean", "sum"])
        print(f"  {st}: сделок {len(g)}, шортов {(g.side < 0).mean():.0%}, средний R {g.R.mean():+.3f}, "
              f"прибыльных {(g.R > 0).mean():.0%}, стоп медиана {g.risk_pct.median() * 100:.1f}% цены; по годам "
              + ", ".join(f"{k}: {v['mean']:+.2f}R × {int(v['size'])}" for k, v in y.iterrows()))
    print("  " + overlaps(df))
    by_m = df.groupby([df.t_out.dt.to_period("M"), "strategy"])["R"].sum().unstack().fillna(0)
    by_d = df.groupby([df.t_out.dt.floor("D"), "strategy"])["R"].sum().unstack().fillna(0)
    if {"bot", "coil"} <= set(by_m.columns):
        worst = by_m.sum(axis=1).nsmallest(5)
        print(f"  корреляция суммы R: по месяцам {by_m.bot.corr(by_m.coil):+.2f}, по дням выхода {by_d.bot.corr(by_d.coil):+.2f}")
        print("  месяцы: оба в минусе " + f"{((by_m.bot < 0) & (by_m.coil < 0)).mean():.0%}, бот в минусе — coil в плюсе "
              f"{((by_m.bot < 0) & (by_m.coil > 0)).mean():.0%}, наоборот {((by_m.bot > 0) & (by_m.coil < 0)).mean():.0%}")
        print("  худшие месяцы вместе (R бот / coil): " + ", ".join(
            f"{k}: {by_m.at[k, 'bot']:+.1f} / {by_m.at[k, 'coil']:+.1f}" for k in worst.index))
    rows = [{"вариант": m.name, **_summary(simulate(df, m))} for m in MIXES]
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
