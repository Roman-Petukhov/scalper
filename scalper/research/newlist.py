"""Дрейф цены нового перпетуала после листинга: шорт через k часов после первой свечи, удержание H часов.

Идея (BitMEX, отчёт по листингам 2025: на новых перпетуалах цена часто пикует в первый день): токены выходят по
завышенным оценкам, а перп даёт ранним держателям способ захеджироваться. Проверяем на всех перпетуалах Binance
архива (включая делистнутые), листинг — первая свечь (с запасом от начала данных, чтобы не принять за листинг старую
монету). Вход по open через k ч, выход по close через H ч; издержки 12 б.п. на круг, funding перпетуала за время
удержания (шорт получает положительную). Вариант «хедж BTC» — к шорту лонг BTC на тот же номинал (бета 1).
Стоп 30% выше входа — по high свечей (шорт на пампе — главный риск идеи). Данные — Binance, а торгуем на Bybit:
листинг на Bybit может быть позже или раньше.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .listings import COST
from .listings2 import funding
from .shard import all_parts, mine, part_path

DELAYS = (1, 6, 24, 72)          # часов от первой свечи до входа
HOLDS = (24, 72, 168, 720)       # часов удержания
STOP = 0.30
START_MARGIN = pd.Timedelta(days=45)       # монета с первой свечой раньше этого — листинг до начала данных
LIQ_USD = 20e6                             # оборот первых 24 ч для «ликвидных» (известен при входе от 24 ч)


def _load(root: Path, sym: str) -> pd.DataFrame | None:
    p = root / f"{sym}-1h.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p, columns=["open_time", "open", "high", "low", "close", "quote_volume"])
    d.index = pd.to_datetime(d["open_time"], unit="ms", utc=True)
    return d.drop(columns=["open_time"])[~d.index.duplicated()].sort_index()


def trade_rows(sym: str, d: pd.DataFrame, btc: pd.DataFrame, fund: pd.Series) -> list[dict]:
    t0 = d.index[0]
    o, hi, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "close"))
    liq24 = float(d["quote_volume"].iloc[:24].sum())
    bo, bc = (btc.reindex(d.index)[k].ffill().to_numpy(dtype="float64") for k in ("open", "close"))
    rows = []
    for k in DELAYS:
        for h in HOLDS:
            if k + h >= len(d) or not (o[k] > 0):
                continue
            e, x = k, k + h
            ret = c[x] / o[e] - 1                                          # цена монеты за удержание
            worst = hi[e:x + 1].max() / o[e] - 1                           # самый сильный рост против шорта
            stopped = worst >= STOP
            short = -STOP if stopped else -ret
            f = float(fund[(fund.index > d.index[e]) & (fund.index <= d.index[x] + pd.Timedelta(hours=1))].sum()) if len(fund) else 0.0
            btc_ret = bc[x] / bo[e] - 1 if bo[e] > 0 else np.nan
            rows.append({"symbol": sym, "t": t0, "k": k, "h": h, "short": short - COST + (0.0 if stopped else f),
                         "short_nostop": -ret - COST + f, "hedged": short - btc_ret * (0 if stopped else 1) - 2 * COST + (0.0 if stopped else f),
                         "worst": worst, "stopped": bool(stopped), "liq24": liq24})
    return rows


def collect(root: Path, syms: list[str]) -> None:
    btc = _load(root, "BTCUSDT")
    if btc is None:
        print("  newlist: нет BTCUSDT", flush=True)
        return
    floor = btc.index[0] + START_MARGIN
    out = []
    for s in mine(syms):
        try:
            d = _load(root, s)
            if d is None or s == "BTCUSDT" or d.index[0] <= floor or len(d) < 24 + 72 + 24:
                continue
            out += trade_rows(s, d, btc, funding(root, s))
        except Exception as e:
            print(f"  newlist {s}: пропуск ({e})", flush=True)
    print(f"  newlist: строк {len(out)}, монет {len({r['symbol'] for r in out})}", flush=True)
    if out:
        pd.DataFrame(out).to_parquet(part_path("newlist"), index=False)


def _t_month(z: pd.DataFrame, col: str) -> float:
    m = z.groupby(z.t.dt.to_period("M"))[col].mean()
    return float(m.mean() / (m.std() / np.sqrt(len(m)))) if len(m) > 2 and m.std() > 0 else np.nan


def _cell(z: pd.DataFrame, col: str) -> str:
    if len(z) < 10:
        return f"n={len(z)}"
    r = z[col]
    return f"{r.mean() * 100:+.1f}% (мед {r.median() * 100:+.1f}, t_мес {_t_month(z, col):.1f}, плюс {np.mean(r > 0):.0%}, n={len(z)})"


def report() -> None:
    parts = all_parts("newlist")
    print("\n===== NEWLIST: шорт нового перпетуала через k часов после листинга, удержание h часов; средний результат "
          "на сделку в % номинала после издержек и funding =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    n = df.drop_duplicates("symbol")
    print(f"  монет {len(n)}, листинги с {n.t.min():%Y-%m} по {n.t.max():%Y-%m}; по годам: "
          + ", ".join(f"{y} {c}" for y, c in n.t.dt.year.value_counts().sort_index().items()))
    for col, nm in (("short", "шорт со стопом 30%"), ("short_nostop", "шорт без стопа"), ("hedged", "шорт + лонг BTC, стоп 30%")):
        print(f"\n  --- {nm} ---")
        rows = []
        for k in DELAYS:
            rows.append({"вход через, ч": k, **{f"{h} ч": _cell(df[(df.k == k) & (df.h == h)], col) for h in HOLDS}})
        print(pd.DataFrame(rows).to_string(index=False))
    z = df[(df.k == 24) & (df.h == 72)]
    print(f"\n  вход через 24 ч, 72 ч: рост против шорта (max high): мед {z.worst.median() * 100:.0f}%, p90 "
          f"{z.worst.quantile(.9) * 100:.0f}%, макс {z.worst.max() * 100:.0f}%; стоп 30% сработал у {z.stopped.mean():.0%}")
    for nm, w in (("2020–2021", z[z.t.dt.year <= 2021]), ("2022–2023", z[(z.t.dt.year >= 2022) & (z.t.dt.year <= 2023)]),
                  ("2024–2026", z[z.t.dt.year >= 2024])):
        print(f"    {nm}: {_cell(w, 'short')}")
    q = z[z.liq24 >= LIQ_USD]
    print(f"    ликвидные (оборот первых 24 ч ≥ ${LIQ_USD / 1e6:.0f}M): {_cell(q, 'short')}")
    q5 = q[q.short < q.short.quantile(0.95)] if len(q) > 20 else q
    print(f"    ликвидные без 5% лучших сделок: {_cell(q5, 'short')}")


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
