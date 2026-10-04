"""
Разлоки: проверка устойчивости шорта перед разлоком (research.unlocks) перед добавлением в бота.

1) Сетка входа/выхода: шорт от T-10 / T-7 / T-5 / T-3 до T / T+1 / T+3 (цены 00:00 UTC, 12 б.п., funding);
   также «против BTC» (шорт монеты + лонг BTC на ту же сумму, издержки x2).
2) Определение события: (минимальный размер, во сколько раз больше обычного дневного разлока) =
   (0.5%, 5), (1%, 3), (1%, 5) — основное, (1%, 10), (2%, 5).
3) Повторяющиеся (тот же протокол давал разлок за 45 дней до) против разовых.
4) Доступность на Bybit: существует ли файл сделок перпетуала на public.bybit.com за день входа T-7.
Основная сделка (зафиксирована до запуска): шорт T-7 → T+1 с funding; её сделки — в unlock_trades.csv для
моделирования счёта.

    python -m research.unlocks2 collect --symbols <все перпетуалы>   (по частям)
    python -m research.unlocks2 report
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import unlocks as U
from .listings2 import _cell
from .shard import all_parts, mine, part_path

ENTRIES = (-10, -7, -5, -3)
EXITS = (0, 1, 3)
DEFS = [(0.005, 5.0), (0.01, 3.0), (0.01, 5.0), (0.01, 10.0), (0.02, 5.0)]
PRIMARY_DEF = (0.01, 5.0)
BYBIT_FILE = "https://public.bybit.com/trading/{s}/{s}{d}.csv.gz"


def events(cache: Path, min_size: float, cliff_x: float) -> pd.DataFrame:
    U.MIN_SIZE, U.CLIFF_X = min_size, cliff_x
    ev = U.unlock_events(cache)
    if len(ev):
        ev = ev.sort_values(["protocol", "t"])
        prev = ev.groupby("protocol")["t"].shift(1)
        ev["recurring"] = (ev["t"] - prev) <= pd.Timedelta(days=45)
        ev["def"] = f"{min_size:.1%}/x{cliff_x:.0f}"
    return ev


def on_bybit(sym: str, day: pd.Timestamp, cache: Path) -> bool:
    f = cache / f"{sym}-bybitday-{day:%Y-%m-%d}.flag"
    if f.exists():
        return f.read_text() == "1"
    ok = False
    for attempt in range(3):
        try:
            req = urllib.request.Request(BYBIT_FILE.format(s=sym, d=f"{day:%Y-%m-%d}"), method="HEAD",
                                         headers={"User-Agent": "flow-scalper-research/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                ok = r.status == 200
            break
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                break
        except Exception:
            continue
    f.write_text("1" if ok else "0")
    return ok


def grid(c: pd.Series, btc: pd.Series, fund: pd.Series, t: pd.Timestamp) -> dict:
    def px(s: pd.Series, d: int) -> float:
        return float(s.get(t + pd.Timedelta(days=d), np.nan))
    row = {}
    for a, b in itertools.product(ENTRIES, EXITS):
        pa, pb, ba, bb = px(c, a), px(c, b), px(btc, a), px(btc, b)
        if not (np.isfinite(pa) and np.isfinite(pb)):
            continue
        f = fund[(fund.index >= t + pd.Timedelta(days=a)) & (fund.index < t + pd.Timedelta(days=b))].sum()
        raw = -(pb / pa - 1) - U.COST + f
        row[f"s{-a}_{b}"] = raw * 1e4
        if np.isfinite(ba) and np.isfinite(bb):
            row[f"h{-a}_{b}"] = (raw + (bb / ba - 1) - U.COST) * 1e4
    return row


def collect(root: Path, syms: list[str]) -> None:
    cache = root / "cache" / "llama"
    cache.mkdir(parents=True, exist_ok=True)
    flags = root / "cache" / "bybitflags"
    flags.mkdir(parents=True, exist_ok=True)
    have = set(syms)
    btc = U.daily_close(root, "BTCUSDT")
    rows = []
    for md, cx in DEFS:
        ev = events(cache, md, cx)
        if not len(ev):
            continue
        ev["symbol"] = ev["ticker"].map(lambda tk: next((s for s in (f"{tk}USDT", f"1000{tk}USDT") if s in have), None))
        ev = ev[ev["symbol"].notna()]
        ev = ev[ev["symbol"].isin(set(mine(sorted(ev["symbol"].unique()))))]
        for s, g in ev.groupby("symbol"):
            c = U.daily_close(root, s)
            if c is None or c.first_valid_index() is None:
                continue
            fund = U.daily_funding(root, s)
            for r in g.itertuples():
                if c.first_valid_index() > r.t - pd.Timedelta(days=30):
                    continue
                m = grid(c, btc, fund, r.t)
                if not m:
                    continue
                by = on_bybit(s, r.t - pd.Timedelta(days=7), flags) if (md, cx) == PRIMARY_DEF else np.nan
                rows.append({"def": f"{md:.1%}/x{cx:.0f}", "symbol": s,
                             "protocol": r.protocol, "t": r.t, "size": r.size, "recurring": r.recurring,
                             "bybit": by, **m})
    print(f"  в части: строк {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("unlocks2"), index=False)


def _per(df: pd.DataFrame, col: str) -> dict:
    out = {}
    for p, (a, b) in U.PER.items():
        x = df[(df.t >= a) & (df.t < b)]
        out[p] = _cell(x.get(col, pd.Series(dtype=float)), x["t"])
    return out


def report() -> None:
    parts = all_parts("unlocks2")
    if not parts:
        print("частей нет")
        return
    res = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    res["t"] = pd.to_datetime(res["t"], utc=True)
    prim_name = f"{PRIMARY_DEF[0]:.1%}/x{PRIMARY_DEF[1]:.0f}"
    prim = res[res["def"] == prim_name].drop_duplicates(["symbol", "t"])
    print(f"===== UNLOCKS2: строк {len(res)}, основное определение {prim_name}: событий {len(prim)}, "
          f"монет {prim.symbol.nunique()}; на Bybit в день входа: {int(prim['bybit'].fillna(False).astype(bool).sum())} =====")
    print("ячейка: среднее, б.п. (t по дням, доля прибыльных, n); s — шорт с funding, h — против BTC")
    rows = []
    for a, b in itertools.product(ENTRIES, EXITS):
        for kind in ("s", "h"):
            rows.append({"сделка": f"{'шорт' if kind == 's' else 'против BTC'} T-{-a} → T{'+' + str(b) if b else ''}",
                         **_per(prim, f"{kind}{-a}_{b}")})
    print("\n1) Сетка входа и выхода (основное определение):")
    print(pd.DataFrame(rows).to_string(index=False))
    rows = []
    for d, g in res.groupby("def"):
        g = g.drop_duplicates(["symbol", "t"])
        rows.append({"определение": d, "событий": len(g), **{f"{k} шорт": v for k, v in _per(g, "s7_1").items()},
                     **{f"{k} против BTC": v for k, v in _per(g, "h7_1").items()}})
    print("\n2) Определение события, основная сделка T-7 → T+1:")
    print(pd.DataFrame(rows).to_string(index=False))
    rows = []
    for name, g in (("разовые", prim[~prim["recurring"].astype(bool)]), ("повторяющиеся", prim[prim["recurring"].astype(bool)]),
                    ("есть на Bybit", prim[prim["bybit"].fillna(False).astype(bool)]),
                    ("нет на Bybit", prim[~prim["bybit"].fillna(False).astype(bool)])):
        rows.append({"группа": name, "событий": len(g), **{f"{k} шорт": v for k, v in _per(g, "s7_1").items()},
                     **{f"{k} против BTC": v for k, v in _per(g, "h7_1").items()}})
    print("\n3–4) Разовые / повторяющиеся, доступность на Bybit (T-7 → T+1):")
    print(pd.DataFrame(rows).to_string(index=False))
    y = prim.groupby(prim["t"].dt.year)[["s7_1", "h7_1"]].agg(["size", "mean"]).round(0)
    print("\nПо годам (T-7 → T+1, б.п.):\n" + y.to_string())
    out = Path(os.environ.get("OUT", "../out"))
    out.mkdir(parents=True, exist_ok=True)
    prim[["t", "symbol", "protocol", "size", "recurring", "bybit", "s7_1", "h7_1"]].sort_values("t").to_csv(
        out / "unlock_trades.csv", index=False)
    print(f"\nunlock_trades.csv: {len(prim)} строк")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 340)
    pd.set_option("display.max_columns", 30)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x])
    else:
        report()
