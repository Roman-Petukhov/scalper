"""Отставание малоликвидных альтов от резкого хода BTC (минуты, а не миллисекунды). Ликвидные монеты повторяют ход BTC
сразу, у мелочи маркет-мейкеры медленнее — ход доходит с задержкой. Опережение Binance → Bybit на миллисекундах у нас
не окупилось (docs/research_report.md, HFT); BTC → мелкий альт на минутах не проверялся.

Свечи 5m Binance, 2023-01 … 2026-09. Событие: 5-минутная доходность BTC по модулю >= k·σ (σ — std 5m-доходностей BTC
за прошлые сутки), не чаще раза в час. На закрытии этой свечи — вход в сторону BTC во все монеты полосы $5–50M оборота
(за 30 прошлых дней), выход через h свечей по закрытию. Отстающие — монеты, чья доходность на той же свече в сторону BTC
меньше 30% хода BTC. Результат события — среднее по монетам (корзина равными весами). Издержки за круг 20 б.п.
(taker 2 × 5.5 + проскальзывание мелочи), ×2 — 40 б.п.
Варианты (заданы до запуска): k 3 / 4 × h 3 / 12 свечей (15 / 60 мин) × все / отстающие = 8. Принятие: средний
результат события после издержек > 0 в каждом периоде IS / VAL / HO (событий >= 20), t по дням >= 3, плюс при ×2.

    python -m research.btclag collect --symbols <1h-all>      (сам качает 5m и удаляет после монеты)
    python -m research.btclag report
"""
from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .oos import PER_ALL
from .ppr import day_t
from .shard import all_parts, mine, part_path
from .spikes_full import adv_daily

START, END = "2023-01", "2026-09"
ADV_LO, ADV_HI = 5e6, 50e6
KS = (3.0, 4.0)
HS = (3, 12)
LAG_SHARE = 0.3
COOLDOWN = 12
COST = 0.0020
M5_ROOT = Path.home() / "m5lag"
VARIANTS = list(itertools.product(KS, HS, ("все", "отстающие")))


def load5(root: Path, sym: str) -> pd.DataFrame | None:
    p = root / f"{sym}-5m.parquet"
    if not p.exists():
        return None
    k = pd.read_parquet(p, columns=["open_time", "close"])
    s = pd.Series(k["close"].to_numpy(dtype="float64"), index=pd.to_datetime(k["open_time"], unit="ms", utc=True))
    return s[~s.index.duplicated()].sort_index().to_frame("close")


def btc_events(btc: pd.Series, k: float) -> pd.DataFrame:
    """Свечи BTC с |доходность| >= k·σ прошлых суток, не чаще раза в COOLDOWN свечей: время, знак, ход."""
    r = btc.pct_change()
    sig = r.rolling(288, min_periods=200).std().shift(1)
    hit = (r.abs() >= k * sig).to_numpy()
    out, last = [], -10**9
    for i in np.flatnonzero(hit):
        if i - last >= COOLDOWN:
            out.append((btc.index[i], float(np.sign(r.iloc[i])), float(r.iloc[i])))
            last = i
    return pd.DataFrame(out, columns=["t", "side", "move"])


def coin_rows(c: pd.Series, ev: pd.DataFrame, adv: pd.Series, sym: str, k: float) -> list[dict]:
    pos = c.index.get_indexer(ev.t)
    rows = []
    a = adv.reindex(ev.t.dt.normalize()).to_numpy()
    x = c.to_numpy()
    for j, (i, row) in enumerate(zip(pos, ev.itertuples(index=False))):
        if i <= 0 or i + max(HS) >= len(x) or not (ADV_LO <= a[j] < ADV_HI):
            continue
        same = row.side * (x[i] / x[i - 1] - 1)
        rec = {"symbol": sym, "t": row.t, "k": k, "lag": bool(same < LAG_SHARE * abs(row.move))}
        for h in HS:
            rec[f"r{h}"] = row.side * (x[i + h] / x[i] - 1)
        rows.append(rec)
    return rows


def collect(root: Path, syms: list[str]) -> None:
    M5_ROOT.mkdir(parents=True, exist_ok=True)
    D.set_host("cdn")
    D.build(["BTCUSDT"], "5m", START, END, M5_ROOT, 16)
    btc = load5(M5_ROOT, "BTCUSDT")
    if btc is None:
        print("  btclag: нет 5m BTC", flush=True)
        return
    evs = {k: btc_events(btc["close"], k) for k in KS}
    print("  btclag: событий BTC " + ", ".join(f"k {k:g} — {len(e)}" for k, e in evs.items()), flush=True)
    rows = []
    for s in mine(syms):
        if s == "BTCUSDT":
            continue
        a = adv_daily(root, s)
        if a is None or not ((a >= ADV_LO) & (a < ADV_HI)).any():
            continue
        try:
            D.build([s], "5m", START, END, M5_ROOT, 16)
            c = load5(M5_ROOT, s)
            if c is not None and len(c) > 5000:
                for k, e in evs.items():
                    rows += coin_rows(c["close"], e, a, s, k)
        except Exception as e:
            print(f"  btclag {s}: пропуск ({e})", flush=True)
        finally:
            for f in list(M5_ROOT.glob(f"{s}-5m.parquet")) + list((M5_ROOT / "cache").glob(f"{s}-*")):
                f.unlink()                                       # место на диске раннера
    print(f"  btclag: строк {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("btclag"), index=False)


def report() -> None:
    print("\n===== BTCLAG: малоликвидные альты догоняют резкий ход BTC (5m; результат события — корзина монеты $5–50M; "
          f"издержки {COST * 1e4:.0f} б.п. за круг) =====")
    parts = all_parts("btclag")
    if not parts:
        print("  btclag: нет данных")
        return
    x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
    passed = []
    for k, h, flt in VARIANTS:
        z = x[(x.k == k) & (x.lag if flt == "отстающие" else True)]
        if z.empty:
            continue
        ev = z.groupby("t").agg(R=(f"r{h}", "mean"), n=(f"r{h}", "size")).reset_index()
        ev["R"] = ev["R"] - COST
        cells, ok = [], []
        for p, (a, b) in PER_ALL.items():
            m = ev[(ev.t >= a) & (ev.t < b)]
            if len(m) == 0:
                continue
            cells.append(f"{p} {m.R.mean() * 1e4:+.1f} б.п. ({len(m)})")
            if len(m) >= 20 and p in ("is", "val", "ho"):
                ok.append(m.R.mean() > 0)
        t = day_t(ev, "R") if len(ev) > 10 else float("nan")
        x2 = (ev.R - COST).mean()
        good = bool(ok) and all(ok) and np.isfinite(t) and t >= 3 and x2 > 0
        print(f"    k {k:g}σ, {h * 5} мин, {flt}: {'принято' if good else 'нет'} — событий {len(ev)}, монет на событие "
              f"{ev.n.median():.0f}, до издержек {(ev.R + COST).mean() * 1e4:+.1f} б.п.\n      {'  '.join(cells)}\n"
              f"      всё время {ev.R.mean() * 1e4:+.1f} б.п. (t {t:.1f}, прибыльных {(ev.R > 0).mean():.0%}); ×2 "
              f"{x2 * 1e4:+.1f} б.п.")
        if good:
            passed.append(f"k {k:g} {h * 5} мин {flt}")
    print("\n  ИТОГ: " + ("прошли — " + "; ".join(passed) if passed else "отставание мелочи от BTC не окупает издержки"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [s for s in a.symbols.split(",") if s])
    else:
        report()
