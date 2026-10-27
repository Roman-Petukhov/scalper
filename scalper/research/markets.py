"""Фондовый рынок и спотовые ETF против крипты (вопрос владельца «как влияют недавно принятые ETF и фондовый рынок»).

Дневки перпетуалов — части research/sectors.py (BTC, ETH, корзина альтов с оборотом >= $5M). Фондовый рынок — FRED
без ключа: S&P 500 (SP500), Nasdaq Composite (NASDAQCOM), VIX (VIXCLS). Дневная свеча крипты — 00:00–00:00 UTC, то
есть день t крипты начинается через 3–4 ч после закрытия США дня t−1.
Задано до запуска:
    K  корреляция BTC и альтов с S&P 500 в тот же день, по периодам — справка
    N  следующий день: S&P 500 за день t вырос > 1% — лонг BTC на день t+1 крипты, упал < −1% — шорт; издержки 11 б.п.
       (часть реакции уже прошла за 3–4 ч до 00:00 — консервативно)
    V  всплеск страха: VIX закрылся >= 30 или вырос за день >= 20% — лонг BTC на 5 / 20 дней против безусловной средней
    E  спотовые ETF в США (первый день торгов): BTC 2024-01-11, ETH 2024-07-23, SOL / LTC / HBAR 2025-10-28,
       XRP 2025-11-13, DOGE 2025-11-24, LINK 2025-12-02 — ход монеты сверх корзины альтов (у BTC — сверх ETH) за
       30 дней до, 7 дней после, с 7 по 30 и с 30 по 90 день — справка (8 событий)
Принятие (N, V): > 0 в каждом периоде IS / VAL / HO (сделок >= 20), t по дням >= 3, > 0 при издержках ×2.

    python -m research.sectors collect --symbols <1h-all> && python -m research.markets report
"""
from __future__ import annotations

import io
import sys
import time
import urllib.request
import warnings

import numpy as np
import pandas as pd

from .oos import PER_ALL
from .ppr import day_t
from .sectors import ADV_MIN, load_daily

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}"
SERIES = {"spx": "SP500", "ndx": "NASDAQCOM", "vix": "VIXCLS"}
COST = 0.0011
MOVE = 0.01
VIX_LEVEL, VIX_JUMP = 30.0, 0.20
ETF = {"BTCUSDT": "2024-01-11", "ETHUSDT": "2024-07-23", "SOLUSDT": "2025-10-28", "LTCUSDT": "2025-10-28",
       "HBARUSDT": "2025-10-28", "XRPUSDT": "2025-11-13", "DOGEUSDT": "2025-11-24", "LINKUSDT": "2025-12-02"}
WINDOWS = {"за 30 дней до": (-30, 0), "0…7 после": (0, 7), "7…30": (7, 30), "30…90": (30, 90)}


def fred(series: str) -> pd.Series:
    for attempt in range(4):
        try:
            req = urllib.request.Request(FRED.format(s=series), headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                df = pd.read_csv(io.BytesIO(r.read()))
            df.columns = ["day", "v"]
            s = pd.Series(pd.to_numeric(df["v"], errors="coerce").to_numpy(), index=pd.to_datetime(df["day"]))
            return s.dropna()
        except Exception as e:
            print(f"  markets: FRED {series} → {e}", flush=True)
            time.sleep(2 ** attempt)
    return pd.Series(dtype="float64")


def verdict(z: pd.DataFrame) -> tuple[bool, str]:
    cells, ok = [], []
    for p, (a, b) in PER_ALL.items():
        m = z[(z.t >= a) & (z.t < b)]
        if m.empty:
            continue
        cells.append(f"{p} {m.X.mean():+.2%} ({len(m)})")
        if len(m) >= 20 and p in ("is", "val", "ho"):
            ok.append(m.X.mean() > 0)
    t = day_t(z, "X") if len(z) > 10 else float("nan")
    x2 = (z.X - COST).mean()
    good = bool(ok) and all(ok) and np.isfinite(t) and t >= 3 and x2 > 0
    return good, f"{'  '.join(cells)}\n        всё время {z.X.mean():+.2%} (t {t:.1f}, в плюсе {(z.X > 0).mean():.0%}); ×2 {x2:+.2%}"


def next_day(btc: pd.Series, spx: pd.Series) -> pd.DataFrame:
    """Сделки N: сигнал — доходность S&P 500 за день t, результат — день t+1 крипты (00:00 t+1 … 00:00 t+2)."""
    sr = spx.pct_change().dropna()
    br = btc.pct_change()
    rows = []
    for d, v in sr.items():
        if abs(v) <= MOVE:
            continue
        t1 = d + pd.Timedelta(days=1)
        nxt = br.get(t1)                                         # свеча t+1: 00:00 t+1 … 00:00 t+2
        if nxt is None or not np.isfinite(nxt):
            continue
        rows.append({"t": t1, "X": float(np.sign(v) * nxt) - COST, "side": int(np.sign(v))})
    return pd.DataFrame(rows)


def vix_events(btc: pd.Series, vix: pd.Series, h: int) -> pd.DataFrame:
    jump = vix.pct_change()
    hit = vix[(vix >= VIX_LEVEL) | (jump >= VIX_JUMP)].index
    fwd = btc.shift(-h) / btc - 1
    base = fwd.mean()
    rows, last = [], None
    for d in hit:
        t = d                                                    # закрытие свечи d — 00:00 d+1, после закрытия США
        if last is not None and (t - last).days < h:
            continue
        v = fwd.get(t)
        if v is None or not np.isfinite(v):
            continue
        rows.append({"t": t, "X": float(v - base) - COST})
        last = t
    return pd.DataFrame(rows)


def etf_study(close: pd.DataFrame, alts: pd.Series) -> None:
    print("\n  --- E: спотовые ETF в США — ход монеты сверх корзины альтов (у BTC — сверх ETH) ---")
    alt_idx = (1 + alts.fillna(0.0)).cumprod()
    for sym, day in ETF.items():
        if sym not in close.columns:
            print(f"    {sym}: нет данных")
            continue
        t0 = pd.Timestamp(day)
        px = close[sym].dropna()
        if sym == "BTCUSDT" and "ETHUSDT" not in close.columns:
            continue
        bench = close["ETHUSDT"].dropna() if sym == "BTCUSDT" else alt_idx
        cells = []
        for name, (a, b) in WINDOWS.items():
            ta, tb = t0 + pd.Timedelta(days=a), t0 + pd.Timedelta(days=b)
            if tb > px.index[-1] or ta not in px.index or tb not in px.index:
                cells.append(f"{name} —")
                continue
            own = px[tb] / px[ta] - 1
            ref = bench.asof(tb) / bench.asof(ta) - 1
            cells.append(f"{name} {own:+.0%} (сверх {own - ref:+.0%})")
        print(f"    {sym[:-4]} {day}: " + "  ".join(cells))


def report() -> None:
    print("\n===== MARKETS: фондовый рынок (S&P 500, Nasdaq, VIX — FRED) и спотовые ETF против крипты =====")
    got = load_daily()
    if got is None:
        print("  markets: нет дневок (research/sectors.py collect)")
        return
    close, qv = got
    r = close.pct_change().where(qv.rolling(30, min_periods=20).mean().shift(1) >= ADV_MIN)
    alts = r.drop(columns=["BTCUSDT", "ETHUSDT"], errors="ignore").mean(axis=1)
    if "BTCUSDT" not in close.columns:
        print("  markets: нет BTCUSDT в дневках")
        return
    btc = close["BTCUSDT"].dropna()
    st = {k: fred(v) for k, v in SERIES.items()}
    print("  FRED: " + ", ".join(f"{k} {len(v)} дней" + (f" ({v.index.min():%Y-%m-%d} … {v.index.max():%Y-%m-%d})" if len(v) else "")
                                  for k, v in st.items()))
    passed = []
    spx = st["spx"]
    if len(spx):
        sr = spx.pct_change()
        br = btc.pct_change().reindex(sr.index)
        ar = alts.reindex(sr.index)
        print("\n  --- K: корреляция с S&P 500 в тот же день (свеча крипты дня t содержит сессию США дня t) ---")
        brs, ars = br, ar
        for p, (a, b) in PER_ALL.items():
            m = (sr.index >= a) & (sr.index < b)
            if m.sum() < 30:
                continue
            print(f"    {p}: BTC {sr[m].corr(brs[m]):+.2f}, альты {sr[m].corr(ars[m]):+.2f}"
                  + (f", Nasdaq–BTC {st['ndx'].pct_change().reindex(sr.index)[m].corr(brs[m]):+.2f}" if len(st["ndx"]) else ""))
        z = next_day(btc, spx)
        good, txt = verdict(z)
        print(f"\n  --- N: S&P 500 за день > ±1% → BTC на следующий день крипты по знаку: {'принято' if good else 'нет'} ---\n"
              f"        {txt}\n        после роста {z[z.side > 0].X.mean():+.2%} ({(z.side > 0).sum()}), после падения "
              f"{z[z.side < 0].X.mean():+.2%} ({(z.side < 0).sum()})")
        if good:
            passed.append("N следующий день")
    vix = st["vix"]
    if len(vix):
        print("\n  --- V: всплеск страха (VIX >= 30 или +20% за день) → BTC сверх безусловной средней ---")
        for h in (5, 20):
            z = vix_events(btc, vix, h)
            if z.empty:
                continue
            good, txt = verdict(z)
            print(f"    {h} дней: {'принято' if good else 'нет'} (событий {len(z)})\n        {txt}")
            if good:
                passed.append(f"V {h} дней")
    etf_study(close, alts)
    print("\n  ИТОГ: " + ("прошли — " + "; ".join(passed) if passed else "фондовый рынок не даёт сделки по крипте, окупающей издержки"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
