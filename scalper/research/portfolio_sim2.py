"""
Счёт $100 на реальной последовательности сделок 2023-01 … 2026-09: четыре стратегии по отдельности и вместе,
с реинвестированием и ограничениями Bybit (минимальный ордер $5, плечо — не больше lev_cap капиталов в позициях).

Стратегии и размер позиции (доля ТЕКУЩЕГО капитала; режимы «умеренный» / «агрессивный»):
    oi_liq   лесенка 5 лимитов (research.export_trades: доходность на номинал сигнала, удержание 4 ч после бара);
             номинал 0.3 / 0.6 капитала, не меньше $25, не больше 3 / 5 позиций одновременно
    spikes   лимит на прострел 6σ (лонг, сделки Bybit; выборка 25 монет — частота занижена/смещена, см. отчёт);
             номинал 0.05 / 0.10 капитала (не меньше $5), в лимитках держим 1 / 3 капитала
    coil     риск 1% / 2% капитала на сделку (номинал = риск / (1.5 ATR / цена)), до 10 дней
    announce шорт после Monitoring Tag (вход через 5 мин, 24 ч) и после делистинга (вход через 5 мин, 4 ч) — только
             события, где монета была на Bybit, результат по ценам Bybit с funding; номинал 0.3 / 0.6 капитала
Сценарий «с поправкой»: из каждой сделки вычитается 30% среднего преимущества её стратегии.
Монте-Карло: 12 случайных месяцев истории (с возвращением) × N путей.

    python -m research.portfolio_sim2 --oi oi_liq_trades.csv --spikes spikes_bybit_trades.csv \\
        --coil coil_trades.csv --announce announce_trades.csv
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

START, END = "2023-01-01", "2026-10-01"
CAPITAL = 100.0
RUIN = 10.0
HAIRCUT = 0.3
MIN_ORDER = 5.0


@dataclass(frozen=True)
class Mode:
    name: str
    oi_frac: float
    oi_max: int
    sp_frac: float
    sp_reserve: float
    coil_risk: float
    ann_frac: float
    lev_cap: float


MODES = [Mode("умеренный", 0.3, 3, 0.05, 1.0, 0.01, 0.3, 2.0),
         Mode("агрессивный", 0.6, 5, 0.10, 3.0, 0.02, 0.6, 5.0)]


def load_events(a: argparse.Namespace) -> pd.DataFrame:
    oi = pd.read_csv(a.oi)
    oi["t"] = pd.to_datetime(oi["t"], utc=True, format="ISO8601")
    e_oi = pd.DataFrame({"t": oi["t"] + pd.Timedelta(hours=1), "end": oi["t"] + pd.Timedelta(hours=5),
                         "symbol": oi["symbol"], "r": oi["L5x0.5"], "risk": np.nan, "kind": "oi_liq"})
    sp = pd.read_csv(a.spikes)
    sp = sp[(sp["m"] == 6.0) & sp["t"].notna()]
    sp["t"] = pd.to_datetime(sp["t"], utc=True, format="ISO8601")
    e_sp = pd.DataFrame({"t": sp["t"], "end": sp["t"] + pd.Timedelta(minutes=60), "symbol": sp["symbol"],
                         "r": sp["r"], "risk": np.nan, "kind": "spikes"})
    co = pd.read_csv(a.coil)
    co["t"] = pd.to_datetime(co["t"], utc=True, format="ISO8601")
    e_co = pd.DataFrame({"t": co["t"] + pd.Timedelta(hours=4), "end": co["t"] + pd.Timedelta(hours=4) * (co["bars"] + 1),
                         "symbol": co["symbol"], "r": co["R"], "risk": co["risk"], "kind": "coil"})
    an = pd.read_csv(a.announce)
    an["t"] = pd.to_datetime(an["t"], utc=True, format="ISO8601")
    an = an[an["bybit"].astype(bool)]
    rows = []
    for cls, col, hold in (("monitoring", "by_d5_24h", 24), ("delist", "by_d5_4h", 4)):
        g = an[(an["cls"] == cls) & an[col].notna()]
        rows.append(pd.DataFrame({"t": g["t"] + pd.Timedelta(minutes=5), "end": g["t"] + pd.Timedelta(hours=hold, minutes=5),
                                  "symbol": g["symbol"], "r": g[col] / 1e4, "risk": np.nan, "kind": "announce"}))
    e_an = pd.concat(rows, ignore_index=True).drop_duplicates(["symbol", "t"])
    ev = pd.concat([e_oi, e_sp, e_co, e_an], ignore_index=True)
    ev["t"] = pd.to_datetime(ev["t"], utc=True)
    ev["end"] = pd.to_datetime(ev["end"], utc=True)
    return ev[(ev["t"] >= START) & (ev["t"] < END)].sort_values("t").reset_index(drop=True)


def haircut(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.copy()
    for k, g in ev.groupby("kind"):
        ev.loc[g.index, "r"] -= HAIRCUT * g["r"].mean()
    return ev


def simulate(ev: pd.DataFrame, mode: Mode, coins: list[str]) -> tuple[pd.Series, bool, int]:
    """Капитал на конец месяцев, флаг разорения, число исполненных сделок."""
    eq = CAPITAL
    pos: list[tuple] = []                     # (конец, номинал, pnl при закрытии, вид, монета)
    month_eq, cur, watched, done = {}, None, set(), 0
    T, E, S, R, K, RK = (ev[c].to_numpy() for c in ("t", "end", "symbol", "r", "risk", "kind"))
    for i in range(len(ev)):
        t = T[i]
        keep = []
        for p in pos:
            if p[0] <= t:
                eq += p[2]
            else:
                keep.append(p)
        pos = keep
        m = pd.Timestamp(t).strftime("%Y-%m")
        if m != cur:
            if cur is not None:
                month_eq[cur] = eq
            cur = m
            order = max(MIN_ORDER, mode.sp_frac * eq)
            watched = set(coins[: int(mode.sp_reserve * eq // order)])
        if eq < RUIN:
            month_eq[cur] = eq
            return pd.Series(month_eq), True, done
        used = sum(p[1] for p in pos)
        kind, sym = RK[i], S[i]
        if any(p[4] == sym and p[3] == kind for p in pos):
            continue
        if kind == "oi_liq":
            if sum(p[3] == "oi_liq" for p in pos) >= mode.oi_max:
                continue
            notional = max(5 * MIN_ORDER, mode.oi_frac * eq)
            pnl = notional * R[i]
        elif kind == "spikes":
            if sym not in watched:
                continue
            notional = max(MIN_ORDER, mode.sp_frac * eq)
            pnl = notional * R[i]
        elif kind == "coil":
            risk_usd = mode.coil_risk * eq
            notional = max(MIN_ORDER, risk_usd / K[i])
            pnl = notional * K[i] * R[i]
        else:
            notional = max(MIN_ORDER, mode.ann_frac * eq)
            pnl = notional * R[i]
        if used + notional > mode.lev_cap * eq:
            continue
        pos.append((E[i], notional, pnl, kind, sym))
        done += 1
    eq += sum(p[2] for p in pos)
    month_eq[cur] = eq
    return pd.Series(month_eq), eq < RUIN, done


def history(ev: pd.DataFrame, mode: Mode, coins: list[str]) -> dict:
    s, ruined, n = simulate(ev, mode, coins)
    eq = pd.concat([pd.Series({"start": CAPITAL}), s])
    mret = eq.pct_change().dropna()
    months = len(mret)
    return {"итог $": round(eq.iloc[-1], 0), "сделок": n, "в мес. (CAGR)": f"{(eq.iloc[-1] / CAPITAL) ** (1 / months) - 1:+.1%}",
            "медиана мес.": f"{mret.median():+.1%}", "худший мес.": f"{mret.min():+.0%}",
            "мес. в минусе": f"{(mret < 0).mean():.0%}", "макс. просадка": f"{(eq / eq.cummax() - 1).min():.0%}",
            "разорён": ruined}


def monte_carlo(ev: pd.DataFrame, mode: Mode, coins: list[str], paths: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    ev = ev.assign(month=ev["t"].dt.tz_localize(None).dt.to_period("M"))
    months = sorted(ev["month"].unique())
    groups = {m: g for m, g in ev.groupby("month")}
    finals, dd50, ruins = [], 0, 0
    base = pd.Timestamp("2000-01-01", tz="UTC")
    for _ in range(paths):
        parts = []
        for j, k in enumerate(rng.choice(len(months), size=12, replace=True)):
            g = groups[months[k]]
            shift = base + pd.DateOffset(months=j) - months[k].start_time.tz_localize("UTC")
            parts.append(g.assign(t=g["t"] + shift, end=g["end"] + shift))
        s, ruined, _ = simulate(pd.concat(parts, ignore_index=True).sort_values("t"), mode, coins)
        eq = pd.concat([pd.Series({"start": CAPITAL}), s])
        finals.append(eq.iloc[-1] / CAPITAL)
        dd50 += (eq / eq.cummax() - 1).min() <= -0.5
        ruins += ruined
    f = np.array(finals)
    return {"год, медиана": f"${CAPITAL * np.median(f):.0f}", "10% худших": f"${CAPITAL * np.quantile(f, 0.1):.0f}",
            "10% лучших": f"${CAPITAL * np.quantile(f, 0.9):.0f}", "≈ в мес.": f"{np.median(f) ** (1 / 12) - 1:+.1%}",
            "P(просадка ≥50%)": f"{dd50 / paths:.0%}", "P(разорение)": f"{ruins / paths:.0%}"}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 20)
    ap = argparse.ArgumentParser()
    for k in ("oi", "spikes", "coil", "announce"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--paths", type=int, default=300)
    a = ap.parse_args()
    ev_all = load_events(a)
    coins = list(pd.Series(ev_all.loc[ev_all.kind == "spikes", "symbol"].unique()).sample(frac=1.0, random_state=1))
    print(f"сделок 2023-01…2026-09: " + ", ".join(f"{k} {v}" for k, v in ev_all["kind"].value_counts().items()))
    for k, g in ev_all.groupby("kind"):
        x = g["r"] * (g["risk"] if k == "coil" else 1.0)
        print(f"  {k}: средняя сделка {x.mean():+.2%} номинала" + (" (R x риск)" if k == "coil" else "") +
              f", прибыльных {np.mean(g['r'] > 0):.0%}")
    sets = {"oi_liq": ["oi_liq"], "spikes": ["spikes"], "coil": ["coil"], "announce": ["announce"],
            "ВСЕ ЧЕТЫРЕ": ["oi_liq", "spikes", "coil", "announce"]}
    for label, adj in (("КАК В БЭКТЕСТЕ", False), ("С ПОПРАВКОЙ −30% преимущества", True)):
        ev = haircut(ev_all) if adj else ev_all
        print(f"\n========== {label}, старт ${CAPITAL:.0f} ==========")
        for mode in MODES:
            hist, mc = [], []
            for name, kinds in sets.items():
                sub = ev[ev["kind"].isin(kinds)]
                hist.append({"стратегия": name, **history(sub, mode, coins)})
                mc.append({"стратегия": name, **monte_carlo(sub, mode, coins, a.paths, 7)})
            print(f"\n--- режим «{mode.name}»: история 45 месяцев ---")
            print(pd.DataFrame(hist).to_string(index=False))
            print(f"--- режим «{mode.name}»: Монте-Карло, 12 случайных месяцев × {a.paths} ---")
            print(pd.DataFrame(mc).to_string(index=False))
