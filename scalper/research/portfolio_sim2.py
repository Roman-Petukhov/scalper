"""
Счёт $100 на реальной последовательности сделок 2023-01 … 2026-09: четыре стратегии по отдельности и вместе,
с реинвестированием и ограничениями Bybit (минимальный ордер $5, плечо — не больше lev_cap капиталов в позициях).

Стратегии и размер позиции (доля ТЕКУЩЕГО капитала; режимы «умеренный» / «агрессивный»):
    oi_liq   лесенка 5 лимитов (research.export_trades: доходность на номинал сигнала, удержание 4 ч после бара);
             номинал 0.3 / 0.6 капитала, не меньше $25, не больше 3 / 5 позиций одновременно
    spikes   лимит на прострел 6σ (лонг, сделки Bybit; выборка 25 монет — частота занижена/смещена, см. отчёт)
             или, с --spikes-full, весь рынок (research.spikes_full, только монеты Bybit); номинал 0.05 / 0.10
             капитала (не меньше $5), в лимитках держим 1 / 3 капитала номинала. Лимитки стоят заранее, поэтому
             исполняются все (их не отменить в обвал) и не проверяются лимитом плеча; остальным стратегиям
             достаётся lev_cap минус резерв под лимитки. Монеты под лимитками — с наибольшим числом прострелов за
             прошлые 90 дней (весь рынок) или фиксированная случайная очередь (выборка 25 монет).
    coil     риск 1% / 2% капитала на сделку (номинал = риск / (1.5 ATR / цена)), до 10 дней
    announce шорт после Monitoring Tag (вход через 5 мин, 24 ч) и после делистинга (вход через 5 мин, 4 ч) — только
             события, где монета была на Bybit, результат по ценам Bybit с funding; номинал 0.3 / 0.6 капитала
    unlock   шорт перпетуала от T-7 до T+1 вокруг разлока >= 1% (research.unlocks2; только монеты, торговавшиеся на
             Bybit в день входа); номинал 0.1 / 0.2 капитала
Сценарий «с поправкой»: из каждой сделки вычитается 30% среднего преимущества её стратегии.
Монте-Карло: 12 случайных месяцев истории (с возвращением) × N путей.

    python -m research.portfolio_sim2 --oi oi_liq_trades.csv --spikes spikes_bybit_trades.csv \\
        --coil coil_trades.csv --announce announce_trades.csv [--unlock unlock_trades.csv]
        [--spikes-full spikes_full_trades.csv --spikes-cfg "6σ прокол 0.2%"] [--start 2025-07-01] [--drop-day 2025-10-10]
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
    unl_frac: float
    lev_cap: float
    exch_lev: float = 0.0          # плечо на бирже: 0 — резерв под лимитки вычитается из lev_cap (номиналом);
                                   # > 0 — маржа (позиции + стоящие лимитки) / exch_lev <= капитала, позиции <= lev_cap


MODES = [Mode("умеренный", 0.3, 3, 0.05, 1.0, 0.01, 0.3, 0.1, 2.0),
         Mode("агрессивный", 0.6, 5, 0.10, 3.0, 0.02, 0.6, 0.2, 5.0)]


def watch_rank(sp: pd.DataFrame) -> np.ndarray:
    """Место монеты в очереди под лимитки на момент сделки: по числу прострелов за прошлые 90 дней (до начала месяца)."""
    month = sp["t"].dt.tz_localize(None).dt.to_period("M").dt.start_time.dt.tz_localize("UTC")
    rank = np.empty(len(sp))
    for m0, idx in sp.groupby(month).groups.items():
        past = sp[(sp["t"] < m0) & (sp["t"] >= m0 - pd.Timedelta(days=90))]["symbol"].value_counts()
        order = {s: i for i, s in enumerate(past.index)}
        rank[sp.index.get_indexer(idx)] = [order.get(s, np.inf) for s in sp.loc[idx, "symbol"]]
    return rank


def load_events(a: argparse.Namespace) -> pd.DataFrame:
    oi = pd.read_csv(a.oi)
    oi["t"] = pd.to_datetime(oi["t"], utc=True, format="ISO8601")
    e_oi = pd.DataFrame({"t": oi["t"] + pd.Timedelta(hours=1), "end": oi["t"] + pd.Timedelta(hours=5),
                         "symbol": oi["symbol"], "r": oi["L5x0.5"], "risk": np.nan, "kind": "oi_liq"})
    if getattr(a, "spikes_full", ""):
        sp = pd.read_csv(a.spikes_full)
        sp = sp[(sp["cfg"] == a.spikes_cfg) & sp["bybit"].astype(bool)].reset_index(drop=True)
        sp["t"] = pd.to_datetime(sp["t"], utc=True, format="ISO8601")
        rank = watch_rank(sp)
    else:
        sp = pd.read_csv(a.spikes)
        sp = sp[(sp["m"] == 6.0) & sp["t"].notna()].reset_index(drop=True)
        sp["t"] = pd.to_datetime(sp["t"], utc=True, format="ISO8601")
        queue = list(pd.Series(sp["symbol"].unique()).sample(frac=1.0, random_state=1))
        rank = sp["symbol"].map({s: i for i, s in enumerate(queue)}).to_numpy(dtype=float)
    e_sp = pd.DataFrame({"t": sp["t"], "end": sp["t"] + pd.Timedelta(minutes=60), "symbol": sp["symbol"],
                         "r": sp["r"], "risk": np.nan, "kind": "spikes", "wrank": rank})
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
    parts = [e_oi, e_sp, e_co, e_an]
    if getattr(a, "unlock", None):
        un = pd.read_csv(a.unlock)
        un["t"] = pd.to_datetime(un["t"], utc=True, format="ISO8601")
        un = un[un["bybit"].fillna(False).astype(bool) & un["s7_1"].notna()]
        parts.append(pd.DataFrame({"t": un["t"] - pd.Timedelta(days=7), "end": un["t"] + pd.Timedelta(days=1),
                                   "symbol": un["symbol"], "r": un["s7_1"] / 1e4, "risk": np.nan, "kind": "unlock"}))
    ev = pd.concat(parts, ignore_index=True)
    ev["t"] = pd.to_datetime(ev["t"], utc=True)
    ev["end"] = pd.to_datetime(ev["end"], utc=True)
    ev = ev[(ev["t"] >= (getattr(a, "start", "") or START)) & (ev["t"] < END)]
    if getattr(a, "drop_day", ""):
        ev = ev[ev["t"].dt.strftime("%Y-%m-%d") != a.drop_day]
    return ev.sort_values("t").reset_index(drop=True)


def haircut(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.copy()
    for k, g in ev.groupby("kind"):
        ev.loc[g.index, "r"] -= HAIRCUT * g["r"].mean()
    return ev


def simulate(ev: pd.DataFrame, mode: Mode) -> tuple[pd.Series, bool, int]:
    """Капитал на конец месяцев, флаг разорения, число исполненных сделок."""
    eq = CAPITAL
    pos: list[tuple] = []                     # (конец, номинал, pnl при закрытии, вид, монета)
    month_eq, cur, n_watch, done = {}, None, 0, 0
    T, E, S, R, K, RK = (ev[c].to_numpy() for c in ("t", "end", "symbol", "r", "risk", "kind"))
    W = ev["wrank"].to_numpy() if "wrank" in ev else np.full(len(ev), np.inf)
    reserve = mode.sp_reserve if (RK == "spikes").any() else 0.0
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
            n_watch = int(mode.sp_reserve * eq // max(MIN_ORDER, mode.sp_frac * eq))
        if eq < RUIN:
            month_eq[cur] = eq
            return pd.Series(month_eq), True, done
        kind, sym = RK[i], S[i]
        if any(p[4] == sym and p[3] == kind for p in pos):
            continue
        if kind == "spikes":                  # лимитка стояла заранее: исполняется всегда, маржа зарезервирована
            if not W[i] < n_watch:
                continue
            notional = max(MIN_ORDER, mode.sp_frac * eq)
            pos.append((E[i], notional, notional * R[i], kind, sym))
            done += 1
            continue
        if kind == "oi_liq":
            if sum(p[3] == "oi_liq" for p in pos) >= mode.oi_max:
                continue
            notional = max(5 * MIN_ORDER, mode.oi_frac * eq)
            pnl = notional * R[i]
        elif kind == "coil":
            risk_usd = mode.coil_risk * eq
            notional = max(MIN_ORDER, risk_usd / K[i])
            pnl = notional * K[i] * R[i]
        elif kind == "unlock":
            notional = max(MIN_ORDER, mode.unl_frac * eq)
            pnl = notional * R[i]
        else:
            notional = max(MIN_ORDER, mode.ann_frac * eq)
            pnl = notional * R[i]
        if mode.exch_lev > 0:
            used = sum(p[1] for p in pos)
            if used + notional > mode.lev_cap * eq or \
                    (used + notional + reserve * eq) / mode.exch_lev > eq:
                continue
        else:
            used = sum(p[1] for p in pos if p[3] != "spikes")
            if used + notional > (mode.lev_cap - reserve) * eq:
                continue
        pos.append((E[i], notional, pnl, kind, sym))
        done += 1
    eq += sum(p[2] for p in pos)
    month_eq[cur] = eq
    return pd.Series(month_eq), eq < RUIN, done


def history(ev: pd.DataFrame, mode: Mode) -> dict:
    s, ruined, n = simulate(ev, mode)
    eq = pd.concat([pd.Series({"start": CAPITAL}), s])
    mret = eq.pct_change().dropna()
    months = len(mret)
    return {"итог $": round(eq.iloc[-1], 0), "сделок": n, "в мес. (CAGR)": f"{(eq.iloc[-1] / CAPITAL) ** (1 / months) - 1:+.1%}",
            "медиана мес.": f"{mret.median():+.1%}", "худший мес.": f"{mret.min():+.0%}",
            "мес. в минусе": f"{(mret < 0).mean():.0%}", "макс. просадка": f"{(eq / eq.cummax() - 1).min():.0%}",
            "разорён": ruined}


def monte_carlo(ev: pd.DataFrame, mode: Mode, paths: int, seed: int) -> dict:
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
        s, ruined, _ = simulate(pd.concat(parts, ignore_index=True).sort_values("t"), mode)
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
    ap.add_argument("--unlock", default="")
    ap.add_argument("--spikes-full", default="")
    ap.add_argument("--spikes-cfg", default="6σ прокол 0.2%")
    ap.add_argument("--start", default="")
    ap.add_argument("--drop-day", default="")
    ap.add_argument("--paths", type=int, default=300)
    a = ap.parse_args()
    ev_all = load_events(a)
    n_months = ev_all["t"].dt.tz_localize(None).dt.to_period("M").nunique()
    print(f"сделок {ev_all['t'].min():%Y-%m}…{ev_all['t'].max():%Y-%m} ({n_months} мес.): " + ", ".join(f"{k} {v}" for k, v in ev_all["kind"].value_counts().items()))
    for k, g in ev_all.groupby("kind"):
        x = g["r"] * (g["risk"] if k == "coil" else 1.0)
        print(f"  {k}: средняя сделка {x.mean():+.2%} номинала" + (" (R x риск)" if k == "coil" else "") +
              f", прибыльных {np.mean(g['r'] > 0):.0%}")
    sets = {k: [k] for k in ("oi_liq", "spikes", "coil", "announce", "unlock") if (ev_all["kind"] == k).any()}
    sets["ВСЕ ВМЕСТЕ"] = list(sets)
    for label, adj in (("КАК В БЭКТЕСТЕ", False), ("С ПОПРАВКОЙ −30% преимущества", True)):
        ev = haircut(ev_all) if adj else ev_all
        print(f"\n========== {label}, старт ${CAPITAL:.0f} ==========")
        for mode in MODES:
            hist, mc = [], []
            for name, kinds in sets.items():
                sub = ev[ev["kind"].isin(kinds)]
                hist.append({"стратегия": name, **history(sub, mode)})
                mc.append({"стратегия": name, **monte_carlo(sub, mode, a.paths, 7)})
            print(f"\n--- режим «{mode.name}»: история {n_months} месяцев ---")
            print(pd.DataFrame(hist).to_string(index=False))
            print(f"--- режим «{mode.name}»: Монте-Карло, 12 случайных месяцев × {a.paths} ---")
            print(pd.DataFrame(mc).to_string(index=False))
