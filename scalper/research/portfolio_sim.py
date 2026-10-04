"""
Моделирование счёта для двух стратегий бота (oi_liq с выносом уровня и лесенкой; ловля прострелов 6σ) на реальной
последовательности сделок 2023-01 … 2026-09, с реинвестированием и ограничениями Bybit.

Правила счёта:
- размер сделки — доля текущего капитала, но не меньше минимума (лесенка oi_liq — 5 лимитов, минимум $25;
  прострел — один лимит, минимум $5);
- oi_liq: не больше max_pos позиций одновременно, позиция живёт 4 ч после закрытия часового бара сигнала;
- прострелы: стоящие лимитки блокируют маржу, поэтому число наблюдаемых монет = резерв × капитал / ордер
  (пересчёт раз в месяц, фиксированный случайный порядок монет), позиция живёт до 60 мин;
- общий номинал открытых позиций <= lev_cap × капитал; иначе сигнал пропускается;
- капитал < $10 — счёт разорён (минимальные ордера уже не поставить).
Сценарий «с поправкой»: из каждой сделки вычитается 30% среднего преимущества стратегии (проскальзывание, отличия
Bybit). Монте-Карло: 12 случайных месяцев истории (с возвращением) × 2000 путей — распределение годового итога,
вероятность просадки >= 50% и разорения.

    python -m research.portfolio_sim --oi <oi_liq_trades.csv> --spikes <spike_trades.csv>
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

START, END = "2023-01-01", "2026-10-01"
RUIN = 10.0
HAIRCUT = 0.3


@dataclass(frozen=True)
class Mode:
    name: str
    oi_frac: float          # номинал лесенки oi_liq, доля капитала
    oi_min: float
    oi_max_pos: int
    sp_frac: float          # номинал одного лимита на прострел, доля капитала
    sp_min: float
    sp_reserve: float       # сколько капиталов можно держать в стоящих лимитках на прострелы
    lev_cap: float          # предел общего номинала открытых позиций, в капиталах


MODES = [Mode("умеренный", 0.20, 25.0, 3, 0.05, 5.0, 1.0, 2.0),
         Mode("агрессивный", 0.60, 25.0, 5, 0.10, 5.0, 3.0, 5.0)]


def events(oi: pd.DataFrame, sp: pd.DataFrame, haircut: bool) -> pd.DataFrame:
    a = pd.DataFrame({"t": oi["t"] + pd.Timedelta(hours=1), "end": oi["t"] + pd.Timedelta(hours=5),
                      "symbol": oi["symbol"], "r": oi["L5x0.5"], "kind": "oi"})
    b = pd.DataFrame({"t": sp["t"], "end": sp["t"] + pd.Timedelta(minutes=60), "symbol": sp["symbol"],
                      "r": sp["r"], "kind": "sp"})
    if haircut:
        a["r"] -= HAIRCUT * a["r"].mean()
        b["r"] -= HAIRCUT * b["r"].mean()
    e = pd.concat([a, b], ignore_index=True)
    return e[(e["t"] >= START) & (e["t"] < END)].sort_values("t").reset_index(drop=True)


def simulate(ev: pd.DataFrame, mode: Mode, capital: float, coin_order: list[str]) -> tuple[pd.Series, bool]:
    """Возвращает капитал на конец каждого месяца и флаг разорения."""
    eq = capital
    open_pos: list[tuple[pd.Timestamp, float, float, str]] = []      # (конец, номинал, доходность, вид)
    month_eq, watched, cur_month = {}, set(), None
    t_arr, end_arr, sym_arr, r_arr, kind_arr = (ev[c].to_numpy() for c in ("t", "end", "symbol", "r", "kind"))
    for i in range(len(ev)):
        t = t_arr[i]
        still = []
        for p in open_pos:                                            # закрываем позиции, завершившиеся до t
            if p[0] <= t:
                eq += p[1] * p[2]
            else:
                still.append(p)
        open_pos = still
        m = pd.Timestamp(t).strftime("%Y-%m")
        if m != cur_month:
            if cur_month is not None:
                month_eq[cur_month] = eq
            cur_month = m
            order = max(mode.sp_min, mode.sp_frac * eq)
            k = int(mode.sp_reserve * eq // order)
            watched = set(coin_order[:k])
        if eq < RUIN:
            month_eq[cur_month] = eq
            return pd.Series(month_eq), True
        used = sum(p[1] for p in open_pos)
        if kind_arr[i] == "oi":
            if sum(1 for p in open_pos if p[3] == "oi") >= mode.oi_max_pos:
                continue
            notional = max(mode.oi_min, mode.oi_frac * eq)
        else:
            if sym_arr[i] not in watched or any(p[3] == "sp:" + sym_arr[i] for p in open_pos):
                continue
            notional = max(mode.sp_min, mode.sp_frac * eq)
        if used + notional > mode.lev_cap * eq:
            continue
        open_pos.append((end_arr[i], notional, r_arr[i], "oi" if kind_arr[i] == "oi" else "sp:" + sym_arr[i]))
    eq += sum(p[1] * p[2] for p in open_pos)
    month_eq[cur_month] = eq
    return pd.Series(month_eq), eq < RUIN


def history_report(ev: pd.DataFrame, mode: Mode, capital: float, coins: list[str]) -> dict:
    s, ruined = simulate(ev, mode, capital, coins)
    eq = pd.concat([pd.Series({"start": capital}), s])
    mret = eq.pct_change().dropna()
    dd = (eq / eq.cummax() - 1).min()
    return {"итог $": round(eq.iloc[-1], 1), "медиана мес.": f"{mret.median():+.1%}",
            "средн. мес.": f"{mret.mean():+.1%}", "лучший мес.": f"{mret.max():+.1%}", "худший мес.": f"{mret.min():+.1%}",
            "мес. в минусе": f"{(mret < 0).mean():.0%}", "макс. просадка": f"{dd:.0%}", "разорён": ruined}


def monte_carlo(ev: pd.DataFrame, mode: Mode, capital: float, coins: list[str], paths: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    ev = ev.assign(month=ev["t"].dt.to_period("M"))
    months = sorted(ev["month"].unique())
    groups = {m: g for m, g in ev.groupby("month")}
    finals, dd50, ruins = [], 0, 0
    for _ in range(paths):
        parts = []
        for j, m in enumerate(rng.choice(len(months), size=12, replace=True)):
            g = groups[months[m]]
            shift = pd.Timestamp("2000-01-01", tz="UTC") + pd.DateOffset(months=j) - g["month"].iloc[0].start_time.tz_localize("UTC")
            parts.append(g.assign(t=g["t"] + shift, end=g["end"] + shift))
        path = pd.concat(parts, ignore_index=True).sort_values("t")
        s, ruined = simulate(path, mode, capital, coins)
        eq = pd.concat([pd.Series({"start": capital}), s])
        finals.append(eq.iloc[-1] / capital)
        dd50 += (eq / eq.cummax() - 1).min() <= -0.5
        ruins += ruined
    f = np.array(finals)
    return {"медиана за год": f"×{np.median(f):.2f}", "10% худших": f"×{np.quantile(f, 0.1):.2f}",
            "10% лучших": f"×{np.quantile(f, 0.9):.2f}", "≈ в мес. (медиана)": f"{np.median(f) ** (1 / 12) - 1:+.1%}",
            "P(просадка ≥50%)": f"{dd50 / paths:.0%}", "P(разорение)": f"{ruins / paths:.0%}"}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 250)
    ap = argparse.ArgumentParser()
    ap.add_argument("--oi", required=True)
    ap.add_argument("--spikes", required=True)
    ap.add_argument("--paths", type=int, default=500)
    a = ap.parse_args()
    oi = pd.read_csv(a.oi, parse_dates=["t"])
    sp = pd.read_csv(a.spikes, parse_dates=["t"])
    coins = list(np.random.default_rng(1).permutation(sorted(sp["symbol"].unique())))
    for haircut in (False, True):
        ev = events(oi, sp, haircut)
        title = "С ПОПРАВКОЙ (−30% преимущества)" if haircut else "КАК В БЭКТЕСТЕ"
        print(f"\n========== {title}: сделок oi_liq {int((ev.kind == 'oi').sum())}, прострелов "
              f"{int((ev.kind == 'sp').sum())} за {START[:7]} … 2026-09 ==========")
        for capital in (50.0, 1000.0):
            rows = []
            for mode in MODES:
                rows.append({"режим": mode.name, **history_report(ev, mode, capital, coins)})
            print(f"\nИстория, старт ${capital:g}:")
            print(pd.DataFrame(rows).to_string(index=False))
            rows = []
            for mode in MODES:
                rows.append({"режим": mode.name, **monte_carlo(ev, mode, capital, coins, a.paths, 7)})
            print(f"Монте-Карло (12 случайных месяцев × {a.paths}), старт ${capital:g}:")
            print(pd.DataFrame(rows).to_string(index=False))
