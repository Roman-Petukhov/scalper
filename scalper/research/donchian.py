"""Тренд каналами вместо линий: ансамбль Дончиана по Zarattini, Pagani & Barbon (2025, «Catching Crypto Trends»).

Для каждого окна N (дней): вход, когда закрытие выше максимума закрытий за N прошлых свечей (шорт — ниже минимума);
выход по закрытию за подтягивающимся стопом — середина канала (между максимумом и минимумом закрытий за N), стоп только
подтягивается. Окна 5, 10, 20, 30, 60, 90, 150, 250, 360 дней; ансамбль — все окна сразу, каждое на 1/9 риска.
Результат сделки — в R, где R — расстояние от входа до стопа на входе, как у бота: комиссия taker на входе и выходе,
funding за время в позиции. Свечи 1d и 4h (окна те же в днях), монеты и порог оборота — как в tline (от $20M в день).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from .broad import ADV_MIN, adv30
from .shard import all_parts, mine, part_path
from .smc import _cell
from .tline import BAR_MIN, PER, TAKER, _per_month, _years, market_context, tf_frame

DAYS = (5, 10, 20, 30, 60, 90, 150, 250, 360)
TFS = ("1d", "4h")


@njit(cache=True)
def _trades(c, fund, hi_n, lo_n, ok, side, fee):
    """Сделки одного окна: (бар входа, бар выхода, R). hi_n / lo_n — максимум / минимум закрытий за N прошлых свечей
    (без текущей), известны на закрытии."""
    n = len(c)
    out = []
    t = 1
    while t < n - 1:
        brk = c[t] > hi_n[t] if side > 0 else c[t] < lo_n[t]
        if not (brk and ok[t]) or np.isnan(hi_n[t]):
            t += 1
            continue
        entry = c[t]
        stop = 0.5 * (hi_n[t] + lo_n[t])
        risk = side * (entry - stop)
        if not (risk > 0):
            t += 1
            continue
        paid = 0.0
        x = n - 1
        for j in range(t + 1, n):
            paid += fund[j]
            mid = 0.5 * (hi_n[j] + lo_n[j])
            if not np.isnan(mid):
                stop = max(stop, mid) if side > 0 else min(stop, mid)
            if side * (c[j] - stop) < 0:
                x = j
                break
        r = (side * (c[x] - entry) - 2 * fee * entry - side * paid * entry) / risk
        out.append((t, x, r))
        t = x + 1
    return out


def coin_trades(root: Path, sym: str, tf: str, ctx: pd.DataFrame | None) -> pd.DataFrame:
    d = tf_frame(root, sym, tf)
    if d is None or len(d) < 200:
        return pd.DataFrame()
    c = d["close"].to_numpy(dtype="float64")
    f = d["funding"].to_numpy(dtype="float64")
    liq = adv30(root, sym).reindex(d.index.floor("D")).to_numpy()
    ok = liq >= ADV_MIN
    per_day = 24 * 60 // BAR_MIN[tf]
    close_t = d.index + pd.Timedelta(minutes=BAR_MIN[tf])
    btc = ctx.reindex(close_t, method="ffill")["btc_trend"].to_numpy() if ctx is not None else np.full(len(c), np.nan)
    rows = []
    cs = d["close"]
    for days in DAYS:
        w = days * per_day
        hi_n = cs.shift(1).rolling(w, min_periods=w).max().to_numpy()
        lo_n = cs.shift(1).rolling(w, min_periods=w).min().to_numpy()
        for side in (1, -1):
            for t, x, r in _trades(c, f, hi_n, lo_n, ok, side, TAKER):
                rows.append({"symbol": sym, "tf": tf, "days": days, "side": side, "t": d.index[t], "R": r,
                             "ret": side * (c[x] / c[t] - 1), "hold_d": (x - t) / per_day, "btc_trend": btc[t]})
    return pd.DataFrame(rows)


def collect(root: Path, syms: list[str]) -> None:
    ctx = market_context(root)
    parts = []
    for tf in TFS:
        for s in mine(syms):
            try:
                x = coin_trades(root, s, tf, ctx)
                if len(x):
                    parts.append(x)
            except Exception as e:
                print(f"  donchian {tf} {s}: пропуск ({e})", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path("donchian"), index=False)


def report() -> None:
    parts = all_parts("donchian")
    print("\n===== DONCHIAN: тренд каналами (Zarattini et al. 2025) — вход за максимумом / минимумом закрытий за N дней, "
          "выход за серединой канала; ячейка — R на сделку (t, прибыльных, сделок в месяц); R/мес — сумма R =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["per"] = ""
    for p, (a, b) in PER.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    print(f"сделок {len(df):,}, монет {df.symbol.nunique()}; данные с 2022 г.: длинным окнам (250–360 дней) нужен год "
          f"истории, в IS их сделок мало")

    def table(items: list[tuple[str, pd.DataFrame]], weight: float = 1.0) -> None:
        rows = []
        for nm, z in items:
            z = z.assign(R3=z["R"] * weight)
            rows.append({"вариант": nm, **{p: _cell(z[z.per == p]) for p in PER},
                         "R/мес IS / VAL / HO": _per_month(z, "R3"),
                         "держим, дней (медиана)": f"{z.hold_d.median():.0f}" if len(z) else "-"})
        print(pd.DataFrame(rows).to_string(index=False))

    for tf in TFS:
        g = df[df.tf == tf]
        if g.empty:
            continue
        for sd, nm in ((1, "лонги"), (-1, "шорты")):
            x = g[g.side == sd]
            print(f"\n  --- {tf}, {nm}: каждое окно отдельно, 1R на сделку ---")
            table([(f"{d_} дней", x[x.days == d_]) for d_ in DAYS])
            print(f"  ансамбль (все окна, каждое на 1/{len(DAYS)} риска), по годам: "
                  f"{_years(x.assign(R3=x['R'] / len(DAYS)), 'R3')}")
        print(f"\n  --- {tf}: ансамбль всех окон, каждое окно на 1/{len(DAYS)} риска ---")
        table([("лонги", g[g.side == 1]), ("лонги, BTC выше EMA50 дневок", g[(g.side == 1) & (g.btc_trend == 1)]),
               ("шорты", g[g.side == -1]), ("шорты, BTC ниже EMA50 дневок", g[(g.side == -1) & (g.btc_trend == -1)]),
               ("лонги + шорты", g),
               ("короткие окна 5–30 дней, обе стороны", g[g.days <= 30]),
               ("длинные окна 60–360 дней, обе стороны", g[g.days >= 60])], weight=1 / len(DAYS))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 300)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x])
    else:
        report()
