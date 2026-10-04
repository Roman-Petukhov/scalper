"""
Пробои с «подтягиванием стопа»: входим во все пробои, ложные режем стопом, трендовые держим, подтягивая стоп.

Входы (без перерисовки):
    lux14 / lux28   пробой линии LuxAlgo Trendlines with Breaks (research.lux_rr.lux_breaks), Slope 1, Atr
    donch20/donch55 close выше максимума (ниже минимума) прошлых N баров
Таймфреймы: 15m (70 монет core16 + ext54), 1h и 4h (725 монет). Вход — по close бара пробоя (тейкер).
Начальный стоп 1.5 ATR(14). Выходы (правила заданы до запуска):
    tp5      тейк 5R (лимит), иначе — по времени (база для сравнения)
    trail3   стоп тянется на 3 ATR за лучшей ценой закрытия с момента входа
    trail6   то же на 6 ATR
    struct   после +1R стоп — под минимум (над максимумом) последних 10 баров, только в сторону прибыли
    hybrid   половина позиции — тейк на +2R, после этого стоп остатка в безубыток и трейлинг 3 ATR
Не дольше 10 дней. Одна позиция на монету. Комиссии Bybit: тейкер 5.5 б.п. (вход, стопы), мейкер 2 б.п. (тейки).
Монеты: оборот за 30 прошлых дней >= $20M. IS -> VAL -> HOLDOUT, t — кластеризованный по дням.

    python -m research.trend_trail collect --symbols15 ... --symbols1h ...   (по частям)
    python -m research.trend_trail report
"""
from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from .broad import ADV_MIN, adv30
from .lux import slope_series
from .lux_rr import PER, _stat, load, lux_breaks
from .shard import all_parts, mine, part_path

ENTRIES = ("lux14", "lux28", "donch20", "donch55")
EXITS = ("tp5", "trail3", "trail6", "struct", "hybrid")
STOP_K = 1.5
TAKER, MAKER = 5.5e-4, 2e-4
BARS_PER_DAY = {"15m": 96, "1h": 24, "4h": 6}


@njit(cache=True)
def run_exits(side, allow, o, h, l, c, atr, mode, max_hold, stop_k, taker, maker):
    """mode: 0 tp5, 1 trail3, 2 trail6, 3 struct, 4 hybrid. Возвращает индекс входа, R после комиссий, баров."""
    m = len(c)
    oi = np.empty(m, np.int64)
    orr = np.empty(m)
    ob = np.empty(m, np.int64)
    k = 0
    busy = -1
    for t in range(m):
        s = side[t]
        if s == 0 or not allow[t] or t <= busy or t + 2 >= m or not (atr[t] > 1e-4 * c[t]):
            continue
        entry = c[t]
        risk = max(stop_k * atr[t], 1e-3 * entry)
        stop = entry - s * risk
        best = entry
        half_done = False
        realized = 0.0                       # R, уже зафиксированный половиной (hybrid)
        size = 1.0
        fees = taker                         # доля цены, комиссия входа
        exit_px = np.nan
        exit_fee = taker
        end = min(t + max_hold, m - 1)
        j = t + 1
        while j <= end:
            # стоп (проверяется первым — худший порядок внутри бара)
            if (s > 0 and l[j] <= stop) or (s < 0 and h[j] >= stop):
                exit_px = min(stop, o[j]) if s > 0 else max(stop, o[j])
                exit_fee = taker
                break
            if mode == 0:
                tp = entry + s * 5.0 * risk
                if (s > 0 and h[j] >= tp) or (s < 0 and l[j] <= tp):
                    exit_px, exit_fee = tp, maker
                    break
            if mode == 4 and not half_done:
                tp2 = entry + s * 2.0 * risk
                if (s > 0 and h[j] >= tp2) or (s < 0 and l[j] <= tp2):
                    half_done = True
                    realized = 0.5 * 2.0
                    fees += 0.5 * maker
                    size = 0.5
                    stop = entry if s > 0 and entry > stop else (entry if s < 0 and entry < stop else stop)
            # обновление стопа по закрытию бара
            if (s > 0 and c[j] > best) or (s < 0 and c[j] < best):
                best = c[j]
            if mode == 1 or mode == 2 or (mode == 4 and half_done):
                dist = (3.0 if mode != 2 else 6.0) * atr[j]
                ns = best - s * dist
                if (s > 0 and ns > stop) or (s < 0 and ns < stop):
                    stop = ns
            elif mode == 3 and s * (c[j] - entry) >= risk:
                lo_n = l[j]
                hi_n = h[j]
                for q in range(max(0, j - 9), j + 1):
                    lo_n = min(lo_n, l[q])
                    hi_n = max(hi_n, h[q])
                ns = lo_n if s > 0 else hi_n
                if (s > 0 and ns > stop) or (s < 0 and ns < stop):
                    stop = ns
            j += 1
        if np.isnan(exit_px):
            j = end
            exit_px = c[end]
            exit_fee = taker
        fees += size * exit_fee
        r = realized + size * s * (exit_px - entry) / risk - fees * entry / risk
        oi[k], orr[k], ob[k] = t, r, j - t
        k += 1
        busy = j
    return oi[:k], orr[:k], ob[:k]


def donchian(h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> np.ndarray:
    hi, lo = h.shift(1).rolling(n).max(), l.shift(1).rolling(n).min()
    up = (c > hi) & (c.shift(1) <= hi.shift(1))
    dn = (c < lo) & (c.shift(1) >= lo.shift(1))
    side = np.zeros(len(c), np.int8)
    side[up.to_numpy()] = 1
    side[dn.to_numpy() & ~up.to_numpy()] = -1
    return side


def bars(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    if tf != "4h":
        return df
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df.resample("4h", label="left", closed="left").agg(agg).dropna(subset=["close"])


def run_symbol(root: Path, sym: str, tf: str) -> list[pd.DataFrame]:
    src = load(root, sym, "15m" if tf == "15m" else "1h")
    if src is None or len(src) < 500:
        return []
    df = bars(src, tf)
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().to_numpy()
    allow = (adv30(root, sym).reindex(df.index, method="ffill") >= ADV_MIN).to_numpy()
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    out = []
    for entry in ENTRIES:
        if entry.startswith("lux"):
            L = int(entry[3:])
            side, _, _ = lux_breaks(h, l, c, slope_series(df, L, 1.0, "atr"), L)
        else:
            side = donchian(df["high"], df["low"], df["close"], int(entry[5:]))
        for mode, ex in enumerate(EXITS):
            i, r, b = run_exits(side, allow, o, h, l, c, atr, mode, 10 * BARS_PER_DAY[tf], STOP_K, TAKER, MAKER)
            if len(i):
                out.append(pd.DataFrame({"tf": tf, "entry": entry, "exit": ex, "t": df.index[i], "R": r,
                                         "bars": b, "symbol": sym}))
    return out


def report() -> None:
    parts = all_parts("trail")
    if not parts:
        print("частей нет")
        return
    tr = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    print(f"===== TREND TRAIL: частей {len(parts)}, сделок {len(tr):,}, монет {tr.symbol.nunique()} =====")
    rows = []
    for (tf, en, ex), g in tr.groupby(["tf", "entry", "exit"]):
        row = {"tf": tf, "вход": en, "выход": ex}
        for p, (a, b) in PER.items():
            st = _stat(g[(g.t >= a) & (g.t < b)])
            row.update({f"{p}_{k}": v for k, v in st.items() if k != "win"})
        row["медиана баров"] = int(g["bars"].median())
        row["доля >= 3R"] = float((g["R"] >= 3).mean())
        rows.append(row)
    res = pd.DataFrame(rows)
    for tf in ("15m", "1h", "4h"):
        x = res[res.tf == tf].sort_values(["вход", "выход"])
        print(f"\n=== {tf}: средний R на сделку после комиссий (t по дням) ===")
        print(x.drop(columns="tf").round(3).to_string(index=False))
    for col in ("is_avgR", "val_avgR", "ho_avgR"):
        if col not in res:
            res[col] = np.nan
    print("\nСравнение выходов (медиана среднего R по всем входам и таймфреймам), IS / VAL / HOLDOUT:")
    print(res.groupby("выход")[["is_avgR", "val_avgR", "ho_avgR"]].median().round(3).to_string())
    for col in ("is_avgR", "val_avgR", "ho_avgR", "is_t_day", "val_t_day", "ho_t_day", "ho_n"):
        if col not in res:
            res[col] = np.nan
    gate = res[(res["is_t_day"] > 2) & (res["val_avgR"] > 0) & (res["val_t_day"] > 1.5)]
    print("\nВОРОТА: IS t > 2 и VAL R > 0, t > 1.5 -> HOLDOUT:")
    print(gate[["tf", "вход", "выход", "ho_n", "ho_avgR", "ho_t_day"]].round(3).to_string(index=False)
          if len(gate) else "  никто не прошёл")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 40)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols15", default="")
    ap.add_argument("--symbols1h", default="")
    a = ap.parse_args()
    root = Path(a.root).expanduser()
    if a.mode == "collect":
        out = []
        for tf, syms in (("15m", a.symbols15), ("1h", a.symbols1h), ("4h", a.symbols1h)):
            for s in mine([x for x in syms.split(",") if x]):
                try:
                    out += run_symbol(root, s, tf)
                except Exception as e:
                    print(f"  {s} {tf}: пропуск ({e})", flush=True)
            print(f"  {tf}: готово", flush=True)
        if out:
            pd.concat(out, ignore_index=True).to_parquet(part_path("trail"), index=False)
    else:
        report()
