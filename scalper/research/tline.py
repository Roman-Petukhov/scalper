"""
Линия тренда по двум точкам на закрытиях свечей: пробой с закреплением, вход по рынку или на ретесте линии,
стоп за ближайшим структурным минимумом / максимумом, выход лесенкой (половина на 3R, половина на 5R).

Таймфреймы: 15m (70 ликвидных монет: core16 + ext54), 1h и 4h (725 монет, 4h — из часовых свечей).
Определения (всё известно на закрытии бара):
    точки линии   локальные экстремумы ЗАКРЫТИЙ (фрактал n = 5 баров, подтверждается через 5 баров); линия для лонга —
                  через две последние подтверждённые вершины закрытий, вторая ниже первой (нисходящая);
                  для шорта — через две последние впадины, вторая выше первой (восходящая)
    пробой        закрытие за линией впервые; «закрепление» — следующая свеча тоже закрылась за линией
    вход          «рынок» — по закрытию свечи сигнала (пробоя или закрепления); «ретест» — лимитка на значении линии
                  в момент сигнала, живёт 12 баров; если цена дошла до первой цели раньше — сделки нет
    стоп          за последним подтверждённым свинг-минимумом (для лонга; фрактал 5 баров по теням) − 0.1 ATR;
                  сделка берётся, если расстояние до стопа от 0.3 до 4 ATR
    выход         половина на 3R, половина на 5R; вариант «БУ» — после первой цели стоп в точку входа; не дольше
                  200 (15m) / 120 (1h) / 60 (4h) баров
    сила пробоя   свеча пробоя: тело >= 60% диапазона, закрытие в крайней четверти, пробой линии >= 0.3 ATR;
                  доля рыночных покупок (для шорта — продаж) >= 55%; и всё вместе с объёмом
    фильтры       объём свечи пробоя >= 1.5 x среднего за 20 свечей; OI вырос за 4 свечи до пробоя (1h / 4h);
                  вариант «по тренду старшего ТФ»: close старшего ТФ (1h для 15m, 4h для 1h, 1d для 4h) выше EMA50
                  для лонга (ниже — для шорта), по последней закрытой свече старшего ТФ
Издержки: вход по рынку 5.5 б.п. (ретест — maker 2), тейки maker 2, стоп / таймаут taker 5.5, funding.
Оборот за 30 дней >= $20M (для 15m — монеты из 70 ликвидных без фильтра).

    python -m research.tline collect --symbols <725 монет> --symbols15 <70 монет>   (по частям)
    python -m research.tline report
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

from .broad import ADV_MIN, adv30, group_of
from .engine import funding_on_bars, resample
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .wave2 import metrics_on_bars

PER = {"is": ("2022-01-01", "2024-07-01"), "val": ("2024-07-01", "2025-07-01"), "ho": ("2025-07-01", "2026-10-01")}
PIV = 5
MAKER, TAKER = 2e-4, 5.5e-4
HOLD = {"15m": 200, "1h": 120, "4h": 60}
HTF = {"15m": "1h", "1h": "4h", "4h": "1D"}
BAR_MIN = {"15m": 15, "1h": 60, "4h": 240}
RETEST_BARS = 12


@njit(cache=True)
def two_targets(o, h, lo, c, fund, e, side, entry, stop, k1, k2, be, max_hold, entry_fee):
    """Позиция открыта на закрытии бара e по цене entry. Половина на k1·R, половина на k2·R; be — стоп в безубыток
    после первой цели. Стоп проверяется раньше тейка в том же баре. Результат в R."""
    n = len(c)
    risk = side * (entry - stop)
    if not (risk > 0):
        return np.nan, e
    tp1, tp2 = entry + side * k1 * risk, entry + side * k2 * risk
    left = 1.0
    r = 0.0
    paid = 0.0
    sp = stop
    for j in range(e + 1, n):
        paid += fund[j] * left
        if (side > 0 and lo[j] <= sp) or (side < 0 and h[j] >= sp):
            ex = min(sp, o[j]) if side > 0 else max(sp, o[j])
            r += left * (side * (ex - entry) - (entry_fee + TAKER) * entry) / risk
            return r - side * paid * entry / risk, j
        if left == 1.0 and ((side > 0 and h[j] >= tp1) or (side < 0 and lo[j] <= tp1)):
            r += 0.5 * (side * (tp1 - entry) - (entry_fee + MAKER) * entry) / risk
            left = 0.5
            if be:
                sp = entry
        if left == 0.5 and ((side > 0 and h[j] >= tp2) or (side < 0 and lo[j] <= tp2)):
            r += 0.5 * (side * (tp2 - entry) - (entry_fee + MAKER) * entry) / risk
            return r - side * paid * entry / risk, j
        if j - e >= max_hold or j == n - 1:
            r += left * (side * (c[j] - entry) - (entry_fee + TAKER) * entry) / risk
            return r - side * paid * entry / risk, j
    return np.nan, n - 1


@njit(cache=True)
def retest_fill(h, lo, e, side, level, tp1_dist, valid):
    """Бар исполнения лимитки на линии (−1 — не исполнилась или цена ушла к первой цели раньше)."""
    n = len(h)
    tp1 = level + side * tp1_dist
    for j in range(e + 1, min(e + valid, n - 1) + 1):
        if (side > 0 and lo[j] < level) or (side < 0 and h[j] > level):
            return j
        if (side > 0 and h[j] >= tp1) or (side < 0 and lo[j] <= tp1):
            return -1
    return -1


def pivots(x: np.ndarray, n: int, high: bool) -> np.ndarray:
    """Индексы экстремумов x (фрактал n) и бар их подтверждения: возвращает массив (индекс, бар подтверждения)."""
    out = []
    for p in range(n, len(x) - n):
        w = x[p - n: p + n + 1]
        if (high and x[p] == w.max()) or (not high and x[p] == w.min()):
            out.append((p, p + n))
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def last_confirmed(piv: np.ndarray, m: int) -> np.ndarray:
    """Для каждого бара — индекс последнего подтверждённого экстремума (−1 — нет)."""
    out = np.full(m, -1, np.int64)
    for p, conf in piv:
        if conf < m:
            out[conf:] = p
    return out


def signals(d: pd.DataFrame) -> list[tuple[int, int, int, float, float]]:
    """(бар пробоя, бар закрепления или −1, сторона, линия на баре пробоя, линия на баре закрепления)."""
    c = d["close"].to_numpy(dtype="float64")
    m = len(c)
    out = []
    for side in (1, -1):
        piv = pivots(c, PIV, high=side > 0)
        used = set()
        k = 0
        for t in range(PIV * 3, m - 1):
            while k + 1 < len(piv) and piv[k + 1][1] <= t - 1:
                k += 1
            if k < 1 or piv[k][1] > t - 1:
                continue
            (i1, _), (i2, _) = piv[k - 1], piv[k]
            if (i1, i2) in used or i2 <= i1:
                continue
            c1, c2 = c[i1], c[i2]
            if not (side * (c1 - c2) > 0):                    # для лонга вторая вершина ниже первой
                continue
            slope = (c2 - c1) / (i2 - i1)
            line_t, line_p = c2 + slope * (t - i2), c2 + slope * (t - 1 - i2)
            if side * (c[t] - line_t) > 0 and side * (c[t - 1] - line_p) <= 0:
                used.add((i1, i2))
                line_n = c2 + slope * (t + 1 - i2)
                out.append((t, t + 1 if side * (c[t + 1] - line_n) > 0 else -1, side, line_t, line_n))
    return out


def tf_frame(root: Path, sym: str, tf: str) -> pd.DataFrame | None:
    p15, p1 = root / f"{sym}-15m.parquet", root / f"{sym}-1h.parquet"
    src = p15 if tf == "15m" else p1
    if not src.exists():
        return None
    df = pd.read_parquet(src)
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop(columns=["open_time"])
    df = df[~df.index.duplicated()].sort_index()
    if tf == "4h":
        df = resample(df, "4h")
    df["funding"] = funding_on_bars(sym, root, df.index, BAR_MIN[tf])
    return df


def htf_trend(d: pd.DataFrame, tf: str) -> np.ndarray:
    """+1 / −1: close последней закрытой свечи старшего ТФ выше / ниже её EMA50 (известно на закрытии бара)."""
    rule = HTF[tf]
    agg = d[["open", "high", "low", "close"]].resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    st = np.sign(agg["close"] - agg["close"].ewm(span=50, adjust=False).mean())
    st.index = st.index + pd.tseries.frequencies.to_offset(rule)            # известно после закрытия
    bar_close = d.index + pd.Timedelta(minutes=BAR_MIN[tf])
    return st.reindex(bar_close, method="ffill").to_numpy()


def coin_trades(root: Path, sym: str, tf: str) -> pd.DataFrame:
    d = tf_frame(root, sym, tf)
    if d is None or len(d) < 500:
        return pd.DataFrame()
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64")
    a = _atr(d).to_numpy()
    sw_lo, sw_hi = last_confirmed(pivots(lo, PIV, False), len(c)), last_confirmed(pivots(hi, PIV, True), len(c))
    trend = htf_trend(d, tf)
    ok = np.ones(len(c), bool)
    if tf != "15m":
        adv = adv30(root, sym)
        ok = adv.reindex(d.index.floor("D")).to_numpy() >= ADV_MIN
    rows = []
    v = d["volume"]
    vol_ratio = (v / v.shift(1).rolling(20).mean()).to_numpy()
    rng = (d["high"] - d["low"]).to_numpy()
    body = np.abs(c - o) / np.where(rng > 0, rng, np.nan)                     # доля тела в свече
    loc = (c - lo) / np.where(rng > 0, rng, np.nan)                         # где закрылась: 1 — у максимума
    buy_share = (d["taker_buy_volume"] / v.replace(0, np.nan)).to_numpy()
    oi_chg = np.full(len(c), np.nan)
    if tf != "15m":
        m = metrics_on_bars(sym, root, d.index, BAR_MIN[tf])
        oi = np.log(m["oi"].replace(0, np.nan))
        oi_chg = (oi - oi.shift(4)).to_numpy()
    for tb, tc, side, line_b, line_c in signals(d):
        for confirm in (False, True):
            e = tc if confirm else tb
            if e < 0 or e >= len(c) - 1 or not ok[e] or not (a[e] > 0):
                continue
            sw = sw_lo[e] if side > 0 else sw_hi[e]
            if sw < 0:
                continue
            stop = (lo[sw] - 0.1 * a[e]) if side > 0 else (hi[sw] + 0.1 * a[e])
            for entry_kind in ("market", "retest"):
                if entry_kind == "market":
                    fill, px, fee = e, c[e], TAKER
                else:
                    # ретест: лимитка на значении линии на баре сигнала
                    level = line_c if confirm else line_b
                    dist = side * (level - stop)
                    if not (dist > 0):
                        continue
                    fill = retest_fill(hi, lo, e, side, level, 3 * dist, RETEST_BARS)
                    if fill < 0:
                        continue
                    px, fee = level, MAKER
                dist_atr = side * (px - stop) / a[e]
                if not (0.3 <= dist_atr <= 4.0):
                    continue
                for be in (False, True):
                    r, ex = two_targets(o, hi, lo, c, f, fill, side, px, stop, 3.0, 5.0, be, HOLD[tf], fee)
                    rows.append({"symbol": sym, "tf": tf, "t": d.index[e], "side": side, "confirm": confirm,
                                 "entry": entry_kind, "be": be, "with_trend": trend[e] == side, "R": r,
                                 "vol_ratio": vol_ratio[tb], "oi_chg": oi_chg[tb], "body": body[tb],
                                 "close_loc": loc[tb] if side > 0 else 1 - loc[tb],
                                 "aggr": buy_share[tb] if side > 0 else 1 - buy_share[tb],
                                 "brk_atr": side * (c[tb] - line_b) / a[tb],
                                 "stop_atr": dist_atr})
    return pd.DataFrame(rows)


def collect(root: Path, syms: list[str], syms15: list[str]) -> None:
    parts = []
    for tf, lst in (("15m", syms15), ("1h", syms), ("4h", syms)):
        for s in mine(lst):
            try:
                x = coin_trades(root, s, tf)
                if len(x):
                    parts.append(x)
            except Exception as e:
                print(f"  {tf} {s}: пропуск ({e})", flush=True)
    print(f"  частей монет-ТФ со сделками: {len(parts)}", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path("tline"), index=False)


def report() -> None:
    parts = all_parts("tline")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["group"] = [group_of(s) for s in df["symbol"]]
    df["per"] = ""
    for p, (a, b) in PER.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    print(f"===== TLINE: сделок {len(df):,}, монет {df.symbol.nunique()}, частей {len(parts)} =====")
    print("ячейка: средний R на сделку (t по дням, прибыльных, сделок в месяц на весь набор монет); выход 1/2 на 3R + 1/2 на 5R")
    rows = []
    for (tf, confirm, entry, be), g in df.groupby(["tf", "confirm", "entry", "be"]):
        for trend_name, gg in (("все", g), ("по тренду старшего ТФ", g[g.with_trend]),
                               ("объём пробоя >= 1.5x", g[g.vol_ratio >= 1.5]), ("OI рос 4 бара", g[g.oi_chg > 0]),
                               ("сильная свеча (тело >= 60%, закрытие у края, пробой >= 0.3 ATR)",
                                g[(g.body >= 0.6) & (g.close_loc >= 0.75) & (g.brk_atr >= 0.3)]),
                               ("агрессоры в сторону пробоя >= 55%", g[g.aggr >= 0.55]),
                               ("сила: свеча + объём + агрессоры", g[(g.body >= 0.6) & (g.close_loc >= 0.75) &
                                                                     (g.brk_atr >= 0.3) & (g.vol_ratio >= 1.5) &
                                                                     (g.aggr >= 0.55)]),
                               ("объём + OI + тренд", g[(g.vol_ratio >= 1.5) & (g.oi_chg > 0) & g.with_trend])):
            if tf == "15m" and "OI" in trend_name:
                continue
            rows.append({"ТФ": tf, "вход": ("закрепление, " if confirm else "пробой, ") +
                         ("рынок" if entry == "market" else "ретест"), "БУ": "да" if be else "нет",
                         "фильтр": trend_name, **{p: _cell(gg[gg.per == p]) for p in PER},
                         "вне подбора, VAL+HO": _cell(gg[(gg.per != "is") & gg.group.isin(["ext54", "fresh"])])})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nпо сторонам (закрепление, ретест, без БУ, все):")
    x = df[df.confirm & (df.entry == "retest") & ~df.be]
    for (tf, sd), g in x.groupby(["tf", "side"]):
        print(f"  {tf} {'лонг' if sd > 0 else 'шорт'}: " + " | ".join(f"{p} {_cell(g[g.per == p])}" for p in PER))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 340)
    pd.set_option("display.max_columns", 30)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--symbols15", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x],
                [x for x in a.symbols15.split(",") if x])
    else:
        report()
