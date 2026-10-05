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
import os
import shutil
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


LINES = ("last2", "clean", "clean3", "major", "zz")
MAJOR_L = 12          # главный экстремум: тень выше (ниже) 12 свечей с каждой стороны
MINOR_N = 3           # точки касания: фрактал 3 свечи
ZZ_K = 3.0            # зигзаг по закрытиям: разворот >= 3 ATR
ZZ_ANCHOR = 60        # опорная вершина — самое высокое закрытие за 60 свечей до неё
ZZ_SPAN = 20          # между точками линии не меньше 20 свечей
ZZ_LIFE = 300         # линия живёт не дольше 300 свечей от опорной вершины


def _lines(c: np.ndarray, atr: np.ndarray, piv: np.ndarray, side: int, mode: str) -> list[tuple | None]:
    """Для каждой подтверждённой вершины k — линия (i1, i2, наклон, касаний) или None.
    last2  — через две последние вершины, вторая ниже первой (для лонга);
    clean  — от самой дальней из 6 предыдущих вершин, при которой ни одно закрытие между точками не выходит за линию;
    clean3 — clean, и на линии не меньше 3 вершин (в пределах 0.25 ATR)."""
    out: list[tuple | None] = [None] * len(piv)
    for k in range(1, len(piv)):
        i2 = piv[k][0]
        cands = [k - 1] if mode == "last2" else range(max(0, k - 6), k)
        for j in cands:
            i1 = piv[j][0]
            if not (side * (c[i1] - c[i2]) > 0) or i2 <= i1:
                continue
            slope = (c[i2] - c[i1]) / (i2 - i1)
            if mode != "last2":
                seg = np.arange(i1 + 1, i2)
                if len(seg) and np.any(side * (c[seg] - (c[i1] + slope * (seg - i1))) > 1e-12):
                    continue
            touches = 0
            for q in range(j, k + 1):
                p = piv[q][0]
                if abs(c[p] - (c[i1] + slope * (p - i1))) <= 0.25 * (atr[p] if atr[p] > 0 else np.inf):
                    touches += 1
            if mode == "clean3" and touches < 3:
                continue
            out[k] = (int(i1), int(i2), slope, touches)
            break                                          # кандидаты идут от самого дальнего
    return out


def major_lines(d: pd.DataFrame) -> list[dict]:
    """Касательные от главных экстремумов по закрытиям (так рисует трейдер): от главной вершины A (закрытие выше MAJOR_L
    закрытий с каждой стороны) — линия через ту из следующих более низких вершин (фрактал MINOR_N), которая даёт самый
    пологий наклон, то есть ни одна вершина после A не выше линии; новая вершина-касание перерисовывает линию.
    Для восходящей линии — зеркально по минимумам. Пробой — первое закрытие за линией (линия известна на t − 1).
    Возвращает по линии на каждый главный экстремум: сторона пробоя, A, B, наклон, бар пробоя (−1 — не пробита),
    значения линии на барах пробоя и закрепления, бар закрепления."""
    c = d["close"].to_numpy(dtype="float64")
    m = len(c)
    out = []
    for side in (1, -1):
        x = c                                               # всё по закрытиям свечей
        majors = pivots(x, MAJOR_L, high=side > 0)
        minors = pivots(x, MINOR_N, high=side > 0)
        for q, (a, conf_a) in enumerate(majors):
            nxt = majors[q + 1][1] if q + 1 < len(majors) else m   # линия живёт до подтверждения следующей вершины
            cand = minors[(minors[:, 0] > a + MINOR_N) & (side * (x[a] - x[minors[:, 0]]) > 0)] if len(minors) else minors
            best, b_best, rec = None, -1, None
            ci = 0
            for t in range(max(conf_a + 1, a + 2), min(nxt + 200, m - 1)):
                while ci < len(cand) and cand[ci][1] <= t - 1:
                    b = cand[ci][0]
                    sl = (x[b] - x[a]) / (b - a)
                    if best is None or side * sl > side * best:      # более пологая касательная
                        best, b_best = sl, b
                    ci += 1
                if best is None or not (side * best < 0):
                    continue
                lt, lp = x[a] + best * (t - a), x[a] + best * (t - 1 - a)
                if side * (c[t] - lt) > 0 and side * (c[t - 1] - lp) <= 0:
                    ln = x[a] + best * (t + 1 - a)
                    rec = {"side": side, "a": int(a), "b": int(b_best), "slope": best, "t": t,
                           "tc": t + 1 if side * (c[t + 1] - ln) > 0 else -1, "line_t": lt, "line_n": ln}
                    break
            if rec is None and best is not None and side * best < 0:
                rec = {"side": side, "a": int(a), "b": int(b_best), "slope": best, "t": -1, "tc": -1,
                       "line_t": np.nan, "line_n": np.nan}
            if rec is not None:
                out.append(rec)
    return out


def zigzag(c: np.ndarray, atr: np.ndarray, k: float) -> tuple[np.ndarray, np.ndarray]:
    """Вершины и впадины зигзага по закрытиям: экстремум подтверждается, когда закрытие ушло от него на k ATR
    (ATR на баре подтверждения). Возвращает (вершины, впадины) как массивы (индекс, бар подтверждения)."""
    highs, lows = [], []
    m = len(c)
    if m == 0:
        return np.zeros((0, 2), np.int64), np.zeros((0, 2), np.int64)
    up = 0                                                  # 0 — направление ещё не задано
    hi_i = lo_i = 0
    for t in range(1, m):
        if c[t] > c[hi_i]:
            hi_i = t
        if c[t] < c[lo_i]:
            lo_i = t
        th = k * atr[t] if atr[t] > 0 else np.inf
        if up >= 0 and c[hi_i] - c[t] >= th and hi_i < t:
            highs.append((hi_i, t))
            up, lo_i = -1, t
        elif up <= 0 and c[t] - c[lo_i] >= th and lo_i < t:
            lows.append((lo_i, t))
            up, hi_i = 1, t
    return (np.array(highs, np.int64).reshape(-1, 2), np.array(lows, np.int64).reshape(-1, 2))


def zz_lines(d: pd.DataFrame) -> list[dict]:
    """Линии по значимым точкам: обе точки — вершины зигзага по закрытиям (разворот >= ZZ_K ATR), первая — самое
    высокое закрытие за ZZ_ANCHOR свечей до неё, между точками >= ZZ_SPAN свечей; из таких вторых точек берётся та,
    что даёт самую пологую касательную (ни одна вершина зигзага после первой не выше линии). Линия живёт до пробоя,
    но не дольше ZZ_LIFE свечей от первой точки; пробой, совпавший по бару с пробоем линии от более ранней опоры,
    не дублируется. Для восходящей линии — зеркально по впадинам. Формат — как у major_lines."""
    c = d["close"].to_numpy(dtype="float64")
    atr = _atr(d).to_numpy() if {"high", "low"} <= set(d.columns) else np.full(len(c), np.inf)
    m = len(c)
    hs, ls = zigzag(c, atr, ZZ_K)
    out = []
    for side, piv in ((1, hs), (-1, ls)):
        seen = set()
        for a, conf_a in piv:
            w = c[max(0, a - ZZ_ANCHOR): a]
            if len(w) < ZZ_ANCHOR or side * (c[a] - (w.max() if side > 0 else w.min())) <= 0:
                continue
            cand = piv[(piv[:, 0] >= a + ZZ_SPAN) & (side * (c[a] - c[piv[:, 0]]) > 0)]
            later = piv[piv[:, 0] > a]
            best, b_best, rec, ci, li = None, -1, None, 0, 0
            for t in range(conf_a + 1, min(a + ZZ_LIFE, m - 1)):
                changed = False
                while li < len(later) and later[li][1] <= t - 1:
                    li += 1
                    changed = True
                while ci < len(cand) and cand[ci][1] <= t - 1:
                    ci += 1
                    changed = True
                if changed:
                    best, b_best = None, -1                 # самая пологая из известных точек, без вершин над линией
                    pts = later[:li, 0]
                    for b in cand[:ci, 0]:
                        sl = (c[b] - c[a]) / (b - a)
                        if np.all(side * (c[pts] - (c[a] + sl * (pts - a))) <= 1e-12):
                            if best is None or side * sl > side * best:
                                best, b_best = sl, int(b)
                if best is None or not (side * best < 0):
                    continue
                lt, lp = c[a] + best * (t - a), c[a] + best * (t - 1 - a)
                if side * (c[t] - lt) > 0 and side * (c[t - 1] - lp) <= 0:
                    ln = c[a] + best * (t + 1 - a)
                    rec = {"side": side, "a": int(a), "b": b_best, "slope": best, "t": t,
                           "tc": t + 1 if side * (c[t + 1] - ln) > 0 else -1, "line_t": lt, "line_n": ln}
                    break
            if rec is None and best is not None and side * best < 0:
                rec = {"side": side, "a": int(a), "b": b_best, "slope": best, "t": -1, "tc": -1,
                       "line_t": np.nan, "line_n": np.nan}
            if rec is not None and (rec["t"] < 0 or rec["t"] not in seen):
                seen.add(rec["t"])
                out.append(rec)
    return out


def signals(d: pd.DataFrame, mode: str = "last2") -> list[tuple[int, int, int, float, float, int, int]]:
    """(бар пробоя, бар закрепления или −1, сторона, линия на баре пробоя, линия на баре закрепления,
    индексы двух точек линии)."""
    if mode in ("major", "zz"):
        recs = major_lines(d) if mode == "major" else zz_lines(d)
        return [(r["t"], r["tc"], r["side"], r["line_t"], r["line_n"], r["a"], r["b"]) for r in recs if r["t"] > 0]
    c = d["close"].to_numpy(dtype="float64")
    atr = _atr(d).to_numpy() if {"high", "low"} <= set(d.columns) else np.full(len(c), np.inf)
    m = len(c)
    out = []
    for side in (1, -1):
        piv = pivots(c, PIV, high=side > 0)
        lines = _lines(c, atr, piv, side, mode)
        used = set()
        k = 0
        for t in range(PIV * 3, m - 1):
            while k + 1 < len(piv) and piv[k + 1][1] <= t - 1:
                k += 1
            if k < 1 or piv[k][1] > t - 1 or lines[k] is None:
                continue
            i1, i2, slope, _ = lines[k]
            if (i1, i2) in used:
                continue
            c2 = c[i2]
            line_t, line_p = c2 + slope * (t - i2), c2 + slope * (t - 1 - i2)
            if side * (c[t] - line_t) > 0 and side * (c[t - 1] - line_p) <= 0:
                used.add((i1, i2))
                line_n = c2 + slope * (t + 1 - i2)
                out.append((t, t + 1 if side * (c[t + 1] - line_n) > 0 else -1, side, line_t, line_n, i1, i2))
    return out


def _extend_daily(sym: str, df: pd.DataFrame) -> pd.DataFrame:
    """Дописать часовые свечи из дневных архивов после конца месячных (для свежих графиков)."""
    from . import data as D
    D.set_host("cdn")
    start = (df.index.max() + pd.Timedelta(hours=1)).normalize()
    days = pd.date_range(start, pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1), freq="D")
    extra = []
    for day in days:
        blob = D._get(f"{D.DAILY}/klines/{sym}/1h/{sym}-1h-{day:%Y-%m-%d}.zip")
        if blob is None:
            continue
        k = D._read_zip_csv(blob, D.KCOLS)
        k.index = pd.to_datetime(pd.to_numeric(k["open_time"]), unit="ms", utc=True)
        extra.append(k[[c_ for c_ in df.columns if c_ in k.columns]].astype("float64"))
    if not extra:
        return df
    out = pd.concat([df, *extra])
    return out[~out.index.duplicated()].sort_index()


def tf_frame(root: Path, sym: str, tf: str, extend: bool = False) -> pd.DataFrame | None:
    p15, p1 = root / f"{sym}-15m.parquet", root / f"{sym}-1h.parquet"
    src = p15 if tf == "15m" else p1
    if not src.exists():
        return None
    df = pd.read_parquet(src)
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop(columns=["open_time"])
    df = df[~df.index.duplicated()].sort_index()
    if extend and tf != "15m":
        df = _extend_daily(sym, df)
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


def market_context(root: Path) -> pd.DataFrame | None:
    """BTC на закрытии каждого часа: выше / ниже EMA50 последней закрытой дневной свечи и изменение за 7 дней."""
    d = tf_frame(root, "BTCUSDT", "1h")
    if d is None:
        return None
    day = d["close"].resample("1D").last()
    trend = np.sign(day - day.ewm(span=50, adjust=False).mean())
    trend.index = trend.index + pd.Timedelta(days=1)                        # известно после закрытия дня
    close_t = d.index + pd.Timedelta(hours=1)
    ctx = pd.DataFrame({"btc_trend": trend.reindex(close_t, method="ffill").to_numpy(),
                        "btc_ret7": (d["close"] / d["close"].shift(24 * 7) - 1).to_numpy()}, index=close_t)
    return ctx


ROOM_LOOKBACK = 300      # свечей назад: уровни зигзага, которые цене предстоит пройти до цели


def strength_inputs(root: Path, sym: str, tf: str, d: pd.DataFrame, atr: np.ndarray) -> dict[str, np.ndarray]:
    """Ряды для признаков силы пробоя (известны на закрытии бара): поток агрессоров, OI, доля часов внутри 4h-свечи,
    в которых перевешивали покупатели."""
    v = d["volume"].to_numpy(dtype="float64")
    tb_ = d["taker_buy_volume"].to_numpy(dtype="float64")
    delta = np.where(v > 0, (2 * tb_ - v), 0.0)                              # покупки − продажи
    roll_d = pd.Series(delta).rolling(20, min_periods=20).sum().to_numpy()
    roll_v = pd.Series(v).rolling(20, min_periods=20).sum().to_numpy()
    c = d["close"].to_numpy(dtype="float64")
    oi = np.full(len(c), np.nan)
    if tf != "15m":
        oi = np.log(metrics_on_bars(sym, root, d.index, BAR_MIN[tf])["oi"].replace(0, np.nan)).to_numpy(dtype="float64")
    buy_hours = np.full(len(c), np.nan)
    if tf == "4h":
        p1 = root / f"{sym}-1h.parquet"
        if p1.exists():
            h = pd.read_parquet(p1, columns=["open_time", "volume", "taker_buy_volume"])
            h.index = pd.to_datetime(h["open_time"], unit="ms", utc=True)
            h = h[~h.index.duplicated()]
            up = (h["taker_buy_volume"] > 0.5 * h["volume"]).astype("float64").where(h["volume"] > 0)
            buy_hours = up.resample("4h").mean().reindex(d.index).to_numpy(dtype="float64")
    return {"v": v, "delta": delta, "cvd20": roll_d / np.where(roll_v > 0, roll_v, np.nan), "oi": oi,
            "buy_hours": buy_hours, "atr": atr, "o": d["open"].to_numpy(dtype="float64"), "c": c,
            "vavg": pd.Series(v).shift(1).rolling(20).mean().to_numpy()}


def strength_row(st: dict[str, np.ndarray], tb: int, tc: int, side: int, entry: float, stop: float,
                 opp: np.ndarray, c: np.ndarray) -> dict[str, float]:
    """Признаки силы на сторону сделки (больше — сильнее для нас).
    effort      log(объём / средний) − log(ход свечи пробоя в ATR): усилие без результата — поглощение
    aggr_c      доля агрессоров нашей стороны в свече закрепления (продолжение потока)
    vol_c       объём свечи закрепления / средний за 20
    move_c      ход свечи закрепления в нашу сторону, ATR
    oi_brk      изменение log OI от свечи до пробоя до закрепления: новые позиции или закрытие старых
    cvd20       дельта агрессоров за 20 свечей до пробоя / объём (на нашей стороне — поток копился заранее)
    hours       доля часов внутри 4h-свечи пробоя, где перевешивала наша сторона (устойчивость, не один всплеск)
    room_r      расстояние от входа до ближайшего встречного уровня зигзага, R (место до цели)"""
    a, o, v, va = st["atr"][tb], st["o"], st["v"], st["vavg"]
    move_b = side * (c[tb] - o[tb]) / a if a > 0 else np.nan
    out = {"effort": np.log(v[tb] / va[tb]) - np.log(move_b) if va[tb] > 0 and v[tb] > 0 and move_b > 0 else np.nan,
           "cvd20": side * st["cvd20"][tb - 1] if tb > 0 else np.nan,
           "hours": st["buy_hours"][tb] if side > 0 else 1 - st["buy_hours"][tb]}
    if tc >= 0:
        out |= {"aggr_c": 0.5 + side * 0.5 * st["delta"][tc] / v[tc] if v[tc] > 0 else np.nan,
                "vol_c": v[tc] / va[tc] if va[tc] > 0 else np.nan,
                "move_c": side * (c[tc] - o[tc]) / st["atr"][tc] if st["atr"][tc] > 0 else np.nan,
                "oi_brk": st["oi"][tc] - st["oi"][tb - 1] if tb > 0 else np.nan}
    else:
        out |= {"aggr_c": np.nan, "vol_c": np.nan, "move_c": np.nan, "oi_brk": np.nan}
    e = tc if tc >= 0 else tb
    risk = side * (entry - stop)
    lv = opp[(opp[:, 1] <= e) & (opp[:, 0] >= e - ROOM_LOOKBACK)][:, 0] if len(opp) else np.empty(0, np.int64)
    ahead = c[lv][side * (c[lv] - entry) > 0] if len(lv) else np.empty(0)
    out["room_r"] = (side * (ahead - entry)).min() / risk if len(ahead) and risk > 0 else np.inf
    return out


def coin_trades(root: Path, sym: str, tf: str, ctx: pd.DataFrame | None = None) -> pd.DataFrame:
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
    bars30 = 30 * 24 * 60 // BAR_MIN[tf]
    ret30 = (d["close"] / d["close"].shift(bars30) - 1).to_numpy()
    close_t = d.index + pd.Timedelta(minutes=BAR_MIN[tf])
    if ctx is not None:
        cx = ctx.reindex(close_t, method="ffill")
        btc_trend, btc_ret7 = cx["btc_trend"].to_numpy(), cx["btc_ret7"].to_numpy()
    else:
        btc_trend = btc_ret7 = np.full(len(c), np.nan)
    oi_chg = np.full(len(c), np.nan)
    if tf != "15m":
        m = metrics_on_bars(sym, root, d.index, BAR_MIN[tf])
        oi = np.log(m["oi"].replace(0, np.nan))
        oi_chg = (oi - oi.shift(4)).to_numpy()
    st = strength_inputs(root, sym, tf, d, a)
    zh, zl = zigzag(c, a, ZZ_K)
    for mode, (tb, tc, side, line_b, line_c, _, _) in ((m_, sg) for m_ in LINES for sg in signals(d, m_)):
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
                for be in (False,):
                    r, ex = two_targets(o, hi, lo, c, f, fill, side, px, stop, 3.0, 5.0, be, HOLD[tf], fee)
                    rows.append({"symbol": sym, "tf": tf, "line": mode, "t": d.index[e], "side": side, "confirm": confirm,
                                 "risk_pct": side * (px - stop) / px,
                                 "entry": entry_kind, "be": be, "with_trend": trend[e] == side, "R": r,
                                 "vol_ratio": vol_ratio[tb], "oi_chg": oi_chg[tb], "body": body[tb],
                                 "close_loc": loc[tb] if side > 0 else 1 - loc[tb],
                                 "aggr": buy_share[tb] if side > 0 else 1 - buy_share[tb],
                                 "brk_atr": side * (c[tb] - line_b) / a[tb],
                                 "stop_atr": dist_atr, "ret30": ret30[e], "btc_trend": btc_trend[e],
                                 "btc_ret7": btc_ret7[e], "hold_bars": ex - fill,
                                 **strength_row(st, tb, tc, side, px, stop, zl if side > 0 else zh, c)})
    return pd.DataFrame(rows)


EXAMPLES = [("NEARUSDT", "1h", 300), ("NEARUSDT", "4h", 200), ("SOLUSDT", "15m", 320), ("SOLUSDT", "1h", 360), ("ETHUSDT", "1h", 360), ("DOGEUSDT", "1h", 360),
            ("ETHUSDT", "4h", 260), ("SOLUSDT", "4h", 260), ("AVAXUSDT", "15m", 320), ("LINKUSDT", "1h", 360),
            ("NEARUSDT", "15m", 400), ("ETHUSDT", "15m", 400), ("DOGEUSDT", "15m", 400)]


def chart(root: Path, sym: str, tf: str, bars: int, out: Path) -> None:
    """Свечи последних `bars` баров, экстремумы закрытий, все линии с пробоем в окне, вход / стоп / цели."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = tf_frame(root, sym, tf, extend=True)
    if d is None or len(d) < bars + 50:
        return
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    a = _atr(d).to_numpy()
    sw_lo, sw_hi = last_confirmed(pivots(lo, PIV, False), len(c)), last_confirmed(pivots(hi, PIV, True), len(c))
    w0 = len(c) - bars
    x = np.arange(len(c))
    fig, ax = plt.subplots(figsize=(18, 8), dpi=110)
    for i in range(w0, len(c)):
        col = "#26a69a" if c[i] >= o[i] else "#ef5350"
        ax.vlines(i, lo[i], hi[i], color=col, linewidth=0.8)
        ax.add_patch(plt.Rectangle((i - 0.35, min(o[i], c[i])), 0.7, max(abs(c[i] - o[i]), 1e-12), color=col))
    zh, zl = zigzag(c, a, ZZ_K)
    for side, mk, col, pv in ((1, "v", "#c62828", zh), (-1, "^", "#2e7d32", zl)):
        pv = pv[pv[:, 0] >= w0]
        ax.scatter(pv[:, 0], c[pv[:, 0]] * (1 + side * 0.004), marker=mk, color=col, s=36,
                   zorder=5, label="вершины зигзага" if side > 0 else "впадины зигзага")
    for tb, tc, side, line_b, line_c, i1, i2 in signals(d, "major"):
        if tb < w0 or i1 < w0 - 200:
            continue
        slope = (c[i2] - c[i1]) / (i2 - i1)
        xs = np.arange(max(i1, w0), tb + 2)
        ax.plot(xs, c[i2] + slope * (xs - i2), color="#9e9e9e", linewidth=0.8, linestyle="--")
    n_tr = 0
    for rec in zz_lines(d):
        i1, i2, side, tb, tc = rec["a"], rec["b"], rec["side"], rec["t"], rec["tc"]
        if i1 < w0 - 300 or (tb > 0 and tb < w0):
            continue
        x_a, x_b = c[i1], c[i2]
        end = tb + 2 if tb > 0 else len(c) - 1
        xs = np.arange(max(i1, w0), end)
        col = "#1565c0" if side > 0 else "#ef6c00"
        ax.plot(xs, x_a + rec["slope"] * (xs - i1), color=col, linewidth=1.8)
        ax.scatter([i1, i2], [x_a, x_b], color=col, s=70, facecolors="none", linewidths=1.6, zorder=6)
        if tb < 0:
            continue
        ax.scatter([tb], [c[tb]], marker="*", color=col, s=150, zorder=7)
        if tc < 0:
            continue
        sw = sw_lo[tc] if side > 0 else sw_hi[tc]
        if sw < 0:
            continue
        stop = (lo[sw] - 0.1 * a[tc]) if side > 0 else (hi[sw] + 0.1 * a[tc])
        entry = c[tc]
        risk = side * (entry - stop)
        if not (0.3 <= risk / a[tc] <= 4.0):
            continue
        r, ex = two_targets(o, hi, lo, c, d["funding"].to_numpy(dtype="float64"), tc, side, entry, stop, 3.0, 5.0,
                            False, HOLD[tf], TAKER)
        ex = int(min(ex, len(c) - 1))
        ax.hlines(entry, tc, ex, color="black", linewidth=1.0)
        ax.hlines(stop, tc, ex, color="#d32f2f", linestyle="--", linewidth=1.0)
        ax.hlines([entry + side * 3 * risk, entry + side * 5 * risk], tc, ex, color="#388e3c", linestyle=":",
                  linewidth=1.0)
        ax.scatter([sw], [lo[sw] if side > 0 else hi[sw]], marker="x", color="#d32f2f", s=50, zorder=7)
        ax.annotate(f"{'L' if side > 0 else 'S'} {r:+.1f}R", (tc, entry), fontsize=9, xytext=(4, 6),
                    textcoords="offset points")
        n_tr += 1
    ax.set_xlim(w0 - 2, len(c) + 2)
    vis = slice(w0, len(c))
    ax.set_ylim(lo[vis].min() * 0.985, hi[vis].max() * 1.015)
    ticks = np.linspace(w0, len(c) - 1, 8).astype(int)
    ax.set_xticks(ticks, [d.index[i].strftime("%m-%d %H:%M") for i in ticks])
    ax.set_title(f"{sym} {tf}: синие / оранжевые — линии по закрытиям через вершины зигзага (разворот >= {ZZ_K:g} ATR, "
                 f"опора — экстремум {ZZ_ANCHOR} свечей, точки >= {ZZ_SPAN} свечей друг от друга); серый пунктир — "
                 f"прежние линии от экстремума {MAJOR_L} свечей;\n"
                 f"○ точки линии, ★ пробой (закрытие за линией), вход — закрытие свечи закрепления; — вход, -- стоп за "
                 f"свингом (×), ··· цели 3R / 5R; сделок {n_tr}")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def collect(root: Path, syms: list[str], syms15: list[str]) -> None:
    tfs = os.environ.get("TL_TFS", "15m,1h,4h").split(",")
    ctx = market_context(root)
    for sym, tf, bars in (EXAMPLES if os.environ.get("TL_CHARTS", "1") == "1" else []):
        if tf in tfs and sym in set(mine(syms15 if tf == "15m" else syms)):
            try:
                chart(root, sym, tf, bars, part_path("tline").parent / "tline_examples" / f"{sym}_{tf}.png")
            except Exception as e:
                print(f"  график {sym} {tf}: {e}", flush=True)
    parts = []
    for tf, lst in (("15m", syms15), ("1h", syms), ("4h", syms)):
        if tf not in tfs:
            continue
        for s in mine(lst):
            try:
                x = coin_trades(root, s, tf, ctx)
                if len(x):
                    parts.append(x)
            except Exception as e:
                print(f"  {tf} {s}: пропуск ({e})", flush=True)
    print(f"  частей монет-ТФ со сделками: {len(parts)}", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path("tline"), index=False)


def diagnose_2026(df: pd.DataFrame, line_name: dict) -> None:
    """Почему шорт на 4h с перевесом продавцов ослаб в 2026: рынок, состояние монеты, исход сделок, помесячно."""
    base = df[(df.tf == "4h") & (df.side == -1) & df.confirm & (df.entry == "market") & (df.aggr >= 0.55)
              & df.line.isin(["clean", "major", "zz"])].copy()
    if not len(base):
        return
    base["year"] = base.t.dt.year
    q = base.loc[base.per == "is", "ret30"].quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
    base["ret30_q"] = np.digitize(base["ret30"], q) + 1
    print("\n=== Разбор 2026: шорт 4h, продавцов >= 55%, закрепление, рынок; ячейка — средний R (n) по годам ===")

    def by_year(g: pd.DataFrame) -> dict:
        y = g.groupby("year")["R"].agg(["size", "mean"])
        return {str(k): f"{v['mean']:+.2f} ({int(v['size'])})" for k, v in y.iterrows()}

    for ln, g in base.groupby("line"):
        print(f"\n  линия: {line_name[ln]}")
        rows = [{"срез": "все", **by_year(g)},
                {"срез": "BTC выше дневной EMA50", **by_year(g[g.btc_trend > 0])},
                {"срез": "BTC ниже дневной EMA50", **by_year(g[g.btc_trend < 0])},
                {"срез": "BTC за 7 дней > 0", **by_year(g[g.btc_ret7 > 0])},
                {"срез": "BTC за 7 дней < 0", **by_year(g[g.btc_ret7 < 0])},
                {"срез": "монета: свой тренд 1d вниз (по тренду)", **by_year(g[g.with_trend])},
                {"срез": "монета: свой тренд 1d вверх", **by_year(g[~g.with_trend])},
                *[{"срез": f"монета за 30 дней: квинтиль {k} (1 — сильнее всех упала)", **by_year(g[g.ret30_q == k])}
                  for k in range(1, 6)]]
        print(pd.DataFrame(rows).fillna("").to_string(index=False))
        ex = g.assign(stop=g.R <= -0.9, tp1=g.R >= 1.3, tp2=g.R >= 3.8).groupby("year")[["stop", "tp1", "tp2"]].mean()
        ex["держим, свечей (медиана)"] = g.groupby("year")["hold_bars"].median()
        ex["стоп, ATR (медиана)"] = g.groupby("year")["stop_atr"].median()
        print("  исход сделок по годам (доли):\n" + ex.round(2).to_string())
        m = g[g.t >= "2025-01-01"].groupby(g.t.dt.strftime("%Y-%m"))["R"].agg(["size", "mean"])
        print("  помесячно 2025–2026: " + ", ".join(f"{k} {v['mean']:+.2f} ({int(v['size'])})" for k, v in m.iterrows()))


STRENGTH = {"effort": "усилие без результата", "aggr_c": "агрессоры в свече закрепления", "vol_c": "объём закрепления",
            "move_c": "ход закрепления, ATR", "oi_brk": "OI за пробой", "cvd20": "дельта за 20 свечей до пробоя",
            "hours": "часы с перевесом внутри 4h", "room_r": "место до встречного уровня, R"}


def strength_report(df: pd.DataFrame, line_name: dict) -> None:
    """Признаки силы пробоя: средний R по квинтилям (границы — по IS) и составной балл, веса которого — знаки ранговой
    корреляции признака с R на IS; балл проверяется на VAL / HOLDOUT и по годам."""
    x0 = df[df.confirm & (df.entry == "market")].copy()
    x0["room_r"] = x0["room_r"].clip(upper=20.0)
    groups = [("4h шорт, агрессоры >= 55%", (x0.tf == "4h") & (x0.side == -1) & (x0.aggr >= 0.55)),
              ("4h обе стороны", x0.tf == "4h"), ("1h обе стороны", x0.tf == "1h")]
    print("\n=== Сила пробоя: средний R по квинтилям признака (1 — слабее для нас, 5 — сильнее; границы по IS) ===")
    for (gname, mask), ln in itertools.product(groups, ("clean", "zz")):
        g = x0[mask & (x0.line == ln)]
        if g.per.eq("is").sum() < 200:
            continue
        print(f"\n  {gname}, линия: {line_name[ln]}")
        rows, w = [], {}
        feats = [f for f in STRENGTH if f in g.columns and g[f].notna().mean() > 0.5]
        for f in feats:
            gi = g[g.per == "is"]
            rho = gi[[f, "R"]].dropna().corr(method="spearman").iloc[0, 1]
            w[f] = np.sign(rho) if abs(rho) >= 0.02 else 0.0
            edges = gi[f].quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
            qq = np.where(g[f].notna(), np.digitize(g[f], edges) + 1, 0)
            for q in range(1, 6):
                gg = g[qq == q]
                rows.append({"признак": STRENGTH[f], "ρ IS": f"{rho:+.3f}", "квинтиль": q,
                             **{p: _cell(gg[gg.per == p]) for p in PER}})
        print(pd.DataFrame(rows).to_string(index=False))
        used = [f for f in feats if w[f] != 0]
        if not used:
            continue
        ref = g[g.per == "is"]
        score = np.zeros(len(g))
        for f in used:                                       # перцентиль по распределению IS, пропуск — середина
            pct = np.searchsorted(np.sort(ref[f].dropna().to_numpy()), g[f].to_numpy(), side="right") / ref[f].notna().sum()
            score += w[f] * np.where(g[f].notna(), pct - 0.5, 0.0)
        g = g.assign(score=score)
        thr = g.loc[g.per == "is", "score"].quantile(0.6)
        top, rest = g[g.score >= thr], g[g.score < thr]
        print(f"  балл ({', '.join(('+' if w[f] > 0 else '−') + f for f in used)}), верхние 40% по IS:")
        print(pd.DataFrame([{"": nm, **{p: _cell(z[z.per == p]) for p in PER}} for nm, z in (("все", g), ("верх 40%", top),
                                                                                               ("остальные", rest))]).to_string(index=False))
        yy = top.groupby(top.t.dt.year)["R"].agg(["size", "mean"])
        print("  верх 40% по годам: " + ", ".join(f"{k}: {v['mean']:+.2f}R ({int(v['size'])})" for k, v in yy.iterrows()))


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
    out = Path(os.environ.get("OUT", "../out")) / "tline_examples"
    for png in Path(os.environ.get("PARTS", "parts")).rglob("tline_examples/*.png"):
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(png, out / png.name)
    print(f"===== TLINE: сделок {len(df):,}, монет {df.symbol.nunique()}, частей {len(parts)} =====")
    print("ячейка: средний R на сделку (t по дням, прибыльных, сделок в месяц на весь набор монет); выход 1/2 на 3R + 1/2 на 5R")
    line_name = {"last2": "2 последние", "clean": "чистая", "clean3": "чистая, 3 касания", "major": "от главного экстремума", "zz": "по значимым точкам"}
    filters = lambda g: (("все", g), ("по тренду старшего ТФ", g[g.with_trend]),
                         ("объём пробоя >= 1.5x", g[g.vol_ratio >= 1.5]), ("OI рос 4 бара", g[g.oi_chg > 0]),
                         ("сильная свеча", g[(g.body >= 0.6) & (g.close_loc >= 0.75) & (g.brk_atr >= 0.3)]),
                         ("агрессоры >= 55%", g[g.aggr >= 0.55]),
                         ("агрессоры >= 55% + тренд", g[(g.aggr >= 0.55) & g.with_trend]),
                         ("сила: свеча + объём + агрессоры", g[(g.body >= 0.6) & (g.close_loc >= 0.75) &
                                                               (g.brk_atr >= 0.3) & (g.vol_ratio >= 1.5) &
                                                               (g.aggr >= 0.55)]))
    rows = []
    for (tf, ln, confirm, entry), g in df.groupby(["tf", "line", "confirm", "entry"]):
        for name, gg in filters(g):
            if tf == "15m" and "OI" in name:
                continue
            rows.append({"ТФ": tf, "линия": line_name[ln], "вход": ("закрепл., " if confirm else "пробой, ") +
                         ("рынок" if entry == "market" else "ретест"), "фильтр": name,
                         **{p: _cell(gg[gg.per == p]) for p in PER},
                         "вне подбора, VAL+HO": _cell(gg[(gg.per != "is") & gg.group.isin(["ext54", "fresh"])])})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== Устойчивость кандидата: агрессоры в сторону пробоя, закрепление, вход по рынку ===")
    rows = []
    x0 = df[df.confirm & (df.entry == "market")]
    for (tf, ln), g in x0[x0.tf.isin(["1h", "4h"])].groupby(["tf", "line"]):
        for thr in (0.50, 0.52, 0.55, 0.58, 0.60):
            gg = g[g.aggr >= thr]
            for sd, nm in ((0, "обе"), (1, "лонг"), (-1, "шорт")):
                x = gg if sd == 0 else gg[gg.side == sd]
                x2 = x.assign(R=x["R"] - 11e-4 / x["risk_pct"])          # издержки x2: ещё 2 x 5.5 б.п.
                rows.append({"ТФ": tf, "линия": line_name[ln], "агрессоры >=": f"{thr:.0%}", "сторона": nm,
                             **{p: _cell(x[x.per == p]) for p in PER},
                             "HO, издержки x2": _cell(x2[x2.per == "ho"])})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nпо годам (4h, агрессоры >= 55%, закрепление, рынок):")
    y = x0[(x0.tf == "4h") & (x0.aggr >= 0.55)]
    for ln, g in y.groupby("line"):
        yy = g.groupby(g.t.dt.year)["R"].agg(["size", "mean"])
        print(f"  {line_name[ln]}: " + ", ".join(f"{k}: {v['mean']:+.2f}R (n={int(v['size'])})" for k, v in yy.iterrows()))
    if "btc_trend" in df.columns:
        diagnose_2026(df, line_name)
    if "effort" in df.columns:
        strength_report(df, line_name)
        ev = df[df.confirm & (df.entry == "market") & df.tf.isin(["1h", "4h"]) & df.line.isin(["clean", "zz"])]
        out = Path(os.environ.get("OUT", "../out"))
        out.mkdir(parents=True, exist_ok=True)
        ev.sort_values("t").to_csv(out / "tline_breakouts.csv", index=False)
        print(f"\ntline_breakouts.csv: {len(ev)} пробоев (1h / 4h, закрепление, рынок) — для разметки стаканом")

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
