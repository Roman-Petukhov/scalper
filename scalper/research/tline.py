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
import json
import os
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from .broad import ADV_MIN, adv30, group_of
from .engine import funding_on_bars
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .wave2 import metrics_on_bars

PER = {"is": ("2022-01-01", "2024-07-01"), "val": ("2024-07-01", "2025-07-01"), "ho": ("2025-07-01", "2026-10-01")}
PIV = 5
MAKER, TAKER = 2e-4, 5.5e-4
HOLD = {"15m": 200, "1h": 120, "2h": 90, "4h": 60, "6h": 50, "12h": 40, "1d": 30}
HTF = {"15m": "1h", "1h": "4h", "2h": "1D", "4h": "1D", "6h": "1D", "12h": "1W", "1d": "1W"}
BAR_MIN = {"15m": 15, "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720, "1d": 1440}
HIGH_TFS = {"2h": "2h", "4h": "4h", "6h": "6h", "12h": "12h", "1d": "1D"}      # собираются из 1h-свечей
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
def target_exit(o, h, lo, c, fund, e, side, entry, stop, k, max_hold, entry_fee, line0, i0, slope, buf):
    """Вся позиция на k·R; ранний выход по закрытию, если свеча закрылась обратно за линию пробоя дальше buf
    (в цене): линия на баре j = line0 + slope·(j − i0). Стоп проверяется раньше тейка. Результат в R."""
    n = len(c)
    risk = side * (entry - stop)
    if not (risk > 0):
        return np.nan, e
    tp = entry + side * k * risk
    paid = 0.0
    for j in range(e + 1, n):
        paid += fund[j]
        if (side > 0 and lo[j] <= stop) or (side < 0 and h[j] >= stop):
            ex = min(stop, o[j]) if side > 0 else max(stop, o[j])
            return (side * (ex - entry) - (entry_fee + TAKER) * entry) / risk - side * paid * entry / risk, j
        if (side > 0 and h[j] >= tp) or (side < 0 and lo[j] <= tp):
            return (side * (tp - entry) - (entry_fee + MAKER) * entry) / risk - side * paid * entry / risk, j
        back = side * (c[j] - (line0 + slope * (j - i0))) < -buf
        if back or j - e >= max_hold or j == n - 1:
            return (side * (c[j] - entry) - (entry_fee + TAKER) * entry) / risk - side * paid * entry / risk, j
    return np.nan, n - 1


@njit(cache=True)
def trail_exit(o, h, lo, c, fund, e, side, entry, stop, atr, k_trail, max_hold, entry_fee):
    """Без цели: стоп подтягивается за лучшим закрытием на k_trail ATR (ATR на баре входа), не отодвигается назад
    (Bulkowski: стоп за свечой пробоя, затем волатильный трейлинг). Стоп проверяется раньше подтяжки. Результат в R."""
    n = len(c)
    risk = side * (entry - stop)
    if not (risk > 0):
        return np.nan, e
    sp = stop
    best = c[e]
    paid = 0.0
    for j in range(e + 1, n):
        paid += fund[j]
        if (side > 0 and lo[j] <= sp) or (side < 0 and h[j] >= sp):
            ex = min(sp, o[j]) if side > 0 else max(sp, o[j])
            return (side * (ex - entry) - (entry_fee + TAKER) * entry) / risk - side * paid * entry / risk, j
        if side * (c[j] - best) > 0:
            best = c[j]
        nsp = best - side * k_trail * atr
        if side * (nsp - sp) > 0:
            sp = nsp
        if j - e >= max_hold or j == n - 1:
            return (side * (c[j] - entry) - (entry_fee + TAKER) * entry) / risk - side * paid * entry / risk, j
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


LINES = ("zz", "s123")      # прежние варианты (зоны, горизонтальные уровни и др.) — в git и docs/research_report.md
SCALE_LINES = {"zz2": 2.0, "zz5": 5.0}   # та же линия на других масштабах зигзага (только 4h): больше сигналов?
MORE_TFS = ("2h", "6h", "12h", "1d")    # соседние с 4h таймфреймы — только правило бота на линии zz
ADV_LO = 5e6                            # монеты с оборотом ниже порога бота (20M) — для корзин ликвидности
MAJOR_L = 12          # главный экстремум: тень выше (ниже) 12 свечей с каждой стороны
MINOR_N = 3           # точки касания: фрактал 3 свечи
ZZ_K = 3.0            # зигзаг по закрытиям: разворот >= 3 ATR
ZZ_ANCHOR = 60        # опорная вершина — самое высокое закрытие за 60 свечей до неё
ZZ_SPAN = 20          # между точками линии не меньше 20 свечей
ZZ_LIFE = 300         # линия живёт не дольше 300 свечей от опорной вершины
FAN_K = (3.0, 6.0)    # «веер»: соседние вершины зигзага на двух масштабах разворота, ATR
FAN2_K = (2.0, 3.0, 6.0)  # «веер» с мелким масштабом (короткие линии внутри движения)


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


def zz_lines(d: pd.DataFrame, log: bool = False, k: float = ZZ_K) -> list[dict]:
    """Линии по значимым точкам: обе точки — вершины зигзага по закрытиям (разворот >= ZZ_K ATR), первая — самое
    высокое закрытие за ZZ_ANCHOR свечей до неё, между точками >= ZZ_SPAN свечей; из таких вторых точек берётся та,
    что даёт самую пологую касательную (ни одна вершина зигзага после первой не выше линии). Линия живёт до пробоя,
    но не дольше ZZ_LIFE свечей от первой точки; пробой, совпавший по бару с пробоем линии от более ранней опоры,
    не дублируется. Для восходящей линии — зеркально по впадинам. Формат — как у major_lines.
    log=True — линия прямая на логарифмической шкале (наклон в долях цены за свечу, как трейдер ведёт её на
    лог-графике): вершины те же, геометрия касания и пробоя — по log(close); line_t / line_n — в цене, slope — в log."""
    price = d["close"].to_numpy(dtype="float64")
    atr = _atr(d).to_numpy() if {"high", "low"} <= set(d.columns) else np.full(len(price), np.inf)
    m = len(price)
    hs, ls = zigzag(price, atr, k)
    c = np.log(price) if log else price
    back = np.exp if log else (lambda v: v)
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
                           "tc": t + 1 if side * (c[t + 1] - ln) > 0 else -1, "line_t": float(back(lt)),
                           "line_n": float(back(ln)), "log": log}
                    break
            if rec is None and best is not None and side * best < 0:
                rec = {"side": side, "a": int(a), "b": b_best, "slope": best, "t": -1, "tc": -1,
                       "line_t": np.nan, "line_n": np.nan, "log": log}
            if rec is not None and (rec["t"] < 0 or rec["t"] not in seen):
                seen.add(rec["t"])
                out.append(rec)
    return out


def s123_lines(d: pd.DataFrame) -> list[dict]:
    """Линия по методу 1-2-3 Сперандео (Bulkowski, «Down-Sloping Trendline Tutorial»): от вершины A (как в zz_lines —
    вершина зигзага, самое высокое закрытие за ZZ_ANCHOR свечей до неё) к самому низкому закрытию B после неё; линия
    разворачивается вверх, пока ни одно закрытие между A и B не окажется над ней (наклон — наибольший из наклонов
    A→j, j в (A, B]). B обновляется, пока цена делает новые минимумы; после B линия может пересечь цену — это пробой
    (закрытие над линией, не раньше чем через свечу после B). Для восходящей линии — зеркально. Формат — как у zz_lines."""
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
            end = min(a + ZZ_LIFE, m - 1)
            js = np.arange(a + 1, end)
            if not len(js):
                continue
            slope_to = (c[js] - c[a]) / (js - a)
            cum = np.maximum.accumulate(side * slope_to) * side     # самый «поворачивающий» наклон до j
            b = a + 1
            rec = None
            for t in range(max(conf_a, a + ZZ_SPAN) + 1, end):
                if side * (c[b] - c[t - 1]) > 0:                    # новый экстремум против линии — новая точка B
                    b = t - 1
                sl = cum[b - a - 1]
                if not (side * sl < 0) or t <= b + 1:
                    continue
                lt, lp = c[a] + sl * (t - a), c[a] + sl * (t - 1 - a)
                if side * (c[t] - lt) > 0 and side * (c[t - 1] - lp) <= 0:
                    ln = c[a] + sl * (t + 1 - a)
                    jb = int(js[int(np.argmax(side * slope_to[: b - a]))])
                    rec = {"side": side, "a": int(a), "b": jb, "slope": sl, "t": t,
                           "tc": t + 1 if side * (c[t + 1] - ln) > 0 else -1, "line_t": float(lt), "line_n": float(ln)}
                    break
            if rec is not None and rec["t"] not in seen:
                seen.add(rec["t"])
                out.append(rec)
    return out


ZONE_W = 1.0          # наклонная зона: ширина 1 ATR
ZONE_BRK = 0.2        # пробой зоны / уровня — закрытие дальше 0.2 ATR за внешним краем; ближе — касание
ZONE_MINI = 3         # мини-вершины: экстремум закрытия среди 3 свечей с каждой стороны


def zone_lines(d: pd.DataFrame, min_touches: int = 0) -> list[dict]:
    """Наклонная зона шириной ZONE_W ATR (сверено с ручной разметкой, tools/linecheck.py).
    Направление — вершины зигзага по закрытиям ZZ_K ATR: от вершины A (самое высокое закрытие за ZZ_ANCHOR свечей)
    зона появляется, когда подтвердилась следующая вершина зигзага ниже A. Внешний край — от крайней тени у A (свечи
    ±2), прижат к теням: ни одна тень после A (до последней подтверждённой мини-вершины) за него не выходит, мини-вершины
    могут заходить внутрь зоны. Пробой — закрытие дальше ZONE_BRK ATR за внешним краем; ближе — касание, свеча станет
    мини-вершиной и край сдвинется. Касания — мини-вершины, закрывшиеся внутри зоны (не ближе 3 свечей друг к другу);
    min_touches — не меньше стольких касаний к моменту пробоя. Для восходящей зоны — зеркально.
    line_t / line_n — внешний край на барах пробоя и следующем (уровень лимитки ретеста)."""
    c = d["close"].to_numpy(dtype="float64")
    hi, lo = d["high"].to_numpy(dtype="float64"), d["low"].to_numpy(dtype="float64")
    atr = _atr(d).to_numpy()
    m = len(c)
    hs, ls = zigzag(c, atr, ZZ_K)
    out = []
    for side, piv in ((1, hs), (-1, ls)):
        ext = hi if side > 0 else lo
        mini = pivots(c, ZONE_MINI, side > 0)
        seen = set()
        for a, conf_a in piv:
            w = c[max(0, a - ZZ_ANCHOR): a]
            if len(w) < ZZ_ANCHOR or side * (c[a] - (w.max() if side > 0 else w.min())) <= 0:
                continue
            big = piv[(piv[:, 0] >= a + ZZ_SPAN) & (side * (c[a] - c[piv[:, 0]]) > 0)]
            if not len(big):
                continue
            b0, conf_b0 = int(big[0][0]), int(big[0][1])
            near = ext[max(0, a - 2): a + 3]
            y0 = near.max() if side > 0 else near.min()
            later = mini[mini[:, 0] > a]
            sl, b_best, wd, rec, li = None, -1, 0.0, None, 0
            for t in range(max(conf_a, conf_b0) + 1, min(a + ZZ_LIFE, m - 1)):
                changed = False
                while li < len(later) and later[li][1] <= t - 1:
                    li, changed = li + 1, True
                if changed or sl is None:
                    last = max(int(later[li - 1][0]) if li else b0, b0)
                    js = np.arange(a + 3, last + 1)
                    if not len(js):
                        continue
                    need = (ext[js] - y0) / (js - a)
                    k = int(np.argmax(side * need))
                    sl, b_best = float(need[k]), int(js[k])
                    wd = ZONE_W * atr[b_best]
                if not (side * sl < 0):
                    continue
                lt, lp = y0 + sl * (t - a), y0 + sl * (t - 1 - a)
                if side * (c[t] - lt) > ZONE_BRK * atr[t] and side * (c[t - 1] - lp) <= ZONE_BRK * atr[t - 1]:
                    tp = later[:li, 0]
                    inside = tp[side * (c[tp] - (y0 + sl * (tp - a))) >= -wd]
                    touches, prev = 1, a                                  # A — первое касание
                    for j in inside:
                        if j - prev >= ZONE_MINI:
                            touches, prev = touches + 1, j
                    ln = y0 + sl * (t + 1 - a)
                    rec = {"side": side, "a": int(a), "b": b_best, "slope": sl, "t": t,
                           "tc": t + 1 if side * (c[t + 1] - ln) > 0 else -1, "line_t": float(lt),
                           "line_n": float(ln), "touches": touches}
                    break
            if rec is not None and rec["t"] not in seen and rec["touches"] >= min_touches:
                seen.add(rec["t"])
                out.append(rec)
    return out


LEVEL_TOL = 0.5       # горизонтальный уровень: развороты в пределах 0.5 ATR от центра — одно скопление
LEVEL_PIV = 10        # развороты для уровней — экстремум закрытий среди 10 свечей с каждой стороны (видимые)
LEVEL_GAP = 20        # касания уровня не ближе 20 свечей друг к другу
LEVEL_LIFE = 300      # уровень живёт 300 свечей после последнего касания (сила уровня затухает)


def level_lines(d: pd.DataFrame, min_touches: int = 3) -> list[dict]:
    """Горизонтальные уровни-зоны по скоплениям разворотов (Chung & Bellotti 2021; Osler 2000): развороты —
    экстремумы закрытий среди LEVEL_PIV свечей с каждой стороны (вершины и впадины вместе); разворот в пределах LEVEL_TOL ATR
    от центра уровня (среднее закрытий касаний) — новое касание, если от прошлого прошло >= LEVEL_GAP свечей.
    Уровень готов с min_touches-го касания (известен с бара его подтверждения). Зона: ближние края — крайние закрытия
    касаний, внешние — крайние тени касаний (максимум сверху, минимум снизу). Пробой вверх — закрытие дальше ZONE_BRK
    ATR над верхним внешним краем, предыдущее закрытие — не выше его; вниз — зеркально. После пробоя уровень снят.
    line_t / line_n — пробитый внешний край (уровень лимитки ретеста)."""
    c = d["close"].to_numpy(dtype="float64")
    hi, lo = d["high"].to_numpy(dtype="float64"), d["low"].to_numpy(dtype="float64")
    atr = _atr(d).to_numpy()
    m = len(c)
    pv = np.concatenate([pivots(c, LEVEL_PIV, True), pivots(c, LEVEL_PIV, False)])
    pv = pv[np.argsort(pv[:, 1], kind="stable")]                      # по бару подтверждения
    levels: list[dict] = []                                           # {"m": [индексы касаний], "last": бар}
    out = []
    k = 0
    for t in range(1, m - 1):
        while k < len(pv) and pv[k][1] <= t - 1:                      # развороты, известные на баре t
            p = int(pv[k][0])
            k += 1
            if not (atr[p] > 0):
                continue
            best, dist = None, LEVEL_TOL * atr[p]
            for lv in levels:
                dd = abs(c[p] - lv["center"])
                if dd <= dist:
                    best, dist = lv, dd
            if best is None:
                levels.append({"m": [p], "center": c[p], "last": p, "up": hi[p], "dn": lo[p]})
            elif p - best["last"] >= LEVEL_GAP:
                best["m"].append(p)
                best["center"] = float(np.mean(c[best["m"]]))
                best["last"], best["up"], best["dn"] = p, max(best["up"], hi[p]), min(best["dn"], lo[p])
        if not levels:
            continue
        alive = []
        for lv in levels:
            if t - lv["last"] > LEVEL_LIFE:
                continue
            fired = False
            if len(lv["m"]) >= min_touches:
                for side, edge in ((1, lv["up"]), (-1, lv["dn"])):
                    if side * (c[t] - edge) > ZONE_BRK * atr[t] and side * (c[t - 1] - edge) <= ZONE_BRK * atr[t - 1]:
                        out.append({"side": side, "a": int(lv["m"][0]), "b": int(lv["m"][-1]), "slope": 0.0, "t": t,
                                    "tc": t + 1 if side * (c[t + 1] - edge) > 0 else -1, "line_t": float(edge),
                                    "line_n": float(edge), "touches": len(lv["m"])})
                        fired = True
                        break
            if not fired:
                alive.append(lv)
        levels = alive
    return out


def fan_lines(d: pd.DataFrame, scales: tuple[float, ...] = FAN_K) -> list[dict]:
    """Линии через две СОСЕДНИЕ вершины зигзага по закрытиям (так трейдер ведёт линию по движению: после пробоя —
    новая, более крутая, от следующей вершины), на масштабах разворота FAN_K ATR одновременно. Нисходящая — если
    вторая вершина ниже первой и ни одно закрытие между ними не выше линии; восходящая — зеркально по впадинам.
    Линия известна после подтверждения второй вершины, живёт до пробоя, но не дольше ZZ_LIFE свечей; пробой,
    совпавший по бару и стороне с уже учтённым, не дублируется. Формат — как у major_lines."""
    c = d["close"].to_numpy(dtype="float64")
    atr = _atr(d).to_numpy() if {"high", "low"} <= set(d.columns) else np.full(len(c), np.inf)
    m = len(c)
    out, seen = [], set()
    for k in scales:
        hs, ls = zigzag(c, atr, k)
        for side, piv in ((1, hs), (-1, ls)):
            for q in range(1, len(piv)):
                a, b, conf_b = int(piv[q - 1][0]), int(piv[q][0]), int(piv[q][1])
                if not (side * (c[a] - c[b]) > 0):
                    continue
                sl = (c[b] - c[a]) / (b - a)
                seg = np.arange(a + 1, b)
                if len(seg) and np.any(side * (c[seg] - (c[a] + sl * (seg - a))) > 1e-12):
                    continue
                rec = {"side": side, "a": a, "b": b, "slope": sl, "t": -1, "tc": -1, "line_t": np.nan,
                       "line_n": np.nan, "k": k}
                for t in range(conf_b + 1, min(a + ZZ_LIFE, m - 1)):
                    lt, lp = c[a] + sl * (t - a), c[a] + sl * (t - 1 - a)
                    if side * (c[t] - lt) > 0:
                        if side * (c[t - 1] - lp) <= 0:
                            ln = c[a] + sl * (t + 1 - a)
                            rec |= {"t": t, "tc": t + 1 if side * (c[t + 1] - ln) > 0 else -1, "line_t": lt,
                                    "line_n": ln}
                        break
                if rec["t"] >= 0 and (rec["t"], side) in seen:
                    continue
                seen.add((rec["t"], side))
                out.append(rec)
    return out


def signals(d: pd.DataFrame, mode: str = "last2") -> list[tuple[int, int, int, float, float, int, int]]:
    """(бар пробоя, бар закрепления или −1, сторона, линия на баре пробоя, линия на баре закрепления,
    индексы двух точек линии)."""
    if mode in SCALE_LINES:
        return [(r["t"], r["tc"], r["side"], r["line_t"], r["line_n"], r["a"], r["b"])
                for r in zz_lines(d, k=SCALE_LINES[mode]) if r["t"] > 0]
    if mode in ("major", "zz", "zzlog", "fan", "fan2", "zone", "zone3", "hl3", "hl4", "s123"):
        recs = {"major": major_lines, "zz": zz_lines, "zzlog": lambda x: zz_lines(x, log=True), "fan": fan_lines,
                "fan2": lambda x: fan_lines(x, FAN2_K), "zone": zone_lines, "zone3": lambda x: zone_lines(x, 3),
                "hl3": lambda x: level_lines(x, 3), "hl4": lambda x: level_lines(x, 4), "s123": s123_lines}[mode](d)
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


def _extend_daily(sym: str, df: pd.DataFrame, interval: str = "1h") -> pd.DataFrame:
    """Дописать свечи из дневных архивов после конца месячных (для свежих графиков)."""
    from . import data as D
    D.set_host("cdn")
    start = (df.index.max() + pd.Timedelta(minutes=1)).normalize()
    days = pd.date_range(start, pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1), freq="D")
    extra = []
    for day in days:
        blob = D._get(f"{D.DAILY}/klines/{sym}/{interval}/{sym}-{interval}-{day:%Y-%m-%d}.zip")
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
    if extend:
        df = _extend_daily(sym, df, "15m" if tf == "15m" else "1h")
    if tf in HIGH_TFS:
        df = df.resample(HIGH_TFS[tf], label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "quote_volume": "sum",
             "count": "sum", "taker_buy_volume": "sum", "taker_buy_quote_volume": "sum"}).dropna(subset=["close"])
    df["funding"] = funding_on_bars(sym, root, df.index, BAR_MIN[tf])
    return df


HTF2 = {"15m": "4h", "1h": "1D", "2h": "1W", "4h": "1W", "6h": "1W", "12h": "1W", "1d": "1W"}          # тренд ещё на ступень старше


def _htf_bars(d: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = d[["open", "high", "low", "close"]].resample("W-MON" if rule == "1W" else rule, label="left",
                                                      closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return agg


def _to_bars(st: pd.Series, d: pd.DataFrame, rule: str, tf: str) -> np.ndarray:
    st.index = st.index + {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "1D": pd.Timedelta(days=1),
                           "1W": pd.Timedelta(days=7)}[rule]
    bar_close = d.index + pd.Timedelta(minutes=BAR_MIN[tf])                  # известно после закрытия свечи ст. ТФ
    return st.reindex(bar_close, method="ffill").to_numpy()


def htf_trend(d: pd.DataFrame, tf: str, rule: str | None = None) -> np.ndarray:
    """+1 / −1: close последней закрытой свечи старшего ТФ выше / ниже её EMA50 (известно на закрытии бара)."""
    rule = rule or HTF[tf]
    agg = _htf_bars(d, rule)
    st = np.sign(agg["close"] - agg["close"].ewm(span=50, adjust=False).mean())
    return _to_bars(st, d, rule, tf)


def htf_structure(d: pd.DataFrame, tf: str, rule: str | None = None) -> np.ndarray:
    """Структурный тренд старшего ТФ по явным вершинам (зигзаг по закрытиям, разворот >= ZZ_K ATR): +1 — последняя
    вершина и последняя впадина выше предыдущих (HH + HL), −1 — обе ниже (LH + LL), 0 — иначе. Известно на
    закрытии свечи старшего ТФ, подтвердившей последнюю точку."""
    rule = rule or HTF[tf]
    agg = _htf_bars(d, rule)
    c = agg["close"].to_numpy(dtype="float64")
    hs, ls = zigzag(c, _atr(agg).to_numpy(), ZZ_K)
    ev = sorted([(int(cf), 1, int(i)) for i, cf in hs] + [(int(cf), -1, int(i)) for i, cf in ls])
    st = np.zeros(len(c))
    last = {1: [], -1: []}
    k = 0
    for t in range(len(c)):
        while k < len(ev) and ev[k][0] <= t:
            last[ev[k][1]].append(c[ev[k][2]])
            k += 1
        if len(last[1]) >= 2 and len(last[-1]) >= 2:
            up_h, up_l = last[1][-1] > last[1][-2], last[-1][-1] > last[-1][-2]
            st[t] = 1 if (up_h and up_l) else -1 if (not up_h and not up_l) else 0
    return _to_bars(pd.Series(st, index=agg.index), d, rule, tf)


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


def filter_inputs(root: Path, sym: str, tf: str, d: pd.DataFrame, atr: np.ndarray) -> dict[str, np.ndarray]:
    """Ряды для признаков «настоящий / ложный пробой» (каждый известен на закрытии своего бара)."""
    c = d["close"].to_numpy(dtype="float64")
    per_day = 24 * 60 // BAR_MIN[tf]
    atr_s = pd.Series(atr)
    cs = pd.Series(c)
    out = {"atr_ratio": (atr_s / atr_s.rolling(100, min_periods=50).mean()).to_numpy(),
           "range10": ((cs.rolling(10).max() - cs.rolling(10).min()) / atr_s).to_numpy(),
           "move10": ((cs - cs.shift(10)) / atr_s).to_numpy(),
           "ret7": (cs / cs.shift(7 * per_day) - 1).to_numpy()}
    f = pd.Series(d["funding"].to_numpy(dtype="float64"))
    out["fund8"] = f.rolling(max(1, 8 * 60 // BAR_MIN[tf]), min_periods=1).sum().to_numpy()
    z = lambda x: ((x - x.rolling(30 * per_day, min_periods=10 * per_day).mean())
                   / x.rolling(30 * per_day, min_periods=10 * per_day).std()).to_numpy()
    if tf != "15m":
        m = metrics_on_bars(sym, root, d.index, BAR_MIN[tf])
        out["lsr_z"] = z(np.log(m["lsr"].replace(0, np.nan)).reset_index(drop=True))
        out["top_z"] = z(np.log(m["top_lsr"].replace(0, np.nan)).reset_index(drop=True))
    else:
        out["lsr_z"] = out["top_z"] = np.full(len(c), np.nan)
    out["spot_aggr"] = out["spot_share_z"] = np.full(len(c), np.nan)
    ps = root / f"{sym}-spot-1h.parquet"
    if tf in ("1h", "4h") and ps.exists():
        sp = pd.read_parquet(ps, columns=["open_time", "quote_volume", "taker_buy_volume", "volume"])
        sp.index = pd.to_datetime(sp["open_time"], unit="ms", utc=True)
        sp = sp[~sp.index.duplicated()].sort_index()
        if tf == "4h":
            sp = sp[["quote_volume", "taker_buy_volume", "volume"]].resample("4h").sum(min_count=1)
        sp = sp.reindex(d.index)
        out["spot_aggr"] = (sp["taker_buy_volume"] / sp["volume"].replace(0, np.nan)).to_numpy(dtype="float64")
        share = np.log(sp["quote_volume"].replace(0, np.nan) / d["quote_volume"].replace(0, np.nan))
        out["spot_share_z"] = z(share.reset_index(drop=True))
    return out


def filter_row(fx: dict[str, np.ndarray], tb: int, e: int, fill: int, side: int, hi: np.ndarray, lo: np.ndarray,
               c: np.ndarray, sw_hi: np.ndarray, sw_lo: np.ndarray, vol_ratio: np.ndarray, buy_share: np.ndarray,
               btc_ret7: np.ndarray) -> dict[str, float]:
    """Признаки на сторону сделки (для лонга как есть, для шорта зеркально):
    atr_ratio   ATR перед пробоем / средний ATR за 100 свечей (< 1 — сжатие)
    range10     диапазон закрытий 10 свечей до пробоя, ATR (меньше — цена прижата к линии)
    approach    ход за 10 свечей до пробоя в сторону пробоя, ATR (плюс — подходит снизу к нисходящей линии)
    hbreak      закрытие входной свечи за последним свингом (горизонтальный уровень пробит вместе с линией)
    crowd_fund  funding за 8 ч против нас (плюс — толпа платит, стоя против пробоя)
    crowd_lsr / crowd_top  z-оценка соотношения лонгов / шортов (всех / топ-трейдеров) против нас
    rs7         доходность монеты за 7 дней минус BTC, в сторону сделки
    spot_aggr   доля агрессоров нашей стороны на споте в свече пробоя; spot_share_z — доля спота в обороте, z
    wait_*      ретест: свечей до исполнения, средний объём и доля агрессоров против нас, пока ждали (без свечи
                исполнения — её объём к моменту входа ещё не известен)"""
    pre = tb - 1
    lvl_i = sw_hi[e] if side > 0 else sw_lo[e]
    lvl = (hi[lvl_i] if side > 0 else lo[lvl_i]) if lvl_i >= 0 else np.nan
    sa = fx["spot_aggr"][tb]
    row = {"atr_ratio": fx["atr_ratio"][pre], "range10": fx["range10"][pre], "approach": side * fx["move10"][pre],
           "hbreak": bool(side * (c[e] - lvl) > 0) if lvl == lvl else False,
           "crowd_fund": -side * fx["fund8"][tb], "crowd_lsr": -side * fx["lsr_z"][tb],
           "crowd_top": -side * fx["top_z"][tb], "rs7": side * (fx["ret7"][e] - btc_ret7[e]),
           "spot_aggr": sa if side > 0 else 1 - sa, "spot_share_z": fx["spot_share_z"][tb],
           "wait_bars": float(fill - e)}
    w = slice(e + 1, fill)
    if fill > e + 1:
        row["wait_vol"] = float(np.nanmean(vol_ratio[w]))
        bs = buy_share[w]
        row["wait_against"] = float(np.nanmean(1 - bs if side > 0 else bs))
    else:
        row["wait_vol"] = row["wait_against"] = np.nan
    return row


def round_step(price: float) -> float:
    """Шаг «круглых» цен: число вида 1 / 2.5 / 5 × 10^k, ближайшее к 0.5% цены (аналог 00 / 50 у Osler 2003)."""
    base = price * 0.005
    k = 10.0 ** np.floor(np.log10(base))
    steps = np.array([1.0, 2.5, 5.0, 10.0]) * k
    return float(steps[np.argmin(np.abs(steps - base))])


def article_outcomes(o, hi, lo, c, f, a, tb: int, e: int, fill: int, side: int, px: float, stop: float, fee: float,
                     hold: int, line_b: float, slope_l: float, ia: int, ib: int) -> dict:
    """Варианты из статей для одной сделки (результат в R; NaN — вариант к сделке неприменим):
    inv_pre — до исполнения ретеста цена закрылась за экстремумом свечи пробоя (Bulkowski: глубокий возврат);
    R3_bx — всё на 3R + выход по закрытию за экстремумом свечи пробоя;
    R3_bs — стоп за свечой пробоя (∓0.1 ATR) вместо свинга, всё на 3R;
    Rtr2 / Rtr3 — стоп за свингом, без цели, трейлинг 2 / 3 ATR от лучшего закрытия;
    Rbs_tr3 — стоп за свечой пробоя + трейлинг 3 ATR (Bulkowski, «Money Management: Stops»);
    Rmm56 / Rmm100 — цель «мерой высоты»: 56% / 100% наибольшего расстояния от линии до цены между второй точкой
    и пробоем, от закрытия свечи пробоя;
    R3_rn — стоп отодвинут за круглое число, если стоял сразу за ним (Osler 2003: там копятся стопы);
    slope_atr — наклон линии, ATR на свечу; inbound_atr — наклон цены за ZZ_ANCHOR свечей до точки A, ATR на свечу."""
    at = a[e]
    out = {"inv_pre": False, "R3_bx": np.nan, "R3_bs": np.nan, "Rtr2": np.nan, "Rtr3": np.nan, "Rbs_tr3": np.nan,
           "Rmm56": np.nan, "Rmm100": np.nan, "R3_rn": np.nan,
           "slope_atr": abs(slope_l) / a[tb] if a[tb] > 0 else np.nan,
           "inbound_atr": (c[ia] - c[max(ia - ZZ_ANCHOR, 0)]) / (ZZ_ANCHOR * a[ia]) if a[ia] > 0 else np.nan}
    brk_ext = lo[tb] if side > 0 else hi[tb]
    if fill > tb + 1:
        out["inv_pre"] = bool(np.any(side * (c[tb + 1: fill] - brk_ext) < 0))
    out["R3_bx"] = target_exit(o, hi, lo, c, f, fill, side, px, stop, 3.0, hold, fee, brk_ext, tb, 0.0, 0.0)[0]
    st_b = brk_ext - side * 0.1 * at
    ok_b = 0.3 * at <= side * (px - st_b) <= 4.0 * at
    if ok_b:
        out["R3_bs"] = two_targets(o, hi, lo, c, f, fill, side, px, st_b, 3.0, 3.0, False, hold, fee)[0]
        out["Rbs_tr3"] = trail_exit(o, hi, lo, c, f, fill, side, px, st_b, at, 3.0, hold, fee)[0]
    out["Rtr2"] = trail_exit(o, hi, lo, c, f, fill, side, px, stop, at, 2.0, hold, fee)[0]
    out["Rtr3"] = trail_exit(o, hi, lo, c, f, fill, side, px, stop, at, 3.0, hold, fee)[0]
    risk = side * (px - stop)
    if ib < tb and risk > 0:
        js = np.arange(ib, tb + 1)
        line = line_b - slope_l * (tb - js)
        far = lo[js] if side > 0 else hi[js]
        height = float(np.max(side * (line - far)))
        for frac, nm in ((0.56, "Rmm56"), (1.0, "Rmm100")):
            k = side * (c[tb] + side * frac * height - px) / risk
            if k >= 1.0:
                out[nm] = two_targets(o, hi, lo, c, f, fill, side, px, stop, k, k, False, hold, fee)[0]
    step = round_step(px)
    rn = np.ceil(stop / step) * step if side > 0 else np.floor(stop / step) * step
    st_rn = stop
    if 0 <= side * (rn - stop) <= 0.25 * at and side * (px - rn) > 0:
        st_rn = rn - side * 0.4 * at
    out["R3_rn"] = two_targets(o, hi, lo, c, f, fill, side, px, st_rn, 3.0, 3.0, False, hold, fee)[0]
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
    trend2 = htf_trend(d, tf, HTF2[tf])
    struct = htf_structure(d, tf)
    ok = np.ones(len(c), bool)
    liq = np.full(len(c), np.nan)
    if tf != "15m":
        liq = adv30(root, sym).reindex(d.index.floor("D")).to_numpy()
        ok = liq >= (ADV_LO if tf == "4h" else ADV_MIN)        # 4h: и неликвиднее порога бота — для корзин
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
    fx = filter_inputs(root, sym, tf, d, a)
    zh, zl = zigzag(c, a, ZZ_K)
    modes = ("zz",) if tf in MORE_TFS else (*LINES, *SCALE_LINES) if tf == "4h" else LINES
    for mode, (tb, tc, side, line_b, line_c, ia, ib) in ((m_, sg) for m_ in modes for sg in signals(d, m_)):
        for confirm in (False,) if (tf in MORE_TFS or mode in SCALE_LINES) else (False, True):
            e = tc if confirm else tb
            if e < 0 or e >= len(c) - 1 or not ok[e] or not (a[e] > 0):
                continue
            if confirm and not liq[e] >= ADV_MIN and tf != "15m":
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
                    r3, _ = two_targets(o, hi, lo, c, f, fill, side, px, stop, 3.0, 3.0, be, HOLD[tf], fee)
                    slope_l = line_c - line_b
                    r3x0, _ = target_exit(o, hi, lo, c, f, fill, side, px, stop, 3.0, HOLD[tf], fee, line_b, tb,
                                          slope_l, 0.0)
                    r3x25, _ = target_exit(o, hi, lo, c, f, fill, side, px, stop, 3.0, HOLD[tf], fee, line_b, tb,
                                           slope_l, 0.25 * a[e])
                    tg = {}
                    if mode == "zz":                                     # другие цели — для отчёта по 1h / 15m
                        for nm, k1, k2, b_ in (("R15", 1.5, 1.5, False), ("R2", 2.0, 2.0, False),
                                               ("R1_2be", 1.0, 2.0, True), ("R15_3be", 1.5, 3.0, True),
                                               ("R2_4be", 2.0, 4.0, True)):
                            tg[nm] = two_targets(o, hi, lo, c, f, fill, side, px, stop, k1, k2, b_, HOLD[tf], fee)[0]
                    if not confirm:                                      # проверки из статей (docs/research_report.md)
                        tg |= article_outcomes(o, hi, lo, c, f, a, tb, e, fill, side, px, stop, fee, HOLD[tf], line_b,
                                               slope_l, int(ia), int(ib))
                    rows.append({"symbol": sym, "tf": tf, "line": mode, "t": d.index[e], "side": side, "confirm": confirm,
                                 "adv": liq[e],
                                 "risk_pct": side * (px - stop) / px,
                                 "entry": entry_kind, "be": be, "with_trend": trend[e] == side, "with_trend2": trend2[e] == side,
                                 "with_struct": struct[e] == side, "R": r, "R3": r3, "R3x0": r3x0, "R3x25": r3x25, **tg,
                                 "vol_ratio": vol_ratio[tb], "oi_chg": oi_chg[tb], "body": body[tb],
                                 "close_loc": loc[tb] if side > 0 else 1 - loc[tb],
                                 "aggr": buy_share[tb] if side > 0 else 1 - buy_share[tb],
                                 "brk_atr": side * (c[tb] - line_b) / a[tb], "rng_atr": (hi[tb] - lo[tb]) / a[tb],
                                 "stop_atr": dist_atr, "ret30": ret30[e], "btc_trend": btc_trend[e],
                                 "btc_ret7": btc_ret7[e], "hold_bars": ex - fill,
                                 **strength_row(st, tb, tc, side, px, stop, zl if side > 0 else zh, c),
                                 **filter_row(fx, tb, e, fill, side, hi, lo, c, sw_hi, sw_lo, vol_ratio, buy_share,
                                              btc_ret7)})
    return pd.DataFrame(rows)


EXAMPLES = [("NEARUSDT", "1h", 300), ("NEARUSDT", "4h", 200), ("SOLUSDT", "15m", 320), ("SOLUSDT", "1h", 360), ("ETHUSDT", "1h", 360), ("DOGEUSDT", "1h", 360),
            ("ETHUSDT", "4h", 260), ("SOLUSDT", "4h", 260), ("AVAXUSDT", "15m", 320), ("LINKUSDT", "1h", 360),
            ("NEARUSDT", "15m", 400), ("ETHUSDT", "15m", 400), ("DOGEUSDT", "15m", 400)]


# окна для сверки с ручной разметкой (NEAR 15m на TradingView: 24.09 вечер — 04.10 ночь, МСК = UTC+3)
WINDOWS = [("NEARUSDT", "15m", "2026-09-24 18:00", "2026-10-04 03:00"),
           ("NEARUSDT", "1h", "2026-09-21 00:00", "2026-10-04 20:00")]


def chart(root: Path, sym: str, tf: str, bars: int, out: Path, mode: str = "fan2", start: str | None = None,
          end: str | None = None) -> None:
    """Свечи последних `bars` баров (или окна start..end), вершины зигзага, линии `mode` с пробоем в окне
    (серым пунктиром — линии по значимым точкам), вход / стоп / цели."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = tf_frame(root, sym, tf, extend=True)
    if d is not None and end is not None:
        d = d[d.index <= pd.Timestamp(end, tz="UTC")]
    if d is not None and start is not None:
        bars = int((d.index >= pd.Timestamp(start, tz="UTC")).sum())
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
    for tb, tc, side, line_b, line_c, i1, i2 in signals(d, "zz" if mode != "zz" else "major"):
        if tb < w0 or i1 < w0 - 200:
            continue
        slope = (c[i2] - c[i1]) / (i2 - i1)
        xs = np.arange(max(i1, w0), tb + 2)
        ax.plot(xs, c[i2] + slope * (xs - i2), color="#9e9e9e", linewidth=0.8, linestyle="--")
    n_tr = 0
    for rec in {"zz": zz_lines, "fan": fan_lines, "major": major_lines,
                "fan2": lambda x: fan_lines(x, FAN2_K), "zone": zone_lines, "zone3": lambda x: zone_lines(x, 3),
                "hl3": lambda x: level_lines(x, 3), "hl4": lambda x: level_lines(x, 4), "s123": s123_lines}[mode](d):
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
    what = {"fan": f"через соседние явные вершины зигзага по закрытиям (разворот >= {' и '.join(f'{k:g}' for k in FAN_K)} ATR)",
            "fan2": f"через соседние явные вершины зигзага по закрытиям (разворот >= {', '.join(f'{k:g}' for k in FAN2_K)} ATR)",
            "zz": f"через вершины зигзага (разворот >= {ZZ_K:g} ATR, опора — экстремум {ZZ_ANCHOR} свечей)",
            "major": f"от главного экстремума ({MAJOR_L} свечей с каждой стороны)"}[mode]
    ax.set_title(f"{sym} {tf}: синие / оранжевые — линии {what}; серый пунктир — для сравнения прежние;\n"
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
    for sym, tf, w0, w1 in (WINDOWS if os.environ.get("TL_CHARTS", "1") == "1" else []):
        if tf in tfs and sym in set(mine(syms15 if tf == "15m" else syms)):
            try:
                chart(root, sym, tf, 0, part_path("tline").parent / "tline_examples" / f"{sym}_{tf}_window.png",
                      start=w0, end=w1)
            except Exception as e:
                print(f"  график {sym} {tf} (окно): {e}", flush=True)
    parts = []
    for tf, lst in (("15m", syms15), ("1h", syms), ("4h", syms), *((t_, syms) for t_ in MORE_TFS)):
        if tf not in tfs:
            continue
        for s in mine(lst):
            try:
                x = coin_trades(root, s, tf, ctx)
                if len(x):
                    parts.append(x)
            except Exception as e:
                print(f"  {tf} {s}: пропуск ({e})", flush=True)
    if "15m" in tfs and "4h" in tfs:
        mt = []
        for s_ in mine(syms15):
            try:
                x = mtf_trades(root, s_)
                if len(x):
                    mt.append(x)
            except Exception as e:
                print(f"  15m-вход {s_}: пропуск ({e})", flush=True)
        if mt:
            pd.concat(mt, ignore_index=True).to_parquet(part_path("tline_mtf"), index=False)
    print(f"  частей монет-ТФ со сделками: {len(parts)}", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path("tline"), index=False)


def diagnose_2026(df: pd.DataFrame, line_name: dict) -> None:
    """Почему шорт на 4h с перевесом продавцов ослаб в 2026: рынок, состояние монеты, исход сделок, помесячно."""
    base = df[(df.tf == "4h") & (df.side == -1) & df.confirm & (df.entry == "market") & (df.aggr >= 0.55)
              & df.line.isin(["clean", "zz", "zzlog", "fan", "fan2"])].copy()
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
    for (gname, mask), ln in itertools.product(groups, ("clean", "zz", "zzlog", "fan", "fan2")):
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


def target_report(df: pd.DataFrame, line_name: dict) -> None:
    """Вся позиция на 3R против лесенки 3R / 5R на одних и тех же сделках. Безубыточная доля прибыльных при 1:3 —
    25% до издержек."""
    print("\n=== Цель: вся позиция на 3R против лесенки (половина 3R, половина 5R); "
          "ячейка — средний R (t, доля прибыльных, сделок в месяц) ===")
    rows = []
    for tf, ln, entry, confirm in itertools.product(("15m", "1h", "4h"), ("clean", "zz", "zzlog", "fan", "fan2"), ("market", "retest"),
                                                    (False, True)):
        g = df[(df.tf == tf) & (df.line == ln) & (df.entry == entry) & (df.confirm == confirm)]
        if not len(g):
            continue
        a55 = g[g.aggr >= 0.55]
        fl = [("агрессоры >= 55%", a55), ("агрессоры >= 55% + тренд", a55[a55.with_trend])]
        if "with_trend2" in g.columns:
            fl += [("агр. + тренд 2 старших ТФ", a55[a55.with_trend & a55.with_trend2]),
                   ("агр. + структура старшего ТФ", a55[a55.with_struct]),
                   ("агр. + структура + тренд 2 ТФ", a55[a55.with_struct & a55.with_trend & a55.with_trend2]),
                   ("тренд 2 старших ТФ, без агрессоров", g[g.with_trend & g.with_trend2])]
        for fname, x in fl:
            for tgt in ("R", "R3"):
                z = x.assign(R=x[tgt])
                rows.append({"ТФ": tf, "линия": line_name[ln], "вход": ("закрепл., " if confirm else "пробой, ") +
                             ("рынок" if entry == "market" else "ретест"), "фильтр": fname,
                             "цель": "3R / 5R" if tgt == "R" else "всё на 3R", **{p: _cell(z[z.per == p]) for p in PER}})
    print(pd.DataFrame(rows).to_string(index=False))
    y = df[(df.tf == "4h") & (df.aggr >= 0.55) & df.with_trend & (df.entry == "retest") & ~df.confirm]
    for ln, g in y.groupby("line"):
        for tgt in ("R", "R3"):
            yy = g.groupby(g.t.dt.year)[tgt].agg(["size", "mean"])
            print(f"  4h, ретест пробоя, агрессоры + тренд, {line_name[ln]}, {'3R / 5R' if tgt == 'R' else 'всё на 3R'}: " +
                  ", ".join(f"{k}: {v['mean']:+.2f}R ({int(v['size'])})" for k, v in yy.iterrows()))


FILTERS = {"atr_ratio": "1 сжатие: ATR / средний за 100", "range10": "1 сжатие: диапазон 10 свечей, ATR",
           "approach": "2 подход к линии: ход 10 свечей, ATR", "crowd_fund": "4 толпа: funding против нас",
           "crowd_lsr": "4 толпа: long/short против нас, z", "crowd_top": "4 толпа: топ-трейдеры против нас, z",
           "oi_chg": "4 толпа: OI за 4 свечи", "spot_aggr": "5 спот: агрессоры нашей стороны",
           "spot_share_z": "5 спот: доля спота в обороте, z", "breadth": "6 ширина: пробоев в ту же сторону − в обратную",
           "rs7": "6 сила к BTC за 7 дней", "wait_bars": "7 ретест: свечей до исполнения",
           "wait_vol": "7 ретест: объём, пока ждали", "wait_against": "7 ретест: агрессоры против нас, пока ждали",
           "cvd20": "сила: дельта за 20 свечей до пробоя"}
# признаки, которые бот может посчитать на закрытии свечи пробоя по данным Binance (без спота, толпы и ожидания ретеста)
LIVE_FEATS = ["aggr", "vol_ratio", "body", "close_loc", "brk_atr", "stop_atr", "ret30", "effort", "cvd20", "atr_ratio",
              "range10", "approach", "rs7", "breadth", "with_trend2"]
MODEL_FEATS = ["aggr", "vol_ratio", "oi_chg", "body", "close_loc", "brk_atr", "stop_atr", "ret30", "effort", "cvd20",
               "atr_ratio", "range10", "approach", "hbreak", "crowd_fund", "crowd_lsr", "crowd_top", "rs7", "spot_aggr",
               "spot_share_z", "breadth", "wait_bars", "wait_vol", "wait_against", "with_trend", "with_trend2"]


def add_breadth(df: pd.DataFrame) -> pd.DataFrame:
    """Ширина рынка на баре пробоя: сколько монет в тот же бар пробили линию («по значимым точкам», «веер» или «чистую»)
    в нашу сторону минус в обратную (без самой монеты)."""
    sig = df[~df.confirm & df.line.isin(["clean", "zz", "zzlog", "fan", "fan2"])][["tf", "t", "symbol", "side"]]
    sig = sig.drop_duplicates()
    cnt = sig.groupby(["tf", "t", "side"]).size().unstack("side", fill_value=0)
    cnt = cnt.reindex(columns=[-1, 1], fill_value=0)
    x = df[["tf", "t", "side"]].join(cnt, on=["tf", "t"])
    up, dn = x[1].fillna(0).to_numpy(), x[-1].fillna(0).to_numpy()
    side = df["side"].to_numpy()
    own = df[["tf", "t", "symbol", "side"]].merge(sig.assign(_own=1.0), how="left",
                                                  on=["tf", "t", "symbol", "side"])["_own"].fillna(0.0).to_numpy()
    return df.assign(breadth=np.where(side > 0, up - own - dn, dn - own - up))


def _years(g: pd.DataFrame, col: str = "R3") -> str:
    yy = g.groupby(g.t.dt.year)[col].agg(["size", "mean"])
    return ", ".join(f"{k}: {v['mean']:+.2f} ({int(v['size'])})" for k, v in yy.iterrows())


LOWTF_TARGETS = (("R3", "всё на 3R"), ("R2", "всё на 2R"), ("R15", "всё на 1.5R"), ("R", "1/2 на 3R + 1/2 на 5R"),
                 ("R1_2be", "1/2 на 1R, стоп в б/у, 1/2 на 2R"), ("R15_3be", "1/2 на 1.5R, б/у, 1/2 на 3R"),
                 ("R2_4be", "1/2 на 2R, б/у, 1/2 на 4R"))


def _per_month(g: pd.DataFrame, col: str) -> str:
    out = []
    for p, (a, b) in PER.items():
        z = g[g.per == p][col].dropna()
        months = (pd.Timestamp(b) - pd.Timestamp(a)).days / 30.4
        out.append(f"{z.sum() / months:+.1f}")
    return " / ".join(out)


def lowtf_report(df: pd.DataFrame) -> None:
    """Как поднять 1h и 15m: база — правило бота (линии по значимым точкам, пробой, агрессоры >= 55% + тренд старшего
    ТФ). Размер стопа (доля комиссий в R), цели и безубыток, совпадение со свежим пробоем 4h в ту же сторону, второй
    старший тренд, время суток. Ячейка — R на сделку; «R/мес» — сумма R в месяц IS / VAL / HO."""
    print("\n=== 1h и 15m: что поднимает результат (база — правило бота) ===")
    base = df[(df.line == "zz") & ~df.confirm & (df.aggr >= 0.55) & df.with_trend & df.tf.isin(["1h", "15m"])].copy()
    if not len(base):
        print("  нет сделок 1h / 15m")
        return
    # пробой 4h известен только после закрытия его свечи: сравниваем время закрытия свечи 4h с закрытием свечи
    # младшего ТФ (раньше бралось открытие 4h — заглядывание до 4 часов вперёд)
    b4 = df[(df.line == "zz") & ~df.confirm & (df.tf == "4h")][["symbol", "side", "t"]].drop_duplicates()
    b4 = b4.assign(t4=(b4.t + pd.Timedelta(minutes=BAR_MIN["4h"])).astype("datetime64[ns, UTC]"))[
        ["symbol", "side", "t4"]].sort_values("t4")
    base = base.assign(tclose=(base.t + pd.to_timedelta(base.tf.map(BAR_MIN), unit="min")).astype(
        "datetime64[ns, UTC]")).sort_values("tclose")
    parts = []
    for (sym, sd), g in base.groupby(["symbol", "side"], sort=False):
        h = b4[(b4.symbol == sym) & (b4.side == sd)]
        if len(h):
            g = pd.merge_asof(g, h[["t4"]], left_on="tclose", right_on="t4", direction="backward")
        else:
            g = g.assign(t4=pd.NaT)
        parts.append(g)
    base = pd.concat(parts, ignore_index=True)
    base["t4"] = pd.to_datetime(base["t4"], utc=True)      # группы без пробоев 4h дают NaT без пояса — приводим
    age_h = (base.tclose - base.t4).dt.total_seconds() / 3600
    hr = base.t.dt.hour
    rules = [("база", lambda g: g)]
    rules += [(f"стоп >= {k:.1f}% цены", lambda g, k=k: g[g.risk_pct >= k / 100]) for k in (0.3, 0.5, 0.8, 1.2)]
    rules += [(f"пробой 4h в ту же сторону, закрылся за {k} ч", lambda g, k=k: g[(age_h.loc[g.index] >= 0) &
                                                                               (age_h.loc[g.index] <= k)])
              for k in (4, 8, 12, 24)]
    rules += [("тренд 2 старших ТФ", lambda g: g[g.with_trend2]),
              ("сессия: Азия 00–07 UTC", lambda g: g[hr.loc[g.index] < 7]),
              ("сессия: Европа 07–13 UTC", lambda g: g[(hr.loc[g.index] >= 7) & (hr.loc[g.index] < 13)]),
              ("сессия: США 13–21 UTC", lambda g: g[(hr.loc[g.index] >= 13) & (hr.loc[g.index] < 21)]),
              ("сессия: 21–24 UTC", lambda g: g[hr.loc[g.index] >= 21]),
              ("стоп >= 0.5% + тренд 2 ТФ", lambda g: g[(g.risk_pct >= 0.005) & g.with_trend2]),
              ("стоп >= 0.5% + пробой 4h за 24 ч", lambda g: g[(g.risk_pct >= 0.005) & (age_h.loc[g.index] >= 0) &
                                                               (age_h.loc[g.index] <= 24)])]
    for tf in ("1h", "15m"):
        for entry in ("retest", "market"):
            g0 = base[(base.tf == tf) & (base.entry == entry)]
            if not len(g0):
                continue
            print(f"\n  --- {tf}, {'ретест' if entry == 'retest' else 'рынок'}: фильтры, всё на 3R ---")
            rows = []
            for nm, f in rules:
                z = f(g0)
                rows.append({"фильтр": nm, **{p: _cell(z[z.per == p].assign(R=z["R3"])) for p in PER},
                             "R/мес IS / VAL / HO": _per_month(z, "R3")})
            print(pd.DataFrame(rows).to_string(index=False))
            print(f"  --- {tf}, {'ретест' if entry == 'retest' else 'рынок'}: цели (все сделки базы / стоп >= 0.5%) ---")
            rows = []
            for col, nm in LOWTF_TARGETS:
                if col not in g0.columns:
                    continue
                for fn_, z in (("все", g0), ("стоп >= 0.5%", g0[g0.risk_pct >= 0.005])):
                    rows.append({"выход": nm, "сделки": fn_, **{p: _cell(z[z.per == p].assign(R=z[col])) for p in PER},
                                 "R/мес IS / VAL / HO": _per_month(z, col)})
            print(pd.DataFrame(rows).to_string(index=False))


def bot_rule_report(df: pd.DataFrame, line_name: dict) -> None:
    """Правило бота (trader/domain/strategy.py) на всех трёх ТФ: обычные линии по значимым точкам, пробой без
    закрепления, агрессоры >= 55%, тренд старшего ТФ, цель — всё на 3R; варианты уверенности свечи и входа."""
    print("\n=== ПРАВИЛО БОТА по таймфреймам: линии по значимым точкам, агрессоры >= 55% + тренд старшего ТФ, "
          "всё на 3R; ячейка — R на сделку (t, прибыльных, сделок в месяц на весь набор монет) ===")
    base0 = df[~df.confirm & (df.aggr >= 0.55) & df.with_trend]
    for ln in LINES:
        if (base0.line == ln).any():
            print(f"\n  --- {line_name.get(ln, ln)} ---")
            _bot_rule_lines(base0[base0.line == ln])


def _bot_rule_lines(base: pd.DataFrame) -> None:
    rules = [("все пробои", lambda g: g),
             ("закрытие в верхней половине (4h по умолчанию)", lambda g: g[g.close_loc >= 0.5]),
             ("закрытие в верхних 20%", lambda g: g[g.close_loc >= 0.8]),
             ("закрытие в верхней трети", lambda g: g[g.close_loc >= 0.67]),
             ("верхние 20% + за линией >= 0.2 ATR", lambda g: g[(g.close_loc >= 0.8) & (g.brk_atr >= 0.2)]),
             ("верхние 20% + за линией >= 0.3 ATR", lambda g: g[(g.close_loc >= 0.8) & (g.brk_atr >= 0.3)])]
    rows = []
    for tf in ("4h", "1h", "15m"):
        for entry in ("retest", "market"):
            g = base[(base.tf == tf) & (base.entry == entry)]
            if not len(g):
                continue
            for nm, f in rules:
                z = f(g).assign(R=lambda x: x["R3"])
                rows.append({"ТФ": tf, "вход": "ретест" if entry == "retest" else "рынок", "свеча": nm,
                             **{p: _cell(z[z.per == p]) for p in PER}})
    print(pd.DataFrame(rows).to_string(index=False))
    for tf in ("4h", "1h", "15m"):
        for entry in ("retest", "market"):
            g = base[(base.tf == tf) & (base.entry == entry)]
            if len(g):
                for nm, f in rules[:2]:
                    print(f"  {tf}, {'ретест' if entry == 'retest' else 'рынок'}, {nm}: {_years(f(g))}")


def article_report(df: pd.DataFrame, line_name: dict) -> None:
    """Проверки из статей (Bulkowski, Osler 2000/2003, Chung & Bellotti 2021) на правиле бота: 4h и 15m, пробой без
    закрепления, агрессоры >= 55% + тренд старшего ТФ, закрытие в верхней половине свечи."""
    if "Rtr3" not in df.columns:
        return
    print("\n=== ИДЕИ ИЗ СТАТЕЙ на правиле бота (агрессоры >= 55% + тренд, закрытие в верхней половине); "
          "ячейка — R на сделку ===")
    base = df[~df.confirm & (df.aggr >= 0.55) & df.with_trend & (df.close_loc >= 0.5)]
    for ln in LINES:
        for tf in ("4h", "15m"):
            for entry in ("retest", "market"):
                g = base[(base.line == ln) & (base.tf == tf) & (base.entry == entry)]
                if not len(g):
                    continue
                rows = []

                def add(nm: str, z: pd.DataFrame, col: str = "R3") -> None:
                    z = z[z[col].notna()].assign(R=lambda x: x[col])
                    rows.append({"вариант": nm, **{p: _cell(z[z.per == p]) for p in PER},
                                 "R/мес IS / VAL / HO": _per_month(z, col)})

                add("база: стоп за свингом, всё на 3R", g)
                if entry == "retest":
                    add("ретест отменён, если до него закрылись за свечой пробоя", g[g.inv_pre.fillna(False).astype(bool) == False])
                add("+ выход по закрытию за свечой пробоя", g, "R3_bx")
                add("стоп за свечой пробоя, 3R", g, "R3_bs")
                add("  то же сделки, стоп за свингом, 3R", g[g.R3_bs.notna()])
                add("стоп за свечой пробоя + трейлинг 3 ATR", g, "Rbs_tr3")
                add("стоп за свингом + трейлинг 2 ATR", g, "Rtr2")
                add("стоп за свингом + трейлинг 3 ATR", g, "Rtr3")
                add("цель мерой высоты 56%", g, "Rmm56")
                add("цель мерой высоты 100%", g, "Rmm100")
                add("  те же сделки, 3R", g[g.Rmm56.notna()])
                add("стоп отодвинут за круглое число", g, "R3_rn")
                for sd, nm in ((1, "лонг"), (-1, "шорт")):
                    x = g[g.side == sd]
                    if len(x) < 30:
                        continue
                    q1, q2 = x.slope_atr.quantile([1 / 3, 2 / 3])
                    add(f"{nm}: пологая линия (нижняя треть наклона)", x[x.slope_atr <= q1])
                    add(f"{nm}: крутая линия (верхняя треть)", x[x.slope_atr > q2])
                    ib = x.inbound_atr.abs()
                    b1, b2 = ib.quantile([1 / 3, 2 / 3])
                    add(f"{nm}: плавный подход к линии (нижняя треть)", x[ib <= b1])
                    add(f"{nm}: крутой подход (верхняя треть)", x[ib > b2])
                print(f"\n  --- {line_name.get(ln, ln)}, {tf}, {'ретест' if entry == 'retest' else 'рынок'} ---")
                print(pd.DataFrame(rows).to_string(index=False))


def _bot_base(df: pd.DataFrame) -> pd.DataFrame:
    """Правило бота: пробой без закрепления, агрессоры >= 55%, тренд старшего ТФ, закрытие в верхней половине свечи."""
    return df[~df.confirm & (df.aggr >= 0.55) & df.with_trend & (df.close_loc >= 0.5)]


def _more_table(items: list[tuple[str, pd.DataFrame]]) -> None:
    rows = []
    for nm, z in items:
        z = z.assign(R=z["R3"])
        rows.append({"вариант": nm, **{p: _cell(z[z.per == p]) for p in PER}, "R/мес IS / VAL / HO": _per_month(z, "R3")})
    print(pd.DataFrame(rows).to_string(index=False))


def more_signals_report(df: pd.DataFrame) -> None:
    """Как получить больше сигналов, не теряя R в месяц: соседние с 4h таймфреймы, другие масштабы зигзага на 4h,
    монеты с оборотом ниже порога бота, фильтры для лонгов. Везде правило бота, всё на 3R."""
    if "adv" not in df.columns:
        return
    base = _bot_base(df)
    liquid = base[base.adv >= ADV_MIN]
    print("\n=== БОЛЬШЕ СИГНАЛОВ: правило бота (агрессоры >= 55% + тренд, закрытие в верхней половине), всё на 3R; "
          "ячейка — R на сделку (t, прибыльных, сделок в месяц); R/мес — сумма R за месяц по всему набору монет ===")
    for entry in ("retest", "market"):
        en = "ретест" if entry == "retest" else "рынок"
        g = liquid[(liquid.entry == entry) & (liquid.line == "zz")]
        print(f"\n  --- 1. таймфреймы, линии по значимым точкам, {en} ---")
        _more_table([(tf, g[g.tf == tf]) for tf in ("2h", "4h", "6h", "12h", "1d") if (g.tf == tf).any()])
        for tf in ("2h", "4h", "6h", "12h", "1d"):
            if (g.tf == tf).any():
                print(f"  {tf} по годам: {_years(g[g.tf == tf])}")

        g4 = liquid[(liquid.entry == entry) & (liquid.tf == "4h")]
        zz = g4[g4.line == "zz"]
        known = set(zip(zz.symbol, zz.t, zz.side))
        items = [("зигзаг 3 ATR (сейчас)", zz)]
        for ln, k in SCALE_LINES.items():
            x = g4[g4.line == ln]
            items += [(f"зигзаг {k:g} ATR", x),
                      ("  из них новые (нет пробоя 3 ATR в ту же свечу)",
                       x[np.array([t not in known for t in zip(x.symbol, x.t, x.side)], dtype=bool)])]
        items.append(("все три масштаба вместе (без повторов)",
                      g4[g4.line.isin(["zz", *SCALE_LINES])].drop_duplicates(["symbol", "t", "side"])))
        items.append(("3 + 5 ATR вместе", g4[g4.line.isin(["zz", "zz5"])].drop_duplicates(["symbol", "t", "side"])))
        print(f"\n  --- 2. несколько масштабов зигзага, 4h, {en} ---")
        _more_table(items)

        b4 = base[(base.entry == entry) & (base.tf == "4h") & (base.line == "zz")]
        print(f"\n  --- 3. ликвидность монеты (средний дневной оборот за 30 дней), 4h, {en} ---")
        _more_table([(f"${lo / 1e6:.0f}M – " + ("∞" if hi == np.inf else f"${hi / 1e6:.0f}M"),
                      b4[(b4.adv >= lo) & (b4.adv < hi)])
                     for lo, hi in ((5e6, 10e6), (10e6, 20e6), (20e6, 50e6), (50e6, 200e6), (200e6, np.inf))]
                    + [("от $10M (порог вдвое ниже)", b4[b4.adv >= 10e6]), ("от $20M (сейчас)", b4[b4.adv >= 20e6])])

        lg = zz[zz.side == 1]
        print(f"\n  --- 4. лонги, 4h, {en} (шорты для сравнения в конце) ---")
        _more_table([("все лонги", lg),
                     ("BTC выше EMA50 дневок", lg[lg.btc_trend == 1]),
                     ("BTC вырос за 7 дней", lg[lg.btc_ret7 > 0]),
                     ("BTC выше EMA50 и вырос за 7 дней", lg[(lg.btc_trend == 1) & (lg.btc_ret7 > 0)]),
                     ("недельный тренд монеты вверх", lg[lg.with_trend2]),
                     ("структура дневок вверх (HH + HL)", lg[lg.with_struct]),
                     ("монета сильнее BTC за 7 дней", lg[lg.rs7 > 0]),
                     ("монета выросла за 30 дней", lg[lg.ret30 > 0]),
                     ("агрессоры >= 58%", lg[lg.aggr >= 0.58]),
                     ("закрытие в верхних 20%", lg[lg.close_loc >= 0.8]),
                     ("BTC выше EMA50 + недельный тренд вверх", lg[(lg.btc_trend == 1) & lg.with_trend2]),
                     ("шорты (как сейчас)", zz[zz.side == -1]),
                     ("шорты, BTC ниже EMA50", zz[(zz.side == -1) & (zz.btc_trend == -1)])])


def conviction_report(df: pd.DataFrame, line_name: dict) -> None:
    """Уверенный пробой: закрытие далеко за линией (ATR), у края свечи, с крупным телом — против пробоев «на чуть-чуть».
    База — сигнал бота: агрессоры >= 55% + тренд старшего ТФ, пробой (без закрепления), цель — всё на 3R."""
    print("\n=== Уверенность свечи пробоя: насколько закрылась за линией (ATR), где закрылась в свече (1 — у края "
          "в сторону пробоя), доля тела; ячейка — R на сделку, всё на 3R ===")
    base = df[(df.aggr >= 0.55) & df.with_trend & ~df.confirm & df.line.isin(["zz", "zzlog"])]
    rules = [("все", lambda g: g)]
    rules += [(f"за линией >= {k:g} ATR", lambda g, k=k: g[g.brk_atr >= k]) for k in (0.1, 0.2, 0.3, 0.5)]
    rules += [(f"закрытие в верхних {round((1 - k) * 100)}% свечи", lambda g, k=k: g[g.close_loc >= k])
              for k in (0.5, 0.67, 0.8)]
    rules += [(f"тело >= {k:.0%}", lambda g, k=k: g[g.body >= k]) for k in (0.5, 0.6)]
    rules += [("за линией < 0.1 ATR (на чуть-чуть)", lambda g: g[g.brk_atr < 0.1]),
              ("закрытие в нижней половине свечи", lambda g: g[g.close_loc < 0.5]),
              ("уверенная: >= 0.2 ATR + верхняя треть", lambda g: g[(g.brk_atr >= 0.2) & (g.close_loc >= 0.67)]),
              ("уверенная: >= 0.2 ATR + верх. треть + тело 50%",
               lambda g: g[(g.brk_atr >= 0.2) & (g.close_loc >= 0.67) & (g.body >= 0.5)])]
    rows = []
    for (tf, ln, entry), g in base.groupby(["tf", "line", "entry"]):
        for nm, f in rules:
            z = f(g).assign(R=lambda x: x["R3"])
            rows.append({"ТФ": tf, "линия": line_name[ln], "вход": "рынок" if entry == "market" else "ретест",
                         "свеча": nm, **{p: _cell(z[z.per == p]) for p in PER}})
    print(pd.DataFrame(rows).to_string(index=False))
    for ln in ("zz", "zzlog"):
        g = base[(base.tf == "4h") & (base.line == ln) & (base.entry == "retest")]
        for nm, f in rules[:1] + [r for r in rules if r[0].startswith("уверенная")]:
            print(f"  4h, ретест, {line_name[ln]}, {nm}: {_years(f(g))}")


def filters_report(df: pd.DataFrame, line_name: dict) -> None:
    """Настоящий или ложный пробой: каждый признак по квинтилям (границы по IS) на базе «4h, ретест пробоя, агрессоры
    >= 55%, тренд 1D», цель — всё на 3R; ранний выход по возврату за линию; модель LightGBM на всех признаках
    (обучение — IS, отбор — верхние 40% прогноза по порогу IS)."""
    df = add_breadth(df)
    for tf in ("4h", "1h", "15m"):
        _filters_tf(df, tf, line_name)


def _filters_tf(df: pd.DataFrame, tf: str, line_name: dict) -> None:
    base_all = df[(df.tf == tf) & (df.entry == "retest") & ~df.confirm]
    if not len(base_all):
        return
    print(f"\n=== Настоящий / ложный пробой: {tf}, ретест пробоя, агрессоры >= 55% + тренд старшего ТФ; "
          f"ячейка — средний R при цели 3R (t, прибыльных, в месяц) ===")
    for ln in ("zz", "zzlog", "fan", "fan2"):
        g = base_all[(base_all.line == ln) & (base_all.aggr >= 0.55) & base_all.with_trend]
        if g.per.eq("is").sum() < 100:
            continue
        print(f"\n  линия: {line_name[ln]}; база: {_cell(g.assign(R=g.R3))}")
        print(f"  по годам: {_years(g)}")
        rows = []
        for f, nm in FILTERS.items():
            if f not in g.columns or g[f].notna().mean() < 0.3:
                continue
            gi = g[g.per == "is"]
            edges = np.unique(gi[f].quantile([0.2, 0.4, 0.6, 0.8]).to_numpy())
            qq = np.where(g[f].notna(), np.digitize(g[f], edges) + 1, 0)
            for q in sorted(set(qq) - {0}):
                gg = g[qq == q].assign(R=lambda x: x.R3)
                rows.append({"признак": nm, "кв.": q, **{p: _cell(gg[gg.per == p]) for p in PER},
                             "2026": _cell(gg[gg.t.dt.year == 2026])})
        for f, nm in (("hbreak", "3 двойной пробой: свинг тоже пробит"),):
            for v in (True, False):
                gg = g[g[f] == v].assign(R=lambda x: x.R3)
                rows.append({"признак": nm, "кв.": "да" if v else "нет", **{p: _cell(gg[gg.per == p]) for p in PER},
                             "2026": _cell(gg[gg.t.dt.year == 2026])})
        print(pd.DataFrame(rows).to_string(index=False))
        print("  8 ранний выход (закрытие обратно за линией), цель 3R:")
        for col, nm in (("R3", "без раннего выхода"), ("R3x0", "выход, если закрытие за линией"),
                        ("R3x25", "выход, если закрытие за линией дальше 0.25 ATR")):
            gg = g.assign(R=g[col])
            print(f"    {nm}: " + ", ".join(f"{p} {_cell(gg[gg.per == p])}" for p in PER) + f"; по годам: {_years(g, col)}")
    try:
        import lightgbm as lgb
    except ImportError:
        print("  lightgbm не установлен — модель пропущена")
        return
    print(f"\n  9 модель LightGBM (все признаки; обучение на IS: {tf}, ретест пробоя, все линии кроме last2, цель 3R):")
    tr_all = base_all[base_all.line.isin(["clean", "zz", "zzlog", "fan", "fan2"])].copy()
    feats = [f for f in MODEL_FEATS if f in tr_all.columns]
    X = tr_all[feats].astype("float64")
    is_m = (tr_all.per == "is").to_numpy()
    y = tr_all["R3"].clip(-1.5, 3.5).to_numpy()
    ok_y = is_m & np.isfinite(y)
    if ok_y.sum() < 1000:
        print("    мало сделок для обучения — модель пропущена")
        return
    params = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 200,
              "bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.8, "lambda_l2": 5.0,
              "verbose": -1, "seed": 7}
    model = lgb.train(params, lgb.Dataset(X[ok_y], y[ok_y]), num_boost_round=300)
    tr_all["score"] = model.predict(X)
    imp = pd.Series(model.feature_importance(), index=feats).sort_values(ascending=False)
    print("    важность: " + ", ".join(f"{k} {v}" for k, v in imp.head(12).items()))
    for ln in ("zz", "zzlog", "fan", "fan2", "clean"):
        g = tr_all[tr_all.line == ln]
        thr = g.loc[g.per == "is", "score"].quantile(0.6)
        for nm, z in (("все пробои линии", g), ("верх 40% модели", g[g.score >= thr]),
                      ("агрессоры >= 55% + тренд", g[(g.aggr >= 0.55) & g.with_trend]),
                      ("агр. + тренд + верх 40% модели", g[(g.aggr >= 0.55) & g.with_trend & (g.score >= thr)])):
            zz = z.assign(R=z.R3)
            print(f"    {line_name[ln]}, {nm}: " + ", ".join(f"{p} {_cell(zz[zz.per == p])}" for p in PER) +
                  f"; по годам: {_years(z)}")
    live_model(tr_all, tf, params, lgb)


def live_model(tr_all: pd.DataFrame, tf: str, params: dict, lgb) -> None:
    """Модель для бота: только LIVE_FEATS, обучение на IS (линии по значимым точкам и соседние, ретест, цель 3R), порог —
    верхние 40% прогноза по IS на базе бота (по значимым точкам, агрессоры >= 55% + тренд). Файл модели и порог —
    в OUT/models для переноса в бота."""
    feats = [f for f in LIVE_FEATS if f in tr_all.columns]
    X = tr_all[feats].astype("float64")
    y = tr_all["R3"].clip(-1.5, 3.5).to_numpy()
    ok_y = (tr_all.per == "is").to_numpy() & np.isfinite(y)
    model = lgb.train(params, lgb.Dataset(X[ok_y], y[ok_y]), num_boost_round=300)
    sc = pd.Series(model.predict(X), index=tr_all.index)
    imp = pd.Series(model.feature_importance(), index=feats).sort_values(ascending=False)
    print(f"\n  9б модель для бота ({len(feats)} признаков, доступных в момент сигнала): важность " +
          ", ".join(f"{k} {v}" for k, v in imp.head(10).items()))
    g = tr_all[(tr_all.line == "zz") & (tr_all.aggr >= 0.55) & tr_all.with_trend].assign(score=sc)
    if not len(g):
        return
    thr = float(g.loc[g.per == "is", "score"].quantile(0.6))
    for nm, z in (("правило бота", g), ("правило бота + верх 40% модели", g[g.score >= thr]),
                  ("правило бота + верх 60% модели", g[g.score >= float(g.loc[g.per == "is", "score"].quantile(0.4))])):
        zz = z.assign(R=z.R3)
        print(f"    {tf}, {nm}: " + ", ".join(f"{p} {_cell(zz[zz.per == p])}" for p in PER) + f"; по годам: {_years(z)}")
    out = Path(os.environ.get("OUT", "../out")) / "models"
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out / f"tline_{tf}.txt"))
    (out / f"tline_{tf}.json").write_text(json.dumps({"tf": tf, "features": feats, "threshold_top40": thr,
                                                      "trained_on": "IS 2022-01..2024-06, retest, R3 clipped"}))


MTF_WINDOW_H = 48       # после сигнала 4h ждём пробой 15m-линии в ту же сторону не дольше 48 ч


def mtf_trades(root: Path, sym: str) -> pd.DataFrame:
    """15m как точка входа по сигналу 4h. Сигнал 4h — пробой линии («по значимым точкам» / «веер») на закрытии
    свечи T; затем первый пробой 15m-линии («веер 2/3/6 ATR» / «по значимым точкам») в ту же сторону в окне
    (T, T + 48 ч]: вход по рынку на закрытии 15m-пробоя или свечи закрепления, стоп за 15m-свингом − 0.1 ATR15.
    Цели: всё на 3R (R по 15m-стопу) и уровень 3R сделки 4h (стоп за 4h-свингом от закрытия 4h-пробоя).
    Для сравнения на тех же сигналах — сделка 4h по рынку на закрытии пробоя, цель 3R."""
    d4, d15 = tf_frame(root, sym, "4h"), tf_frame(root, sym, "15m")
    if d4 is None or d15 is None or len(d4) < 500 or len(d15) < 2000:
        return pd.DataFrame()
    o4, h4, l4, c4 = (d4[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    a4 = _atr(d4).to_numpy()
    bs4 = (d4["taker_buy_volume"] / d4["volume"].replace(0, np.nan)).to_numpy()
    tr4 = htf_trend(d4, "4h")
    slo4, shi4 = last_confirmed(pivots(l4, PIV, False), len(c4)), last_confirmed(pivots(h4, PIV, True), len(c4))
    o, h, lo, c = (d15[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f15 = d15["funding"].to_numpy(dtype="float64")
    a = _atr(d15).to_numpy()
    bs = (d15["taker_buy_volume"] / d15["volume"].replace(0, np.nan)).to_numpy()
    slo, shi = last_confirmed(pivots(lo, PIV, False), len(c)), last_confirmed(pivots(h, PIV, True), len(c))
    close15 = d15.index + pd.Timedelta(minutes=15)
    sig15 = {m: sorted(signals(d15, m)) for m in ("fan2", "zz")}
    tb15 = {m: np.array([x[0] for x in v], dtype=np.int64) for m, v in sig15.items()}
    rows = []
    for m4 in ("zz", "fan"):
        for tb, _, side, *_ in signals(d4, m4):
            if not (a4[tb] > 0):
                continue
            T = d4.index[tb] + pd.Timedelta(hours=4)
            sw4 = slo4[tb] if side > 0 else shi4[tb]
            if sw4 < 0:
                continue
            stop4 = (l4[sw4] - 0.1 * a4[tb]) if side > 0 else (h4[sw4] + 0.1 * a4[tb])
            risk4 = side * (c4[tb] - stop4)
            if not (0.3 <= risk4 / a4[tb] <= 4.0):
                continue
            tp4 = c4[tb] + side * 3 * risk4
            r4, _ = two_targets(o4, h4, l4, c4, d4["funding"].to_numpy(dtype="float64"), tb, side, c4[tb], stop4,
                                3.0, 3.0, False, HOLD["4h"], TAKER)
            aggr4 = bs4[tb] if side > 0 else 1 - bs4[tb]
            i0 = int(close15.searchsorted(T, side="right"))
            i1 = int(close15.searchsorted(T + pd.Timedelta(hours=MTF_WINDOW_H), side="right"))
            for m15 in ("fan2", "zz"):
                arr = tb15[m15]
                j0 = int(np.searchsorted(arr, i0))
                hit = None
                while j0 < len(arr) and arr[j0] < i1:
                    if sig15[m15][j0][2] == side:
                        hit = sig15[m15][j0]
                        break
                    j0 += 1
                if hit is None:
                    continue
                t15, tc15 = hit[0], hit[1]
                for confirm in (False, True):
                    e = tc15 if confirm else t15
                    if e < 0 or e >= len(c) - 1 or not (a[e] > 0):
                        continue
                    sw = slo[e] if side > 0 else shi[e]
                    if sw < 0:
                        continue
                    stop = (lo[sw] - 0.1 * a[e]) if side > 0 else (h[sw] + 0.1 * a[e])
                    risk = side * (c[e] - stop)
                    if not (0.3 <= risk / a[e] <= 4.0) or side * (tp4 - c[e]) <= risk:
                        continue
                    k4 = side * (tp4 - c[e]) / risk
                    r3, _ = two_targets(o, h, lo, c, f15, e, side, c[e], stop, 3.0, 3.0, False, 16 * HOLD["4h"], TAKER)
                    rk, _ = two_targets(o, h, lo, c, f15, e, side, c[e], stop, k4, k4, False, 16 * HOLD["4h"], TAKER)
                    rows.append({"symbol": sym, "t": T, "side": side, "line4": m4, "line15": m15, "confirm": confirm,
                                 "aggr4": aggr4, "with_trend": tr4[tb] == side,
                                 "aggr15": bs[t15] if side > 0 else 1 - bs[t15],
                                 "wait_h": (d15.index[t15] + pd.Timedelta(minutes=15) - T) / pd.Timedelta(hours=1),
                                 "k4": k4, "R4": r4, "R3_15": r3, "Rk_15": rk})
    return pd.DataFrame(rows)


def mtf_report(line_name: dict) -> None:
    parts = all_parts("tline_mtf")
    if not parts:
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["per"] = ""
    for p, (a, b) in PER.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    print(f"\n=== 15m как точка входа по сигналу 4h (70 монет; сигналов {len(df)}) — ячейка: средний R (t, прибыльных, в месяц) ===")
    print("R4 — та же сделка на 4h (рынок на закрытии пробоя, 3R); R3_15 — вход на 15m, всё на 3R по 15m-стопу;"
          " Rk_15 — вход на 15m, цель — уровень 3R сделки 4h (медиана k указана)")
    rows = []
    for (l4, l15, cf), g in df.groupby(["line4", "line15", "confirm"]):
        for fname, x in (("все сигналы 4h", g), ("агрессоры 4h >= 55% + тренд", g[(g.aggr4 >= 0.55) & g.with_trend]),
                         ("+ агрессоры 15m >= 55%", g[(g.aggr4 >= 0.55) & g.with_trend & (g.aggr15 >= 0.55)])):
            for col in ("R4", "R3_15", "Rk_15"):
                z = x.assign(R=x[col])
                rows.append({"4h": line_name[l4], "15m": line_name[l15], "вход 15m": "закрепл." if cf else "пробой",
                             "фильтр": fname, "сделка": col, "k": f"{x.k4.median():.1f}" if col == "Rk_15" else "",
                             **{p: _cell(z[z.per == p]) for p in PER}, "2026": _cell(z[z.t.dt.year == 2026])})
    print(pd.DataFrame(rows).to_string(index=False))


def entry_by_candle_report(df: pd.DataFrame, line_name: dict) -> None:
    """Длинная свеча пробоя: вход по рынку на закрытии пробоя против лимитки на ретесте линии. Сравнение по сигналам:
    у ретеста неисполненная лимитка — 0R (сделки нет), поэтому средний R ретеста считается на сигнал. Гибрид — по
    рынку, если диапазон свечи пробоя < порога (ATR), иначе ретест; порог — квантиль, выбранный на IS."""
    if "rng_atr" not in df.columns:
        return
    key = ["symbol", "tf", "line", "t", "side"]
    print("\n=== Длинная свеча пробоя: рынок или ретест (агрессоры >= 55% + тренд, цель 3R; ретест без исполнения = 0R) ===")
    for tf, ln in itertools.product(("4h", "1h"), ("zz", "zzlog", "fan")):
        g = df[(df.tf == tf) & (df.line == ln) & ~df.confirm & (df.aggr >= 0.55) & df.with_trend]
        mk = g[g.entry == "market"].drop_duplicates(key)
        rt = g[g.entry == "retest"].drop_duplicates(key)[key + ["R3"]].rename(columns={"R3": "R3_rt"})
        x = mk.merge(rt, on=key, how="left")
        if x.per.eq("is").sum() < 100:
            continue
        x["filled"] = x["R3_rt"].notna()
        x["R3_rt0"] = x["R3_rt"].fillna(0.0)
        edges = x.loc[x.per == "is", "rng_atr"].quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
        x["q"] = np.digitize(x["rng_atr"], edges) + 1
        print(f"\n  {tf}, {line_name[ln]}: квинтили диапазона свечи пробоя (ATR), границы IS {np.round(edges, 2).tolist()}")
        rows = []
        for q, gq in x.groupby("q"):
            for nm, col in (("рынок", "R3"), ("ретест", "R3_rt0")):
                z = gq.assign(R=gq[col])
                rows.append({"кв.": q, "вход": nm, "исполнено": f"{gq.filled.mean():.0%}" if col == "R3_rt0" else "100%",
                             **{p: _cell(z[z.per == p]) for p in PER}, "2026": _cell(z[z.t.dt.year == 2026])})
        print(pd.DataFrame(rows).to_string(index=False))
        best = None
        xi = x[x.per == "is"]
        for qq in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            thr = xi["rng_atr"].quantile(qq)
            v = np.where(xi["rng_atr"] < thr, xi["R3"], xi["R3_rt0"]).mean()
            if best is None or v > best[0]:
                best = (v, qq, thr)
        _, qq, thr = best
        hyb = x.assign(R=np.where(x["rng_atr"] < thr, x["R3"], x["R3_rt0"]))
        print(f"  гибрид (рынок, если свеча < {thr:.2f} ATR — квантиль {qq:.0%} по IS, иначе ретест): " +
              ", ".join(f"{p} {_cell(hyb[hyb.per == p])}" for p in PER) + f"; по годам: {_years(hyb, 'R')}")
        for nm, col in (("всё по рынку", "R3"), ("всё ретест", "R3_rt0")):
            z = x.assign(R=x[col])
            print(f"  {nm}: " + ", ".join(f"{p} {_cell(z[z.per == p])}" for p in PER) + f"; по годам: {_years(z, 'R')}")


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
    more = df
    if "adv" in df.columns:                      # прежние отчёты — на прежнем наборе: 15m / 4h, линии бота, порог оборота
        df = df[df.tf.isin(["15m", "1h", "4h"]) & df.line.isin(LINES) & ~(df.adv < ADV_MIN)].reset_index(drop=True)
    out = Path(os.environ.get("OUT", "../out")) / "tline_examples"
    for png in Path(os.environ.get("PARTS", "parts")).rglob("tline_examples/*.png"):
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(png, out / png.name)
    print(f"===== TLINE: сделок {len(df):,}, монет {df.symbol.nunique()}, частей {len(parts)} =====")
    print("ячейка: средний R на сделку (t по дням, прибыльных, сделок в месяц на весь набор монет); выход 1/2 на 3R + 1/2 на 5R")
    line_name = {"last2": "2 последние", "clean": "чистая", "clean3": "чистая, 3 касания", "major": "от главного экстремума", "zz": "по значимым точкам", "zzlog": "по значимым точкам, лог-шкала", "fan": "веер: соседние вершины", "fan2": "веер 2/3/6 ATR", "zone": "наклонная зона 1 ATR", "zone3": "наклонная зона, 3+ касания", "hl3": "горизонтальный уровень, 3+ касания", "hl4": "горизонтальный уровень, 4+ касания", "s123": "линия 1-2-3 Сперандео"}
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
    if "R3" in df.columns:
        target_report(df, line_name)
    if "R3" in df.columns and "brk_atr" in df.columns:
        bot_rule_report(df, line_name)
        more_signals_report(more)
        article_report(df, line_name)
        lowtf_report(df)
        conviction_report(df, line_name)
    if "R3x0" in df.columns:
        filters_report(df, line_name)
    mtf_report(line_name)
    entry_by_candle_report(df, line_name)
    if "effort" in df.columns:
        strength_report(df, line_name)
        ev = df[df.confirm & (df.entry == "market") & (df.tf == "4h") & df.line.isin(["clean", "zz", "zzlog", "fan", "fan2"])]
        if len(ev):
            out = Path(os.environ.get("OUT", "../out"))
            out.mkdir(parents=True, exist_ok=True)
            cols = ["symbol", "tf", "line", "t", "side", "R", "aggr", "with_trend", "per"]
            ev[cols].sort_values("t").to_parquet(out / "tline_breakouts.parquet", index=False)
            print(f"\ntline_breakouts.parquet: {len(ev)} пробоев 4h (закрепление, рынок) — для разметки стаканом")

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
