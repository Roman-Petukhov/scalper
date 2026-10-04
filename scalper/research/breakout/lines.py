"""
Линии тренда и их пробои — алгоритмически и без заглядывания в будущее.

Разворотная точка (фрактал) масштаба n: high[i] — максимум окна [i-n, i+n]. Она становится ИЗВЕСТНОЙ только на
баре i+n (как и человеку: вершину видно, когда цена уже ушла от неё). Линия сопротивления проходит через две
последние известные вершины, поддержки — через две последние известные впадины (любой наклон).
Пробой: закрытие бара за линией (впервые с момента её построения). Сопротивление пробито вверх -> лонг,
поддержка вниз -> шорт. Каждая линия даёт не больше одного события.
"""
from __future__ import annotations

import numpy as np
from numba import njit

# признаки линии в момент пробоя (порядок колонок в выходной матрице)
LINE_FEATS = ("slope_atr", "span", "age", "touches", "dist_atr", "pivot_gap_atr")


@njit(cache=True)
def _is_pivot(x, i, n, high):
    lo, hi = i - n, i + n
    if lo < 0 or hi >= len(x):
        return False
    v = x[i]
    for k in range(lo, hi + 1):
        if k == i:
            continue
        if high:
            if (k < i and x[k] > v) or (k > i and x[k] >= v):
                return False
        else:
            if (k < i and x[k] < v) or (k > i and x[k] <= v):
                return False
    return True


@njit(cache=True)
def trend_breaks(high, low, close, atr, n, touch_tol=0.25):
    """Возвращает:
    event[t] in {-1,0,1} — пробой поддержки (шорт) / сопротивления (лонг) на закрытии бара t;
    feats[t, :] — признаки пробитой линии (LINE_FEATS), NaN где события нет;
    res_line[t], sup_line[t] — значение актуальной (ещё не пробитой) линии на баре t, NaN если её нет."""
    m = len(close)
    event = np.zeros(m, np.int8)
    feats = np.full((m, 6), np.nan)
    res_line = np.full(m, np.nan)
    sup_line = np.full(m, np.nan)
    # сопротивление
    r_i1, r_h1, r_i2, r_h2 = -1, 0.0, -1, 0.0
    r_ok, r_touch, r_last_touch = False, 0, -10
    # поддержка
    s_i1, s_l1, s_i2, s_l2 = -1, 0.0, -1, 0.0
    s_ok, s_touch, s_last_touch = False, 0, -10
    for t in range(m):
        c = t - n                        # кандидат в разворот, подтверждаемый на баре t
        if c >= n and _is_pivot(high, c, n, True):
            r_i1, r_h1, r_i2, r_h2 = r_i2, r_h2, c, high[c]
            r_ok = r_i1 >= 0
            r_touch, r_last_touch = 0, -10
            if r_ok:                     # если закрытия уже были выше новой линии — она недействительна
                sl = (r_h2 - r_h1) / (r_i2 - r_i1)
                for k in range(r_i2 + 1, t):
                    if close[k] > r_h1 + sl * (k - r_i1):
                        r_ok = False
                        break
        if c >= n and _is_pivot(low, c, n, False):
            s_i1, s_l1, s_i2, s_l2 = s_i2, s_l2, c, low[c]
            s_ok = s_i1 >= 0
            s_touch, s_last_touch = 0, -10
            if s_ok:
                sl = (s_l2 - s_l1) / (s_i2 - s_i1)
                for k in range(s_i2 + 1, t):
                    if close[k] < s_l1 + sl * (k - s_i1):
                        s_ok = False
                        break
        a = atr[t]
        if r_ok and a > 0:
            sl = (r_h2 - r_h1) / (r_i2 - r_i1)
            lv = r_h1 + sl * (t - r_i1)
            res_line[t] = lv
            if close[t] > lv:
                event[t] = 1
                feats[t, 0] = sl / a
                feats[t, 1] = r_i2 - r_i1
                feats[t, 2] = t - r_i2
                feats[t, 3] = r_touch
                feats[t, 4] = (close[t] - lv) / a
                feats[t, 5] = (r_h1 - r_h2) / a
                r_ok = False
            elif high[t] >= lv - touch_tol * a and t - r_last_touch > 2:
                r_touch += 1
                r_last_touch = t
        if s_ok and a > 0:
            sl = (s_l2 - s_l1) / (s_i2 - s_i1)
            lv = s_l1 + sl * (t - s_i1)
            sup_line[t] = lv
            if close[t] < lv:
                if event[t] == 0:        # одновременный пробой обеих линий (редко) — берём сопротивление
                    event[t] = -1
                    feats[t, 0] = sl / a
                    feats[t, 1] = s_i2 - s_i1
                    feats[t, 2] = t - s_i2
                    feats[t, 3] = s_touch
                    feats[t, 4] = (lv - close[t]) / a
                    feats[t, 5] = (s_l2 - s_l1) / a
                s_ok = False
            elif low[t] <= lv + touch_tol * a and t - s_last_touch > 2:
                s_touch += 1
                s_last_touch = t
    return event, feats, res_line, sup_line


@njit(cache=True)
def triple_barrier(idx, side, open_, high, low, close, atr, tp_mult, sl_mult, max_hold, cost_bps):
    """Вход по close[idx] (рыночный), тейк tp_mult*ATR, стоп sl_mult*ATR, иначе выход по close через max_hold баров.
    Если в одном баре достижимы стоп и тейк — считаем стоп. Возвращает чистую доходность (доли) и длительность."""
    k = len(idx)
    ret = np.full(k, np.nan)
    bars = np.zeros(k, np.int32)
    m = len(close)
    for e in range(k):
        t = idx[e]
        s = side[e]
        a = atr[t]
        if t + 1 >= m or not (a > 0):
            continue
        entry = close[t]
        tp = entry + s * tp_mult * a
        sl = entry - s * sl_mult * a
        exit_px = np.nan
        j = t + 1
        end = min(t + max_hold, m - 1)
        while j <= end:
            if s > 0:
                if low[j] <= sl:
                    exit_px = min(sl, open_[j])          # гэп через стоп — исполнение хуже уровня
                    break
                if high[j] >= tp:
                    exit_px = tp
                    break
            else:
                if high[j] >= sl:
                    exit_px = max(sl, open_[j])
                    break
                if low[j] <= tp:
                    exit_px = tp
                    break
            j += 1
        if np.isnan(exit_px):
            j = end
            exit_px = close[end]
        ret[e] = s * (exit_px / entry - 1.0) - 2 * cost_bps * 1e-4
        bars[e] = j - t
    return ret, bars
