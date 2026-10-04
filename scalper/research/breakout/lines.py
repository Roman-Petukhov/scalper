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


@njit(cache=True)
def retest_barrier(idx, side, line_val, slope, open_, high, low, close, atr, wait, tp_mult, sl_mult, max_hold,
                   maker_bps, taker_bps, through):
    """Вход лимитом на ретесте пробитой линии: после пробоя на баре t ставим заявку по цене линии
    (линия продолжается со своим наклоном) на wait баров. Исполнение — только если цена прошла СКВОЗЬ уровень.
    Стоп sl_mult*ATR за уровнем входа (рыночный), тейк tp_mult*ATR (лимит), иначе выход по close через max_hold.
    В баре исполнения тейк не проверяется, стоп — проверяется (худший порядок).
    Возвращает чистую доходность (NaN — не исполнилось) и число баров от пробоя до выхода/снятия заявки."""
    k = len(idx)
    ret = np.full(k, np.nan)
    bars = np.zeros(k, np.int32)
    m = len(close)
    mk, tk = maker_bps * 1e-4, taker_bps * 1e-4
    for e in range(k):
        t = idx[e]
        s = side[e]
        a = atr[t]
        bars[e] = min(wait, m - 1 - t)
        if not (a > 0) or not np.isfinite(line_val[e]):
            continue
        fill_j = -1
        entry = 0.0
        for j in range(t + 1, min(t + wait, m - 1) + 1):
            lv = line_val[e] + slope[e] * (j - t)
            if (s > 0 and low[j] < lv * (1 - through)) or (s < 0 and high[j] > lv * (1 + through)):
                fill_j = j
                entry = lv
                break
        if fill_j < 0:
            continue
        tp = entry + s * tp_mult * a
        sl = entry - s * sl_mult * a
        if (s > 0 and low[fill_j] <= sl) or (s < 0 and high[fill_j] >= sl):
            ret[e] = s * (sl / entry - 1.0) - mk - tk
            bars[e] = fill_j - t
            continue
        exit_px = np.nan
        fee = tk
        j = fill_j + 1
        end = min(fill_j + max_hold, m - 1)
        while j <= end:
            if s > 0:
                if low[j] <= sl:
                    exit_px = min(sl, open_[j])
                    break
                if high[j] > tp * (1 + through):
                    exit_px = tp
                    fee = mk
                    break
            else:
                if high[j] >= sl:
                    exit_px = max(sl, open_[j])
                    break
                if low[j] < tp * (1 - through):
                    exit_px = tp
                    fee = mk
                    break
            j += 1
        if np.isnan(exit_px):
            j = end
            exit_px = close[end]
        ret[e] = s * (exit_px / entry - 1.0) - mk - fee
        bars[e] = j - t
    return ret, bars


@njit(cache=True)
def luxalgo_breaks(high, low, close, atr, length, mult, touch_tol=0.25):
    """Метод «Trendlines with Breaks» (идея — LuxAlgo, CC BY-NC-SA 4.0; здесь собственная реализация метода, не кода):
    от каждой подтверждённой вершины (pivot high, подтверждение через length баров) линия опускается с наклоном
    ATR(length)/length*mult за бар, от впадины — поднимается. Пробой — первое закрытие за линией после её построения.
    Отличие от оригинала: до первой вершины линии нет (в оригинале upper=0 даёт ложные ранние сигналы).
    Возвращает то же, что trend_breaks: event, feats (LINE_FEATS), текущие значения верхней и нижней линий."""
    m = len(close)
    event = np.zeros(m, np.int8)
    feats = np.full((m, 6), np.nan)
    up_line = np.full(m, np.nan)
    lo_line = np.full(m, np.nan)
    upper = np.nan
    lower = np.nan
    slope_ph = 0.0
    slope_pl = 0.0
    ph_bar, pl_bar = -1, -1
    upos, dnos = 0, 0
    up_touch, dn_touch, up_last, dn_last = 0, 0, -10, -10
    for t in range(m):
        c = t - length
        ph = c >= length and _is_pivot(high, c, length, True)
        pl = c >= length and _is_pivot(low, c, length, False)
        slope = atr[t] / length * mult
        if ph:
            slope_ph = slope
            upper = high[c]
            ph_bar = c
            upos = 0
            up_touch, up_last = 0, -10
        elif not np.isnan(upper):
            upper -= slope_ph
        if pl:
            slope_pl = slope
            lower = low[c]
            pl_bar = c
            dnos = 0
            dn_touch, dn_last = 0, -10
        elif not np.isnan(lower):
            lower += slope_pl
        a = atr[t]
        up_ev = False
        dn_ev = False
        if not np.isnan(upper):
            uv = upper - slope_ph * length            # значение линии на текущем баре
            up_line[t] = uv
            if not ph and upos == 0:
                if close[t] > uv:
                    upos = 1
                    up_ev = True
                elif a > 0 and high[t] >= uv - touch_tol * a and t - up_last > 2:
                    up_touch += 1
                    up_last = t
        if not np.isnan(lower):
            lv = lower + slope_pl * length
            lo_line[t] = lv
            if not pl and dnos == 0:
                if close[t] < lv:
                    dnos = 1
                    dn_ev = True
                elif a > 0 and low[t] <= lv + touch_tol * a and t - dn_last > 2:
                    dn_touch += 1
                    dn_last = t
        if up_ev and not dn_ev and a > 0:
            event[t] = 1
            feats[t, 0] = -slope_ph / a
            feats[t, 1] = length
            feats[t, 2] = t - ph_bar
            feats[t, 3] = up_touch
            feats[t, 4] = (close[t] - up_line[t]) / a
            feats[t, 5] = (high[ph_bar] - up_line[t]) / a
        elif dn_ev and not up_ev and a > 0:
            event[t] = -1
            feats[t, 0] = slope_pl / a
            feats[t, 1] = length
            feats[t, 2] = t - pl_bar
            feats[t, 3] = dn_touch
            feats[t, 4] = (lo_line[t] - close[t]) / a
            feats[t, 5] = (lo_line[t] - low[pl_bar]) / a
    return event, feats, up_line, lo_line
