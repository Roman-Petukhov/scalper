"""
Бэктест с лимитными входами (maker) для частых интрадей-стратегий на барах.

Модель исполнения (консервативная):
- сигнал на закрытии бара t выставляет лимит по цене L, ордер живёт ttl баров (t+1 .. t+ttl);
- лонг-лимит исполнен в баре j, только если low[j] < L * (1 - through): цена прошла СКВОЗЬ уровень,
  а не коснулась его (иначе мы могли стоять в конце очереди и не получить исполнение);
- цена входа = L (стратегии ставят лонг-лимит не выше текущей цены, шорт-лимит — не ниже);
- в баре исполнения тейк-профит не проверяется, а стоп проверяется (худший порядок событий внутри бара);
- если в одном баре достижимы и стоп, и тейк — считается стоп;
- стоп исполняется рыночно: по уровню стопа или по open, если бар открылся с гэпом за стопом;
- тейк-профит — лимит (maker), исполняется только при проходе сквозь уровень;
- выход по таймеру — рыночный по close.
Комиссии: maker_bps на лимитные исполнения, taker_bps на рыночные (включая проскальзывание).
Funding: позиция, открытая на начало бара с выплатой, платит pos * rate.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def limit_machine(side, limit_px, tp_px, sl_px, ttl, max_hold, open_, high, low, close, funding,
                  maker_bps, taker_bps, through):
    """side[t] in {-1,0,1}: желаемое направление по сигналу на закрытии t; limit_px/tp_px/sl_px — уровни
    (NaN tp/sl = не используется). Возвращает доходность по барам (на капитал, позиция 1x),
    позицию на закрытии бара, число сделок и массив причин выхода (1 tp, 2 sl, 3 time)."""
    n = len(close)
    ret = np.zeros(n)
    pos_end = np.zeros(n)
    exit_kind = np.zeros(n, np.int8)
    trades = 0
    pend_side = 0.0
    pend_px = 0.0
    pend_tp = np.nan
    pend_sl = np.nan
    pend_left = 0
    cur = 0.0
    entry = 0.0
    tp = np.nan
    sl = np.nan
    held = 0
    mk = maker_bps * 1e-4
    tk = taker_bps * 1e-4
    for t in range(n):
        # funding за удержание позиции с прошлого закрытия
        if t > 0 and pos_end[t - 1] != 0.0:
            ret[t] -= pos_end[t - 1] * funding[t]
        if cur != 0.0:
            held += 1
            px_prev = close[t - 1]
            done = False
            # стоп (рыночный)
            if not np.isnan(sl):
                if cur > 0 and low[t] <= sl:
                    xp = min(sl, open_[t])
                    ret[t] += (xp / px_prev - 1.0) - tk
                    exit_kind[t] = 2
                    done = True
                elif cur < 0 and high[t] >= sl:
                    xp = max(sl, open_[t])
                    ret[t] += -(xp / px_prev - 1.0) - tk
                    exit_kind[t] = 2
                    done = True
            # тейк (лимит, проход сквозь уровень)
            if not done and not np.isnan(tp):
                if cur > 0 and high[t] > tp * (1.0 + through):
                    ret[t] += (tp / px_prev - 1.0) - mk
                    exit_kind[t] = 1
                    done = True
                elif cur < 0 and low[t] < tp * (1.0 - through):
                    ret[t] += -(tp / px_prev - 1.0) - mk
                    exit_kind[t] = 1
                    done = True
            # таймер (рыночный по close)
            if not done and held >= max_hold:
                ret[t] += cur * (close[t] / px_prev - 1.0) - tk
                exit_kind[t] = 3
                done = True
            if done:
                cur = 0.0
            else:
                ret[t] += cur * (close[t] / px_prev - 1.0)
        elif pend_side != 0.0:
            filled = False
            if pend_side > 0 and low[t] < pend_px * (1.0 - through):
                filled = True
            elif pend_side < 0 and high[t] > pend_px * (1.0 + through):
                filled = True
            if filled:
                ep = pend_px
                cur = pend_side
                entry = ep
                tp = pend_tp
                sl = pend_sl
                held = 0
                trades += 1
                pend_side = 0.0
                # в баре входа: сначала возможный стоп (худший случай), иначе переоценка до close
                hit_sl = False
                if not np.isnan(sl):
                    if cur > 0 and low[t] <= sl:
                        hit_sl = True
                        ret[t] += (min(sl, ep) / ep - 1.0) - mk - tk
                    elif cur < 0 and high[t] >= sl:
                        hit_sl = True
                        ret[t] += -(max(sl, ep) / ep - 1.0) - mk - tk
                if hit_sl:
                    exit_kind[t] = 2
                    cur = 0.0
                else:
                    ret[t] += cur * (close[t] / ep - 1.0) - mk
            else:
                pend_left -= 1
                if pend_left <= 0:
                    pend_side = 0.0
        # новый сигнал выставляет лимит, если нет позиции и нет активного ордера
        if cur == 0.0 and pend_side == 0.0 and side[t] != 0 and not np.isnan(limit_px[t]):
            pend_side = float(side[t])
            pend_px = limit_px[t]
            pend_tp = tp_px[t]
            pend_sl = sl_px[t]
            pend_left = ttl
        pos_end[t] = cur
    return ret, pos_end, trades, exit_kind
