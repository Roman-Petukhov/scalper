"""Правило сигнала — то же, что в исследовании (research.tline, docs/research_report.md):
линия по закрытиям через значимые вершины (зигзаг 3 ATR), пробой — первое закрытие за линией; доля агрессоров
в сторону пробоя >= порога; пробой по тренду старшего ТФ (close последней закрытой свечи старшего ТФ выше /
ниже EMA50); стоп — за последним свингом (фрактал 5 по теням) ∓ 0.1 ATR, 0.3–4 ATR от цены; цель — target_r·R.
Вход — по настройке: лимитка на линии (ретест), по рынку на закрытии свечи пробоя или гибрид (свеча пробоя от
hybrid_range_atr ATR — по рынку: после длинной свечи цена часто не возвращается к линии; короче — ретест).

Построение линий берётся из research.tline (один источник правды с бэктестом); линии — в обычной шкале (лог-вариант — LOG_LINES).
Пометка extra["level"]: закрытие пробоя прошло и горизонтальный уровень (2+ разворота закрытий в 0.5 ATR за 300 свечей) —
не фильтр, а признак для сравнения по журналу (research/oos.py, docs/knowledge.md).
Пометки extra["touch3"] / extra["long_line"] — канон Murphy / Bulkowski (research/tlcanon.py): 3+ касания линии и линия
длиннее медианы бэктеста. В бэктесте 4h лучше остальных в IS / VAL / HO (t 1.7 / 1.8), но не приняты — метки для
журнала, не фильтры. extra["round"] — закрытие пробоя прошло крупный круглый уровень (Osler; research/roundnum.py:
4h лучше в IS / VAL / HO, t 1.6) — метка без надбавки к риску."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.smc import _atr
from research.tline import (LEVEL_PIV, MAKER, PIV, TAKER, crossed_round, htf_trend, last_confirmed, level_hit,
                            line_touches, pivots, retest_fill, two_targets, zz_lines)

from collections.abc import Callable
from datetime import datetime, timedelta

from .models import EntryKind, EntryPolicy, Settings, Side, Signal, Timeframe, TradePlan

REQUIRED = ("open", "high", "low", "close", "volume", "taker_buy_volume")
STOP_ATR = (0.3, 4.0)
LOG_LINES = False           # обычные линии: по бэктесту 4h сильнее лог-шкалы (HOLDOUT +0.40R против +0.32R)
LONG_LINE_BARS = {Timeframe.H4: 126}    # медиана длины линии сделок бота в IS (research/tlcanon.py); 15m — 110 свечей
LONG_LINE_DEFAULT = 110


def _with_probe_bar(d: pd.DataFrame) -> pd.DataFrame:
    """Построитель линий проверяет пробой до предпоследнего бара (ему нужна следующая свеча для «закрепления»).
    Чтобы оценить последнюю закрытую свечу, добавляем копию её закрытия как условную следующую свечу —
    на решение по последней свече она не влияет (используется только для признака закрепления, который здесь
    не нужен)."""
    last = d.iloc[[-1]].copy()
    last.index = last.index + (d.index[-1] - d.index[-2])
    for k in ("open", "high", "low"):
        last[k] = last["close"]
    return pd.concat([d, last])


def detect(d: pd.DataFrame, tf: Timeframe, symbol: str, settings: Settings) -> list[Signal]:
    """Сигналы на последней закрытой свече `d` (индекс — открытие свечи UTC, только закрытые свечи)."""
    missing = [k for k in REQUIRED if k not in d.columns]
    if missing:
        raise ValueError(f"нет колонок {missing}")
    if len(d) < 400:
        return []
    x = _with_probe_bar(d)
    last = len(d) - 1
    recs = [r for r in zz_lines(x, log=LOG_LINES) if r["t"] == last]
    if not recs:
        return []
    return _signals_at(d, _Bars(d, tf), recs, last, tf, symbol, settings)


class _Bars:
    """Массивы свечей и индикаторы для правила — считаются один раз на ряд."""

    def __init__(self, d: pd.DataFrame, tf: Timeframe) -> None:
        self.o, self.hi, self.lo, self.c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
        self.atr = _atr(d).to_numpy()
        self.trend = htf_trend(d, tf.value)
        self.buy = d["taker_buy_volume"].to_numpy(dtype="float64") / np.where(d["volume"] > 0, d["volume"], np.nan)
        self.sw_lo = last_confirmed(pivots(self.lo, PIV, False), len(self.c))
        self.sw_hi = last_confirmed(pivots(self.hi, PIV, True), len(self.c))
        self.lv_hi, self.lv_lo = pivots(self.c, LEVEL_PIV, True), pivots(self.c, LEVEL_PIV, False)


def _signals_at(d: pd.DataFrame, b: _Bars, recs: list[dict], last: int, tf: Timeframe, symbol: str,
                settings: Settings) -> list[Signal]:
    """Сигналы по пробоям `recs` на свече `last` (не больше одного на сторону)."""
    o, hi, lo, c, atr = b.o, b.hi, b.lo, b.c, b.atr
    if not (atr[last] > 0):
        return []
    trend, buy, sw_lo, sw_hi = b.trend, b.buy, b.sw_lo, b.sw_hi
    tp = settings.p(tf)
    out, seen = [], set()
    for r in recs:
        side = Side(r["side"])
        if side in seen or not tp.sides.allows(side):
            continue
        slope_atr = abs(float(r["slope"])) / atr[last]                       # наклон линии, ATR за свечу
        if tp.max_slope_atr > 0 and slope_atr > tp.max_slope_atr:
            continue
        aggr = buy[last] if side is Side.LONG else 1 - buy[last]
        if not (aggr >= tp.min_aggr) or trend[last] != int(side):
            continue
        brk = int(side) * (c[last] - float(r["line_t"])) / atr[last]          # насколько закрылась за линией
        rng = hi[last] - lo[last]
        loc = ((c[last] - lo[last]) if side is Side.LONG else (hi[last] - c[last])) / rng if rng > 0 else 0.0
        if brk < tp.min_break_atr or loc < tp.min_close_loc:
            continue                                                           # пробой «на чуть-чуть»
        sw = sw_lo[last] if side is Side.LONG else sw_hi[last]
        if sw < 0:
            continue
        stop = lo[sw] - 0.1 * atr[last] if side is Side.LONG else hi[sw] + 0.1 * atr[last]
        range_atr = (hi[last] - lo[last]) / atr[last]
        market = (tp.entry_policy is EntryPolicy.MARKET
                  or (tp.entry_policy is EntryPolicy.HYBRID and range_atr >= tp.hybrid_range_atr))
        if market:
            kind, entry = EntryKind.MARKET, c[last]
        else:
            kind, entry = EntryKind.RETEST, float(r["line_t"])
        risk = int(side) * (entry - stop)
        if not (STOP_ATR[0] <= risk / atr[last] <= STOP_ATR[1]):
            continue
        plan = TradePlan(kind, float(entry), float(stop), float(entry + int(side) * tp.target_r * risk),
                         tp.retest_bars if kind is EntryKind.RETEST else 0)
        a, bb = r["a"], r["b"]
        lv = level_hit(c, atr, last, int(side), b.lv_hi, b.lv_lo)
        y = np.log(c) if LOG_LINES else c
        touches = line_touches(y, atr, a, last, int(side), float(y[a]), float(r["slope"]))
        line_bars = last - a
        out.append(Signal(symbol=symbol, timeframe=tf, side=side, bar_time=d.index[last].to_pydatetime(),
                          close=float(c[last]), line_value=float(r["line_t"]),
                          line_points=((d.index[a].to_pydatetime(), float(c[a])), (d.index[bb].to_pydatetime(), float(c[bb]))),
                          aggr=float(aggr), range_atr=float(range_atr), plan=plan,
                          extra={"log_line": LOG_LINES, "break_atr": round(float(brk), 3), "close_loc": round(float(loc), 3),
                                 "slope_atr": round(float(slope_atr), 4),
                                 "level": lv is not None,
                                 "touches": touches, "line_bars": int(line_bars), "touch3": touches >= 3,
                                 "long_line": line_bars > LONG_LINE_BARS.get(tf, LONG_LINE_DEFAULT),
                                 "round": bool(c[last - 1] > 0 and crossed_round(float(c[last - 1]), float(c[last]))),
                                 **({"level_px": lv[0], "level_n": lv[1]} if lv is not None else {})}))
        seen.add(side)
    return out


def replay(d: pd.DataFrame, tf: Timeframe, symbol: str, settings: Settings, since: datetime,
           keep: Callable[[Signal], bool] | None = None) -> list[tuple[datetime, Side, float]]:
    """Правило бота задним числом: все сигналы со свечой не раньше `since` и их итог в R по свечам `d` — вход по плану
    (ретест — лимитка на линии, пока не истекла и цена не ушла к цели), стоп раньше цели в той же свече, срок
    max_hold_bars, комиссии как в бэктесте, без funding и доливки. Сделки, ещё открытые на последней свече, и
    неисполненные лимитки не попадают. `keep` — дополнительный отбор сигналов (подтверждение старшего ТФ)."""
    if len(d) < 400 or any(k not in d.columns for k in REQUIRED):
        return []
    x = _with_probe_bar(d)
    n = len(d)
    by_t: dict[int, list[dict]] = {}
    for r in zz_lines(x, log=LOG_LINES):
        if 0 < r["t"] < n - 1 and d.index[r["t"]] >= since:
            by_t.setdefault(r["t"], []).append(r)
    if not by_t:
        return []
    b = _Bars(d, tf)
    tp = settings.p(tf)
    fund = np.zeros(n)
    out = []
    for t in sorted(by_t):
        for sig in _signals_at(d, b, by_t[t], t, tf, symbol, settings):
            if keep is not None and not keep(sig):
                continue
            side, plan = int(sig.side), sig.plan
            risk = side * (plan.entry - plan.stop)
            if plan.entry_kind is EntryKind.MARKET:
                fill, fee = t, TAKER
            else:
                fill, fee = retest_fill(b.hi, b.lo, t, side, plan.entry, tp.target_r * risk, plan.valid_bars), MAKER
                if fill < 0:
                    continue
                if (side > 0 and b.lo[fill] <= plan.stop) or (side < 0 and b.hi[fill] >= plan.stop):
                    out.append((sig.bar_time, sig.side, (side * (plan.stop - plan.entry) - (fee + TAKER) * plan.entry) / risk))
                    continue
            r, ex = two_targets(b.o, b.hi, b.lo, b.c, fund, fill, side, plan.entry, plan.stop, tp.target_r, tp.target_r,
                                False, tp.max_hold_bars, fee)
            if not np.isfinite(r) or (ex >= n - 1 and ex - fill < tp.max_hold_bars):
                continue                                                   # ещё открыта
            out.append((sig.bar_time, sig.side, float(r)))
    return out


def htf_breakouts(d: pd.DataFrame, tf: Timeframe) -> list[tuple[datetime, Side]]:
    """Пробои линий старшего ТФ (то же построение, что в detect, без фильтров свечи): время закрытия свечи пробоя и
    сторона. `d` — только закрытые свечи."""
    if len(d) < 400:
        return []
    x = _with_probe_bar(d)
    step = timedelta(minutes=tf.minutes)
    return [((d.index[r["t"]] + step).to_pydatetime(), Side(r["side"])) for r in zz_lines(x, log=LOG_LINES)
            if 0 < r["t"] < len(d)]


def htf_confirmation(breakouts: list[tuple[datetime, Side]], side: Side, at: datetime, hours: int) -> float | None:
    """Сколько часов назад закрылась свеча последнего пробоя старшего ТФ в сторону `side` (не позже `at` — закрытия
    свечи сигнала и не раньше `hours` часов до него); None — такого пробоя нет."""
    ages = [(at - t).total_seconds() / 3600 for t, sd in breakouts if sd is side and t <= at]
    ages = [h for h in ages if h <= hours]
    return min(ages) if ages else None
