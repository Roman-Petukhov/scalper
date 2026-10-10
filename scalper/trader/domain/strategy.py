"""Правило сигнала — то же, что в исследовании (research.tline, docs/research_report.md):
линия по закрытиям через значимые вершины (зигзаг 3 ATR), пробой — первое закрытие за линией; доля агрессоров
в сторону пробоя >= порога; пробой по тренду старшего ТФ (close последней закрытой свечи старшего ТФ выше /
ниже EMA50); стоп — за последним свингом (фрактал 5 по теням) ∓ 0.1 ATR, 0.3–4 ATR от цены; цель — target_r·R.
Вход — по настройке: лимитка на линии (ретест), по рынку на закрытии свечи пробоя или гибрид (свеча пробоя от
hybrid_range_atr ATR — по рынку: после длинной свечи цена часто не возвращается к линии; короче — ретест).

Построение линий берётся из research.tline (один источник правды с бэктестом); линии — в обычной шкале (лог-вариант — LOG_LINES).
Пометка extra["level"]: закрытие пробоя прошло и горизонтальный уровень (2+ разворота закрытий в 0.5 ATR за 300 свечей) —
не фильтр, а признак для сравнения по журналу (research/oos.py, docs/knowledge.md)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.smc import _atr
from research.tline import LEVEL_PIV, PIV, htf_trend, last_confirmed, level_hit, pivots, zz_lines

from datetime import datetime, timedelta

from .models import EntryKind, EntryPolicy, Settings, Side, Signal, Timeframe, TradePlan

REQUIRED = ("open", "high", "low", "close", "volume", "taker_buy_volume")
STOP_ATR = (0.3, 4.0)
LOG_LINES = False           # обычные линии: по бэктесту 4h сильнее лог-шкалы (HOLDOUT +0.40R против +0.32R)


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
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    atr = _atr(d).to_numpy()
    if not (atr[last] > 0):
        return []
    trend = htf_trend(d, tf.value)
    buy = d["taker_buy_volume"].to_numpy(dtype="float64") / np.where(d["volume"] > 0, d["volume"], np.nan)
    sw_lo = last_confirmed(pivots(lo, PIV, False), len(c))
    sw_hi = last_confirmed(pivots(hi, PIV, True), len(c))
    lv_hi, lv_lo = pivots(c, LEVEL_PIV, True), pivots(c, LEVEL_PIV, False)
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
        a, b = r["a"], r["b"]
        lv = level_hit(c, atr, last, int(side), lv_hi, lv_lo)
        out.append(Signal(symbol=symbol, timeframe=tf, side=side, bar_time=d.index[last].to_pydatetime(),
                          close=float(c[last]), line_value=float(r["line_t"]),
                          line_points=((d.index[a].to_pydatetime(), float(c[a])), (d.index[b].to_pydatetime(), float(c[b]))),
                          aggr=float(aggr), range_atr=float(range_atr), plan=plan,
                          extra={"log_line": LOG_LINES, "break_atr": round(float(brk), 3), "close_loc": round(float(loc), 3),
                                 "slope_atr": round(float(slope_atr), 4),
                                 "level": lv is not None,
                                 **({"level_px": lv[0], "level_n": lv[1]} if lv is not None else {})}))
        seen.add(side)
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
