"""Правило сигнала — то же, что в исследовании (research.tline, docs/research_report.md):
линия по закрытиям через значимые вершины (зигзаг 3 ATR), пробой — первое закрытие за линией; доля агрессоров
в сторону пробоя >= порога; пробой по тренду старшего ТФ (close последней закрытой свечи старшего ТФ выше /
ниже EMA50); стоп — за последним свингом (фрактал 5 по теням) ∓ 0.1 ATR, 0.3–4 ATR от цены; цель — target_r·R.
Вход — по настройке: лимитка на линии (ретест, по умолчанию), по рынку на закрытии свечи пробоя или гибрид
(свеча пробоя короче hybrid_range_atr ATR — по рынку, длиннее — ретест).

Построение линий берётся из research.tline (один источник правды с бэктестом); линии — на лог-шкале."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.smc import _atr
from research.tline import PIV, htf_trend, last_confirmed, pivots, zz_lines

from .models import EntryKind, EntryPolicy, Settings, Side, Signal, Timeframe, TradePlan

REQUIRED = ("open", "high", "low", "close", "volume", "taker_buy_volume")
STOP_ATR = (0.3, 4.0)
LOG_LINES = True            # линии прямые на логарифмической шкале (как трейдер ведёт их на лог-графике)


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
    out, seen = [], set()
    for r in recs:
        side = Side(r["side"])
        if side in seen:
            continue
        aggr = buy[last] if side is Side.LONG else 1 - buy[last]
        if not (aggr >= settings.min_aggr) or trend[last] != int(side):
            continue
        sw = sw_lo[last] if side is Side.LONG else sw_hi[last]
        if sw < 0:
            continue
        stop = lo[sw] - 0.1 * atr[last] if side is Side.LONG else hi[sw] + 0.1 * atr[last]
        range_atr = (hi[last] - lo[last]) / atr[last]
        market = (settings.entry_policy is EntryPolicy.MARKET
                  or (settings.entry_policy is EntryPolicy.HYBRID and range_atr < settings.hybrid_range_atr))
        if market:
            kind, entry = EntryKind.MARKET, c[last]
        else:
            kind, entry = EntryKind.RETEST, float(r["line_t"])
        risk = int(side) * (entry - stop)
        if not (STOP_ATR[0] <= risk / atr[last] <= STOP_ATR[1]):
            continue
        plan = TradePlan(kind, float(entry), float(stop), float(entry + int(side) * settings.target_r * risk),
                         settings.retest_bars if kind is EntryKind.RETEST else 0)
        a, b = r["a"], r["b"]
        out.append(Signal(symbol=symbol, timeframe=tf, side=side, bar_time=d.index[last].to_pydatetime(),
                          close=float(c[last]), line_value=float(r["line_t"]),
                          line_points=((d.index[a].to_pydatetime(), float(c[a])), (d.index[b].to_pydatetime(), float(c[b]))),
                          aggr=float(aggr), range_atr=float(range_atr), plan=plan, extra={"log_line": LOG_LINES}))
        seen.add(side)
    return out
