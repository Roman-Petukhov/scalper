"""Данные графика сигнала: свечи, трендовая линия, уровни входа / стопа / цели.

Живой график (сигнал ждёт входа или сделка открыта) — последние свечи с текущей, обновляется. Архивный (сигнал истёк,
пропущен или сделка закрыта) — застывшее окно вокруг сигнала (начало линии … выход из сделки, research-выгрузка
trader/application/export.py), чтобы линия, пробой, вход и выход оставались на своих свечах."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from ..domain.execution import Trade
from ..domain.models import Signal

AHEAD_BARS = 6              # линию продлеваем на несколько свечей вперёд


def _ts(t: datetime | pd.Timestamp) -> int:
    return int(pd.Timestamp(t).timestamp())


def _on_bar(t: datetime | None, step: int) -> int | None:
    """Время события — на открытие свечи, в которой оно случилось (маркер ставится на свечу)."""
    return None if t is None else _ts(t) // step * step


def chart_payload(signal: Signal, bars: pd.DataFrame, trade: Trade | None = None, frozen: bool = False) -> dict:
    """JSON для Lightweight Charts: время — секунды UTC; линия — по двум опорным точкам на закрытиях,
    продлена до текущей свечи и на AHEAD_BARS вперёд (в архиве — только до последней свечи окна)."""
    candles = [{"time": _ts(t), "open": r.open, "high": r.high, "low": r.low, "close": r.close}
               for t, r in zip(bars.index, bars.itertuples(index=False))]
    (t1, v1), (t2, v2) = signal.line_points
    s1, s2 = _ts(t1), _ts(t2)
    log = bool(signal.extra.get("log_line"))
    step = signal.timeframe.minutes * 60
    ahead = 0 if frozen else AHEAD_BARS
    end = (candles[-1]["time"] if candles else _ts(signal.bar_time)) + ahead * step
    start = max(s1, candles[0]["time"]) if candles else s1
    line = []
    if end >= start and s2 != s1:
        # точки линии — ровно на свечах (и на будущих свечах): ось времени графика идёт по свечам, точка между
        # свечами добавила бы на ось лишнее деление и изогнула линию
        last = candles[-1]["time"] if candles else _ts(signal.bar_time)
        ts = np.array([c["time"] for c in candles if c["time"] >= start]
                      + [last + k * step for k in range(1, ahead + 1)], dtype=np.int64)
        frac = (ts - s1) / (s2 - s1)
        vals = v1 * (v2 / v1) ** frac if log else v1 + (v2 - v1) * frac
        line = [{"time": int(t), "value": float(v)} for t, v in zip(ts, vals)]
    p = trade or signal.plan
    marks = []
    if trade is not None and trade.filled_at is not None:
        marks.append({"time": _on_bar(trade.filled_at, step), "kind": "entry",
                      "price": trade.result.entry_fill if trade.result else trade.price})
    if trade is not None and trade.result is not None:
        marks.append({"time": _on_bar(trade.result.closed_at, step), "kind": "exit", "price": trade.result.exit})
    return {
        "symbol": signal.symbol, "tf": signal.timeframe.value, "side": int(signal.side),
        "active": signal.status.value in ("new", "taken"),             # истёкший / пропущенный — без уровней
        "frozen": frozen,
        "show_levels": signal.status.value in ("new", "taken") or (frozen and trade is not None
                                                                   and trade.filled_at is not None),
        "marks": [m for m in marks if candles and candles[0]["time"] <= m["time"] <= candles[-1]["time"]],
        "candles": candles, "line": line, "log": log,
        "breakout": _ts(signal.bar_time),
        "level": signal.extra.get("level_px"),                         # горизонтальный уровень пробоя («+ уровень»)
        "levels": {"entry": p.price if trade else p.entry, "stop": p.stop, "target": p.target},
        "entry_label": "вход" if trade is None else ("лимитка" if trade.kind.value == "retest" else "вход (рынок)"),
    }
