"""Данные живого графика сигнала: свечи, трендовая линия, уровни входа / стопа / цели."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from ..domain.execution import Trade
from ..domain.models import Signal

AHEAD_BARS = 6              # линию продлеваем на несколько свечей вперёд


def _ts(t: datetime | pd.Timestamp) -> int:
    return int(pd.Timestamp(t).timestamp())


def chart_payload(signal: Signal, bars: pd.DataFrame, trade: Trade | None = None) -> dict:
    """JSON для Lightweight Charts: время — секунды UTC; линия — по двум опорным точкам на закрытиях,
    продлена до текущей свечи и на AHEAD_BARS вперёд."""
    candles = [{"time": _ts(t), "open": r.open, "high": r.high, "low": r.low, "close": r.close}
               for t, r in zip(bars.index, bars.itertuples(index=False))]
    (t1, v1), (t2, v2) = signal.line_points
    s1, s2 = _ts(t1), _ts(t2)
    log = bool(signal.extra.get("log_line"))
    step = signal.timeframe.minutes * 60
    end = (candles[-1]["time"] if candles else _ts(signal.bar_time)) + AHEAD_BARS * step
    start = max(s1, candles[0]["time"]) if candles else s1
    line = []
    if end > start and s2 != s1:
        # точки линии — ровно на свечах (и на будущих свечах): ось времени графика идёт по свечам, точка между
        # свечами добавила бы на ось лишнее деление и изогнула линию
        last = candles[-1]["time"] if candles else _ts(signal.bar_time)
        ts = np.array([c["time"] for c in candles if c["time"] >= start]
                      + [last + k * step for k in range(1, AHEAD_BARS + 1)], dtype=np.int64)
        frac = (ts - s1) / (s2 - s1)
        vals = v1 * (v2 / v1) ** frac if log else v1 + (v2 - v1) * frac
        line = [{"time": int(t), "value": float(v)} for t, v in zip(ts, vals)]
    p = trade or signal.plan
    return {
        "symbol": signal.symbol, "tf": signal.timeframe.value, "side": int(signal.side),
        "active": signal.status.value in ("new", "taken"),             # истёкший / пропущенный — без уровней
        "candles": candles, "line": line, "log": log,
        "breakout": _ts(signal.bar_time),
        "level": signal.extra.get("level_px"),                         # горизонтальный уровень пробоя («+ уровень»)
        "levels": {"entry": p.price if trade else p.entry, "stop": p.stop, "target": p.target},
        "entry_label": "вход" if trade is None else ("лимитка" if trade.kind.value == "retest" else "вход (рынок)"),
    }
