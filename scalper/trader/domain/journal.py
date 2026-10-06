"""Журнал сделок: итоги на бирже против бэктеста по каждому таймфрейму. Чистая логика без хранилищ."""
from __future__ import annotations

from dataclasses import dataclass

from .execution import Trade, TradeStatus
from .models import EntryKind, Timeframe

# R на сделку по бэктесту на последнем периоде (HOLDOUT, 2025.07–2026.09), правило бота по умолчанию:
# 4h — ретест, research/tline.py (followup_report); 15m — пологие шорты, топ-150, research/wide15.py
BACKTEST_R: dict[Timeframe, float] = {Timeframe.H4: 0.45, Timeframe.M15: 0.08}
MIN_TRADES = 30             # раньше сравнивать с бэктестом бессмысленно: разброс одной сделки — несколько R


@dataclass(frozen=True)
class JournalEntry:
    trade: Trade
    timeframe: Timeframe


@dataclass(frozen=True)
class TfStats:
    timeframe: Timeframe
    closed: int                 # сделок с известным итогом
    avg_r: float | None
    sum_r: float
    win_share: float | None
    avg_slippage_r: float | None
    retest_orders: int          # лимиток ретеста с решённой судьбой (исполнилась или снята)
    retest_fill: float | None   # доля исполнившихся
    backtest_r: float | None

    @property
    def enough(self) -> bool:
        return self.closed >= MIN_TRADES

    @property
    def need(self) -> int:
        return max(0, MIN_TRADES - self.closed)


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def tf_stats(entries: list[JournalEntry]) -> list[TfStats]:
    out = []
    for tf in Timeframe:
        own = [e.trade for e in entries if e.timeframe is tf]
        rs = [r for r in (t.r_multiple for t in own) if r is not None]
        slips = [s for s in (t.slippage_r for t in own) if s is not None]
        retest = [t for t in own if t.kind is EntryKind.RETEST and t.status is not TradeStatus.PLACED]
        filled = [t for t in retest if t.filled_at is not None or t.status in
                  (TradeStatus.FILLED, TradeStatus.CLOSED, TradeStatus.TIMED_OUT)]
        out.append(TfStats(tf, len(rs), _mean(rs), sum(rs), _mean([float(r > 0) for r in rs]), _mean(slips),
                           len(retest), len(filled) / len(retest) if retest else None, BACKTEST_R.get(tf)))
    return out
