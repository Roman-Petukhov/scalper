"""Теневые лонги: сигналы, которые бот НЕ торгует, но ведёт в журнале, чтобы проверить вперёд то, что на истории
не прошло или прошло без плато (docs/knowledge.md, «Чем заменить агрессоров для лонгов 4h», research/longfilt.py).

AGGR525 — лонговый пробой линии 4h со всеми фильтрами бота, кроме порога агрессоров: доля покупок от 52.5% (у бота —
55%). Бэктест (998 лонгов): добавка 52.5–55% — 123 сделки, +8.5R, IS +0.07 / VAL +0.32 / HO +0.14 / 2026 +0.89R; соседние
пороги (50% — IS −0.23R, 55% — IS −0.16R) плато не дают, 2026 тянет режим года. Решение — по живым данным.

Итог теневого сигнала считается по правилу бота (стоп, цель target_r·R, срок max_hold_bars, лимитка ретеста со сроком)
по закрытым свечам, без funding и доливки — как research/strategy.replay."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from research.tline import MAKER, TAKER, retest_fill, two_targets

from .models import EntryKind, Settings, Side, SideFilter, Signal, Timeframe
from .strategy import detect

AGGR525 = "aggr525"
SHADOW_KINDS = {AGGR525: "Лонг 4h, агрессоры 52.5–55%"}
SHADOW_AGGR = {Timeframe.H4: 0.525}
BACKTEST_NOTE = {AGGR525: "бэктест: IS +0.07 / VAL +0.32 / HO +0.14R на сделку, на реале ждём около половины"}
MIN_CLOSED = 20             # раньше этого числа закрытых теневых сделок среднее ничего не говорит


def candidates(d: pd.DataFrame, tf: Timeframe, symbol: str, settings: Settings) -> list[tuple[str, Signal]]:
    """Теневые лонги на последней закрытой свече: те, которых бот при текущих настройках не покажет."""
    thr = SHADOW_AGGR.get(tf)
    if thr is None:
        return []
    wide = detect(d, tf, symbol, settings.with_tf(tf, min_aggr=thr, sides=SideFilter.LONG))
    if not wide:
        return []
    real = {x.bar_time for x in detect(d, tf, symbol, settings) if x.side is Side.LONG}
    return [(AGGR525, x) for x in wide if x.side is Side.LONG and x.bar_time not in real]


@dataclass(frozen=True)
class Outcome:
    state: str                  # "closed" — сделка закрыта, "unfilled" — лимитка не исполнилась
    r: float | None
    closed_at: datetime


def resolve(sig: Signal, bars: pd.DataFrame, settings: Settings, now: datetime) -> Outcome | None:
    """Итог теневого сигнала по закрытым свечам `bars` (с открытия свечи сигнала); None — ещё не решён."""
    tp = settings.p(sig.timeframe)
    step = timedelta(minutes=sig.timeframe.minutes)
    d = bars[bars.index >= pd.Timestamp(sig.bar_time)]
    if len(d) == 0 or d.index[0].to_pydatetime() != sig.bar_time:
        return None
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    n, side, plan = len(d), int(sig.side), sig.plan
    risk = side * (plan.entry - plan.stop)
    if not risk > 0:
        return Outcome("unfilled", None, sig.bar_time + step)
    if plan.entry_kind is EntryKind.MARKET:
        fill, fee = 0, TAKER
    else:
        fill, fee = retest_fill(hi, lo, 0, side, plan.entry, tp.target_r * risk, plan.valid_bars), MAKER
        if fill < 0:
            return Outcome("unfilled", None, sig.valid_until()) if now >= sig.valid_until() else None
        if (side > 0 and lo[fill] <= plan.stop) or (side < 0 and hi[fill] >= plan.stop):
            r = (side * (plan.stop - plan.entry) - (fee + TAKER) * plan.entry) / risk
            return Outcome("closed", float(r), d.index[fill].to_pydatetime() + step)
    r, ex = two_targets(o, hi, lo, c, np.zeros(n), fill, side, plan.entry, plan.stop, tp.target_r, tp.target_r, False,
                        tp.max_hold_bars, fee)
    if not np.isfinite(r) or (ex >= n - 1 and ex - fill < tp.max_hold_bars):
        return None                                                  # ещё открыта
    return Outcome("closed", float(r), d.index[ex].to_pydatetime() + step)


@dataclass(frozen=True)
class ShadowRow:
    kind: str
    signal: Signal
    state: str                  # "open" | "closed" | "unfilled"
    r: float | None = None
    closed_at: datetime | None = None


@dataclass(frozen=True)
class ShadowStats:
    kind: str
    label: str
    total: int
    open: int
    unfilled: int
    closed: int
    avg_r: float | None
    sum_r: float
    win_share: float | None
    spring_closed: int          # из закрытых — после выноса минимума
    spring_avg_r: float | None
    note: str

    @property
    def enough(self) -> bool:
        return self.closed >= MIN_CLOSED

    @property
    def need(self) -> int:
        return max(0, MIN_CLOSED - self.closed)


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def shadow_stats(rows: list[ShadowRow]) -> list[ShadowStats]:
    out = []
    for kind, label in SHADOW_KINDS.items():
        own = [r for r in rows if r.kind == kind]
        done = [r for r in own if r.state == "closed" and r.r is not None]
        rs = [r.r for r in done]
        sp = [r.r for r in done if r.signal.spring]
        out.append(ShadowStats(kind, label, len(own), sum(r.state == "open" for r in own),
                               sum(r.state == "unfilled" for r in own), len(done), _mean(rs), sum(rs),
                               _mean([float(x > 0) for x in rs]), len(sp), _mean(sp), BACKTEST_NOTE[kind]))
    return out
