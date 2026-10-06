"""Журнал сделок: итоги на бирже против бэктеста по каждому таймфрейму. Чистая логика без хранилищ."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .execution import Trade, TradeStatus
from .models import EntryKind, Timeframe

# R на сделку по бэктесту на последнем периоде (HOLDOUT, 2025.07–2026.09), правило бота по умолчанию:
# 4h — ретест, research/tline.py (followup_report); 15m — пологие шорты, топ-150, research/wide15.py
BACKTEST_R: dict[Timeframe, float] = {Timeframe.H4: 0.45, Timeframe.M15: 0.08}
# худшая просадка бэктеста в R (4h, 2020–2026, research/hedged.py): чтобы показать, во что она обходится при риске
BACKTEST_DD_R: dict[Timeframe, float] = {Timeframe.H4: 13.3}
MIN_TRADES = 30             # раньше сравнивать с бэктестом бессмысленно: разброс одной сделки — несколько R
# Детектор «стратегия перестала работать». Норма на реале — половина бэктеста (docs/knowledge.md: вне выборки
# остаётся 50–57%). Сделка моделируется как стоп (−1R) или цель (+target R) с такой вероятностью, чтобы среднее было
# этой половиной; по 4000 таким последовательностям той же длины видно, насколько плохими бывают среднее и просадка
# просто от невезения. Хуже, чем в 90% случаев — «присмотреться», хуже, чем в 97.5% — «остановить».
EXPECT_SHARE = 0.5
HEALTH_MIN_TRADES = 20
HEALTH_SIMS = 4000
WATCH_Q, STOP_Q = 0.90, 0.975


class Health(str, Enum):
    EARLY = "early"             # сделок мало для вывода
    OK = "ok"                   # в пределах обычного невезения
    WATCH = "watch"             # хуже, чем в 90% случаев
    STOP = "stop"               # хуже, чем в 97.5% случаев: стратегия, похоже, не работает


@dataclass(frozen=True)
class HealthCheck:
    level: Health
    trades: int
    expected_r: float           # ожидание на реале, R на сделку
    live_r: float | None
    worst_mean_r: float | None  # граница «присмотреться» для среднего
    drawdown_r: float | None    # текущая худшая просадка, R
    drawdown_limit_r: float | None   # граница «присмотреться» для просадки

    @property
    def text(self) -> str:
        if self.level is Health.EARLY:
            return f"Оценка — после {HEALTH_MIN_TRADES} сделок (сейчас {self.trades})."
        base = (f"Ожидание на реале {self.expected_r:+.2f}R (половина бэктеста). Обычное невезение: среднее не ниже "
                f"{self.worst_mean_r:+.2f}R, просадка не глубже {self.drawdown_limit_r:.1f}R.")
        return {Health.OK: "В пределах нормы. ", Health.WATCH: "Хуже, чем в 90% случаев — присмотреться. ",
                Health.STOP: "Хуже, чем в 97.5% случаев: похоже, стратегия не работает — стоит выключить авто. "}[
            self.level] + base


def _max_dd(eq: np.ndarray) -> np.ndarray:
    """Худшая просадка накопленного R по строкам."""
    z = np.concatenate([np.zeros((eq.shape[0], 1)), eq], axis=1)
    return (np.maximum.accumulate(z, axis=1) - z).max(axis=1)


def health(rs: list[float], backtest_r: float, target_r: float) -> HealthCheck:
    """rs — итоги закрытых сделок в R по порядку закрытия."""
    n = len(rs)
    mu = EXPECT_SHARE * backtest_r
    if n < HEALTH_MIN_TRADES:
        return HealthCheck(Health.EARLY, n, mu, float(np.mean(rs)) if rs else None, None, None, None)
    p = min(max((mu + 1.0) / (target_r + 1.0), 0.0), 1.0)       # доля целей при стопе −1R и цели +target R
    rng = np.random.default_rng(n)
    sims = np.where(rng.random((HEALTH_SIMS, n)) < p, target_r, -1.0)
    means, dds = sims.mean(axis=1), _max_dd(np.cumsum(sims, axis=1))
    live = np.asarray(rs, dtype="float64")
    live_mean, live_dd = float(live.mean()), float(_max_dd(np.cumsum(live)[None, :])[0])
    worst_share = max(float((means > live_mean).mean()), float((dds < live_dd).mean()))   # хуже скольких случаев
    level = Health.STOP if worst_share >= STOP_Q else Health.WATCH if worst_share >= WATCH_Q else Health.OK
    return HealthCheck(level, n, mu, live_mean, float(np.quantile(means, 1 - WATCH_Q)), live_dd,
                       float(np.quantile(dds, WATCH_Q)))


@dataclass(frozen=True)
class JournalEntry:
    trade: Trade
    timeframe: Timeframe
    at_level: bool = False      # сигнал с пометкой «+ уровень» (Signal.at_level)


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
    health: HealthCheck | None = None
    level_closed: int = 0       # из них с пометкой «+ уровень»
    level_avg_r: float | None = None
    plain_avg_r: float | None = None
    hedged_closed: int = 0      # из них с хеджем BTC (оценка итога хеджа известна)
    hedged_trade_r: float | None = None     # сами сделки
    hedged_total_r: float | None = None     # сделка + хедж

    @property
    def enough(self) -> bool:
        return self.closed >= MIN_TRADES

    @property
    def need(self) -> int:
        return max(0, MIN_TRADES - self.closed)


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _closed(entries: list[JournalEntry], tf: Timeframe) -> list[JournalEntry]:
    done = [e for e in entries if e.timeframe is tf and e.trade.result is not None and e.trade.r_multiple is not None]
    return sorted(done, key=lambda e: e.trade.result.closed_at)


def closed_rs(entries: list[JournalEntry], tf: Timeframe) -> list[float]:
    """Итоги закрытых сделок ТФ в R по порядку закрытия."""
    return [e.trade.r_multiple for e in _closed(entries, tf)]


def tf_stats(entries: list[JournalEntry], target_r: dict[Timeframe, float] | None = None) -> list[TfStats]:
    out = []
    for tf in Timeframe:
        own = [e.trade for e in entries if e.timeframe is tf]
        done = _closed(entries, tf)
        rs = [e.trade.r_multiple for e in done]
        lv = [e.trade.r_multiple for e in done if e.at_level]
        slips = [s for s in (t.slippage_r for t in own) if s is not None]
        retest = [t for t in own if t.kind is EntryKind.RETEST and t.status is not TradeStatus.PLACED]
        filled = [t for t in retest if t.filled_at is not None or t.status in
                  (TradeStatus.FILLED, TradeStatus.CLOSED, TradeStatus.TIMED_OUT)]
        hd = [e.trade for e in done if e.trade.hedge_r is not None]
        bt = BACKTEST_R.get(tf)
        hc = health(rs, bt, (target_r or {}).get(tf, 3.0)) if bt is not None else None
        out.append(TfStats(tf, len(rs), _mean(rs), sum(rs), _mean([float(r > 0) for r in rs]), _mean(slips),
                           len(retest), len(filled) / len(retest) if retest else None, bt, hc,
                           len(lv), _mean(lv), _mean([e.trade.r_multiple for e in done if not e.at_level]),
                           len(hd), _mean([t.r_multiple for t in hd]), _mean([t.r_multiple + t.hedge_r for t in hd])))
    return out
