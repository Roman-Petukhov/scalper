"""Сущности торговой панели: таймфреймы, настройки, сигналы и план сделки. Без сети и хранилищ."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import Enum


MAX_LEVERAGE = 10                       # верхняя граница плеча в панели
MARKET_VALID_BARS = 2                   # вход по рынку возможен ещё 2 свечи после свечи пробоя


class Timeframe(str, Enum):
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"

    @property
    def minutes(self) -> int:
        return {"15m": 15, "1h": 60, "4h": 240}[self.value]


class Mode(str, Enum):
    MANUAL = "manual"
    AUTO = "auto"


class Side(int, Enum):
    LONG = 1
    SHORT = -1

    @property
    def label(self) -> str:
        return "лонг" if self is Side.LONG else "шорт"


class EntryKind(str, Enum):
    MARKET = "market"          # по рынку на закрытии свечи пробоя
    RETEST = "retest"          # лимитка на линии (свеча пробоя длинная)


class EntryPolicy(str, Enum):
    """Как входить по сигналу. По бэктесту 4h (docs/research_report.md) ретест лучше на HOLDOUT и в 2026;
    гибрид по длине свечи не лучше входа по рынку."""
    RETEST = "retest"
    MARKET = "market"
    HYBRID = "hybrid"          # свеча пробоя длиннее hybrid_range_atr — ретест, иначе по рынку


class SignalStatus(str, Enum):
    NEW = "new"                # ждёт решения (ручной режим) или исполнения (авто)
    TAKEN = "taken"            # принят: ручной вход или ордер отправлен
    SKIPPED = "skipped"        # пропущен вручную
    EXPIRED = "expired"        # ретест не пришёл за отведённое время


@dataclass(frozen=True)
class Settings:
    mode: Mode = Mode.MANUAL
    timeframes: frozenset[Timeframe] = frozenset({Timeframe.H4})
    risk_pct: float = 1.0               # риск на сделку, % капитала
    leverage: int = 5                   # плечо на бирже и потолок номинала позиции (×капитал); риск задаёт стоп
    max_positions: int = 5
    daily_loss_pct: float = 4.0         # дневной лимит убытка, % капитала: дальше авто не открывает
    min_aggr: float = 0.55              # доля агрессоров в сторону пробоя
    min_break_atr: float = 0.0          # уверенный пробой: закрытие за линией не ближе, чем столько ATR
    min_close_loc: float = 0.0          # и у края свечи в сторону пробоя: 0 — у противоположного, 1 — у края
    target_r: float = 3.0
    entry_policy: EntryPolicy = EntryPolicy.RETEST
    hybrid_range_atr: float = 2.5       # для гибрида: свеча пробоя длиннее (в ATR) — ретест, короче — по рынку
    retest_bars: int = 12               # сколько свечей ждём ретест
    min_turnover_usd: float = 20e6      # оборот монеты за 24 ч

    def __post_init__(self) -> None:
        if not 0.05 <= self.risk_pct <= 5.0:
            raise ValueError("риск на сделку — от 0.05% до 5%")
        if not (isinstance(self.leverage, int) and 1 <= self.leverage <= MAX_LEVERAGE):
            raise ValueError(f"плечо — целое от 1× до {MAX_LEVERAGE}×")
        if not 1 <= self.max_positions <= 50:
            raise ValueError("одновременных позиций — от 1 до 50")
        if not 0.5 <= self.daily_loss_pct <= 30.0:
            raise ValueError("дневной лимит убытка — от 0.5% до 30%")
        if not 0.5 <= self.min_aggr <= 0.8:
            raise ValueError("порог агрессоров — от 50% до 80%")
        if not 0.0 <= self.min_break_atr <= 1.0:
            raise ValueError("закрытие за линией — от 0 до 1 ATR")
        if not 0.0 <= self.min_close_loc <= 0.9:
            raise ValueError("место закрытия в свече — от 0% до 90%")
        if not 1.0 <= self.target_r <= 10.0:
            raise ValueError("цель — от 1R до 10R")
        if not 0.5 <= self.hybrid_range_atr <= 10.0:
            raise ValueError("порог длинной свечи — от 0.5 до 10 ATR")

    def toggle(self, tf: Timeframe) -> Settings:
        tfs = set(self.timeframes)
        tfs.symmetric_difference_update({tf})
        return replace(self, timeframes=frozenset(tfs))


@dataclass(frozen=True)
class TradePlan:
    entry_kind: EntryKind
    entry: float
    stop: float
    target: float
    valid_bars: int                     # для ретеста — сколько свечей живёт лимитка

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry - self.stop)

    @property
    def risk_pct_of_price(self) -> float:
        return self.risk_per_unit / self.entry * 100


@dataclass(frozen=True)
class Signal:
    symbol: str
    timeframe: Timeframe
    side: Side
    bar_time: datetime                  # открытие свечи пробоя (UTC)
    close: float
    line_value: float                   # значение линии на свече пробоя
    line_points: tuple[tuple[datetime, float], tuple[datetime, float]]
    aggr: float                         # доля агрессоров в сторону пробоя
    range_atr: float                    # диапазон свечи пробоя, ATR
    plan: TradePlan
    status: SignalStatus = SignalStatus.NEW
    id: int | None = None
    created_at: datetime | None = None
    chart_path: str | None = None
    note: str = ""
    extra: dict = field(default_factory=dict, compare=False)

    def valid_until(self) -> datetime:
        """До какого момента по сигналу ещё можно входить: вход по рынку — MARKET_VALID_BARS свечи после свечи
        пробоя, ретест — пока живёт лимитка (valid_bars свечей)."""
        bars = 1 + (self.plan.valid_bars if self.plan.valid_bars > 0 else MARKET_VALID_BARS)
        return self.bar_time + timedelta(minutes=self.timeframe.minutes * bars)

    @property
    def key(self) -> tuple[str, str, str, int]:
        return self.symbol, self.timeframe.value, self.bar_time.isoformat(), int(self.side)
