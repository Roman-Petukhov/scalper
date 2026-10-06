"""Сущности торговой панели: таймфреймы, настройки, сигналы и план сделки. Без сети и хранилищ."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import Enum


MAX_LEVERAGE = 10                       # верхняя граница плеча в панели
MARKET_VALID_BARS = 2                   # вход по рынку возможен ещё 2 свечи после свечи пробоя


class Timeframe(str, Enum):
    """Таймфреймы панели. 1h убран по решению трейдера (на бэктесте — лишь тонкий плюс)."""
    M15 = "15m"
    H4 = "4h"

    @property
    def minutes(self) -> int:
        return {"15m": 15, "4h": 240}[self.value]


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


class SideFilter(str, Enum):
    """Какие стороны торговать на таймфрейме."""
    BOTH = "both"
    LONG = "long"
    SHORT = "short"

    def allows(self, side: Side) -> bool:
        return self is SideFilter.BOTH or (self is SideFilter.LONG) == (side is Side.LONG)


class SignalStatus(str, Enum):
    NEW = "new"                # ждёт решения (ручной режим) или исполнения (авто)
    TAKEN = "taken"            # принят: ручной вход или ордер отправлен
    SKIPPED = "skipped"        # пропущен вручную
    EXPIRED = "expired"        # ретест не пришёл за отведённое время
    CLOSED = "closed"          # сделка по сигналу завершена: стоп, цель, срок или снята на бирже


@dataclass(frozen=True)
class TfParams:
    """Правило входа и риск для одного таймфрейма."""
    risk_pct: float = 1.0               # риск на сделку, % капитала
    entry_policy: EntryPolicy = EntryPolicy.RETEST
    min_aggr: float = 0.55              # доля агрессоров в сторону пробоя
    target_r: float = 3.0
    min_close_loc: float = 0.0          # закрытие у края свечи в сторону пробоя: 0 — фильтра нет, 1 — у самого края
    min_break_atr: float = 0.0          # закрытие за линией не ближе, чем столько ATR
    hybrid_range_atr: float = 2.5       # гибрид: свеча пробоя длиннее (в ATR) — ретест, короче — по рынку
    retest_bars: int = 12               # сколько свечей ждём ретест
    htf_confirm_h: int = 0              # только если старший ТФ (HTF_CONFIRM) пробил линию в ту же сторону не раньше
                                        # стольких часов назад (по закрытию его свечи); 0 — без этого условия
    sides: SideFilter = SideFilter.BOTH
    max_slope_atr: float = 0.0          # линия не круче стольких ATR за свечу (пологие линии надёжнее); 0 — любая
    top_n: int = 0                      # только столько самых ликвидных монет (по обороту за 24 ч); 0 — все с порогом
    max_hold_bars: int = 60             # позиция закрывается по рынку через столько свечей (как в бэктесте)
    max_positions: int = 0              # не больше стольких позиций этого ТФ сразу (внутри общего лимита); 0 — только общий

    def __post_init__(self) -> None:
        if not 0.05 <= self.risk_pct <= 5.0:
            raise ValueError("риск на сделку — от 0.05% до 5%")
        if not 0.5 <= self.min_aggr <= 0.8:
            raise ValueError("порог агрессоров — от 50% до 80%")
        if not 1.0 <= self.target_r <= 10.0:
            raise ValueError("цель — от 1R до 10R")
        if not 0.0 <= self.min_close_loc <= 0.9:
            raise ValueError("место закрытия в свече — от 0% до 90%")
        if not 0.0 <= self.min_break_atr <= 1.0:
            raise ValueError("закрытие за линией — от 0 до 1 ATR")
        if not 0.5 <= self.hybrid_range_atr <= 10.0:
            raise ValueError("порог длинной свечи — от 0.5 до 10 ATR")
        if not (isinstance(self.retest_bars, int) and 1 <= self.retest_bars <= 48):
            raise ValueError("ожидание ретеста — от 1 до 48 свечей")
        if not (isinstance(self.htf_confirm_h, int) and 0 <= self.htf_confirm_h <= 48):
            raise ValueError("окно пробоя старшего ТФ — от 0 до 48 часов")
        if not 0.0 <= self.max_slope_atr <= 1.0:
            raise ValueError("наклон линии — от 0 до 1 ATR за свечу")
        if not (isinstance(self.top_n, int) and 0 <= self.top_n <= 1000):
            raise ValueError("число монет — от 0 (все) до 1000")
        if not (isinstance(self.max_hold_bars, int) and 1 <= self.max_hold_bars <= 1000):
            raise ValueError("срок сделки — от 1 до 1000 свечей")
        if not (isinstance(self.max_positions, int) and 0 <= self.max_positions <= 50):
            raise ValueError("позиций таймфрейма — от 0 (только общий лимит) до 50")


# Лучшее по бэктесту (docs/research_report.md): 4h — основная стратегия (ретест, закрытие в верхней половине свечи);
# 15m: обычный пробой в ноль (и после свежего пробоя 4h тоже), шорт от пологой линии — тонкий плюс: порог наклона
# 0.025 ATR/свечу — нижняя треть по IS (2022–2024.06), вход по рынку. На 725 монетах с местом в рейтинге оборота на день
# сигнала (research/wide15.py) топ-150 лучше топ-70 (HO +0.08R против +0.02R на сделку), но сигналов ~100 в месяц и
# держатся до 2 суток — свой лимит 3 позиции, чтобы 15m не занимал места 4h. Риск 1% / 0.25%.
DEFAULT_TF_PARAMS: dict[Timeframe, TfParams] = {
    Timeframe.H4: TfParams(risk_pct=1.0, min_close_loc=0.5),
    Timeframe.M15: TfParams(risk_pct=0.25, min_close_loc=0.5, entry_policy=EntryPolicy.MARKET, sides=SideFilter.SHORT,
                            max_slope_atr=0.025, top_n=150, max_hold_bars=200, max_positions=3),
}
HTF_CONFIRM: dict[Timeframe, Timeframe] = {Timeframe.M15: Timeframe.H4}   # чей пробой подтверждает сигнал младшего ТФ
RETIRED_TIMEFRAMES = ("1h",)            # были в панели раньше: сигналы и настройки этих ТФ при загрузке убираются
# один раз при запуске перевести таймфрейм на новую стратегию в сохранённых настройках: правило — из DEFAULT_TF_PARAMS,
# сигналы ищутся, автоторговля по нему выключена до проверки на демо; дальше трейдер меняет всё сам
STRATEGY_RESETS: dict[str, Timeframe] = {"m15_gentle_shorts_2026_10": Timeframe.M15}
# один раз при запуске поменять общие поля в сохранённых настройках (дальше — как поставит трейдер)
SETTINGS_ONCE: dict[str, dict[str, object]] = {"max_positions_12_2026_10": {"max_positions": 12}}
# один раз при запуске поменять поля правила таймфрейма в сохранённых настройках
TF_SETTINGS_ONCE: dict[str, tuple[Timeframe, dict[str, object]]] = {
    "m15_top150_pos3_2026_10": (Timeframe.M15, {"top_n": 150, "max_positions": 3})}
TF_FIELDS = tuple(TfParams.__dataclass_fields__)


@dataclass(frozen=True)
class Settings:
    mode: Mode = Mode.MANUAL
    timeframes: frozenset[Timeframe] = frozenset({Timeframe.H4})
    auto_timeframes: frozenset[Timeframe] = frozenset({Timeframe.H4})   # по каким ТФ автобот входит сам
    tf_params: dict[Timeframe, TfParams] = field(default_factory=lambda: dict(DEFAULT_TF_PARAMS))
    leverage: int = 5                   # плечо на бирже и потолок номинала позиции (×капитал); риск задаёт стоп
    max_positions: int = 12             # 4h: обычно открыто 3, в пике истории 12; лимит 5 терял ~14% сигналов
    daily_loss_pct: float = 4.0         # дневной лимит убытка, % капитала: дальше авто не открывает
    min_turnover_usd: float = 20e6      # оборот монеты за 24 ч

    def __post_init__(self) -> None:
        if not (isinstance(self.leverage, int) and 1 <= self.leverage <= MAX_LEVERAGE):
            raise ValueError(f"плечо — целое от 1× до {MAX_LEVERAGE}×")
        if not 1 <= self.max_positions <= 50:
            raise ValueError("одновременных позиций — от 1 до 50")
        if not 0.5 <= self.daily_loss_pct <= 30.0:
            raise ValueError("дневной лимит убытка — от 0.5% до 30%")
        missing = set(Timeframe) - set(self.tf_params)
        if missing:
            object.__setattr__(self, "tf_params", {**{t: DEFAULT_TF_PARAMS[t] for t in missing}, **self.tf_params})

    def p(self, tf: Timeframe) -> TfParams:
        """Правило входа и риск таймфрейма."""
        return self.tf_params[tf]

    def with_tf(self, tf: Timeframe, **fields) -> Settings:
        return replace(self, tf_params={**self.tf_params, tf: replace(self.tf_params[tf], **fields)})

    def toggle_auto(self, tf: Timeframe) -> Settings:
        tfs = set(self.auto_timeframes)
        tfs.symmetric_difference_update({tf})
        return replace(self, auto_timeframes=frozenset(tfs))

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
    def at_level(self) -> bool:
        """Пробой линии совпал с пробоем горизонтального уровня (кандидат по research/oos.py: не фильтр, а пометка
        для сравнения по журналу)."""
        return bool(self.extra.get("level"))

    @property
    def key(self) -> tuple[str, str, str, int]:
        return self.symbol, self.timeframe.value, self.bar_time.isoformat(), int(self.side)
