"""Сущности торговой панели: таймфреймы, настройки, сигналы и план сделки. Без сети и хранилищ."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import Enum


MAX_LEVERAGE = 12                       # верхняя граница общего плеча в панели
MAX_TF_LEVERAGE = 25                    # своё плечо таймфрейма: у 15m стоп ~1.5% цены, ликвидация при 15× — ~6.5%
MAX_HEDGE_LEVERAGE = 50                 # плечо позиции BTC-хеджа (BTCUSDT на Bybit допускает до 100×)
MARKET_VALID_BARS = 2                   # вход по рынку возможен ещё 2 свечи после свечи пробоя


class Timeframe(str, Enum):
    """Таймфреймы панели. 1h убран по решению трейдера (на бэктесте — лишь тонкий плюс); 5m — тоже (2026-10: живая
    проверка около нуля, на бэктесте в минусе — docs/knowledge.md, research/lowtf.py)."""
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
    """Как входить по сигналу. По бэктесту 4h (research/entrycandle.py, docs/knowledge.md) ретест лучше входа по рынку
    на HOLDOUT и в 2026, а после длинной свечи пробоя ретест исполняется лишь в половине случаев — там рынок лучше во всех
    периодах. Гибрид совмещает: длинная свеча — по рынку, остальные — ретест (+0.02…+0.05R на сигнал к ретесту)."""
    RETEST = "retest"
    MARKET = "market"
    HYBRID = "hybrid"          # свеча пробоя не короче hybrid_range_atr — по рынку, иначе ретест


class Sizing(str, Enum):
    """Как считать размер сделки."""
    RISK = "risk"              # по риску до стопа: убыток по стопу = risk_pct% капитала (как в бэктесте, итог в R ровный)
    MARGIN = "margin"          # как на Bybit: маржа = margin_pct% капитала, позиция = маржа × плечо; убыток зависит от стопа


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
    hybrid_range_atr: float = 2.15      # гибрид: свеча пробоя от стольких ATR — по рынку, короче — ретест (q80 по IS)
    retest_bars: int = 12               # сколько свечей ждём ретест
    htf_confirm_h: int = 0              # только если старший ТФ (HTF_CONFIRM) пробил линию в ту же сторону не раньше
                                        # стольких часов назад (по закрытию его свечи); 0 — без этого условия
    sides: SideFilter = SideFilter.BOTH
    max_slope_atr: float = 0.0          # линия не круче стольких ATR за свечу (пологие линии надёжнее); 0 — любая
    top_n: int = 0                      # только столько самых ликвидных монет (по обороту за 24 ч); 0 — все с порогом
    max_hold_bars: int = 60             # позиция закрывается по рынку через столько свечей (как в бэктесте)
    max_positions: int = 0              # не больше стольких позиций этого ТФ сразу (внутри общего лимита); 0 — только общий
    hedge_btc: bool = False             # к сделке — обратная позиция BTC на бету монеты (domain/hedge.py)
    margin_pct: float = 10.0            # размер «как на Bybit»: маржа сделки, % капитала (Settings.sizing = MARGIN)
    leverage: int = 0                   # своё плечо ТФ; 0 — общее (Settings.leverage). Риск не меняет, только маржу
    pyramid: bool = False               # на +1.5R долить 0.5 позиции, стоп всей позиции — в точку входа (research/pyramid.py)
    tag_risk_x: float = 1.0             # за каждую метку канона (3+ касания, длинная линия) риск / маржа ×; обе — ×²;
                                        # итог не выше 6% риска (100% маржи). 1 — без надбавки (research/tlcanon.py)
    long_risk_x: float = 1.0            # риск / маржа лонгов × (после надбавки за метки); 1 — как у шортов. На 15m лонги
                                        # хуже шортов в каждом периоде (research/sides15.py), в IS / VAL / HO — в минусе
    ladder_x: float = 1.0               # лестница риска: после сделки этого ТФ с итогом >= +2R риск × это число (до
                                        # ladder_steps ступеней), после убыточной — снова база; 1 — выключена
    ladder_steps: int = 3               # ступеней вместе с базой: 3 — база, ×x, ×x² (research/crisk.py, 15m)
    squeeze_risk_x: float = 1.0         # риск / маржа × для пробоя после сжатия волатильности (ATR пробоя <= половины ATR
                                        # опоры линии); 1 — без надбавки (research/squeezesize.py, 4h ×1.5)

    def sized_for(self, touch3: bool, long_line: bool, side: Side | None = None, step: int = 0,
                  squeezed: bool = False) -> tuple[float, float]:
        """(риск %, маржа %) сделки с учётом ступени лестницы, надбавок за метки канона и сжатие, пониженного риска
        лонгов; итог не выше 6% риска (100% маржи)."""
        k = self.tag_risk_x ** (int(touch3) + int(long_line)) * self.ladder_x ** min(max(step, 0), self.ladder_steps - 1)
        k *= self.squeeze_risk_x if squeezed else 1.0
        risk, margin = min(self.risk_pct * k, 6.0), min(self.margin_pct * k, 100.0)
        if side is Side.LONG:
            risk, margin = risk * self.long_risk_x, margin * self.long_risk_x
        return risk, margin

    def __post_init__(self) -> None:
        if not 0.05 <= self.risk_pct <= 6.0:
            raise ValueError("риск на сделку — от 0.05% до 6%")
        if not 1.0 <= self.squeeze_risk_x <= 2.0:
            raise ValueError("надбавка за сжатие — от ×1 (выключена) до ×2")
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
        if not 0.1 <= self.margin_pct <= 100.0:
            raise ValueError("маржа на сделку — от 0.1% до 100% капитала")
        if not (isinstance(self.leverage, int) and 0 <= self.leverage <= MAX_TF_LEVERAGE):
            raise ValueError(f"плечо таймфрейма — целое от 1× до {MAX_TF_LEVERAGE}× (0 — общее)")
        if not isinstance(self.pyramid, bool):
            raise ValueError("доливка — да или нет")
        if not isinstance(self.hedge_btc, bool):
            raise ValueError("хедж BTC — да или нет")
        if not 1.0 <= self.tag_risk_x <= 2.0:
            raise ValueError("надбавка за метку линии — от ×1 до ×2")
        if not 0.1 <= self.long_risk_x <= 1.0:
            raise ValueError("риск лонгов — от ×0.1 до ×1 от риска шортов")
        if not 1.0 <= self.ladder_x <= 2.0:
            raise ValueError("шаг лестницы риска — от ×1 (выключена) до ×2")
        if not (isinstance(self.ladder_steps, int) and 1 <= self.ladder_steps <= 5):
            raise ValueError("ступеней лестницы — от 1 до 5")


# Лучшее по бэктесту (docs/research_report.md, docs/knowledge.md): 4h — основная стратегия (гибрид: ретест, а после
# свечи пробоя от 2.15 ATR — по рынку; закрытие в верхней половине свечи);
# 15m: обычный пробой в ноль (и после свежего пробоя 4h тоже), шорт от пологой линии — тонкий плюс: порог наклона
# 0.025 ATR/свечу — нижняя треть по IS (2022–2024.06), вход по рынку. На 725 монетах с местом в рейтинге оборота на день
# сигнала (research/wide15.py) топ-150 лучше топ-70 (HO +0.08R против +0.02R на сделку), но сигналов ~100 в месяц и
# держатся до 2 суток — свой лимит 3 позиции, чтобы 15m не занимал места 4h. Риск 1% / 0.25%.
DEFAULT_TF_PARAMS: dict[Timeframe, TfParams] = {
    # надбавка ×1.25 за каждую метку канона (обе — ×1.56) — по решению трейдера; в бэктесте метки не приняты (t 1.7–1.8)
    Timeframe.H4: TfParams(risk_pct=1.0, min_close_loc=0.5, entry_policy=EntryPolicy.HYBRID, pyramid=True,
                           tag_risk_x=1.25, squeeze_risk_x=1.5),
    Timeframe.M15: TfParams(risk_pct=0.25, min_close_loc=0.5, entry_policy=EntryPolicy.MARKET, sides=SideFilter.SHORT,
                            max_slope_atr=0.025, top_n=150, max_hold_bars=200, max_positions=3, leverage=15,
                            tag_risk_x=1.25,         # надбавка за метки — по решению трейдера, на 15m в бэктесте ≈ 0
                            long_risk_x=0.5,         # лонги на 15m хуже шортов в каждом периоде, в IS / VAL / HO в минусе
                            ladder_x=1.5),           # лестница 0.25 → 0.375 → 0.56%: лучше фиксированного риска того же
                                                     # среднего во всех периодах, просадка не выше (research/crisk.py)
}
HTF_CONFIRM: dict[Timeframe, Timeframe] = {Timeframe.M15: Timeframe.H4}   # чей пробой подтверждает сигнал младшего ТФ
RETIRED_TIMEFRAMES = ("1h", "5m")       # были в панели раньше: сигналы и настройки этих ТФ при загрузке убираются
# один раз при запуске перевести таймфрейм на новую стратегию в сохранённых настройках: правило — из DEFAULT_TF_PARAMS,
# сигналы ищутся, автоторговля по нему выключена до проверки на демо; дальше трейдер меняет всё сам
STRATEGY_RESETS: dict[str, Timeframe] = {"m15_gentle_shorts_2026_10": Timeframe.M15}
# один раз при запуске поменять общие поля в сохранённых настройках (дальше — как поставит трейдер)
SETTINGS_ONCE: dict[str, dict[str, object]] = {"max_positions_12_2026_10": {"max_positions": 12}}
# один раз при запуске поменять поля правила таймфрейма в сохранённых настройках
TF_SETTINGS_ONCE: dict[str, tuple[Timeframe, dict[str, object]]] = {
    "m15_top150_pos3_2026_10": (Timeframe.M15, {"top_n": 150, "max_positions": 3}),
    "h4_tag_risk_2026_10": (Timeframe.H4, {"tag_risk_x": 1.25}),
    "m15_tag_risk_2026_10": (Timeframe.M15, {"tag_risk_x": 1.25}),
    "m15_long_risk_2026_10": (Timeframe.M15, {"long_risk_x": 0.5}),
    "h4_squeeze_risk_2026_10": (Timeframe.H4, {"squeeze_risk_x": 1.5})}
# лестница риска включается в сохранённых настройках один раз вместе с базовым риском (по решению трейдера 2026-10-26:
# 15m — база 2%, ступени 2 / 3 / 4.5%); дальше настройки панели не трогаем
LADDER_ONCE: dict[str, tuple[Timeframe, float, float]] = {"m15_ladder_2026_10b": (Timeframe.M15, 1.5, 2.0)}
TF_FIELDS = tuple(TfParams.__dataclass_fields__)


@dataclass(frozen=True)
class Settings:
    mode: Mode = Mode.MANUAL
    timeframes: frozenset[Timeframe] = frozenset({Timeframe.H4})
    auto_timeframes: frozenset[Timeframe] = frozenset({Timeframe.H4})   # по каким ТФ автобот входит сам
    tf_params: dict[Timeframe, TfParams] = field(default_factory=lambda: dict(DEFAULT_TF_PARAMS))
    leverage: int = 5                   # плечо на бирже и потолок номинала позиции (×капитал); риск задаёт стоп
    hedge_leverage: int = 0             # плечо позиции BTC-хеджа; 0 — как у сделок. Меняет только маржу под хедж
    max_positions: int = 12             # 4h: обычно открыто 3, в пике истории 12; лимит 5 терял ~14% сигналов
    daily_loss_pct: float = 4.0         # дневной лимит убытка, % капитала: дальше авто не открывает
    min_turnover_usd: float = 20e6      # оборот монеты за 24 ч
    sizing: Sizing = Sizing.RISK        # размер сделки: по риску до стопа или маржа × плечо, как на Bybit

    def __post_init__(self) -> None:
        if not (isinstance(self.leverage, int) and 1 <= self.leverage <= MAX_LEVERAGE):
            raise ValueError(f"плечо — целое от 1× до {MAX_LEVERAGE}×")
        if not (isinstance(self.hedge_leverage, int) and 0 <= self.hedge_leverage <= MAX_HEDGE_LEVERAGE):
            raise ValueError(f"плечо хеджа — целое от 1× до {MAX_HEDGE_LEVERAGE}× (0 — как у сделок)")
        if not 1 <= self.max_positions <= 50:
            raise ValueError("одновременных позиций — от 1 до 50")
        if not 0.5 <= self.daily_loss_pct <= 30.0:
            raise ValueError("дневной лимит убытка — от 0.5% до 30%")
        missing = set(Timeframe) - set(self.tf_params)
        if missing:
            object.__setattr__(self, "tf_params", {**{t: DEFAULT_TF_PARAMS[t] for t in missing}, **self.tf_params})

    @property
    def hedge_lev(self) -> int:
        """Плечо позиции BTC-хеджа: своё или, если не задано, как у сделок."""
        return self.hedge_leverage or self.leverage

    @property
    def hedging(self) -> bool:
        """Хедж BTC включён хотя бы для одного таймфрейма."""
        return any(p.hedge_btc for p in self.tf_params.values())

    def p(self, tf: Timeframe) -> TfParams:
        """Правило входа и риск таймфрейма."""
        return self.tf_params[tf]

    def lev(self, tf: Timeframe) -> int:
        """Плечо сделок таймфрейма: своё или, если не задано, общее."""
        return self.tf_params[tf].leverage or self.leverage

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
    def touch3(self) -> bool:
        """Линия с 3+ касаниями (research/tlcanon.py: метка для журнала, не фильтр)."""
        return bool(self.extra.get("touch3"))

    @property
    def long_line(self) -> bool:
        """Линия длиннее медианы бэктеста (research/tlcanon.py: метка для журнала, не фильтр)."""
        return bool(self.extra.get("long_line"))

    @property
    def squeezed(self) -> bool:
        """Пробой после сжатия волатильности: ATR свечи пробоя не больше SQUEEZE_MAX от ATR опоры линии
        (research/squeezesize.py: на 4h +1.0R против +0.35R; надбавка к риску — TfParams.squeeze_risk_x)."""
        return bool(self.extra.get("squeezed"))

    @property
    def round_level(self) -> bool:
        """Закрытие пробоя прошло крупный круглый уровень (research/roundnum.py: метка, без надбавки к риску)."""
        return bool(self.extra.get("round"))

    @property
    def tags(self) -> tuple[str, ...]:
        """Пометки сигнала для уведомлений и панели."""
        return tuple(name for name, on in (("+ уровень", self.at_level), ("3+ касания", self.touch3),
                                           ("длинная линия", self.long_line), ("+ круглый", self.round_level),
                                           ("сжатие", self.squeezed)) if on)

    @property
    def key(self) -> tuple[str, str, str, int]:
        return self.symbol, self.timeframe.value, self.bar_time.isoformat(), int(self.side)
