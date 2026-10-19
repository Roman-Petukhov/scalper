"""Третья стратегия: пробой линии на дневках, бумажный счёт (research/htftrend.py, docs/knowledge.md).

Правило бота 4h, перенесённое на дневки, со стопом от линии вместо свинга — лучший вариант третьего прогона:
    линия       зигзаг 3 ATR по закрытиям (research.tline.zz_lines), пробой — первое закрытие дня за линией
    фильтры     неделя по направлению (закрытие против EMA50 последней закрытой недели), закрытие в своей половине
                свечи, средний оборот за 30 прошлых дней от $20M; агрессоры не фильтруются — записываются для сравнения
    вход        лимитка на линии 12 дней; цена дошла до цели раньше — отмена
    стоп, цель  1.5 ATR за линией, цель 3R, долив +0.5 на +1.5R (стоп — на вход), срок 60 дней
Бэктест перпетуалов 2021-01 … 2026-09: +0.58R на сделку (t 4.3), ~8 сделок в месяц, 41% прибыльных, все
периоды в плюсе (VAL — +0.11R). Исход сделки каждый день пересчитывается по истории свечей тем же кодом, что в бэктесте
(research.dconf.simulate_bar), поэтому состояние — только сетап: линия, стоп, ATR. Funding — отдельно, за время
удержания (долив в нём не учитывается).

Счёт в деньгах (по просьбе владельца): риск 6% капитала на сделку, плечо 12, хедж BTC. Размер — как у бота
(domain/execution.py): номинал = риск / расстояние до стопа, не больше плеча × капитал; маржа сделки и хеджа (бета ×
номинал) — не больше 95% свободной, иначе сделка урезается, а без свободной маржи пропускается. Капитал на день
входа — старт 1.0 плюс итог сделок, закрытых раньше; свободная маржа — капитал минус маржа открытых. Хедж — обратная
позиция BTCUSDT на бету монеты к BTC за 180 дней, по закрытиям дня входа и выхода (точного часа ретеста дневки не
знают), комиссия taker на вход и выход. На дневках хедж в бэктесте не проверялся — проверен только на 4h.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from enum import Enum

import numpy as np
import pandas as pd

from research.dconf import Exit, simulate_bar
from research.smc import _atr
from research.tline import MAKER, TAKER, htf_trend, retest_fill, zz_lines

from .execution import HEDGE_FEE
from .hedge import beta as hedge_beta
from .strategy import _with_probe_bar

HISTORY_DAYS = 1000                 # дневных свечей на монету: неделе нужна EMA50, зигзагу — опорные вершины
MIN_BARS = 400                      # как у detect: меньше — линии и тренд недели ещё не устоялись
REQUIRED = ("open", "high", "low", "close", "quote_volume", "volume", "taker_buy_volume")


@dataclass(frozen=True)
class DailyBreakRule:
    key: str
    title: str
    backtest: str                   # что было на бэктесте — для сравнения в панели
    min_adv: float = 20e6
    stop_atr: float = 1.5
    target_r: float = 3.0
    retest_bars: int = 12
    add_at: float = 1.5
    add_size: float = 0.5
    max_hold: int = 60
    risk: float = 0.06              # доля капитала под риском на сделку
    leverage: int = 12              # потолок номинала (× капитал) и маржа = номинал / плечо
    hedge: bool = True              # обратная позиция BTC на бету монеты
    beta_days: int = 180
    margin_use: float = 0.95        # запас на комиссии, как у бота

    @property
    def exit(self) -> Exit:
        return Exit(self.target_r, (self.add_at,), self.add_size, self.max_hold)


RULES: tuple[DailyBreakRule, ...] = (
    DailyBreakRule("d1_line15", "Пробой линии на дневках, стоп 1.5 ATR, 3R",
                   "бэктест перпетуалов 2021–2026: +0.58R на сделку, ~8 сделок в месяц, 41% в плюсе; хедж на дневках не проверялся"),
)


class Status(str, Enum):
    PENDING = "pending"             # лимитка на линии ждёт ретеста
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"         # ретеста не было за retest_bars дней или цена ушла к цели без нас
    SKIPPED = "skipped"             # ретест был, но свободной маржи на сделку не осталось


@dataclass(frozen=True)
class DailySetup:
    rule: str
    symbol: str
    side: int                       # +1 лонг, −1 шорт
    day: date                       # день пробоя (открытие дневной свечи UTC)
    level: float                    # линия на дне пробоя — цена лимитки
    stop: float
    atr: float
    aggr: float                     # доля агрессоров в сторону пробоя
    slope_atr: float
    status: Status = Status.PENDING
    fill_day: date | None = None
    exit_day: date | None = None
    r: float | None = None          # итог (CLOSED) или оценка по последнему закрытию (OPEN), funding включён
    funding_r: float = 0.0
    id: int | None = None
    risk_abs: float | None = None   # риск сделки в долях стартового капитала (задаётся на дне входа)
    notional: float | None = None   # номинал в долях стартового капитала
    beta: float | None = None       # бета к BTC; None — без хеджа
    btc_in: float | None = None
    btc_out: float | None = None
    hedge_pnl: float = 0.0          # итог хеджа (CLOSED) или оценка (OPEN), доли стартового капитала

    @property
    def risk(self) -> float:
        return self.side * (self.level - self.stop)

    @property
    def sized(self) -> bool:
        return self.risk_abs is not None and self.notional is not None

    @property
    def pnl(self) -> float | None:
        """Итог (или оценка) сделки с хеджем в долях стартового капитала."""
        return self.r * self.risk_abs + self.hedge_pnl if self.sized and self.r is not None else None

    def margin(self, leverage: int) -> float:
        return (self.notional or 0.0) * (1 + (self.beta or 0.0)) / leverage


def _ohlc(d: pd.DataFrame) -> tuple[np.ndarray, ...]:
    return tuple(d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))


def find(d: pd.DataFrame, symbol: str, rule: DailyBreakRule) -> list[DailySetup]:
    """Сетапы на последней закрытой дневке `d` (индекс — открытие дня UTC, только закрытые свечи)."""
    missing = [k for k in REQUIRED if k not in d.columns]
    if missing:
        raise ValueError(f"нет колонок {missing}")
    if len(d) < MIN_BARS:
        return []
    last = len(d) - 1
    recs = [r for r in zz_lines(_with_probe_bar(d)) if r["t"] == last]
    if not recs:
        return []
    _, hi, lo, c = _ohlc(d)
    atr = float(_atr(d).to_numpy()[last])
    adv = d["quote_volume"].rolling(30, min_periods=20).mean().shift(1).to_numpy()[last]
    if not (atr > 0) or not (adv >= rule.min_adv):
        return []
    trend = htf_trend(d, "1d")
    vol = float(d["volume"].iloc[-1])
    buy = float(d["taker_buy_volume"].iloc[-1]) / vol if vol > 0 else float("nan")
    rng = hi[last] - lo[last]
    out, seen = [], set()
    for r in recs:
        side = int(r["side"])
        if side in seen or trend[last] != side or not (rng > 0):
            continue
        loc = (c[last] - lo[last]) / rng if side > 0 else (hi[last] - c[last]) / rng
        if loc < 0.5:
            continue
        level = float(r["line_t"])
        out.append(DailySetup(rule.key, symbol, side, d.index[last].date(), level, level - side * rule.stop_atr * atr,
                              atr, buy if side > 0 else 1 - buy, abs(float(r["slope"])) / atr))
        seen.add(side)
    return out


def advance(d: pd.DataFrame, s: DailySetup, rule: DailyBreakRule) -> DailySetup:
    """Состояние сетапа по закрытым дневкам `d` (с дня пробоя и дальше): ждёт ретеста, в позиции, закрыт или отменён.
    Funding здесь не трогается — его добавляет use case за время удержания."""
    if s.status in (Status.CLOSED, Status.CANCELLED, Status.SKIPPED):
        return s
    days = [x.date() for x in d.index]
    if s.day not in days:
        return s
    t = days.index(s.day)
    o, hi, lo, c = _ohlc(d)
    fill = retest_fill(hi, lo, t, s.side, s.level, rule.target_r * s.risk, rule.retest_bars)
    if fill < 0:
        cut = min(t + rule.retest_bars, len(c) - 1)
        tp = s.level + s.side * rule.target_r * s.risk
        ran = any((s.side > 0 and hi[j] >= tp) or (s.side < 0 and lo[j] <= tp) for j in range(t + 1, cut + 1))
        done = ran or len(c) - 1 >= t + rule.retest_bars
        return replace(s, status=Status.CANCELLED if done else Status.PENDING)
    fd = days[fill]
    if (s.side > 0 and lo[fill] <= s.stop) or (s.side < 0 and hi[fill] >= s.stop):    # выбит в день исполнения
        r = (s.side * (s.stop - s.level) - (MAKER + TAKER) * s.level) / s.risk
        return replace(s, status=Status.CLOSED, fill_day=fd, exit_day=fd, r=r + s.funding_r)
    # лишняя свеча NaN: simulate_bar закрывает позицию на последней свече данных — здесь это значит «ещё открыта»
    nan = np.array([np.nan])
    o2, h2, l2, c2 = (np.r_[x, nan] for x in (o, hi, lo, c))
    r, j = simulate_bar(o2, h2, l2, c2, np.zeros(len(c2)), fill, s.side, s.level, s.stop, s.atr, rule.exit, MAKER)
    if 0 <= j < len(c):
        return replace(s, status=Status.CLOSED, fill_day=fd, exit_day=days[j], r=float(r) + s.funding_r)
    mark = (s.side * (c[-1] - s.level) - (MAKER + TAKER) * s.level) / s.risk
    return replace(s, status=Status.OPEN, fill_day=fd, r=float(mark) + s.funding_r)


def funding_r(s: DailySetup, funding: float) -> float:
    """Funding за удержание в R: лонг платит при ставке > 0 (сумма ставок — доля номинала)."""
    return -s.side * funding * s.level / s.risk if s.risk > 0 else 0.0


def equity_on(setups: list[DailySetup], day: date) -> float:
    """Капитал на начало дня day: старт 1.0 плюс итог сделок, закрытых раньше."""
    return 1.0 + sum(s.pnl for s in setups if s.status is Status.CLOSED and s.sized and s.pnl is not None
                     and s.exit_day is not None and s.exit_day < day)


def margin_on(setups: list[DailySetup], day: date, leverage: int) -> float:
    """Маржа сделок (с хеджем), открытых на день day."""
    return sum(s.margin(leverage) for s in setups if s.sized and s.fill_day is not None and s.fill_day <= day
               and (s.exit_day is None or s.exit_day >= day))


def size(s: DailySetup, equity: float, used: float, rule: DailyBreakRule, beta: float | None) -> DailySetup:
    """Размер сделки на дне входа: риск rule.risk капитала, номинал не больше плеча × капитал и не больше того, что
    помещается в свободную маржу вместе с хеджем. Нет места — SKIPPED."""
    b = beta if rule.hedge and beta is not None and beta > 0 else None
    notional = rule.risk * equity * s.level / s.risk if s.risk > 0 else 0.0
    notional = min(notional, rule.leverage * equity,
                   max(rule.margin_use * (equity - used), 0.0) * rule.leverage / (1 + (b or 0.0)))
    if not notional > 0:
        return replace(s, status=Status.SKIPPED, exit_day=s.fill_day, r=None, risk_abs=0.0, notional=0.0, beta=b)
    return replace(s, risk_abs=notional * s.risk / s.level, notional=notional, beta=b)


def hedge_pnl(s: DailySetup, btc_now: float | None) -> float:
    """Итог хеджа: обратная позиция BTC на бету × номинал, комиссия taker на вход и выход."""
    if not s.beta or not s.notional or not s.btc_in or not btc_now:
        return 0.0
    h = s.beta * s.notional
    return -s.side * h * (btc_now / s.btc_in - 1) - 2 * HEDGE_FEE * h


def beta_before(coin: pd.Series, btc: pd.Series, day: date, rule: DailyBreakRule) -> float | None:
    """Бета монеты к BTC по дневкам, закрытым до дня day."""
    cut = pd.Timestamp(day, tz="UTC")
    return hedge_beta(coin[coin.index < cut], btc[btc.index < cut], rule.beta_days)


@dataclass(frozen=True)
class DailyBook:
    closed: int
    mean_r: float | None
    total_r: float                  # сумма R закрытых сделок
    win: float | None               # доля прибыльных
    dd_r: float                     # худшая просадка суммы R по закрытым сделкам
    open_r: float                   # оценка открытых по последнему закрытию
    open: list[DailySetup]
    pending: list[DailySetup]
    recent: list[DailySetup]        # последние закрытые, новые первыми
    since: date | None              # первый сетап
    total: float = 0.0              # итог закрытых в долях стартового капитала (с хеджем)
    open_pnl: float = 0.0
    hedge: float = 0.0              # вклад хеджа в итог закрытых
    dd: float = 0.0                 # худшая просадка капитала по закрытым сделкам
    skipped: int = 0


def book(setups: list[DailySetup], recent: int = 8) -> DailyBook:
    """Сводка счёта по сетапам правила (в любом порядке)."""
    closed = sorted((s for s in setups if s.status is Status.CLOSED and s.r is not None),
                    key=lambda s: (s.exit_day or s.day, s.day))
    rs = [s.r for s in closed]
    peak = cum = dd = 0.0
    for r in rs:
        cum += r
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    eq = peak_eq = 1.0
    dd_eq = 0.0
    for s in closed:
        eq += s.pnl or 0.0
        peak_eq = max(peak_eq, eq)
        dd_eq = max(dd_eq, 1 - eq / peak_eq)
    opened = sorted((s for s in setups if s.status is Status.OPEN), key=lambda s: s.fill_day or s.day)
    return DailyBook(len(rs), float(np.mean(rs)) if rs else None, float(sum(rs)),
                     float(np.mean([r > 0 for r in rs])) if rs else None, dd,
                     float(sum(s.r or 0.0 for s in opened)), opened,
                     sorted((s for s in setups if s.status is Status.PENDING), key=lambda s: s.day),
                     closed[::-1][:recent], min((s.day for s in setups), default=None),
                     float(sum(s.pnl or 0.0 for s in closed)), float(sum(s.pnl or 0.0 for s in opened)),
                     float(sum(s.hedge_pnl for s in closed)), dd_eq,
                     sum(s.status is Status.SKIPPED for s in setups))
