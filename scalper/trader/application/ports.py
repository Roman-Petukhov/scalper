"""Контракты внешних зависимостей. Сценарии зависят только от них, реализации — в infrastructure."""
from __future__ import annotations

from typing import Protocol

import pandas as pd

from datetime import datetime

from ..domain.execution import Account, ClosedPnl, HedgeLeg, Instrument, OrderRequest, Trade, TradeResult, TradeStatus
from ..domain.journal import JournalEntry
from ..domain.models import Settings, Signal, SignalStatus, Timeframe
from ..domain.daily_break import DailySetup
from ..domain.trend import TrendDay


class MarketData(Protocol):
    async def universe(self, min_turnover_usd: float) -> list[str]:
        """Монеты (перпетуалы USDT) с оборотом за 24 ч не ниже порога, от самых ликвидных."""

    async def closed_bars(self, symbol: str, tf: Timeframe) -> pd.DataFrame:
        """Только закрытые свечи; индекс — открытие свечи UTC; колонки open, high, low, close, volume,
        taker_buy_volume."""

    async def live_bars(self, symbol: str, tf: Timeframe, limit: int = 300) -> pd.DataFrame:
        """Последние свечи вместе с текущей, ещё не закрытой (для живого графика)."""

    async def bars_between(self, symbol: str, tf: Timeframe, start: datetime, end: datetime) -> pd.DataFrame:
        """Свечи с открытием в [start, end] (последняя может быть не закрыта), не больше 1500 — для выгрузки."""

    async def daily_bars(self, symbol: str, limit: int) -> pd.DataFrame:
        """Закрытые дневные свечи, не больше limit; индекс — открытие дня UTC; колонки open, high, low, close,
        volume, quote_volume, taker_buy_volume."""

    async def funding_sum(self, symbol: str, start: datetime, end: datetime) -> float:
        """Сумма ставок funding с выплатой в (start, end]."""


class SignalRepository(Protocol):
    def add(self, signal: Signal) -> Signal | None:
        """Сохранить; None — такой сигнал (монета, ТФ, свеча, сторона) уже есть."""

    def get(self, signal_id: int) -> Signal | None: ...

    def recent(self, limit: int = 100, timeframes: set[Timeframe] | None = None,
               statuses: set[SignalStatus] | None = None) -> list[Signal]: ...

    def set_status(self, signal_id: int, status: SignalStatus, note: str = "") -> Signal | None: ...

    def set_chart(self, signal_id: int, path: str) -> None: ...


class SettingsRepository(Protocol):
    def load(self) -> Settings: ...

    def save(self, settings: Settings) -> None: ...


class ChartRenderer(Protocol):
    def render(self, signal: Signal, bars: pd.DataFrame) -> str:
        """Нарисовать график сигнала; вернуть путь к PNG."""


class Notifier(Protocol):
    async def signal(self, signal: Signal, chart_path: str | None, panel_url: str) -> None: ...

    async def text(self, message: str) -> None: ...


class Broker(Protocol):
    """Биржа для исполнения (Bybit). Все суммы в USDT."""
    network: str                                          # "demo" / "live"

    async def account(self) -> Account:
        """Капитал, свободная маржа, открытые позиции и монеты с неисполненными лимитками входа."""

    async def instrument(self, symbol: str) -> Instrument | None:
        """Правила монеты; None — такой USDT-перпетуал на бирже не торгуется."""

    async def price(self, symbol: str) -> float: ...

    async def place(self, order: OrderRequest) -> str:
        """Отправить ордер входа со стопом и целью на бирже; вернуть id ордера."""

    async def open_order_ids(self) -> set[str]: ...

    async def cancel(self, symbol: str, order_id: str) -> None: ...

    async def adjust(self, symbol: str, qty: float, leverage: int) -> None:
        """Рыночный ордер без стопа и цели: qty > 0 — купить, < 0 — продать (позиция хеджа BTC)."""

    async def close_position(self, symbol: str) -> None:
        """Закрыть позицию по монете по рынку (reduce-only); стоп и цель биржа снимает сама."""

    async def protect(self, symbol: str, stop: float, target: float) -> None:
        """Переставить стоп и цель позиции по монете — на весь её объём."""

    async def closed_pnl(self, symbol: str, since: datetime, until: datetime) -> list[ClosedPnl]:
        """Закрытия позиций по монете за окно (не длиннее 7 дней — ограничение биржи)."""

    async def closed_pnl_all(self, since: datetime, until: datetime) -> list[ClosedPnl]:
        """Закрытия позиций по всем монетам за окно (не длиннее 7 дней), все страницы."""

    async def close(self) -> None: ...


class TradeRepository(Protocol):
    def add_trade(self, trade: Trade) -> Trade: ...

    def trades_for(self, signal_ids: list[int]) -> dict[int, Trade]: ...

    def pending_trades(self) -> list[Trade]: ...

    def filled_trades(self) -> list[Trade]:
        """Сделки со статусом «на бирже», новые первыми."""

    def set_trade_status(self, trade_id: int, status: TradeStatus) -> None: ...

    def set_trade_filled(self, trade_id: int, at: datetime) -> None:
        """Вход исполнился: статус «на бирже» и время исполнения."""

    def set_trade_hedge(self, trade_id: int, hedge: HedgeLeg) -> None: ...

    def set_trade_pyramid(self, trade_id: int, qty: float, price: float) -> None:
        """Доливка сделана (qty > 0) или стоп переставлен в безубыток без доливки (qty = 0)."""

    def unsettled_trades(self) -> list[Trade]:
        """Исполненные сделки без записанного итога (на бирже или закрытые по сроку), новые первыми."""

    def settle_trade(self, trade_id: int, status: TradeStatus, result: TradeResult | None) -> None:
        """Записать итог (None — биржа итога не вернула, больше не ищем)."""

    def lost_trades(self, filled_since: datetime) -> list[Trade]:
        """Исполненные не раньше filled_since сделки с записанным «нет данных» (статус «закрыта», итога нет)."""

    def reopen_trade(self, trade_id: int) -> None:
        """Вернуть сделку в «на бирже» без итога — итог будет искаться заново."""

    def journal(self, limit: int = 500) -> list[JournalEntry]:
        """Сделки с таймфреймом сигнала, новые первыми."""

    def kv_get(self, key: str) -> str | None: ...

    def kv_set(self, key: str, value: str) -> None: ...


class SentimentData(Protocol):
    async def fear_greed(self, days: int = 8) -> list[tuple[int, str]]:
        """Индекс страха и жадности: (значение 0…100, подпись), новые первыми."""

    async def premium_now(self) -> dict[str, tuple[float, float]]:
        """По монетам: (последняя ставка funding за 8 ч, базис фьючерса к споту), доли."""

    async def btc_long_short(self, days: int = 8) -> list[float]:
        """Лонги / шорты по счетам на BTC за последние дни, старые первыми."""

    async def btc_top_long_short(self, days: int = 8) -> list[float]:
        """Лонги / шорты по позициям крупных трейдеров на BTC, старые первыми."""

    async def btc_open_interest(self, days: int = 8) -> list[float]:
        """Открытый интерес BTC, $, по дням, старые первыми."""

    async def btc_dominance(self) -> float | None:
        """Доля BTC в капитализации рынка, %."""

    async def btc_dvol(self, days: int = 8) -> list[float]:
        """Индекс волатильности опционов BTC (DVOL) по дням, старые первыми."""


class TrendRepository(Protocol):
    def trend_last(self, rule: str) -> TrendDay | None:
        """Последний шаг бумажного счёта стратегии."""

    def trend_save(self, day: TrendDay) -> None: ...

    def trend_history(self, rule: str, limit: int = 400) -> list[TrendDay]:
        """Шаги счёта, старые первыми."""


class DailyBreakRepository(Protocol):
    def dbreak_add(self, setup: DailySetup) -> DailySetup | None:
        """Новый сетап; None — такой (правило, монета, день, сторона) уже есть."""

    def dbreak_update(self, setup: DailySetup) -> None: ...

    def dbreak_active(self, rule: str) -> list[DailySetup]:
        """Ждут ретеста или в позиции."""

    def dbreak_recent(self, rule: str, limit: int = 500) -> list[DailySetup]:
        """Все сетапы, новые первыми."""

    def kv_get(self, key: str) -> str | None: ...

    def kv_set(self, key: str, value: str) -> None: ...


class StablecoinData(Protocol):
    async def supply(self) -> pd.Series:
        """Суммарное предложение долларовых стейблкоинов по дням, $; индекс — день UTC."""
