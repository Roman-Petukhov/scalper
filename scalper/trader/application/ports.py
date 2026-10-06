"""Контракты внешних зависимостей. Сценарии зависят только от них, реализации — в infrastructure."""
from __future__ import annotations

from typing import Protocol

import pandas as pd

from datetime import datetime

from ..domain.execution import Account, ClosedPnl, Instrument, OrderRequest, Trade, TradeResult, TradeStatus
from ..domain.journal import JournalEntry
from ..domain.models import Settings, Signal, SignalStatus, Timeframe


class MarketData(Protocol):
    async def universe(self, min_turnover_usd: float) -> list[str]:
        """Монеты (перпетуалы USDT) с оборотом за 24 ч не ниже порога, от самых ликвидных."""

    async def closed_bars(self, symbol: str, tf: Timeframe) -> pd.DataFrame:
        """Только закрытые свечи; индекс — открытие свечи UTC; колонки open, high, low, close, volume,
        taker_buy_volume."""

    async def live_bars(self, symbol: str, tf: Timeframe, limit: int = 300) -> pd.DataFrame:
        """Последние свечи вместе с текущей, ещё не закрытой (для живого графика)."""


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

    async def close_position(self, symbol: str) -> None:
        """Закрыть позицию по монете по рынку (reduce-only); стоп и цель биржа снимает сама."""

    async def closed_pnl(self, symbol: str, since: datetime, until: datetime) -> list[ClosedPnl]:
        """Закрытия позиций по монете за окно (не длиннее 7 дней — ограничение биржи)."""

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

    def unsettled_trades(self) -> list[Trade]:
        """Исполненные сделки без записанного итога (на бирже или закрытые по сроку), новые первыми."""

    def settle_trade(self, trade_id: int, status: TradeStatus, result: TradeResult | None) -> None:
        """Записать итог (None — биржа итога не вернула, больше не ищем)."""

    def journal(self, limit: int = 500) -> list[JournalEntry]:
        """Сделки с таймфреймом сигнала, новые первыми."""

    def kv_get(self, key: str) -> str | None: ...

    def kv_set(self, key: str, value: str) -> None: ...
