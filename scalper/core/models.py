"""Базовые структуры данных: рыночные события, сигналы, ордера, сделки."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ---------- рыночные события ----------

@dataclass(slots=True)
class Trade:
    ts: float            # секунды (exchange time)
    price: float
    qty: float
    buyer_maker: bool    # True => агрессор продавец

    @property
    def sign(self) -> int:
        return -1 if self.buyer_maker else 1


@dataclass(slots=True)
class Book:
    ts: float
    bids: list[tuple[float, float]]   # [(price, qty)] от лучшей цены
    asks: list[tuple[float, float]]


@dataclass(slots=True)
class Liquidation:
    ts: float
    side: str            # сторона ордера ликвидации: SELL = ликвидировали лонг, BUY = шорт
    price: float
    qty: float


@dataclass(slots=True)
class MarkPrice:
    ts: float
    mark: float
    index: float
    funding: float


MarketEvent = Trade | Book | Liquidation | MarkPrice


# ---------- торговые сущности ----------

class Side(int, Enum):
    LONG = 1
    SHORT = -1

    @property
    def order_side(self) -> str:          # сторона ордера на вход
        return "buy" if self is Side.LONG else "sell"

    @property
    def close_side(self) -> str:          # сторона ордера на выход
        return "sell" if self is Side.LONG else "buy"


@dataclass(slots=True)
class Signal:
    ts: float
    side: Side
    setup: str           # momentum | fade
    price: float         # референсная цена на момент сигнала
    stop_dist: float     # в цене
    tp_dist: float
    reason: str = ""


class OrderStatus(str, Enum):
    NEW = "NEW"
    PARTIAL = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"     # в т.ч. GTX, который бы исполнился как тейкер
    EXPIRED = "EXPIRED"

    @property
    def is_final(self) -> bool:
        return self in (OrderStatus.FILLED, OrderStatus.CANCELED,
                        OrderStatus.REJECTED, OrderStatus.EXPIRED)


class OrderKind(str, Enum):
    LIMIT = "limit"
    MARKET = "market"
    STOP = "stop"            # STOP_MARKET reduceOnly


@dataclass(slots=True)
class OrderCmd:
    """Команда брокеру. cid генерирует менеджер позиции, поэтому submit не блокирует."""
    action: str              # place | cancel
    cid: str
    kind: OrderKind = OrderKind.LIMIT
    side: str = ""           # buy | sell
    qty: float = 0.0
    price: float = 0.0       # для limit
    stop_price: float = 0.0  # для stop
    post_only: bool = False
    reduce_only: bool = False


@dataclass(slots=True)
class OrderUpdate:
    cid: str
    status: OrderStatus
    filled: float            # накопленный объём
    avg_price: float
    last_qty: float = 0.0    # объём последнего исполнения
    last_price: float = 0.0
    fee: float = 0.0         # комиссия последнего исполнения (USDT)
    ts: float = 0.0
    reason: str = ""


@dataclass(slots=True)
class PositionSync:
    """Сверка с биржей (только live): фактический размер позиции."""
    ts: float
    qty: float               # со знаком: >0 лонг, <0 шорт
    entry_price: float


@dataclass
class TradeRecord:
    symbol: str
    side: str
    setup: str
    entry_ts: float
    exit_ts: float
    qty: float
    entry_price: float
    exit_price: float
    gross_pnl: float
    fees: float
    net_pnl: float
    exit_reason: str
    mfe: float               # max favorable excursion, в долях цены
    mae: float               # max adverse excursion
    signal_reason: str = ""
    extra: dict = field(default_factory=dict)
