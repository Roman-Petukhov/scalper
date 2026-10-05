"""Перевод сигнала в биржевой ордер: размер от риска, округление под правила монеты и защитные проверки.
Чистая логика без сети — биржа подставляется через порт Broker (application/ports.py)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

from .models import EntryKind, Settings, Side, Signal, TradePlan
from .sizing import position_size

MIN_RR = 1.5                # рыночный вход: если цена ушла и до цели осталось меньше 1.5 стопа — не входим
DEFAULT_MIN_NOTIONAL = 5.0  # $, минимальный ордер Bybit для USDT-перпетуалов


class ExecutionRefused(Exception):
    """Ордер не отправлен по правилам риска; текст — причина для трейдера."""


@dataclass(frozen=True)
class Instrument:
    symbol: str                 # как в сигнале: ETHUSDT
    qty_step: float
    min_qty: float
    tick: float
    max_leverage: float
    min_notional: float = DEFAULT_MIN_NOTIONAL


@dataclass(frozen=True)
class Position:
    symbol: str
    side: Side
    qty: float
    entry: float
    mark: float
    upnl: float                 # нереализованный PnL, $
    stop: float | None = None
    target: float | None = None

    @property
    def notional(self) -> float:
        return self.qty * self.mark


@dataclass(frozen=True)
class Account:
    equity: float               # капитал с учётом открытых позиций, $
    available: float            # свободно под новые позиции, $
    upnl: float                 # нереализованный PnL по всем позициям, $
    positions: tuple[Position, ...] = ()
    pending_symbols: frozenset[str] = frozenset()   # монеты с неисполненной лимиткой входа

    @property
    def busy_symbols(self) -> set[str]:
        return {p.symbol for p in self.positions} | set(self.pending_symbols)

    @property
    def open_count(self) -> int:
        return len(self.busy_symbols)


@dataclass(frozen=True)
class OrderRequest:
    signal_id: int
    symbol: str
    side: Side
    kind: EntryKind
    qty: float
    price: float                # лимит для ретеста; для рыночного — ориентир (текущая цена)
    stop: float
    target: float
    leverage: int
    expires_at: datetime | None  # лимитка ретеста снимается после этого времени
    risk_usd: float
    extra: dict = field(default_factory=dict, compare=False)

    @property
    def client_id(self) -> str:
        """Метка ордера на бирже: повторная отправка того же сигнала будет отклонена биржей."""
        return f"tt-{self.signal_id}"

    def describe(self) -> str:
        kind = "лимит" if self.kind is EntryKind.RETEST else "рынок"
        return (f"{self.side.label} {self.symbol} · {kind} {_fmt(self.qty)} @ {_fmt(self.price)} · "
                f"стоп {_fmt(self.stop)} · цель {_fmt(self.target)} · риск ${self.risk_usd:.2f}")


def _fmt(x: float) -> str:
    return f"{x:.6g}"


def round_down(x: float, step: float) -> float:
    if step <= 0:
        return x
    n = math.floor(x / step + 1e-9)
    decimals = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
    return round(n * step, decimals)


def round_price(x: float, tick: float) -> float:
    if tick <= 0:
        return x
    decimals = max(0, -int(math.floor(math.log10(tick)))) if tick < 1 else 0
    return round(round(x / tick) * tick, decimals)


def target_reached(side: Side, target: float, price: float) -> bool:
    """Цена дошла до цели: лимитка ретеста больше не нужна (в бэктесте такой вход не делается)."""
    return int(side) * (price - target) >= 0


def day_loss_hit(equity: float, day_start_equity: float | None, daily_loss_pct: float) -> bool:
    return day_start_equity is not None and day_start_equity > 0 and \
        equity <= day_start_equity * (1 - daily_loss_pct / 100)


def build_order(signal: Signal, settings: Settings, account: Account, instrument: Instrument, price: float,
                day_start_equity: float | None, now: datetime, tf_open: int = 0) -> OrderRequest:
    """Ордер по сигналу или ExecutionRefused с понятной причиной. tf_open — сколько из занятых монет заняты
    сделками того же таймфрейма."""
    if signal.id is None:
        raise ExecutionRefused("сигнал не сохранён")
    sym, side, plan = signal.symbol, signal.side, signal.plan
    if sym in account.busy_symbols:
        raise ExecutionRefused(f"по {sym} уже есть позиция или ордер")
    if account.open_count >= settings.max_positions:
        raise ExecutionRefused(f"открыто {account.open_count} из {settings.max_positions} позиций")
    tf_limit = settings.p(signal.timeframe).max_positions
    if tf_limit and tf_open >= tf_limit:
        raise ExecutionRefused(f"по {signal.timeframe.value} открыто {tf_open} из {tf_limit} позиций")
    if day_loss_hit(account.equity, day_start_equity, settings.daily_loss_pct):
        raise ExecutionRefused(f"дневной стоп −{settings.daily_loss_pct:g}% достигнут, до 00:00 UTC новых входов нет")
    if account.equity <= 0:
        raise ExecutionRefused("на счёте нет средств")
    s = int(side)
    if (price - plan.stop) * s <= 0:
        raise ExecutionRefused("цена уже за стопом — сигнал устарел")
    if (plan.target - price) * s <= 0:
        raise ExecutionRefused("цена уже дошла до цели — сигнал устарел")

    if plan.entry_kind is EntryKind.MARKET:
        entry = price
        if (plan.target - entry) * s / ((entry - plan.stop) * s) < MIN_RR:
            raise ExecutionRefused(f"цена ушла от точки входа: до цели меньше {MIN_RR:g} стопа")
        expires = None
    else:
        entry = plan.entry
        tf_min = signal.timeframe.minutes
        expires = signal.bar_time + timedelta(minutes=tf_min * (1 + plan.valid_bars))
        if expires <= now:
            raise ExecutionRefused("время на ретест уже вышло")

    entry = round_price(entry, instrument.tick)
    stop, target = round_price(plan.stop, instrument.tick), round_price(plan.target, instrument.tick)
    risk_pct = settings.p(signal.timeframe).risk_pct
    sized = position_size(account.equity, risk_pct,
                          TradePlan(plan.entry_kind, entry, stop, target, plan.valid_bars), settings.leverage)
    lev = int(max(1, min(settings.leverage, instrument.max_leverage)))
    margin_cap = max(account.available, 0.0) * 0.95 * lev / entry      # запас 5% на комиссию и проскальзывание
    qty = round_down(min(sized.qty, margin_cap), instrument.qty_step)
    if margin_cap < instrument.min_qty:
        raise ExecutionRefused("не хватает свободной маржи на счёте")
    if qty < instrument.min_qty or qty <= 0:
        raise ExecutionRefused(f"при риске {risk_pct:g}% объём меньше минимального для {sym} "
                               f"({_fmt(instrument.min_qty)})")
    if qty * entry < instrument.min_notional:
        raise ExecutionRefused(f"ордер меньше минимальных ${instrument.min_notional:g}")
    return OrderRequest(signal.id, sym, side, plan.entry_kind, qty, entry, stop, target, lev, expires,
                        qty * abs(entry - stop))


class TradeStatus(str, Enum):
    PLACED = "placed"           # ордер входа на бирже (лимитка ждёт ретеста)
    FILLED = "filled"           # вход исполнен, позиция со стопом и целью на бирже
    EXPIRED = "expired"         # лимитка не исполнилась вовремя (или цена ушла к цели без ретеста) и снята
    TIMED_OUT = "timed_out"     # позиция закрыта по рынку: истёк срок сделки (max_hold_bars)
    CANCELLED = "cancelled"     # снята вручную на бирже


@dataclass(frozen=True)
class Trade:
    signal_id: int
    symbol: str
    side: Side
    kind: EntryKind
    qty: float
    price: float
    stop: float
    target: float
    order_id: str
    network: str                # "demo" / "live"
    status: TradeStatus = TradeStatus.PLACED
    expires_at: datetime | None = None
    created_at: datetime | None = None
    id: int | None = None

    @staticmethod
    def from_order(req: OrderRequest, order_id: str, network: str, now: datetime) -> Trade:
        status = TradeStatus.PLACED if req.kind is EntryKind.RETEST else TradeStatus.FILLED
        return Trade(req.signal_id, req.symbol, req.side, req.kind, req.qty, req.price, req.stop, req.target,
                     order_id, network, status, req.expires_at, now)
