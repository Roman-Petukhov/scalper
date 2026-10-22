"""Перевод сигнала в биржевой ордер: размер от риска, округление под правила монеты и защитные проверки.
Чистая логика без сети — биржа подставляется через порт Broker (application/ports.py)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

from .models import EntryKind, Settings, Side, Signal, Sizing, TradePlan
from .sizing import position_size

MIN_RR = 1.5                # рыночный вход: если цена ушла и до цели осталось меньше 1.5 стопа — не входим
PYRAMID_AT_R = 1.5          # доливка (research/pyramid.py, P2): цена прошла +1.5R от входа —
PYRAMID_SIZE = 0.5          # докупаем 0.5 исходного объёма, стоп всей позиции — в точку входа
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
    cut: str = ""               # почему риск меньше заданного (упёрлись в плечо или свободную маржу)

    @property
    def client_id(self) -> str:
        """Метка ордера на бирже: повторная отправка того же сигнала будет отклонена биржей."""
        return f"tt-{self.signal_id}"

    def describe(self) -> str:
        kind = "лимит" if self.kind is EntryKind.RETEST else "рынок"
        return (f"{self.side.label} {self.symbol} · {kind} {_fmt(self.qty)} @ {_fmt(self.price)} · "
                f"стоп {_fmt(self.stop)} · цель {_fmt(self.target)} · маржа ${self.qty * self.price / self.leverage:.2f} "
                f"× {self.leverage} · риск ${self.risk_usd:.2f}"
                + (f" ({self.cut})" if self.cut else ""))


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
                day_start_equity: float | None, now: datetime, tf_open: int = 0, hedge_beta: float = 0.0,
                hedge_lev: int = 0) -> OrderRequest:
    """Ордер по сигналу или ExecutionRefused с понятной причиной. tf_open — сколько из занятых монет заняты
    сделками того же таймфрейма; hedge_beta — к сделке добавится хедж BTC на бету × номинал, ему тоже нужна маржа
    (по плечу хеджа hedge_lev; 0 — как у сделки)."""
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
    tp = settings.p(signal.timeframe)
    risk_pct, margin_pct = tp.sized_for(signal.touch3, signal.long_line, signal.side)   # метки линии, лонги ×long_risk_x
    lev = int(max(1, min(settings.lev(signal.timeframe), instrument.max_leverage)))
    if settings.sizing is Sizing.MARGIN:                                 # как на Bybit: маржа × плечо
        want_qty = account.equity * margin_pct / 100 * lev / entry
        lev_capped, size_label = False, f"при марже {margin_pct:g}%"
    else:                                                                # убыток по стопу = risk_pct% капитала
        sized = position_size(account.equity, risk_pct,
                              TradePlan(plan.entry_kind, entry, stop, target, plan.valid_bars), lev)
        want_qty = sized.qty
        lev_capped, size_label = sized.risk_usd < account.equity * risk_pct / 100 * 0.999, f"при риске {risk_pct:g}%"
    # запас 5% на комиссию и проскальзывание; хедж BTC займёт ещё hedge_beta × номинал / плечо хеджа
    hl = hedge_lev or lev
    margin_cap = max(account.available, 0.0) * 0.95 / (entry * (1 / lev + max(hedge_beta, 0.0) / hl))
    qty = round_down(min(want_qty, margin_cap), instrument.qty_step)
    if margin_cap < instrument.min_qty:
        raise ExecutionRefused("не хватает свободной маржи на счёте")
    if qty < instrument.min_qty or qty <= 0:
        raise ExecutionRefused(f"{size_label} объём меньше минимального для {sym} ({_fmt(instrument.min_qty)})")
    if qty * entry < instrument.min_notional:
        raise ExecutionRefused(f"ордер меньше минимальных ${instrument.min_notional:g}")
    cut = ""
    if qty < 0.9 * want_qty or lev_capped:
        if settings.sizing is Sizing.MARGIN:
            cut = f"маржа ${qty * entry / lev:.2f} вместо ${want_qty * entry / lev:.2f}: не хватило свободной маржи"
        elif lev_capped and want_qty <= margin_cap:
            cut = f"вместо ${account.equity * risk_pct / 100:.2f}: номинал упёрся в плечо {lev}×"
        else:
            cut = f"вместо ${account.equity * risk_pct / 100:.2f}: не хватило свободной маржи"
        if hedge_beta > 0 and qty < 0.9 * want_qty:
            cut += " с учётом хеджа BTC"
    return OrderRequest(signal.id, sym, side, plan.entry_kind, qty, entry, stop, target, lev, expires,
                        qty * abs(entry - stop), cut=cut)


HEDGE_FEE = 5.5e-4          # тейкер Bybit на сторону: оценка комиссий хеджа BTC в журнале (domain/hedge.py)


@dataclass(frozen=True)
class HedgeLeg:
    """Хедж сделки позицией BTC (domain/hedge.py): бета на момент входа и цена BTC, когда хедж добавлен и снят."""
    beta: float
    btc_in: float | None = None
    btc_out: float | None = None


class TradeStatus(str, Enum):
    PLACED = "placed"           # ордер входа на бирже (лимитка ждёт ретеста)
    FILLED = "filled"           # вход исполнен, позиция со стопом и целью на бирже
    EXPIRED = "expired"         # лимитка не исполнилась вовремя (или цена ушла к цели без ретеста) и снята
    TIMED_OUT = "timed_out"     # позиция закрыта по рынку: истёк срок сделки (max_hold_bars)
    CANCELLED = "cancelled"     # снята вручную на бирже
    CLOSED = "closed"           # позиция закрыта (стоп, цель или вручную), итог записан


@dataclass(frozen=True)
class ClosedPnl:
    """Запись биржи о закрытии позиции (или её части): сторона позиции, объём, средние цены, итог с комиссиями."""
    side: Side
    qty: float
    entry: float
    exit: float
    pnl: float                  # $, по данным биржи (за вычетом комиссий)
    closed_at: datetime
    symbol: str = ""            # монета записи (если известна): запись из общей выдачи по всем монетам


@dataclass(frozen=True)
class TradeResult:
    entry_fill: float           # средняя цена входа на бирже
    exit: float                 # средняя цена выхода
    pnl_usd: float
    closed_at: datetime


@dataclass(frozen=True)
class Fill:
    """Исполнение ордера на бирже: сторона ордера (лонг — покупка), объём, цена, комиссия ($), сколько закрыло."""
    side: Side
    qty: float
    price: float
    fee: float
    closed_qty: float           # > 0 — исполнение закрывало позицию
    at: datetime


def settle_fills(fills: list[Fill], side: Side, entry: float, since: datetime, until: datetime) -> TradeResult | None:
    """Итог по исполнениям, когда биржа не отдаёт запись о закрытии: закрывающие исполнения против стороны сделки
    в окне [since, until), вход — средняя открывающих; объём, которого среди открывающих в окне нет (вход раньше окна
    в 7 дней, а в окне только доливка), — по цене ордера entry. Комиссии — всех исполнений в окне."""
    win = [f for f in fills if since <= f.at < until and f.qty > 0]
    close = [f for f in win if f.side is not side and f.closed_qty > 0]
    if not close:
        return None
    opened = [f for f in win if f.side is side and f.closed_qty <= 0]
    q = sum(f.closed_qty for f in close)
    q_in = sum(f.qty for f in opened)
    q_old = max(q - q_in, 0.0)                                  # вход раньше окна
    entry_fill = (sum(f.price * f.qty for f in opened) + entry * q_old) / (q_in + q_old)
    exit_ = sum(f.price * f.closed_qty for f in close) / q
    pnl = int(side) * (exit_ - entry_fill) * q - sum(f.fee for f in opened + close)
    return TradeResult(entry_fill=entry_fill, exit=exit_, pnl_usd=pnl, closed_at=max(f.at for f in close))


def settle(records: list[ClosedPnl], side: Side, since: datetime, until: datetime) -> TradeResult | None:
    """Итог сделки из записей биржи: только своя сторона и закрытия в окне [since, until); цены — средние по объёму."""
    own = [r for r in records if r.side is side and since <= r.closed_at < until and r.qty > 0]
    if not own:
        return None
    qty = sum(r.qty for r in own)
    return TradeResult(entry_fill=sum(r.entry * r.qty for r in own) / qty, exit=sum(r.exit * r.qty for r in own) / qty,
                       pnl_usd=sum(r.pnl for r in own), closed_at=max(r.closed_at for r in own))


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
    filled_at: datetime | None = None       # когда вход исполнился (рынок — сразу, ретест — когда заметили)
    result: TradeResult | None = None
    hedge: HedgeLeg | None = None
    add_qty: float | None = None            # доливка: None — ещё не было, 0 — стоп в безубыток без доливки
    add_price: float | None = None          # цена, по которой доливали (ориентир — mark в момент доливки)

    @property
    def hedge_r(self) -> float | None:
        """Итог хеджа BTC в R сделки (оценка по ценам BTC при добавлении и снятии хеджа, с комиссиями тейкера):
        BTC против стороны сделки на бету × номинал входа."""
        h = self.hedge
        if h is None or not h.btc_in or not h.btc_out or self.qty * self.risk_per_unit <= 0:
            return None
        notional = h.beta * self.qty * self.price
        pnl = -int(self.side) * notional * (h.btc_out / h.btc_in - 1) - 2 * HEDGE_FEE * notional
        return pnl / (self.qty * self.risk_per_unit)

    @property
    def opened_at(self) -> datetime | None:
        return self.filled_at or self.created_at

    @property
    def risk_per_unit(self) -> float:
        return abs(self.price - self.stop)

    @property
    def r_multiple(self) -> float | None:
        """Итог в R: R — риск по плану (объём × расстояние от входа до стопа)."""
        if self.result is None or self.qty * self.risk_per_unit <= 0:
            return None
        return self.result.pnl_usd / (self.qty * self.risk_per_unit)

    @property
    def pyramid_level(self) -> float:
        """Цена, на которой доливаем: +PYRAMID_AT_R R от входа по плану."""
        return self.price + int(self.side) * PYRAMID_AT_R * self.risk_per_unit

    @property
    def slippage_r(self) -> float | None:
        """Проскальзывание входа в R: плюс — вошли хуже плана. С доливкой биржа отдаёт среднюю цену двух входов —
        цена первого входа восстанавливается по объёмам."""
        if self.result is None or self.risk_per_unit <= 0:
            return None
        fill = self.result.entry_fill
        if self.add_qty and self.add_price and self.qty > 0:
            fill = (fill * (self.qty + self.add_qty) - self.add_price * self.add_qty) / self.qty
        return int(self.side) * (fill - self.price) / self.risk_per_unit

    @property
    def exit_reason(self) -> str:
        if self.status is TradeStatus.TIMED_OUT:
            return "по сроку"
        if self.result is None:
            return "нет данных"
        x, r = self.result.exit, self.risk_per_unit
        if r > 0 and abs(x - self.target) <= 0.25 * r:
            return "цель"
        if r > 0 and abs(x - self.stop) <= 0.25 * r:
            return "стоп"
        if r > 0 and self.add_qty is not None and abs(x - self.price) <= 0.25 * r:
            return "безубыток"
        return "вручную"

    @staticmethod
    def from_order(req: OrderRequest, order_id: str, network: str, now: datetime) -> Trade:
        status = TradeStatus.PLACED if req.kind is EntryKind.RETEST else TradeStatus.FILLED
        return Trade(req.signal_id, req.symbol, req.side, req.kind, req.qty, req.price, req.stop, req.target,
                     order_id, network, status, req.expires_at, now,
                     filled_at=now if status is TradeStatus.FILLED else None)


def pyramid_due(trade: Trade, mark: float) -> bool:
    """Пора доливать: доливки ещё не было, а цена прошла +PYRAMID_AT_R R от входа."""
    return trade.add_qty is None and trade.risk_per_unit > 0 and \
        int(trade.side) * (mark - trade.pyramid_level) >= 0


def pyramid_qty(trade: Trade, instrument: Instrument, mark: float, available: float, leverage: int) -> tuple[float, str]:
    """Объём доливки (0 — не доливаем) и причина, если не доливаем: шаг и минимум монеты, свободная маржа."""
    want = round_down(PYRAMID_SIZE * trade.qty, instrument.qty_step)
    if want < instrument.min_qty or want <= 0 or want * mark < instrument.min_notional:
        return 0.0, "объём доливки меньше минимального ордера"
    cap = round_down(max(available, 0.0) * 0.95 * leverage / mark, instrument.qty_step)
    if cap < want:
        return 0.0, "не хватает свободной маржи"
    return want, ""
