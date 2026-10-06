"""Хедж BTC (research/hedged.py): к каждой сделке — противоположная позиция BTCUSDT на бету монеты к BTC.
По бэктесту 4h (2020–2026): R в месяц +3.1 → +2.6, разброс месяца 7.2 → 4.2R, худший месяц −9.0 → −4.4R.
На бирже позиция по монете одна (режим одной позиции), поэтому хеджи сделок складываются в одну позицию BTCUSDT,
которая раз в минуту подгоняется под сумму. Чистая логика без сети."""
from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .execution import HEDGE_FEE, Account, HedgeLeg, Instrument, round_down
from .models import Settings, Side

__all__ = ["HEDGE_FEE", "HEDGE_SYMBOL", "HedgeLeg", "beta", "held_qty", "hedge_leverage", "rebalance", "target_qty",
           "without_hedge"]

HEDGE_SYMBOL = "BTCUSDT"
BETA_BARS = 360                 # 4h-свечей: 60 дней, как в бэктесте
BETA_MIN_BARS = 120             # меньше — бета ненадёжна, сделка без хеджа
BETA_RANGE = (0.0, 3.0)         # отрицательная или огромная бета — шум данных, режем
REBALANCE_SHARE = 0.10          # докупать / продавать BTC, только если разница больше 10% нужной позиции


def beta(coin_close: pd.Series, btc_close: pd.Series, bars: int = BETA_BARS) -> float | None:
    """Бета логдоходностей монеты к BTC по общим свечам за последние bars; None — данных мало."""
    both = pd.concat([coin_close, btc_close], axis=1, join="inner").dropna()
    r = np.log(both[both > 0].dropna()).diff().dropna().iloc[-bars:]
    if len(r) < BETA_MIN_BARS:
        return None
    var = float(r.iloc[:, 1].var())
    if not var > 0:
        return None
    b = float(r.cov().iloc[0, 1]) / var
    return min(max(b, BETA_RANGE[0]), BETA_RANGE[1]) if math.isfinite(b) else None


def target_qty(legs: Iterable[tuple[Side, float, float]], btc_price: float) -> float:
    """Нужная позиция BTC в монетах (плюс — лонг) по сделкам (сторона, номинал позиции $, бета)."""
    if not btc_price > 0:
        return 0.0
    return sum(-int(side) * b * notional for side, notional, b in legs) / btc_price


def rebalance(target: float, current: float, inst: Instrument, btc_price: float) -> float:
    """Сколько BTC купить (плюс) или продать (минус), чтобы позиция стала target; 0 — оставить как есть.
    Мелкая разница (меньше REBALANCE_SHARE нужной позиции, шага объёма или минимального ордера) не торгуется."""
    delta = target - current
    if target == 0:
        return -current
    size = round_down(abs(delta), inst.qty_step)
    if size < inst.min_qty or size * btc_price < inst.min_notional or abs(delta) < REBALANCE_SHARE * abs(target):
        return 0.0
    return math.copysign(size, delta)


def hedge_leverage(settings: Settings, inst: Instrument) -> int:
    """Плечо, которое ставится на BTCUSDT: из настроек, но не выше потолка монеты на бирже."""
    return int(max(1, min(settings.hedge_lev, inst.max_leverage)))


def held_qty(acc: Account) -> float:
    """Позиция BTCUSDT на счёте в монетах со знаком."""
    return sum(int(p.side) * p.qty for p in acc.positions if p.symbol == HEDGE_SYMBOL)


def without_hedge(acc: Account) -> Account:
    """Счёт без позиции хеджа: она не занимает место среди позиций сделок."""
    return Account(acc.equity, acc.available, acc.upnl, tuple(p for p in acc.positions if p.symbol != HEDGE_SYMBOL),
                   acc.pending_symbols - {HEDGE_SYMBOL})
