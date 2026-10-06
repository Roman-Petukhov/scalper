"""Размер позиции от риска до стопа."""
from __future__ import annotations

from dataclasses import dataclass

from .models import TradePlan


@dataclass(frozen=True)
class Size:
    qty: float              # монет
    notional: float         # $
    risk_usd: float


def position_size(equity: float, risk_pct: float, plan: TradePlan, max_leverage: float = 5.0) -> Size:
    """Количество, при котором стоп стоит risk_pct% капитала; номинал не больше max_leverage·капитала
    (тогда риск меньше заданного)."""
    if equity <= 0 or plan.risk_per_unit <= 0:
        return Size(0.0, 0.0, 0.0)
    risk_usd = equity * risk_pct / 100
    qty = risk_usd / plan.risk_per_unit
    cap = equity * max_leverage / plan.entry
    qty = min(qty, cap)
    return Size(qty, qty * plan.entry, qty * plan.risk_per_unit)


TYPICAL_HEDGE_BETA = 1.3        # медиана беты монет бота к BTC (research/hedged.py)


@dataclass(frozen=True)
class Exposure:
    """Во что обходится одна сделка при типичном стопе: доли капитала."""
    stop_pct: float         # стоп, % от цены входа (типичный для ТФ)
    loss_pct: float         # убыток по стопу, % капитала (меньше риска, если номинал упёрся в плечо)
    notional_x: float       # номинал позиции, × капитала
    margin_pct: float       # маржа сделки (и хеджа BTC, если включён), % капитала
    fits: int               # столько таких сделок помещается в маржу одновременно


def exposure(risk_pct: float, stop_pct: float, leverage: float, hedge: bool) -> Exposure:
    notional_x = min(risk_pct / stop_pct, leverage) if stop_pct > 0 else 0.0
    margin = notional_x * (1 + (TYPICAL_HEDGE_BETA if hedge else 0.0)) / leverage * 100
    fits = int(95 // margin) if margin > 0 else 0              # 5% капитала — запас на комиссии
    return Exposure(stop_pct, notional_x * stop_pct, notional_x, margin, fits)


def risk_notes(risk_pct: float, daily_loss_pct: float, exp: Exposure, backtest_dd_r: float | None) -> list[str]:
    """Предупреждения о сочетании риска, плеча и дневного лимита — простыми словами."""
    notes = []
    if risk_pct >= daily_loss_pct:
        notes.append(f"Один стоп ({risk_pct:g}%) не меньше дневного лимита ({daily_loss_pct:g}%): после первого убытка "
                     "новых входов до конца дня не будет.")
    if backtest_dd_r is not None and risk_pct * backtest_dd_r >= 25:
        notes.append(f"Худшая просадка бэктеста — {backtest_dd_r:g}R (стопов): при риске {risk_pct:g}% это "
                     f"−{min(risk_pct * backtest_dd_r, 100):.0f}% капитала.")
    if exp.loss_pct < 0.9 * risk_pct:
        notes.append(f"Номинал упирается в плечо: при стопе ~{exp.stop_pct:.1f}% реальный риск ~{exp.loss_pct:.1f}%, "
                     "а не заданный.")
    if exp.fits < 2:
        notes.append("Маржи хватает только на одну такую сделку: следующие будут урезаны или не откроются. "
                     "Снизьте риск или поднимите плечо.")
    return notes
