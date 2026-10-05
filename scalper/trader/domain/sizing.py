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
