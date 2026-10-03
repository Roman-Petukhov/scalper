"""Сайзинг позиции и защитные лимиты (дневной убыток, серия убытков)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from .models import Signal


@dataclass
class Instrument:
    symbol: str
    tick_size: float
    step_size: float
    min_notional: float

    def round_price(self, p: float, mode: str = "nearest") -> float:
        t = self.tick_size
        k = p / t
        k = math.floor(k + 1e-9) if mode == "down" else math.ceil(k - 1e-9) if mode == "up" else round(k)
        return round(k * t, 10)

    def round_qty(self, q: float) -> float:
        s = self.step_size
        return round(math.floor(q / s + 1e-9) * s, 10)


def utc_day(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


class RiskManager:
    def __init__(self, cfg, equity: float, on_halt=None):
        self.r = cfg.risk
        self.equity = equity
        self.day = ""
        self.day_start_equity = equity
        self.day_pnl = 0.0
        self.consec_losses = 0
        self.paused_until = 0.0
        self.halted_day = ""
        self.on_halt = on_halt or (lambda reason: None)

    def _roll_day(self, ts: float) -> None:
        d = utc_day(ts)
        if d != self.day:
            self.day = d
            self.day_start_equity = self.equity
            self.day_pnl = 0.0

    def can_trade(self, ts: float) -> tuple[bool, str]:
        self._roll_day(ts)
        if self.halted_day == self.day:
            return False, "daily_loss_halt"
        if ts < self.paused_until:
            return False, "loss_streak_pause"
        if self.equity <= 0:
            return False, "no_equity"
        return True, ""

    def size(self, sig: Signal, entry_price: float, inst: Instrument) -> tuple[float, str]:
        risk_usdt = self.equity * self.r.risk_per_trade
        qty = risk_usdt / sig.stop_dist
        max_qty = self.equity * self.r.max_leverage_used / entry_price
        qty = inst.round_qty(min(qty, max_qty))
        if qty <= 0:
            return 0.0, "qty_zero"
        if qty * entry_price < inst.min_notional:
            # минимальный номинал больше, чем позволяет риск: пропускаем, а не раздуваем риск
            return 0.0, f"below_min_notional({qty * entry_price:.1f}<{inst.min_notional})"
        return qty, ""

    def on_trade_closed(self, net_pnl: float, ts: float) -> None:
        self._roll_day(ts)
        self.equity += net_pnl
        self.day_pnl += net_pnl
        self.consec_losses = self.consec_losses + 1 if net_pnl < 0 else 0
        if self.day_pnl <= -self.r.max_daily_loss * self.day_start_equity and self.halted_day != self.day:
            self.halted_day = self.day
            self.on_halt(f"Дневной лимит убытка: {self.day_pnl:.2f} USDT. Торговля остановлена до конца UTC-суток.")
        if self.consec_losses >= self.r.max_consecutive_losses:
            self.paused_until = ts + self.r.pause_after_losses_s
            self.consec_losses = 0
            self.on_halt(f"{self.r.max_consecutive_losses} убытков подряд: пауза "
                         f"{self.r.pause_after_losses_s // 60:.0f} мин.")
