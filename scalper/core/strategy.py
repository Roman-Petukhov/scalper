"""
Flow Scalper: два сетапа на признаках потока ордеров.

1. Momentum ignition: всплеск агрессивной дельты, подтверждённый стаканом,
   пробивает микро-диапазон. Входим по направлению потока.
2. Exhaustion fade: цена растянута от VWAP (или прошёл каскад ликвидаций),
   агрессоры давят, а цена стоит (поглощение), стакан разворачивается.
   Входим против движения.
"""
from __future__ import annotations

from collections import deque

from .features import Snapshot
from .models import Side, Signal


class FlowScalper:
    def __init__(self, cfg):
        self.s = cfg.strategy
        self.windows = sorted(int(w) for w in cfg.features.delta_windows_s)
        self.w_fast, self.w_mid = self.windows[0], self.windows[1]
        self.tick = cfg.instrument.tick_size
        self.last_eval = 0.0
        self.last_trade_ts = -1e18
        self.entries: deque[float] = deque()
        self.stats: dict[str, int] = {}

    def _count(self, key: str) -> None:
        self.stats[key] = self.stats.get(key, 0) + 1

    def notify_entry(self, ts: float) -> None:
        self.last_trade_ts = ts
        self.entries.append(ts)

    # ---------- фильтры ----------

    def _filters_ok(self, x: Snapshot) -> bool:
        f = self.s.filters
        if not x.ready:
            return False
        if x.ts - self.last_trade_ts < f.cooldown_s:
            return self._reject("cooldown")
        while self.entries and self.entries[0] < x.ts - 3600:
            self.entries.popleft()
        if len(self.entries) >= f.max_trades_per_hour:
            return self._reject("hourly_limit")
        if x.spread_bps > f.max_spread_bps:
            return self._reject("spread")
        if not (f.min_vol_bps_per_min <= x.vol_bps_min <= f.max_vol_bps_per_min):
            return self._reject("vol_regime")
        if x.vol_ratio > f.max_vol_ratio:
            return self._reject("vol_shock")
        if self.s.require_book and x.obi is None:
            return self._reject("no_book")
        return True

    def _reject(self, why: str) -> bool:
        self._count("reject_" + why)
        return False

    def _funding_ok(self, side: Side, x: Snapshot) -> bool:
        lim = self.s.filters.max_abs_funding
        # высокий положительный funding = толпа в лонгах, лонг не добавляем
        if side is Side.LONG and x.funding > lim:
            return False
        if side is Side.SHORT and x.funding < -lim:
            return False
        return True

    # ---------- сетапы ----------

    def _momentum(self, x: Snapshot) -> Signal | None:
        m = self.s.momentum
        zf, zm = x.z[self.w_fast], x.z[self.w_mid]
        for side in (Side.LONG, Side.SHORT):
            d = side.value
            if zf * d < m.z_fast or zm * d < m.z_mid:
                continue
            if x.obi is not None and x.obi * d < m.obi_min:
                self._count("mom_obi_block")
                continue
            brk = m.min_breakout_ticks * self.tick
            if side is Side.LONG and x.price < x.range_hi + brk:
                continue
            if side is Side.SHORT and x.price > x.range_lo - brk:
                continue
            if x.vwap_std > 0 and (x.price - x.vwap) * d > m.max_vwap_dev_sigma * x.vwap_std:
                self._count("mom_overextended")
                continue
            if not self._funding_ok(side, x):
                continue
            sp = x.sigma_price(m.horizon_s)
            return self._make(x, side, "momentum", m.stop_sigma * sp, m.tp_sigma * sp,
                              f"zf={zf:.1f} zm={zm:.1f} obi={_f(x.obi)} brk")
        return None

    def _fade(self, x: Snapshot) -> Signal | None:
        fd = self.s.fade
        zm = x.z[self.w_mid]
        for side in (Side.LONG, Side.SHORT):
            d = side.value
            # для лонга: цена ниже VWAP - kσ ИЛИ прошёл каскад ликвидаций лонгов
            stretched = x.vwap_std > 0 and (x.vwap - x.price) * d >= fd.band_sigma * x.vwap_std
            liq = (x.liq_long if side is Side.LONG else x.liq_short) >= fd.liq_min_notional
            if not (stretched or liq):
                continue
            # поглощение: агрессоры давят против нас (zm*d отрицателен), а цена почти стоит
            if zm * d > -fd.absorb_z:
                continue
            if x.move15_sigma * d < -fd.absorb_max_move_sigma:
                continue
            if x.obi is not None and x.obi * d < fd.obi_flip:
                self._count("fade_obi_block")
                continue
            if not self._funding_ok(side, x):
                continue
            sp = x.sigma_price(fd.horizon_s)
            why = "liq" if liq else "band"
            return self._make(x, side, "fade", fd.stop_sigma * sp, fd.tp_sigma * sp,
                              f"{why} zm={zm:.1f} mv={x.move15_sigma:.2f} obi={_f(x.obi)}")
        return None

    def _make(self, x: Snapshot, side: Side, setup: str, stop: float, tp: float, why: str) -> Signal:
        stop = max(stop, x.price * self.s.min_stop_bps / 1e4)
        tp = max(tp, x.price * self.s.min_tp_bps / 1e4)
        self._count("signal_" + setup)
        return Signal(ts=x.ts, side=side, setup=setup, price=x.price,
                      stop_dist=stop, tp_dist=tp, reason=why)

    def evaluate(self, x: Snapshot) -> Signal | None:
        if x.ts - self.last_eval < self.s.eval_interval_s:
            return None
        self.last_eval = x.ts
        if not self._filters_ok(x):
            return None
        sig = None
        if self.s.momentum.enabled:
            sig = self._momentum(x)
        if sig is None and self.s.fade.enabled:
            sig = self._fade(x)
        return sig


def _f(v: float | None) -> str:
    return "na" if v is None else f"{v:.2f}"
