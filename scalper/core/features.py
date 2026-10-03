"""
Инкрементальные признаки микроструктуры.

Все вычисления опираются на посекундные корзины, которые закрываются по времени
событий (exchange time). Поэтому бэктест, paper и live дают одинаковые значения
на одинаковом потоке данных.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from .models import Book, Liquidation, MarkPrice, Trade

MAX_GAP_FILL = 3600  # сколько пустых секунд досчитывать после разрыва данных


class RollingStats:
    """Скользящее среднее и стандартное отклонение по фиксированному числу значений."""

    __slots__ = ("buf", "n", "s", "ss", "_since_rebuild")

    def __init__(self, n: int):
        self.buf: deque[float] = deque(maxlen=n)
        self.n = n
        self.s = 0.0
        self.ss = 0.0
        self._since_rebuild = 0

    def push(self, x: float) -> None:
        if len(self.buf) == self.n:
            old = self.buf[0]
            self.s -= old
            self.ss -= old * old
        self.buf.append(x)
        self.s += x
        self.ss += x * x
        self._since_rebuild += 1
        if self._since_rebuild >= 10_000:  # гасим накопление ошибки округления
            self.s = math.fsum(self.buf)
            self.ss = math.fsum(v * v for v in self.buf)
            self._since_rebuild = 0

    def __len__(self) -> int:
        return len(self.buf)

    @property
    def mean(self) -> float:
        return self.s / len(self.buf) if self.buf else 0.0

    @property
    def std(self) -> float:
        k = len(self.buf)
        if k < 2:
            return 0.0
        var = (self.ss - self.s * self.s / k) / (k - 1)
        return math.sqrt(var) if var > 0 else 0.0

    def z(self, x: float) -> float:
        sd = self.std
        return (x - self.mean) / sd if sd > 0 else 0.0


@dataclass(slots=True)
class SecBucket:
    sec: int
    signed: float = 0.0     # агрессивная дельта, в базовой валюте
    volume: float = 0.0
    pv: float = 0.0         # Σ (p - ref) * q, для VWAP
    p2v: float = 0.0        # Σ (p - ref)² * q
    high: float = -math.inf
    low: float = math.inf
    close: float = 0.0


@dataclass
class Snapshot:
    ts: float
    ready: bool
    price: float
    bid: float
    ask: float
    spread_bps: float
    delta: dict[int, float]
    z: dict[int, float]
    sigma_1s: float              # σ лог-доходности за 1 секунду
    vol_bps_min: float           # σ за минуту, б.п.
    vol_ratio: float             # σ_fast / σ_slow
    vwap: float
    vwap_std: float
    range_hi: float
    range_lo: float
    move15_sigma: float          # сдвиг цены за 15s в единицах σ
    obi: float | None
    microprice: float | None
    liq_long: float              # USDT ликвидированных лонгов за окно
    liq_short: float
    funding: float
    extra: dict = field(default_factory=dict)

    def sigma_price(self, horizon_s: float) -> float:
        """Ожидаемый разброс цены (в цене) на горизонте."""
        return self.price * self.sigma_1s * math.sqrt(horizon_s)


class Features:
    def __init__(self, cfg, tick_size: float):
        f = cfg.features
        self.tick = tick_size
        self.windows: list[int] = sorted(int(w) for w in f.delta_windows_s)
        self.max_w = max(self.windows)
        self.range_w = int(f.range_window_s)
        self.range_ex = int(f.range_exclude_s)
        self.vwap_w = int(f.vwap_window_s)
        self.warmup_s = float(f.warmup_s)
        self.obi_levels = int(f.obi_levels)
        self.obi_weights = [f.obi_decay ** i for i in range(self.obi_levels)]
        self.obi_hl = float(f.obi_smooth_halflife_s)
        self.liq_window = float(f.liq_window_s)
        self.a_fast = 1 - 0.5 ** (1 / f.vol_halflife_fast_s)
        self.a_slow = 1 - 0.5 ** (1 / f.vol_halflife_slow_s)

        keep = max(self.max_w, self.range_w + self.range_ex, 16) + 2
        self.secs: deque[SecBucket] = deque(maxlen=keep)
        self.vwap_buf: deque[SecBucket] = deque()
        self.vwap_pv = self.vwap_v = self.vwap_p2v = 0.0
        self.vwap_ref: float | None = None
        self._vwap_since_rebuild = 0
        self.zstats = {w: RollingStats(int(f.z_history_s)) for w in self.windows}

        self.cur: SecBucket | None = None
        self.first_ts: float | None = None
        self.last_ts = 0.0
        self.last_price = 0.0
        self.last_buyer_maker = False
        self.var_fast = 0.0
        self.var_slow = 0.0
        self.n_returns = 0

        self.book: Book | None = None
        self.obi_raw: float | None = None
        self.obi_s: float | None = None
        self.obi_ts = 0.0
        self.liqs: deque[tuple[float, str, float]] = deque()
        self.funding = 0.0
        self.mark = 0.0

    # ---------- приём событий ----------

    def on_trade(self, t: Trade) -> None:
        if self.first_ts is None:
            self.first_ts = t.ts
            self.vwap_ref = t.price
        self.advance(t.ts)
        b = self.cur
        if b is None:
            b = self.cur = SecBucket(int(t.ts), close=t.price)
        q = t.qty
        b.signed += q if not t.buyer_maker else -q
        b.volume += q
        dp = t.price - self.vwap_ref
        b.pv += dp * q
        b.p2v += dp * dp * q
        if t.price > b.high:
            b.high = t.price
        if t.price < b.low:
            b.low = t.price
        b.close = t.price
        self.last_price = t.price
        self.last_buyer_maker = t.buyer_maker
        self.last_ts = t.ts

    def on_book(self, bk: Book) -> None:
        self.advance(bk.ts)
        self.book = bk
        n = min(self.obi_levels, len(bk.bids), len(bk.asks))
        if n == 0:
            return
        w = self.obi_weights
        bq = sum(w[i] * bk.bids[i][1] for i in range(n))
        aq = sum(w[i] * bk.asks[i][1] for i in range(n))
        raw = (bq - aq) / (bq + aq) if bq + aq > 0 else 0.0
        self.obi_raw = raw
        if self.obi_s is None:
            self.obi_s = raw
        else:
            dt = max(bk.ts - self.obi_ts, 0.0)
            a = 1 - 0.5 ** (dt / self.obi_hl) if self.obi_hl > 0 else 1.0
            self.obi_s += a * (raw - self.obi_s)
        self.obi_ts = bk.ts

    def on_liquidation(self, lq: Liquidation) -> None:
        self.advance(lq.ts)
        self.liqs.append((lq.ts, lq.side, lq.price * lq.qty))

    def on_mark(self, m: MarkPrice) -> None:
        self.advance(m.ts)
        self.funding = m.funding
        self.mark = m.mark

    def on_event(self, ev) -> None:
        if isinstance(ev, Trade):
            self.on_trade(ev)
        elif isinstance(ev, Book):
            self.on_book(ev)
        elif isinstance(ev, Liquidation):
            self.on_liquidation(ev)
        elif isinstance(ev, MarkPrice):
            self.on_mark(ev)

    # ---------- посекундные корзины ----------

    def advance(self, ts: float) -> None:
        """Закрыть все секунды до ts (включая пустые)."""
        if self.cur is None:
            return
        sec = int(ts)
        if sec <= self.cur.sec:
            return
        self._close_bucket(self.cur)
        gap = sec - self.cur.sec - 1
        close = self.cur.close
        start = self.cur.sec + 1 + max(0, gap - MAX_GAP_FILL)
        for s in range(start, sec):
            self._close_bucket(SecBucket(s, close=close))
        self.cur = SecBucket(sec, close=close)

    def _close_bucket(self, b: SecBucket) -> None:
        prev_close = self.secs[-1].close if self.secs else b.close
        self.secs.append(b)
        # волатильность
        if prev_close > 0 and b.close > 0:
            r = math.log(b.close / prev_close)
            r2 = r * r
            if self.n_returns == 0:
                self.var_fast = self.var_slow = r2
            else:
                self.var_fast += self.a_fast * (r2 - self.var_fast)
                self.var_slow += self.a_slow * (r2 - self.var_slow)
            self.n_returns += 1
        # история дельт для z-score
        n = len(self.secs)
        for w in self.windows:
            if n >= w:
                s = 0.0
                for i in range(n - w, n):
                    s += self.secs[i].signed
                self.zstats[w].push(s)
        # скользящий VWAP
        self.vwap_buf.append(b)
        self.vwap_pv += b.pv
        self.vwap_v += b.volume
        self.vwap_p2v += b.p2v
        while self.vwap_buf and self.vwap_buf[0].sec <= b.sec - self.vwap_w:
            old = self.vwap_buf.popleft()
            self.vwap_pv -= old.pv
            self.vwap_v -= old.volume
            self.vwap_p2v -= old.p2v
        self._vwap_since_rebuild += 1
        if self._vwap_since_rebuild >= 3600:
            self.vwap_pv = math.fsum(x.pv for x in self.vwap_buf)
            self.vwap_v = math.fsum(x.volume for x in self.vwap_buf)
            self.vwap_p2v = math.fsum(x.p2v for x in self.vwap_buf)
            self._vwap_since_rebuild = 0

    # ---------- снимок для стратегии ----------

    def best_bid_ask(self, ts: float) -> tuple[float, float]:
        bk = self.book
        if bk and bk.bids and bk.asks and ts - bk.ts < 1.0:
            bid, ask = bk.bids[0][0], bk.asks[0][0]
            # стакан приходит раз в 100мс; сделка свежее, сдвигаем границы по ней
            p = self.last_price
            if p and self.last_ts > bk.ts:
                if p >= ask:
                    ask, bid = p, max(bid, p - self.tick)
                elif p <= bid:
                    bid, ask = p, min(ask, p + self.tick)
            return bid, ask
        p = self.last_price
        if self.last_buyer_maker:          # сделка ударила в бид
            return p, p + self.tick
        return p - self.tick, p

    def snapshot(self, ts: float) -> Snapshot:
        self.advance(ts)
        p = self.last_price
        bid, ask = self.best_bid_ask(ts)
        mid = (bid + ask) / 2 if bid and ask else p
        spread_bps = (ask - bid) / mid * 1e4 if mid else 0.0

        secs = self.secs
        n = len(secs)
        cur_signed = self.cur.signed if self.cur else 0.0
        delta, z = {}, {}
        for w in self.windows:
            s = cur_signed
            for i in range(max(0, n - (w - 1)), n):
                s += secs[i].signed
            delta[w] = s
            z[w] = self.zstats[w].z(s)

        sigma = math.sqrt(self.var_fast) if self.var_fast > 0 else 0.0
        sigma_slow = math.sqrt(self.var_slow) if self.var_slow > 0 else 0.0
        vol_bps_min = sigma * math.sqrt(60) * 1e4
        vol_ratio = sigma / sigma_slow if sigma_slow > 0 else 1.0

        if self.vwap_v > 0:
            m = self.vwap_pv / self.vwap_v
            var = self.vwap_p2v / self.vwap_v - m * m
            vwap = self.vwap_ref + m
            vwap_std = math.sqrt(var) if var > 0 else 0.0
        else:
            vwap, vwap_std = p, 0.0

        lo_i = max(0, n - self.range_ex - self.range_w)
        hi_i = max(0, n - self.range_ex)
        rng = [secs[i] for i in range(lo_i, hi_i) if secs[i].volume > 0]
        range_hi = max((b.high for b in rng), default=p)
        range_lo = min((b.low for b in rng), default=p)

        ref15 = secs[n - 15].close if n >= 15 else (secs[0].close if n else p)
        sp15 = p * sigma * math.sqrt(15)
        move15 = (p - ref15) / sp15 if sp15 > 0 else 0.0

        cutoff = ts - self.liq_window
        while self.liqs and self.liqs[0][0] < cutoff:
            self.liqs.popleft()
        liq_long = sum(nv for _, side, nv in self.liqs if side == "SELL")
        liq_short = sum(nv for _, side, nv in self.liqs if side == "BUY")

        obi = micro = None
        bk = self.book
        if bk and bk.bids and bk.asks and ts - bk.ts < 2.0:
            obi = self.obi_s
            bp, bq = bk.bids[0]
            ap, aq = bk.asks[0]
            if bq + aq > 0:
                micro = (bp * aq + ap * bq) / (bq + aq)

        ready = (
            self.first_ts is not None
            and ts - self.first_ts >= self.warmup_s
            and self.n_returns > 60
            and len(self.zstats[self.windows[0]]) > 60
        )
        return Snapshot(ts=ts, ready=ready, price=p, bid=bid, ask=ask, spread_bps=spread_bps,
                        delta=delta, z=z, sigma_1s=sigma, vol_bps_min=vol_bps_min,
                        vol_ratio=vol_ratio, vwap=vwap, vwap_std=vwap_std,
                        range_hi=range_hi, range_lo=range_lo, move15_sigma=move15,
                        obi=obi, microprice=micro, liq_long=liq_long, liq_short=liq_short,
                        funding=self.funding)
