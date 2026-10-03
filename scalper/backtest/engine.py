"""Бэктест: реплей событий через ту же сессию, что и в live, с paper-брокером."""
from __future__ import annotations

import logging
import time
from typing import Iterable

from core.models import Trade
from core.risk import Instrument
from core.session import TradingSession
from exchange.broker_paper import PaperBroker

log = logging.getLogger("backtest")


def run_backtest(cfg, events: Iterable, total: int | None = None, progress: bool = True) -> TradingSession:
    inst = Instrument(cfg.exchange.symbol, cfg.instrument.tick_size,
                      cfg.instrument.step_size, cfg.instrument.min_notional)
    equity = float(cfg.paper.start_equity)
    broker = PaperBroker(cfg, inst.tick_size, inst.step_size, equity)
    session = TradingSession(cfg, inst, broker, equity)
    t0 = time.time()
    n = 0
    last = None
    for ev in events:
        session.on_event(ev)
        last = ev
        n += 1
        if progress and n % 500_000 == 0:
            pct = f"{n / total * 100:5.1f}%" if total else ""
            print(f"\r  {pct} {n:,} событий, сделок: {len(session.manager.trades)}, "
                  f"{n / (time.time() - t0):,.0f} ev/s", end="", flush=True)
    if progress:
        print()
    # закрываем незавершённую позицию по последней цене
    if last is not None and not session.manager.is_flat:
        session.manager.flatten(last.ts, "end_of_data")
        for k in range(1, 50):
            ts = last.ts + k * 0.1
            broker.on_event(Trade(ts, session.features.last_price, 0.0, False))
            session.manager.on_market(session.features.snapshot(ts))
            if session.manager.is_flat:
                break
    session.elapsed = time.time() - t0
    session.events = n
    return session
