"""Теневые сигналы: запись и закрытие по закрытым свечам (trader/domain/shadow.py). Не влияют на торговлю."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from ..domain.models import Timeframe
from ..domain.shadow import resolve
from .ports import MarketData, SettingsRepository, ShadowRepository

log = logging.getLogger(__name__)


class ShadowTracker:
    def __init__(self, market: MarketData, repo: ShadowRepository, settings: SettingsRepository) -> None:
        self.market, self.repo, self.settings = market, repo, settings

    async def update(self, tf: Timeframe) -> int:
        """Закрыть теневые сигналы ТФ, итог которых уже известен по закрытым свечам; вернуть, сколько закрыто."""
        s, now, step = self.settings.load(), datetime.now(timezone.utc), timedelta(minutes=tf.minutes)
        done = 0
        for shadow_id, kind, sig in self.repo.open_shadow(tf):
            try:
                bars = await self.market.bars_between(sig.symbol, tf, sig.bar_time, now)
                bars = bars[bars.index + step <= now]                 # только закрытые свечи
                out = resolve(sig, bars, s, now)
            except Exception:
                log.exception("теневой сигнал %s %s", sig.symbol, kind)
                continue
            if out is not None:
                self.repo.finish_shadow(shadow_id, out.state, out.r, out.closed_at)
                done += 1
        return done
