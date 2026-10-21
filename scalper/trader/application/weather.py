"""«Погода рынка» раз в сутки (domain/weather.py): режим рынка, настроение толпы и правило бота задним числом за
последние дни рядом с живым журналом. Считается в фоне, чтобы не задерживать минутный цикл; снимок — в kv."""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from datetime import datetime, time, timedelta, timezone

import pandas as pd

from ..domain.models import HTF_CONFIRM, Settings, Signal, Timeframe
from ..domain.strategy import htf_breakouts, htf_confirmation, replay
from ..domain.weather import Mood, Weather, alts_vs_btc, replay_stats, ret, share_below_ema
from .ports import MarketData, SentimentData, SettingsRepository, TradeRepository

log = logging.getLogger(__name__)
KEY = "weather"
BTC = "BTCUSDT"
COINS = 100                     # режим — по самым ликвидным монетам
DAYS = 70                       # дневных свечей: EMA50 + неделя назад + запас
REPLAY_DAYS = 30
WARMUP = 400                    # правилу нужно столько свечей до сигнала (strategy.detect)
BARS = 1500                     # столько свечей отдаёт closed_bars
AFTER_CLOSE = timedelta(minutes=10)
RETRY = timedelta(minutes=30)
REPLAY_TFS = (Timeframe.H4, Timeframe.M15, Timeframe.M5)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def replay_since(tf: Timeframe, now: datetime) -> datetime:
    """Начало окна: 30 дней или сколько позволяют 1500 свечей после разгона правила (15m — около 11 дней)."""
    return now - min(timedelta(days=REPLAY_DAYS), timedelta(minutes=tf.minutes * (BARS - WARMUP)))


class WeatherService:
    def __init__(self, market: MarketData, mood: SentimentData | None, trades: TradeRepository,
                 settings: SettingsRepository, clock: Callable[[], datetime] = _utcnow, concurrency: int = 4) -> None:
        self.market, self.mood, self.trades, self.settings = market, mood, trades, settings
        self.clock = clock
        self.sem = asyncio.Semaphore(concurrency)
        self._tried: datetime | None = None
        self._task: asyncio.Task | None = None
        self.last_error: str | None = None

    def current(self) -> Weather | None:
        raw = self.trades.kv_get(KEY)
        if not raw:
            return None
        try:
            return Weather.from_json(json.loads(raw))
        except (ValueError, KeyError, TypeError):
            log.exception("погода: снимок не читается")
            return None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def due(self, now: datetime) -> bool:
        if self.running or (self._tried is not None and now - self._tried < RETRY):
            return False
        w = self.current()
        if w is None:
            return True
        today_close = datetime.combine(now.date(), time(), timezone.utc) + AFTER_CLOSE
        return now >= today_close and w.computed_at < today_close

    async def maybe_run(self) -> None:
        """Для минутного цикла: раз в сутки после закрытия дня (или если снимка нет) — запустить расчёт в фоне."""
        now = self.clock()
        if self.due(now):
            self._tried = now
            self._task = asyncio.create_task(self._guarded())

    async def _guarded(self) -> None:
        try:
            w = await self.run()
            self.trades.kv_set(KEY, json.dumps(w.to_json()))
            self.last_error = None
        except Exception as e:
            self.last_error = str(e)[:200]
            log.exception("погода рынка")

    async def _daily(self, symbol: str) -> pd.Series | None:
        async with self.sem:
            try:
                return (await self.market.daily_bars(symbol, DAYS))["close"].astype(float)
            except Exception as e:
                log.warning("погода: дневки %s — %s", symbol, e)
                return None

    async def run(self) -> Weather:
        now = self.clock()
        s = self.settings.load()
        universe = await self.market.universe(s.min_turnover_usd)
        alts = [x for x in universe if x != BTC][:COINS]
        btc, *closes = await asyncio.gather(self._daily(BTC), *(self._daily(x) for x in alts))
        closes = [c for c in closes if c is not None and len(c)]
        if btc is None or not closes:
            raise RuntimeError("нет дневных свечей BTC или альтов")
        mood = await self._mood(universe[:COINS])
        reps = []
        for tf in REPLAY_TFS:
            if tf in s.timeframes:
                reps.append(await self._replay(tf, universe, s, now))
        return Weather(day=now.date(), computed_at=now, coins=len(closes),
                       below_ema=share_below_ema(closes), below_ema_week=share_below_ema(closes, lag=7),
                       alts_vs_btc_30d=alts_vs_btc(closes, btc), btc_7d=ret(btc, 7), btc_30d=ret(btc, 30),
                       mood=mood, replay=tuple(reps))

    async def _mood(self, symbols: list[str]) -> Mood:
        if self.mood is None:
            return Mood()
        fg, fund, ls = await asyncio.gather(self.mood.fear_greed(), self.mood.funding_now(), self.mood.btc_long_short(),
                                            return_exceptions=True)
        out: dict = {}
        if isinstance(fg, list) and fg:
            out |= {"fear_greed": fg[0][0], "fear_greed_label": fg[0][1],
                    "fear_greed_week": fg[7][0] if len(fg) > 7 else None}
        if isinstance(fund, dict):
            rates = [fund[x] for x in symbols if x in fund]
            if rates:
                rates.sort()
                out |= {"funding_median": rates[len(rates) // 2],
                        "funding_pos_share": sum(r > 0 for r in rates) / len(rates)}
        if isinstance(ls, list) and ls:
            out |= {"btc_long_short": ls[-1], "btc_long_short_week": ls[0] if len(ls) > 7 else None}
        for nm, x in (("страх и жадность", fg), ("funding", fund), ("лонги/шорты BTC", ls)):
            if isinstance(x, Exception):
                log.warning("погода: %s — %s", nm, x)
        return Mood(**out)

    async def _replay(self, tf: Timeframe, universe: list[str], s: Settings, now: datetime):
        since = replay_since(tf, now)
        tp = s.p(tf)
        symbols = universe[: tp.top_n] if tp.top_n else universe
        htf = HTF_CONFIRM.get(tf) if tp.htf_confirm_h else None

        async def one(symbol: str) -> list[float]:
            async with self.sem:
                try:
                    bars = await self.market.closed_bars(symbol, tf)
                    hbars = await self.market.closed_bars(symbol, htf) if htf is not None else None
                except Exception as e:
                    log.warning("погода: свечи %s %s — %s", symbol, tf.value, e)
                    return []
            keep = None
            if hbars is not None:
                brk = htf_breakouts(hbars, htf)

                def keep(sig: Signal) -> bool:
                    at = sig.bar_time + timedelta(minutes=tf.minutes)
                    return htf_confirmation(brk, sig.side, at, tp.htf_confirm_h) is not None
            res = await asyncio.to_thread(replay, bars, tf, symbol, s, since, keep)
            return [r for _, _, r in res]

        parts = await asyncio.gather(*(one(x) for x in symbols))
        trades = [r for p in parts for r in p]
        live = [e.trade.r_multiple for e in self.trades.journal()
                if e.timeframe is tf and e.trade.r_multiple is not None and e.trade.opened_at is not None
                and e.trade.opened_at >= since]
        return replay_stats(tf.value, since, trades, live)
