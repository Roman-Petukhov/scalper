"""«Погода рынка» раз в сутки (domain/weather.py): режим рынка, настроение толпы и правило бота задним числом за
последние дни рядом с живым журналом. Считается в фоне, чтобы не задерживать минутный цикл; снимок — в kv."""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from datetime import datetime, time, timedelta, timezone

import pandas as pd

from ..domain.liquidations import LiqTotals
from ..domain.models import HTF_CONFIRM, Settings, Signal, Timeframe
from ..domain.strategy import htf_breakouts, htf_confirmation, replay
from ..domain.weather import Assessment, Mood, Weather, alts_vs_btc, assess, replay_stats, ret, share_below_ema
from .ports import MarketData, Notifier, SentimentData, SettingsRepository, StablecoinData, TradeRepository

log = logging.getLogger(__name__)
KEY = "weather"
DOM_KEY = "weather:dominance"   # своя история доминации BTC (у CoinGecko бесплатно — только текущая)
DOM_KEEP = 30
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


async def _none() -> None:
    return None


def replay_since(tf: Timeframe, now: datetime) -> datetime:
    """Начало окна: 30 дней или сколько позволяют 1500 свечей после разгона правила (15m — около 11 дней)."""
    return now - min(timedelta(days=REPLAY_DAYS), timedelta(minutes=tf.minutes * (BARS - WARMUP)))


class WeatherService:
    def __init__(self, market: MarketData, mood: SentimentData | None, trades: TradeRepository,
                 settings: SettingsRepository, clock: Callable[[], datetime] = _utcnow, concurrency: int = 4,
                 stables: StablecoinData | None = None,
                 liquidations: Callable[[], LiqTotals | None] | None = None,
                 notifier: Notifier | None = None) -> None:
        self.market, self.mood, self.trades, self.settings = market, mood, trades, settings
        self.stables, self.liquidations, self.notifier = stables, liquidations, notifier
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

    def assessment(self) -> Assessment | None:
        w = self.current()
        return assess(w) if w is not None else None

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
            raw = json.dumps(w.to_json())
            self.trades.kv_set(KEY, raw)
            self.trades.kv_set(f"{KEY}:{w.day.isoformat()}", raw)          # история: проверить потом на живых данных
            liq = self.liquidations() if self.liquidations is not None else None
            if liq is not None:
                self.trades.kv_set(f"{KEY}:liq:{w.day.isoformat()}", json.dumps(
                    {"long_usd": liq.long_usd, "short_usd": liq.short_usd, "since": liq.since.isoformat()}))
            if self.notifier is not None:
                try:
                    await self.notifier.text(assess(w).text())
                except Exception:
                    log.exception("погода: уведомление")
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
        mood = await self._mood(universe[:COINS], now)
        reps = []
        for tf in REPLAY_TFS:
            if tf in s.timeframes:
                reps.append(await self._replay(tf, universe, s, now))
        return Weather(day=now.date(), computed_at=now, coins=len(closes),
                       below_ema=share_below_ema(closes), below_ema_week=share_below_ema(closes, lag=7),
                       alts_vs_btc_30d=alts_vs_btc(closes, btc), btc_7d=ret(btc, 7), btc_30d=ret(btc, 30),
                       mood=mood, replay=tuple(reps))

    async def _mood(self, symbols: list[str], now: datetime) -> Mood:
        if self.mood is None:
            return Mood()
        m = self.mood
        names = ("страх и жадность", "funding и базис", "лонги/шорты BTC", "крупные трейдеры BTC", "открытый интерес BTC",
                 "доминация BTC", "DVOL", "стейблкоины")
        calls = (m.fear_greed(), m.premium_now(), m.btc_long_short(), m.btc_top_long_short(), m.btc_open_interest(),
                 m.btc_dominance(), m.btc_dvol(), self.stables.supply() if self.stables is not None else _none())
        fg, prem, ls, top, oi, dom, dvol, stab = await asyncio.gather(*calls, return_exceptions=True)
        for nm, x in zip(names, (fg, prem, ls, top, oi, dom, dvol, stab)):
            if isinstance(x, Exception):
                log.warning("погода: %s — %s", nm, x)
        out: dict = {}
        if isinstance(fg, list) and fg:
            out |= {"fear_greed": fg[0][0], "fear_greed_label": fg[0][1],
                    "fear_greed_week": fg[7][0] if len(fg) > 7 else None}
        if isinstance(prem, dict):
            fund = sorted(prem[x][0] for x in symbols if x in prem)
            basis = sorted(prem[x][1] for x in symbols if x in prem)
            if fund:
                out |= {"funding_median": fund[len(fund) // 2], "funding_pos_share": sum(r > 0 for r in fund) / len(fund),
                        "basis_median": basis[len(basis) // 2]}
            if BTC in prem:
                out["basis_btc"] = prem[BTC][1]
        for key, x in (("btc_long_short", ls), ("top_long_short", top)):
            if isinstance(x, list) and x:
                out |= {key: x[-1], f"{key}_week": x[0] if len(x) > 7 else None}
        if isinstance(oi, list) and len(oi) > 7 and oi[0] > 0:
            out["oi_btc_7d"] = oi[-1] / oi[0] - 1
        if isinstance(dvol, list) and dvol:
            out |= {"dvol": dvol[-1], "dvol_week": dvol[0] if len(dvol) > 7 else None}
        if isinstance(dom, float):
            out |= {"dominance": dom, "dominance_week": self._dominance_week(now, dom)}
        if isinstance(stab, pd.Series) and len(stab) > 30 and stab.iloc[-31] > 0:
            out["stables_30d"] = float(stab.iloc[-1] / stab.iloc[-31] - 1)
        return Mood(**out)

    def _dominance_week(self, now: datetime, value: float) -> float | None:
        """Запомнить сегодняшнюю доминацию и вернуть значение неделю назад (±1 день), если оно было записано."""
        try:
            hist = json.loads(self.trades.kv_get(DOM_KEY) or "{}")
        except ValueError:
            hist = {}
        hist[now.date().isoformat()] = value
        hist = dict(sorted(hist.items())[-DOM_KEEP:])
        self.trades.kv_set(DOM_KEY, json.dumps(hist))
        week = now.date() - timedelta(days=7)
        for d in (week, week - timedelta(days=1), week + timedelta(days=1)):
            if d.isoformat() in hist:
                return float(hist[d.isoformat()])
        return None

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
