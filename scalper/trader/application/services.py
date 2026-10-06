"""Сценарии: сканирование рынка после закрытия свечи, решения по сигналам, изменение настроек."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

import pandas as pd

from ..domain.models import HTF_CONFIRM, Mode, Settings, Signal, SignalStatus, Timeframe
from ..domain.strategy import detect, htf_breakouts, htf_confirmation
from ..domain.execution import ExecutionRefused
from .execution import Executor, _short
from .ports import ChartRenderer, MarketData, Notifier, SettingsRepository, SignalRepository

log = logging.getLogger(__name__)

CATCHUP_BARS = 12                       # после перерыва проверяем до стольких прошлых свечей (окно ретеста 4h)
ALERT_EVERY = timedelta(hours=3)        # тревога о сбое скана — не чаще раза в 3 часа на ТФ
ERROR_SHARE = 0.2                       # ошибок больше 20% монет — скан считается сбойным


@dataclass
class ScanReport:
    timeframe: Timeframe
    symbols: int
    signals: list[Signal]
    errors: int
    started: datetime
    finished: datetime


class Scanner:
    def __init__(self, market: MarketData, signals: SignalRepository, settings: SettingsRepository,
                 charts: ChartRenderer, notifier: Notifier | None, panel_url: str, concurrency: int = 8,
                 executor: Executor | None = None, heartbeat: Callable[[], Awaitable[None]] | None = None) -> None:
        self.market, self.signals, self.settings = market, signals, settings
        self.executor, self.heartbeat = executor, heartbeat
        self.charts, self.notifier, self.panel_url = charts, notifier, panel_url
        self.sem = asyncio.Semaphore(concurrency)
        self.last: dict[Timeframe, ScanReport] = {}
        self._alerted: dict[Timeframe, datetime] = {}

    async def _one(self, symbol: str, tf: Timeframe, s: Settings, back: int = 0) -> list[Signal]:
        """Сигналы на последней закрытой свече и на `back` свечах до неё (после перерыва); с прошлых свечей —
        только те, по которым время на вход ещё не вышло."""
        async with self.sem:
            bars = await self.market.closed_bars(symbol, tf)
        now = datetime.now(timezone.utc)
        found: list[tuple[Signal, pd.DataFrame]] = []
        for j in range(min(back, max(len(bars) - 1, 0)), -1, -1):
            sub = bars.iloc[: len(bars) - j] if j else bars
            for sig in detect(sub, tf, symbol, s):
                if j and now >= sig.valid_until():
                    continue
                found.append((replace(sig, extra={**sig.extra, "caught_up": True}) if j else sig, sub))
        hours, htf = s.p(tf).htf_confirm_h, HTF_CONFIRM.get(tf)
        if found and hours and htf is not None:                     # только вслед за свежим пробоем старшего ТФ
            async with self.sem:
                hbars = await self.market.closed_bars(symbol, htf)
            brk = htf_breakouts(hbars, htf)
            kept = []
            for sig, sub in found:
                age = htf_confirmation(brk, sig.side, sig.bar_time + timedelta(minutes=tf.minutes), hours)
                if age is not None:
                    kept.append((replace(sig, extra={**sig.extra, "htf": htf.value, "htf_age_h": round(age, 1)}), sub))
            found = kept
        new = []
        for sig, sub in found:
            saved = self.signals.add(sig)
            if saved is None:
                continue
            try:
                path = await asyncio.to_thread(self.charts.render, saved, sub)
                self.signals.set_chart(saved.id, path)
                saved = replace(saved, chart_path=path)
            except Exception:
                log.exception("график %s %s", symbol, tf.value)
            new.append(saved)
        return new

    async def scan(self, tf: Timeframe, back: int = 0) -> ScanReport:
        """Один проход по рынку для таймфрейма (back — сколько прошлых свечей проверить после перерыва). Сигналы
        появляются, только если ТФ включён в настройках. Сбой — тревога в уведомления, иначе отметка heartbeat."""
        s = self.settings.load()
        started = datetime.now(timezone.utc)
        if tf not in s.timeframes:
            rep = ScanReport(tf, 0, [], 0, started, started)
            self.last[tf] = rep
            return rep
        try:
            symbols = await self.market.universe(s.min_turnover_usd)
        except Exception as e:
            log.exception("список монет %s", tf.value)
            await self._alert(tf, f"не получил список монет с Binance ({_short(e)})")
            raise
        if s.p(tf).top_n:
            symbols = symbols[: s.p(tf).top_n]                      # только самые ликвидные — как в бэктесте ТФ
        res = await asyncio.gather(*(self._one(x, tf, s, back) for x in symbols), return_exceptions=True)
        found, errors = [], 0
        for sym, r in zip(symbols, res):
            if isinstance(r, Exception):
                errors += 1
                log.warning("скан %s %s: %s", sym, tf.value, r)
            else:
                found += r
        if self.notifier is not None:
            for sig in found:
                try:
                    await self.notifier.signal(sig, sig.chart_path, self.panel_url)
                except Exception:
                    log.exception("уведомление %s", sig.symbol)
        if (s.mode is Mode.AUTO and tf in s.auto_timeframes and self.executor is not None
                and self.executor.connected):
            found = [await self._auto(sig) for sig in found]
        rep = ScanReport(tf, len(symbols), found, errors, started, datetime.now(timezone.utc))
        self.last[tf] = rep
        log.info("скан %s: монет %d, сигналов %d, ошибок %d%s", tf.value, len(symbols), len(found), errors,
                 f", проверено и {back} прошлых свечей" if back else "")
        if not symbols:
            await self._alert(tf, "ни одной монеты с нужным оборотом — проверь доступ к Binance")
        elif errors > ERROR_SHARE * len(symbols):
            await self._alert(tf, f"не загрузились свечи {errors} из {len(symbols)} монет")
        elif self.heartbeat is not None:
            try:
                await self.heartbeat()
            except Exception:
                log.exception("heartbeat")
        return rep

    async def _alert(self, tf: Timeframe, what: str) -> None:
        """Тревога о сбое скана — в уведомления, не чаще ALERT_EVERY на таймфрейм."""
        now = datetime.now(timezone.utc)
        if self.notifier is None or now - self._alerted.get(tf, now - ALERT_EVERY) < ALERT_EVERY:
            return
        self._alerted[tf] = now
        try:
            await self.notifier.text(f"⚠️ Скан {tf.value}: {what}. Сигналы могут не приходить.")
        except Exception:
            log.exception("уведомление о сбое")


    async def _auto(self, sig: Signal) -> Signal:
        """Авто-режим: бот сам отправляет ордер; отказ по правилам риска — сигнал пропущен с причиной."""
        assert self.executor is not None and sig.id is not None
        if datetime.now(timezone.utc) >= sig.valid_until():          # сигнал со свечи, пропущенной в перерыве
            return self.signals.set_status(sig.id, SignalStatus.EXPIRED, "бот не вошёл: время на вход вышло") or sig
        try:
            await self.executor.execute(sig)
            msg = f"Бот вошёл: {sig.symbol} {sig.timeframe.value} {sig.side.label}"
        except ExecutionRefused as e:
            self.signals.set_status(sig.id, SignalStatus.SKIPPED, f"бот не вошёл: {e}")
            msg = f"Бот не вошёл в {sig.symbol} {sig.timeframe.value}: {e}"
        if self.notifier is not None:
            try:
                await self.notifier.text(msg)
            except Exception:
                log.exception("уведомление")
        return self.signals.get(sig.id) or sig


class SignalDecisions:
    """Ручной режим: трейдер принимает или пропускает сигнал. В авто-режиме решение принимает исполнитель."""

    def __init__(self, signals: SignalRepository, settings: SettingsRepository,
                 executor: Executor | None = None) -> None:
        self.signals, self.settings, self.executor = signals, settings, executor

    def _check(self, signal_id: int) -> Signal:
        sig = self.signals.get(signal_id)
        if sig is None:
            raise LookupError("сигнал не найден")
        if sig.status is not SignalStatus.NEW:
            raise ValueError(f"сигнал уже обработан: {sig.status.value}")
        s = self.settings.load()
        if s.mode is not Mode.MANUAL and sig.timeframe in s.auto_timeframes:
            raise ValueError("включён авто-режим: сигналы этого таймфрейма исполняет бот")
        return sig

    async def take(self, signal_id: int) -> Signal:
        """«Вхожу»: с подключённой биржей — ордер со стопом и целью на Bybit; без неё — только отметка.
        ValueError — ордер не отправлен (причина в тексте), сигнал остаётся новым."""
        sig = self._check(signal_id)
        if datetime.now(timezone.utc) >= sig.valid_until():
            self.signals.set_status(signal_id, SignalStatus.EXPIRED, "время на вход вышло")
            raise ValueError("Сигнал устарел: время на вход вышло")
        if self.executor is not None and self.executor.connected:
            try:
                await self.executor.execute(sig)
            except ExecutionRefused as e:
                raise ValueError(f"Ордер не отправлен: {e}") from e
            out = self.signals.get(signal_id)
        else:
            out = self.signals.set_status(signal_id, SignalStatus.TAKEN, "вход вручную · биржа не подключена")
        assert out is not None
        return out

    def expire_stale(self, now: datetime | None = None) -> int:
        """Новые сигналы, по которым время входа вышло, помечаем «истёк» — чтобы не висели с точкой входа."""
        now = now or datetime.now(timezone.utc)
        n = 0
        for sig in self.signals.recent(500):
            if sig.status is SignalStatus.NEW and sig.id is not None and now >= sig.valid_until():
                self.signals.set_status(sig.id, SignalStatus.EXPIRED, "время на вход вышло")
                n += 1
        return n

    def skip(self, signal_id: int) -> Signal:
        self._check(signal_id)
        out = self.signals.set_status(signal_id, SignalStatus.SKIPPED, "пропущен")
        assert out is not None
        return out


class SettingsService:
    def __init__(self, repo: SettingsRepository) -> None:
        self.repo = repo

    def get(self) -> Settings:
        return self.repo.load()

    def toggle_auto_timeframe(self, tf: Timeframe) -> Settings:
        s = self.repo.load().toggle_auto(tf)
        self.repo.save(s)
        return s

    def toggle_timeframe(self, tf: Timeframe) -> Settings:
        s = self.repo.load().toggle(tf)
        self.repo.save(s)
        return s

    def set_mode(self, mode: Mode) -> Settings:
        s = replace(self.repo.load(), mode=mode)
        self.repo.save(s)
        return s

    def reset_defaults(self) -> Settings:
        """Все настройки — к стандартным (лучшие по бэктесту, риск 1%, плечо 5×). Режим ручной / авто и
        включённые таймфреймы не меняем: это выбор трейдера, а не параметры правила."""
        cur = self.repo.load()
        s = replace(Settings(), mode=cur.mode, timeframes=cur.timeframes, auto_timeframes=cur.auto_timeframes)
        self.repo.save(s)
        return s

    GLOBAL_FIELDS = {"leverage", "max_positions", "daily_loss_pct"}

    def update(self, global_fields: dict, per_tf: dict[Timeframe, dict]) -> Settings:
        """Сохранить общие поля (плечо, позиции, дневной стоп) и правило каждого ТФ одним действием; диапазоны
        проверяют Settings и TfParams — при ошибке ничего не сохраняется."""
        bad = set(global_fields) - self.GLOBAL_FIELDS
        if bad:
            raise ValueError(f"неизвестные поля: {sorted(bad)}")
        s = replace(self.repo.load(), **global_fields)
        for tf, fields in per_tf.items():
            s = s.with_tf(tf, **fields)
        self.repo.save(s)
        return s
