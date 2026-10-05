"""Сценарии: сканирование рынка после закрытия свечи, решения по сигналам, изменение настроек."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from ..domain.models import EntryPolicy, Mode, Settings, Signal, SignalStatus, Timeframe
from ..domain.strategy import detect
from ..domain.execution import ExecutionRefused
from .execution import Executor
from .ports import ChartRenderer, MarketData, Notifier, SettingsRepository, SignalRepository

log = logging.getLogger(__name__)


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
                 executor: Executor | None = None) -> None:
        self.market, self.signals, self.settings = market, signals, settings
        self.executor = executor
        self.charts, self.notifier, self.panel_url = charts, notifier, panel_url
        self.sem = asyncio.Semaphore(concurrency)
        self.last: dict[Timeframe, ScanReport] = {}

    async def _one(self, symbol: str, tf: Timeframe, s: Settings) -> list[Signal]:
        async with self.sem:
            bars = await self.market.closed_bars(symbol, tf)
        new = []
        for sig in detect(bars, tf, symbol, s):
            saved = self.signals.add(sig)
            if saved is None:
                continue
            try:
                path = await asyncio.to_thread(self.charts.render, saved, bars)
                self.signals.set_chart(saved.id, path)
                saved = replace(saved, chart_path=path)
            except Exception:
                log.exception("график %s %s", symbol, tf.value)
            new.append(saved)
        return new

    async def scan(self, tf: Timeframe) -> ScanReport:
        """Один проход по рынку для таймфрейма. Сигналы появляются, только если ТФ включён в настройках."""
        s = self.settings.load()
        started = datetime.now(timezone.utc)
        if tf not in s.timeframes:
            rep = ScanReport(tf, 0, [], 0, started, started)
            self.last[tf] = rep
            return rep
        symbols = await self.market.universe(s.min_turnover_usd)
        res = await asyncio.gather(*(self._one(x, tf, s) for x in symbols), return_exceptions=True)
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
        if s.mode is Mode.AUTO and self.executor is not None and self.executor.connected:
            found = [await self._auto(sig) for sig in found]
        rep = ScanReport(tf, len(symbols), found, errors, started, datetime.now(timezone.utc))
        self.last[tf] = rep
        log.info("скан %s: монет %d, сигналов %d, ошибок %d", tf.value, len(symbols), len(found), errors)
        return rep


    async def _auto(self, sig: Signal) -> Signal:
        """Авто-режим: бот сам отправляет ордер; отказ по правилам риска — сигнал пропущен с причиной."""
        assert self.executor is not None and sig.id is not None
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
        if self.settings.load().mode is not Mode.MANUAL:
            raise ValueError("включён авто-режим: сигналы исполняет бот")
        return sig

    async def take(self, signal_id: int) -> Signal:
        """«Вхожу»: с подключённой биржей — ордер со стопом и целью на Bybit; без неё — только отметка.
        ValueError — ордер не отправлен (причина в тексте), сигнал остаётся новым."""
        sig = self._check(signal_id)
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

    def toggle_timeframe(self, tf: Timeframe) -> Settings:
        s = self.repo.load().toggle(tf)
        self.repo.save(s)
        return s

    def set_mode(self, mode: Mode) -> Settings:
        s = replace(self.repo.load(), mode=mode)
        self.repo.save(s)
        return s

    def set_entry_policy(self, policy: EntryPolicy) -> Settings:
        s = replace(self.repo.load(), entry_policy=policy)
        self.repo.save(s)
        return s

    def update(self, **fields: float | int) -> Settings:
        """Числовые параметры риска и правила; Settings проверяет допустимые диапазоны."""
        allowed = {"risk_pct", "max_positions", "daily_loss_pct", "min_aggr", "target_r", "hybrid_range_atr"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"неизвестные поля: {sorted(bad)}")
        s = replace(self.repo.load(), **fields)
        self.repo.save(s)
        return s
