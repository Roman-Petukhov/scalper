"""Мост «бот → облако»: раз в сутки кладёт в ветку репозитория снимок состояния (сигналы, итоги сделок в R, настройки,
погода рынка) и забирает оттуда недельный отчёт облачного разбора, чтобы переслать его в Telegram.

Репозиторий публичный, поэтому в снимок не попадает ничего про деньги и доступ: ни объёмов позиций, ни номиналов, ни
прибыли в $, ни баланса, ни ключей. Только то, что и так видно по рынку (цены, время, сторона) и итоги в R."""
from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import logging
from collections.abc import Callable
from datetime import datetime, time, timedelta, timezone

from ..domain.execution import Trade
from ..domain.models import Settings, Signal
from .export import _plain, _signal_row
from .ports import Notifier, SettingsRepository, SignalRepository, StatePublisher, TradeRepository

log = logging.getLogger(__name__)
STATE_PATH = "state/latest.json.gz"
REPORT_PATH = "reports/latest.md"
REPORT_KEY = "bridge:report"        # хэш уже отправленного отчёта
DAY_KEY = "bridge:day"
SNAPSHOT_DAYS = 120
SNAPSHOT_MAX = 4000
AFTER_CLOSE = timedelta(minutes=30)  # после дневной свечи и погоды рынка (она считается с 00:10)
RETRY = timedelta(minutes=30)
TELEGRAM_LIMIT = 3500
MONEY_FIELDS = ("qty", "notional_usd", "risk_usd", "pnl_usd")


def _trade_row(t: Trade) -> dict:
    """Сделка без денег: цены, время, итог в R. Объём, номинал, риск и прибыль в $ в снимок не попадают."""
    r, h = t.result, t.hedge
    return {"status": t.status.value, "kind": t.kind.value, "network": t.network, "price": t.price, "stop": t.stop,
            "target": t.target, "filled_at": t.filled_at.isoformat() if t.filled_at else None,
            "entry_fill": r.entry_fill if r else None, "exit": r.exit if r else None,
            "closed_at": r.closed_at.isoformat() if r else None, "r": t.r_multiple, "slippage_r": t.slippage_r,
            "exit_reason": t.exit_reason if r or t.status.value == "timed_out" else None,
            "hedge_beta": h.beta if h else None, "hedge_r": t.hedge_r, "pyramid": t.add_qty is not None}


def summary(rows: list[dict]) -> dict:
    """Итоги закрытых сделок по ТФ и стороне: число, сумма и среднее R."""
    out: dict[str, dict[str, dict]] = {}
    for x in rows:
        r = (x.get("trade") or {}).get("r")
        if r is None:
            continue
        cell = out.setdefault(x["tf"], {}).setdefault("long" if x["side"] > 0 else "short", {"n": 0, "sum_r": 0.0})
        cell["n"] += 1
        cell["sum_r"] += r
    for sides in out.values():
        for cell in sides.values():
            cell["sum_r"] = round(cell["sum_r"], 3)
            cell["avg_r"] = round(cell["sum_r"] / cell["n"], 3)
    return out


def build_snapshot(signals: list[Signal], trades: dict[int, Trade], settings: Settings, weather: dict | None,
                   now: datetime) -> dict:
    since = now - timedelta(days=SNAPSHOT_DAYS)
    rows = []
    for s in signals:
        if s.created_at is not None and s.created_at < since:
            continue
        row = _signal_row(s)
        t = trades.get(s.id) if s.id is not None else None
        row["trade"] = _trade_row(t) if t is not None else None
        rows.append(row)
        if len(rows) >= SNAPSHOT_MAX:
            break
    st = {k: _plain(getattr(settings, k)) for k in ("mode", "timeframes", "auto_timeframes", "leverage",
                                                     "hedge_leverage", "max_positions", "daily_loss_pct",
                                                     "min_turnover_usd", "sizing")}
    st["tf_params"] = {tf.value: _plain(p.__dict__) for tf, p in settings.tf_params.items()}
    return {"version": 1, "generated_at": now.isoformat(), "days": SNAPSHOT_DAYS, "settings": st, "weather": weather,
            "summary": summary(rows), "signals_n": len(rows), "signals": rows}


def encode(snapshot: dict) -> bytes:
    return gzip.compress(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")).encode(), mtime=0)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class BridgeService:
    def __init__(self, publisher: StatePublisher, signals: SignalRepository, trades: TradeRepository,
                 settings: SettingsRepository, kv, notifier: Notifier | None = None,
                 clock: Callable[[], datetime] = _utcnow, weather_key: str = "weather") -> None:
        self.publisher, self.signals, self.trades, self.settings = publisher, signals, trades, settings
        self.kv, self.notifier, self.clock, self.weather_key = kv, notifier, clock, weather_key
        self._tried: datetime | None = None
        self._task: asyncio.Task | None = None
        self.last_error: str | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def due(self, now: datetime) -> bool:
        if self.running or (self._tried is not None and now - self._tried < RETRY):
            return False
        close = datetime.combine(now.date(), time(), timezone.utc) + AFTER_CLOSE
        return now >= close and self.kv.kv_get(DAY_KEY) != now.date().isoformat()

    async def maybe_run(self) -> None:
        """Для минутного цикла: раз в сутки после закрытия дня — отправить снимок и проверить отчёт, в фоне."""
        now = self.clock()
        if self.due(now):
            self._tried = now
            self._task = asyncio.create_task(self._guarded(now))

    async def _guarded(self, now: datetime) -> None:
        try:
            await self.push(now)
            self.kv.kv_set(DAY_KEY, now.date().isoformat())
            await self.pull_report()
            self.last_error = None
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {str(e)[:160]}"
            log.exception("мост: снимок состояния")

    async def push(self, now: datetime) -> None:
        sigs = self.signals.recent(SNAPSHOT_MAX)
        trades = self.trades.trades_for([s.id for s in sigs if s.id is not None])
        raw = self.kv.kv_get(self.weather_key)
        try:
            weather = json.loads(raw) if raw else None
        except ValueError:
            weather = None
        snap = build_snapshot(sigs, trades, self.settings.load(), weather, now)
        await self.publisher.put(STATE_PATH, encode(snap), f"bot state {now:%Y-%m-%d}")

    async def pull_report(self) -> None:
        """Новый недельный отчёт облачного разбора — один раз в Telegram (хэш запоминается)."""
        text = await self.publisher.get(REPORT_PATH)
        if not text or not text.strip():
            return
        digest = hashlib.sha256(text.encode()).hexdigest()
        if self.kv.kv_get(REPORT_KEY) == digest:
            return
        if self.notifier is not None:
            body = text.strip()
            if len(body) > TELEGRAM_LIMIT:
                body = body[:TELEGRAM_LIMIT].rsplit("\n", 1)[0] + f"\n… полный отчёт: {REPORT_PATH} в ветке состояния"
            await self.notifier.text("Недельный разбор бота\n\n" + body)
        self.kv.kv_set(REPORT_KEY, digest)
