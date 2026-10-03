"""Режимы реального времени: paper (симуляция на живых котировках) и live (реальные ордера)."""
from __future__ import annotations

import asyncio
import csv
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from backtest.recorder import Recorder
from core.config import ROOT, env
from core.models import Trade, TradeRecord
from core.risk import Instrument
from core.session import TradingSession
from exchange.binance_ws import MarketStream
from exchange.broker_paper import PaperBroker
from notify.telegram import Telegram, fmt_close, fmt_daily, fmt_open

log = logging.getLogger("runner")

KILL_FILE = ROOT / "STOP"   # создайте файл STOP рядом с main.py: бот закроет позицию и выключится


class Journal:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, r: TradeRecord) -> None:
        row = {k: v for k, v in asdict(r).items() if k != "extra"}
        new = not self.path.exists()
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)


async def run_realtime(cfg, live: bool, record: bool = False) -> None:
    mode = "LIVE" if live else "PAPER"
    tg = None
    if cfg.telegram.enabled and env("TELEGRAM_TOKEN") and env("TELEGRAM_CHAT_ID"):
        tg = Telegram(env("TELEGRAM_TOKEN"), env("TELEGRAM_CHAT_ID"),
                      prefix=f"[{mode} {cfg.exchange.symbol}] ")
        tg.start()
    elif cfg.telegram.enabled:
        log.warning("telegram.enabled=true, но TELEGRAM_TOKEN/TELEGRAM_CHAT_ID не заданы в .env")

    journal = Journal(ROOT / cfg.logging.dir / f"trades_{mode.lower()}.csv")
    day_trades: list[TradeRecord] = []

    def notify(kind: str, payload) -> None:
        if kind == "close":
            journal.write(payload)
            day_trades.append(payload)
        if kind == "alert":
            log.warning("ALERT: %s", payload)
        if tg is None:
            return
        if kind == "open":
            tg.send(fmt_open(payload))
        elif kind == "close":
            tg.send(fmt_close(payload))
        elif kind == "alert":
            tg.send(f"⚠️ {payload}")

    # ---------- брокер ----------
    if live:
        from exchange.broker_live import LiveBroker
        key, secret = _keys(cfg.exchange.env)
        broker = LiveBroker(cfg, key, secret, on_alert=lambda m: notify("alert", m))
        inst = await broker.start()
        equity = await broker.fetch_equity()
        data_env = cfg.exchange.env
        log.info("LIVE %s (%s): equity=%.2f USDT, tick=%s step=%s minNotional=%s",
                 cfg.exchange.symbol, cfg.exchange.env, equity, inst.tick_size, inst.step_size,
                 inst.min_notional)
        if equity <= 0:
            raise SystemExit("Баланс USDT на фьючерсном счёте равен нулю.")
    else:
        inst = Instrument(cfg.exchange.symbol, cfg.instrument.tick_size,
                          cfg.instrument.step_size, cfg.instrument.min_notional)
        equity = float(cfg.paper.start_equity)
        broker = PaperBroker(cfg, inst.tick_size, inst.step_size, equity)
        data_env = cfg.exchange.paper_data_env

    session = TradingSession(cfg, inst, broker, equity, notify=notify)

    # часы в домене биржевого времени: local + сглаженный сдвиг
    offset = {"v": 0.0, "n": 0}

    def clock() -> float:
        return time.time() + offset["v"]

    broker.clock = clock
    stats = {"events": 0, "last_ev": time.time()}

    def on_event(ev) -> None:
        if isinstance(ev, Trade):
            d = ev.ts - time.time()
            offset["v"] = d if offset["n"] == 0 else offset["v"] + 0.05 * (d - offset["v"])
            offset["n"] += 1
        stats["events"] += 1
        stats["last_ev"] = time.time()
        session.on_event(ev)

    recorder = Recorder(ROOT / cfg.recorder.dir, cfg.exchange.symbol, cfg.recorder.rotate_minutes) if record else None
    stream = MarketStream(cfg, data_env, on_event, raw_sink=recorder.write if recorder else None)

    hello = (f"Старт {mode}: {cfg.exchange.symbol}, данные {data_env}, капитал {equity:.2f} USDT, "
             f"прогрев {cfg.features.warmup_s}s")
    log.info(hello)
    if tg:
        tg.send(hello)

    async def ticker() -> None:
        """Тик раз в секунду: тайм-стопы и выходы работают, даже когда рынок затих."""
        last_status = 0.0
        last_report_day = datetime.now(timezone.utc).date()
        while True:
            await asyncio.sleep(1.0)
            now = clock()
            if session.features.last_price:
                snap = session.features.snapshot(now)
                session.manager.on_market(snap)
            if KILL_FILE.exists():
                log.warning("найден файл STOP: выключаюсь")
                notify("alert", "Найден файл STOP: закрываю позицию и выключаюсь.")
                return
            if time.time() - stats["last_ev"] > 60:
                log.warning("нет рыночных данных > 60s")
            if time.time() - last_status >= 60:
                last_status = time.time()
                s = session.last_snap
                eq = broker.equity if hasattr(broker, "equity") else session.risk.equity
                if s is not None:
                    log.info("px=%.2f vol=%.1fbp/m z5=%.1f obi=%s ready=%s | state=%s trades=%d eq=%.2f | %s",
                             s.price, s.vol_bps_min, s.z[min(s.z)], f"{s.obi:.2f}" if s.obi is not None else "na",
                             s.ready, session.manager.state, len(session.manager.trades), eq,
                             _top(session.strategy.stats))
            today = datetime.now(timezone.utc).date()
            if today != last_report_day and datetime.now(timezone.utc).hour >= cfg.telegram.daily_report_utc_hour:
                last_report_day = today
                if tg:
                    tg.send(fmt_daily(day_trades, session.risk.equity))
                day_trades.clear()

    tasks = [asyncio.create_task(stream.run(), name="ws"), asyncio.create_task(ticker(), name="ticker")]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            if t.exception():
                log.error("задача %s упала: %r", t.get_name(), t.exception())
                notify("alert", f"Сбой задачи {t.get_name()}: {t.exception()!r}")
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        log.info("остановка: закрываю позицию...")
        session.trading_enabled = False
        session.manager.flatten(clock(), "shutdown")
        for _ in range(100):           # ждём до 10 секунд, пока позиция закроется
            if session.manager.is_flat:
                break
            await asyncio.sleep(0.1)
        if not session.manager.is_flat:
            log.error("позиция не закрыта за 10с, проверьте биржу вручную!")
            notify("alert", "Позиция не закрылась при остановке. Проверьте биржу вручную!")
        stream.stop()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if live:
            await broker.cancel_all()
            await broker.close()
        if recorder:
            recorder.close()
        trades = session.manager.trades
        net = sum(t.net_pnl for t in trades)
        log.info("итог сессии: сделок %d, PnL %+.4f USDT", len(trades), net)
        if tg:
            tg.send(f"Остановлен. Сделок за сессию: {len(trades)}, PnL {net:+.4f} USDT")
            await tg.stop()
        if KILL_FILE.exists():
            KILL_FILE.unlink()


async def run_recorder(cfg, minutes: float | None) -> None:
    rec = Recorder(ROOT / cfg.recorder.dir, cfg.exchange.symbol, cfg.recorder.rotate_minutes)
    stream = MarketStream(cfg, cfg.exchange.paper_data_env, on_event=lambda ev: None, raw_sink=rec.write)
    task = asyncio.create_task(stream.run())
    t0 = time.time()
    try:
        while minutes is None or time.time() - t0 < minutes * 60:
            await asyncio.sleep(30)
            log.info("записано сообщений: %d", rec.count)
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        stream.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        rec.close()
        log.info("запись завершена: %d сообщений", rec.count)


def _keys(environment: str) -> tuple[str, str]:
    prefix = "BINANCE_DEMO" if environment == "demo" else "BINANCE"
    key, secret = env(f"{prefix}_API_KEY"), env(f"{prefix}_API_SECRET")
    if not key or not secret:
        raise SystemExit(f"Не заданы {prefix}_API_KEY / {prefix}_API_SECRET в .env")
    return key, secret


def _top(stats: dict, n: int = 4) -> str:
    items = sorted(stats.items(), key=lambda kv: -kv[1])[:n]
    return " ".join(f"{k}={v}" for k, v in items)
