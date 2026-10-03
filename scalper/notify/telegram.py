"""Telegram-уведомления: сделки, алерты, дневной отчёт. Отправка в фоне, не блокирует торговлю."""
from __future__ import annotations

import asyncio
import html
import logging
from datetime import datetime, timezone

import aiohttp

from core.models import TradeRecord

log = logging.getLogger("telegram")


class Telegram:
    def __init__(self, token: str, chat_id: str, prefix: str = ""):
        self.url = f"https://api.telegram.org/bot{token}/sendMessage"
        self.chat_id = chat_id
        self.prefix = prefix
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=200)
        self.task: asyncio.Task | None = None

    def start(self) -> None:
        self.task = asyncio.create_task(self._sender(), name="telegram")

    async def stop(self, timeout: float = 5.0) -> None:
        try:
            await asyncio.wait_for(self.queue.join(), timeout)
        except asyncio.TimeoutError:
            pass
        if self.task:
            self.task.cancel()

    def send(self, text: str) -> None:
        try:
            self.queue.put_nowait(f"{self.prefix}{text}")
        except asyncio.QueueFull:
            log.warning("telegram queue full, dropping message")

    async def _sender(self) -> None:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as s:
            while True:
                text = await self.queue.get()
                for attempt in range(3):
                    try:
                        async with s.post(self.url, json={"chat_id": self.chat_id, "text": text,
                                                          "parse_mode": "HTML",
                                                          "disable_web_page_preview": True}) as r:
                            if r.status == 429:
                                data = await r.json()
                                await asyncio.sleep(data.get("parameters", {}).get("retry_after", 3))
                                continue
                            if r.status != 200:
                                log.warning("telegram %s: %s", r.status, (await r.text())[:200])
                            break
                    except Exception as e:
                        log.warning("telegram send failed: %s", e)
                        await asyncio.sleep(2 * (attempt + 1))
                self.queue.task_done()
                await asyncio.sleep(0.05)


# ---------- форматирование ----------

def fmt_open(p: dict) -> str:
    arrow = "🟢 LONG" if p["side"] == "LONG" else "🔴 SHORT"
    return (f"{arrow} <b>{p['setup']}</b>\n"
            f"Вход: {p['price']:.2f} × {p['qty']}\n"
            f"Стоп: {p['stop']:.2f} · TP: {p['tp']:.2f}\n"
            f"<i>{html.escape(p['reason'])}</i>")


def fmt_close(r: TradeRecord) -> str:
    icon = "✅" if r.net_pnl > 0 else "❌"
    return (f"{icon} Закрыто {r.side} <b>{r.setup}</b> ({r.exit_reason})\n"
            f"{r.entry_price:.2f} → {r.exit_price:.2f} · {r.exit_ts - r.entry_ts:.0f}с\n"
            f"PnL: <b>{r.net_pnl:+.4f}</b> USDT (комиссии {r.fees:.4f})")


def fmt_daily(trades: list[TradeRecord], equity: float) -> str:
    n = len(trades)
    net = sum(t.net_pnl for t in trades)
    wins = sum(1 for t in trades if t.net_pnl > 0)
    wr = wins / n * 100 if n else 0.0
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return (f"📊 <b>Итоги {day}</b>\nСделок: {n} · winrate {wr:.0f}%\n"
            f"PnL: <b>{net:+.2f}</b> USDT · капитал {equity:.2f}")
