"""Уведомления в Telegram: график сигнала и ссылка на панель. Управление — в веб-панели."""
from __future__ import annotations

import httpx

from ..domain.models import EntryKind, Signal


def caption(s: Signal, panel_url: str) -> str:
    p = s.plan
    entry = "лимитка на ретесте линии" if p.entry_kind is EntryKind.RETEST else "по рынку"
    return (f"<b>{s.symbol}</b> · {s.timeframe.value} · {s.side.label}{' · + уровень' if s.at_level else ''}\n"
            f"Вход: {entry} {p.entry:.6g}\nСтоп: {p.stop:.6g} ({p.risk_pct_of_price:.2f}%)\n"
            f"Цель: {p.target:.6g}\nАгрессоры: {s.aggr:.0%}\n"
            f"<a href=\"{panel_url.rstrip('/')}/#signal-{s.id}\">Открыть в панели</a>")


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, client: httpx.AsyncClient | None = None) -> None:
        self.base = f"https://api.telegram.org/bot{token}"
        self.chat_id = chat_id
        self.client = client or httpx.AsyncClient(timeout=20.0)

    async def signal(self, signal: Signal, chart_path: str | None, panel_url: str) -> None:
        text = caption(signal, panel_url)
        if chart_path:
            with open(chart_path, "rb") as fh:
                r = await self.client.post(f"{self.base}/sendPhoto",
                                           data={"chat_id": self.chat_id, "caption": text, "parse_mode": "HTML"},
                                           files={"photo": fh})
        else:
            r = await self.client.post(f"{self.base}/sendMessage",
                                       data={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"})
        r.raise_for_status()

    async def text(self, message: str) -> None:
        r = await self.client.post(f"{self.base}/sendMessage", data={"chat_id": self.chat_id, "text": message})
        r.raise_for_status()
