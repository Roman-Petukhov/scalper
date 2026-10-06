"""Пинг внешнего монитора (healthchecks.io, Uptime Kuma и т. п.) после каждого удачного скана: если пинги
перестали приходить — сервер лёг или бот завис, монитор сам пришлёт письмо или сообщение."""
from __future__ import annotations

import httpx


class HttpHeartbeat:
    def __init__(self, url: str, client: httpx.AsyncClient | None = None) -> None:
        self.url = url
        self.client = client or httpx.AsyncClient(timeout=10)

    async def __call__(self) -> None:
        (await self.client.get(self.url)).raise_for_status()

    async def close(self) -> None:
        await self.client.aclose()
