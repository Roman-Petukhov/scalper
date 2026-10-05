"""Push-уведомления браузера (Web Push, VAPID): приходят на телефон и компьютер, даже когда панель закрыта.
Ключи VAPID создаются один раз и лежат в каталоге данных; подписки устройств — в SQLite."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Protocol

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from pywebpush import WebPushException, webpush

from ..domain.models import EntryKind, Signal

log = logging.getLogger(__name__)


class SubscriptionStore(Protocol):
    def subscriptions(self) -> list[dict]: ...

    def remove_subscription(self, endpoint: str) -> None: ...


def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def load_or_create_keys(path: Path) -> tuple[Path, str]:
    """PEM закрытого ключа P-256 и открытый ключ в формате applicationServerKey (несжатая точка, base64url)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        key = ec.generate_private_key(ec.SECP256R1())
        path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
        try:
            path.chmod(0o600)
        except OSError:
            pass
    key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    pub = key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    return path, _b64url(pub)


def payload(s: Signal) -> dict:
    p = s.plan
    kind = "ретест" if p.entry_kind is EntryKind.RETEST else "рынок"
    return {"title": f"{s.symbol} · {s.timeframe.value} · {s.side.label}",
            "body": f"Вход ({kind}) {p.entry:.6g} · стоп {p.stop:.6g} · цель {p.target:.6g}\nАгрессоры {s.aggr:.0%}",
            "tag": f"signal-{s.id}", "url": f"/#signal-{s.id}"}


class WebPushNotifier:
    def __init__(self, store: SubscriptionStore, key_path: Path, contact: str = "mailto:panel@localhost") -> None:
        self.store = store
        self.key_path, self.public_key = load_or_create_keys(key_path)
        self.contact = contact

    def _send_one(self, sub: dict, data: str) -> bool:
        """True — доставлено; подписки, которые сервис push больше не знает (404 / 410), удаляются."""
        try:
            webpush(sub, data, vapid_private_key=str(self.key_path), vapid_claims={"sub": self.contact},
                    ttl=3600, timeout=10)
            return True
        except WebPushException as e:
            code = e.response.status_code if e.response is not None else None
            if code in (404, 410):
                self.store.remove_subscription(sub["endpoint"])
                log.info("push: подписка устарела и удалена")
            else:
                log.warning("push: не доставлено (%s)", code or e)
            return False

    async def send(self, message: dict) -> int:
        data = json.dumps(message, ensure_ascii=False)
        subs = self.store.subscriptions()
        res = await asyncio.gather(*(asyncio.to_thread(self._send_one, s, data) for s in subs))
        return sum(res)

    async def signal(self, signal: Signal, chart_path: str | None, panel_url: str) -> None:
        await self.send(payload(signal))

    async def text(self, message: str) -> None:
        await self.send({"title": "Трендовые пробои", "body": message, "tag": "info", "url": "/"})


class MultiNotifier:
    """Несколько каналов сразу: ошибка одного не мешает остальным."""

    def __init__(self, *channels) -> None:
        self.channels = [c for c in channels if c is not None]

    async def signal(self, signal: Signal, chart_path: str | None, panel_url: str) -> None:
        for c in self.channels:
            try:
                await c.signal(signal, chart_path, panel_url)
            except Exception:
                log.exception("уведомление через %s", type(c).__name__)

    async def text(self, message: str) -> None:
        for c in self.channels:
            try:
                await c.text(message)
            except Exception:
                log.exception("уведомление через %s", type(c).__name__)
