"""Конфигурация сервера из переменных окружения (файл .env на VPS; секреты в репозиторий не попадают)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    panel_password: str
    session_secret: str
    data_dir: Path
    panel_url: str
    telegram_token: str | None
    telegram_chat_id: str | None
    scan_delay_s: float           # пауза после закрытия свечи: Binance дописывает свечу 1–5 с
    scheduler: bool
    heartbeat_url: str | None = None   # необязательно: пинг после каждого удачного скана (healthchecks.io и т. п.)
    backup_days: int = 14              # сколько дневных копий базы хранить в data/backups

    @staticmethod
    def from_env() -> AppConfig:
        pw = os.environ.get("PANEL_PASSWORD", "")
        secret = os.environ.get("SESSION_SECRET", "")
        if len(pw) < 10:
            raise RuntimeError("PANEL_PASSWORD не задан или короче 10 символов")
        if len(secret) < 32:
            raise RuntimeError("SESSION_SECRET не задан или короче 32 символов (openssl rand -hex 32)")
        return AppConfig(panel_password=pw, session_secret=secret,
                         data_dir=Path(os.environ.get("DATA_DIR", "data")),
                         panel_url=os.environ.get("PANEL_URL", "http://localhost:8000"),
                         telegram_token=os.environ.get("TELEGRAM_TOKEN") or None,
                         telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID") or None,
                         scan_delay_s=float(os.environ.get("SCAN_DELAY_S", "20")),
                         scheduler=os.environ.get("SCHEDULER", "1") == "1",
                         heartbeat_url=os.environ.get("HEARTBEAT_URL") or None,
                         backup_days=int(os.environ.get("BACKUP_DAYS", "14")))
