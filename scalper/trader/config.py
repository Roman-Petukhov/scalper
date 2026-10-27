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
    bridge_token: str | None = None    # мост «бот → облако»: токен GitHub с записью в одну ветку; нет — моста нет
    bridge_repo: str = "Roman-Petukhov/scalper"
    bridge_branch: str = "bot-state"
    coingecko_key: str | None = None   # необязательно: демо-ключ CoinGecko для скринера (без него — общий лимит)
    anthropic_key: str | None = None   # необязательно: ключ Anthropic API для вкладки «Тезисы» (без него — только индикаторы)
    theses_model: str = "claude-opus-5-5"

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
                         backup_days=int(os.environ.get("BACKUP_DAYS", "14")),
                         bridge_token=os.environ.get("BRIDGE_GITHUB_TOKEN") or None,
                         bridge_repo=os.environ.get("BRIDGE_REPO", "Roman-Petukhov/scalper"),
                         bridge_branch=os.environ.get("BRIDGE_BRANCH", "bot-state"),
                         coingecko_key=os.environ.get("COINGECKO_API_KEY") or None,
                         anthropic_key=os.environ.get("ANTHROPIC_API_KEY") or None,
                         theses_model=os.environ.get("THESES_MODEL") or "claude-opus-5-5")
