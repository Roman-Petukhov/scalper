"""Запуск панели на своём компьютере: настройки в файле panel.env рядом с проектом (создаётся при первом запуске),
сервер слушает только этот компьютер (127.0.0.1), браузер открывается сам.

    python -m trader.local
"""
from __future__ import annotations

import getpass
import os
import secrets
import shutil
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / "panel.env"
HOST, PORT = "127.0.0.1", 8000


def _read_env(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _first_run() -> None:
    print("Первый запуск: задайте пароль для входа в панель (не короче 10 символов).")
    while True:
        pw = getpass.getpass("Пароль: ")
        if len(pw) < 10:
            print("Слишком короткий, нужно минимум 10 символов.")
            continue
        if getpass.getpass("Ещё раз: ") != pw:
            print("Пароли не совпали, попробуйте снова.")
            continue
        break
    ENV_FILE.write_text(
        "# Настройки панели на этом компьютере. Файл не попадает в git.\n"
        f"PANEL_PASSWORD={pw}\nSESSION_SECRET={secrets.token_hex(32)}\n"
        f"PANEL_URL=http://{HOST}:{PORT}\nDATA_DIR={ROOT / 'data_panel'}\n"
        "# Необязательно: дублировать сигналы в Telegram\nTELEGRAM_TOKEN=\nTELEGRAM_CHAT_ID=\nSCAN_DELAY_S=20\n",
        encoding="utf-8")
    try:
        os.chmod(ENV_FILE, 0o600)
    except OSError:
        pass
    print(f"Сохранено в {ENV_FILE.name}. Сменить пароль — удалите этот файл и запустите снова.")


def _browsers() -> list[str]:
    """Chromium-браузеры, которые умеют открывать страницу отдельным окном (--app)."""
    found = []
    if sys.platform == "win32":
        for base in (os.environ.get("PROGRAMFILES(X86)", ""), os.environ.get("PROGRAMFILES", ""),
                     os.environ.get("LOCALAPPDATA", "")):
            for rel in (r"Microsoft\Edge\Application\msedge.exe", r"Google\Chrome\Application\chrome.exe",
                        r"Yandex\YandexBrowser\Application\browser.exe"):
                p = Path(base) / rel
                if base and p.exists():
                    found.append(str(p))
    elif sys.platform == "darwin":
        for app in ("Google Chrome", "Microsoft Edge", "Yandex"):
            p = Path("/Applications") / f"{app}.app" / "Contents" / "MacOS" / app
            if p.exists():
                found.append(str(p))
    else:
        found += [p for p in (shutil.which(x) for x in ("google-chrome", "chromium", "microsoft-edge")) if p]
    return found


def open_app_window(url: str) -> None:
    """Открыть панель отдельным окном без адресной строки, как приложение; если Chromium-браузера нет — вкладкой."""
    for exe in _browsers():
        try:
            subprocess.Popen([exe, f"--app={url}", "--window-size=1280,860"], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            return
        except OSError:
            continue
    webbrowser.open(url)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not ENV_FILE.exists():
        _first_run()
    for k, v in _read_env(ENV_FILE).items():
        os.environ.setdefault(k, v)
    import uvicorn
    url = f"http://{HOST}:{PORT}"
    print(f"\nПанель: {url}  (остановить — Ctrl+C)\nПервый скан — сразу после ближайшего закрытия свечи "
          f"включённого таймфрейма; кнопка «Скан» в панели проверит рынок сейчас.\n")
    threading.Timer(2.5, lambda: open_app_window(url)).start()
    uvicorn.run("trader.web.app:main", factory=True, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
