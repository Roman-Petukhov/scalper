#!/usr/bin/env bash
# Запуск панели на Mac / Linux: ./run_panel.sh
set -euo pipefail
cd "$(dirname "$0")"
if command -v git >/dev/null && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Обновляю панель с GitHub..."
  git pull --ff-only --quiet || echo "Обновиться не удалось (нет сети или есть свои правки) — запускаю текущую версию."
fi
PY=$(command -v python3 || command -v python)
if [ ! -d .venv ]; then
  echo "Создаю окружение Python..."
  "$PY" -m venv .venv
fi
source .venv/bin/activate
echo "Проверяю зависимости..."
python -m pip install -q --disable-pip-version-check -r requirements-trader.txt
python -m trader.local
