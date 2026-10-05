#!/usr/bin/env bash
# Запуск панели на Mac / Linux: ./run_panel.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python)
if [ ! -d .venv ]; then
  echo "Создаю окружение Python..."
  "$PY" -m venv .venv
fi
source .venv/bin/activate
echo "Проверяю зависимости..."
python -m pip install -q --disable-pip-version-check -r requirements-trader.txt
python -m trader.local
