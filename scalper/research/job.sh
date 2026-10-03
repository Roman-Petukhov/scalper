#!/usr/bin/env bash
# Задание для GitHub Actions (.github/workflows/research.yml); запускается из папки scalper/.
set -euo pipefail
EXTRA=$(python -m research.universe --print-symbols)
echo "===== DATA: новые монеты (Binance, свечи + funding + OI) ====="
python -m research.data --root ~/rs --start 2022-01 --end 2026-09 --metrics --workers 48 --symbols "$EXTRA" | grep -v "файлов$"
echo "===== UNIVERSE: проверенные стратегии на новых монетах ====="
python -m research.universe --root ~/rs --out ../out
