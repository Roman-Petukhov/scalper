#!/usr/bin/env bash
# Задание для GitHub Actions (.github/workflows/research.yml); запускается из папки scalper/.
set -euo pipefail
EXTRA=$(python -m research.universe --print-symbols)
python -m research.data --root ~/rs --start 2022-01 --end 2026-09 --metrics --workers 48 --symbols "$EXTRA" | grep -v "файлов$" | tail -2
echo "===== PORTFOLIO: 16 исходных + новые монеты ====="
python -m research.portfolio --root ~/rs --out ../out
