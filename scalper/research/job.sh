#!/usr/bin/env bash
# Задание для GitHub Actions (.github/workflows/research.yml); запускается из папки scalper/.
set -euo pipefail
python -m research.data --root ~/rs --start 2022-01 --end 2026-09 --metrics --premium --workers 32 | grep -v "файлов$"
echo "===== DEEP: кандидаты волн 1-2 ====="
python -m research.deep --root ~/rs --out ../out
echo "===== WAVE 3 ====="
python -m research.wave3 --root ~/rs --out ../out
