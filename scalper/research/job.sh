#!/usr/bin/env bash
# Задание для GitHub Actions (.github/workflows/research.yml); запускается из папки scalper/.
set -euo pipefail
ALL=$(python -m research.universe --print-symbols --all)
python -m research.data --root ~/rs --start 2022-01 --end 2026-09 --metrics --workers 48 --symbols "$ALL" | grep -v "файлов$" | tail -1
python -m research.data --root ~/rs --interval 1h --start 2022-01 --end 2026-09 --spot --workers 48 --symbols "$ALL" | grep -v "файлов$" | tail -1
echo "===== SPOT FILTER: oi_liq + поток агрессоров спота (70 монет Binance) ====="
python -m research.spotfilter --root ~/rs
