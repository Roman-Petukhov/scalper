#!/usr/bin/env bash
# Задание для GitHub Actions (.github/workflows/research.yml); запускается из папки scalper/.
set -euo pipefail
pip install -q lightgbm scikit-learn
ALL=$(python -m research.universe --print-symbols --all)
python -m research.data --root ~/rs --start 2022-01 --end 2026-09 --metrics --workers 48 --symbols "$ALL" | grep -v "файлов$" | tail -1
echo "===== BREAKOUT: события пробоев линий тренда (5m, 70 монет) ====="
python -m research.breakout.dataset --root ~/rs --out ~/bo --symbols "$ALL" --workers 4 | tail -3
echo "===== BREAKOUT: мета-разметка (LightGBM) ====="
python -m research.breakout.model --events ~/bo --out ../out
