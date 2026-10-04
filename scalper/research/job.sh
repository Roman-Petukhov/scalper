#!/usr/bin/env bash
# Задание для GitHub Actions (.github/workflows/research.yml); запускается из папки scalper/.
set -euo pipefail
pip install -q lightgbm scikit-learn
ALL=$(python -m research.universe --print-symbols --all)
python -m research.data --root ~/rs --start 2022-01 --end 2026-09 --metrics --workers 48 --symbols "$ALL" | grep -v "файлов$" | tail -1
for DET in luxalgo two_pivot; do
  echo "===== BREAKOUT [$DET]: события пробоев (5m, 70 монет) ====="
  python -m research.breakout.dataset --root ~/rs --out ~/bo_$DET --symbols "$ALL" --workers 4 --detector $DET 2>/dev/null | tail -2
  echo "===== BREAKOUT [$DET]: мета-разметка, варианты входа: рынок / ретест (maker) / 50 на 50 ====="
  python -m research.breakout.model --events ~/bo_$DET --out ../out/$DET
done
