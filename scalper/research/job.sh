#!/usr/bin/env bash
# Задание для GitHub Actions (.github/workflows/research.yml); запускается из папки scalper/.
set -euo pipefail
python -m research.data --root ~/rs --start 2022-01 --end 2026-09 --metrics --workers 32
python -m research.wave2 --root ~/rs --out ../out
