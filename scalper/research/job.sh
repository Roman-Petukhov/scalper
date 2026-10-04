#!/usr/bin/env bash
# Задание для GitHub Actions (.github/workflows/research.yml); запускается из папки scalper/.
set -euo pipefail
echo "===== HL WHALES: копирование сделок кошельков Hyperliquid (отбор до T, проверка после T) ====="
python -m research.hl_whales --cache ~/bn/cache/hl --wallets 200 2>&1 | grep -v "Pandas4Warning\|pd.concat"
