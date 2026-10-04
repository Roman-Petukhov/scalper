#!/usr/bin/env bash
# Задание для .github/workflows/research.yml; запускается из папки scalper/.
#   bash research/job.sh collect   — своя часть (SHARD из NSHARDS, см. research/shard.py), результаты в $PARTS
#   bash research/job.sh report    — сборка итогов из всех частей
set -euo pipefail
MODE=${1:-collect}
FILTER="Pandas4Warning\|pd.concat"
case "$MODE" in
  collect)
    echo "===== HL WHALES (часть ${SHARD:-0}/${NSHARDS:-1}): сделки кошельков Hyperliquid ====="
    python -m research.hl_whales collect --wallets 500 2>&1 | grep -v "$FILTER"
    ;;
  report)
    echo "===== HL WHALES: копирование кошельков Hyperliquid (отбор до T, проверка после T) ====="
    python -m research.hl_whales report 2>&1 | grep -v "$FILTER"
    ;;
  *) echo "usage: job.sh collect|report"; exit 2 ;;
esac
