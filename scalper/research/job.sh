#!/usr/bin/env bash
# Задание для .github/workflows/research.yml; запускается из папки scalper/.
#   bash research/job.sh collect   — своя часть (SHARD из NSHARDS, см. research/shard.py), результаты в $PARTS
#   bash research/job.sh report    — сборка итогов из всех частей
set -euo pipefail
MODE=${1:-collect}
FILTER="Pandas4Warning\|pd.concat\|файлов$\|мес. свечей"
case "$MODE" in
  collect)
    python -m research.prepare 1h-qualified 15m-core70 2>&1 | grep -v "$FILTER"
    echo "===== LUX R:R (часть ${SHARD:-0}/${NSHARDS:-1}) ====="
    python -m research.lux_rr collect --symbols15 "$(cat /tmp/syms_15m-core70.txt)" \
      --symbols1h "$(cat /tmp/syms_1h-qualified.txt)" 2>&1 | grep -v "$FILTER"
    ;;
  report)
    echo "===== LUX R:R: LuxAlgo Trendlines with Breaks, выход 1:3 / 1:5, вход по пробою и на ретесте ====="
    python -m research.lux_rr report 2>&1 | grep -v "$FILTER"
    ;;
  *) echo "usage: job.sh collect|report"; exit 2 ;;
esac
