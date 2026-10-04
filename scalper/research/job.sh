#!/usr/bin/env bash
# Задание для .github/workflows/research.yml; запускается из папки scalper/.
#   bash research/job.sh collect   — своя часть (SHARD из NSHARDS, см. research/shard.py), результаты в $PARTS
#   bash research/job.sh report    — сборка итогов из всех частей
# TASKS — какие исследования гонять в этом прогоне (по порядку).
set -euo pipefail
TASKS=${TASKS:-"prep listings2 trail mom"}
MODE=${1:-collect}
FILTER="Pandas4Warning\|pd.concat\|файлов$\|мес. свечей"

hl_collect() { python -m research.hl_whales collect --wallets 500; }
hl_report() {
  echo "===== HL WHALES: копирование кошельков Hyperliquid (отбор до T, проверка после T) ====="
  python -m research.hl_whales report
}
lux_collect() {
  python -m research.prepare 1h-qualified 15m-core70
  python -m research.lux_rr collect --symbols15 "$(cat /tmp/syms_15m-core70.txt)" --symbols1h "$(cat /tmp/syms_1h-qualified.txt)"
}
lux_report() {
  echo "===== LUX R:R: LuxAlgo Trendlines with Breaks, выход 1:3 / 1:5, вход по пробою и на ретесте ====="
  python -m research.lux_rr report
}

prep_collect() { python -m research.prepare 1h-qualified 15m-core70 1h-all; }
prep_report() { :; }
listings2_collect() { python -m research.listings2 collect --symbols "$(cat /tmp/syms_1h-all.txt)"; }
listings2_report() {
  echo "===== LISTINGS2: шорты по анонсам Binance — funding, цены Bybit, шорт после листинга ====="
  python -m research.listings2 report
}
trail_collect() {
  python -m research.trend_trail collect --symbols15 "$(cat /tmp/syms_15m-core70.txt)" --symbols1h "$(cat /tmp/syms_1h-qualified.txt)"
}
trail_report() {
  echo "===== TREND TRAIL: пробои LuxAlgo и Дончиана с подтягиванием стопа (15m / 1h / 4h) ====="
  python -m research.trend_trail report
}
mom_collect() { python -m research.momentum collect --symbols "$(cat /tmp/syms_1h-qualified.txt)"; }
mom_report() {
  echo "===== MOMENTUM: межмонетный (неделя) и по времени (CTA, день) ====="
  python -m research.momentum report
}

case "$MODE" in
  collect|report) ;;
  *) echo "usage: job.sh collect|report"; exit 2 ;;
esac
for t in $TASKS; do
  echo "----- $t $MODE (часть ${SHARD:-0}/${NSHARDS:-1}) -----"
  "${t}_${MODE}" 2>&1 | { grep -v "$FILTER" || true; }
done
