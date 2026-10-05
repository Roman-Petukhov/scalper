#!/usr/bin/env bash
# Внутри контейнера research.sh: репозиторий в /repo (только чтение), данные в /root/bn, результат в /out.
set -euo pipefail
apt-get update -qq >/dev/null && apt-get install -y -qq libgomp1 >/dev/null     # lightgbm
pip install -q --root-user-action=ignore -r requirements.txt lightgbm
FILTER="Pandas4Warning\|pd.concat\|файлов$\|мес. свечей"

if [ "$WHAT" = "tradfi" ]; then
  python -m tools.tradfi --root /root/bn/tradfi --out /out 2>&1 | tee /out/log.txt
  exit 0
fi

# 1. данные — одним процессом (общий кеш), без OI/LSR-метрик: правилу бота они не нужны
echo "== данные"
NSHARDS=1 SHARD=0 RESEARCH_METRICS=0 python -m research.prepare 1h-qualified 15m-core70 2>&1 \
  | { grep -v "$FILTER" || true; } | tee /out/prepare.txt
# 2. сделки — JOBS процессов, каждый со своей частью монет (research/shard.py)
echo "== сделки: $JOBS частей"
export NSHARDS=$JOBS TL_CHARTS=0 PARTS=/out/parts OUT=/out
pids=()
for i in $(seq 0 $((JOBS - 1))); do
  SHARD=$i python -m research.tline collect --symbols "$(cat /tmp/syms_1h-qualified.txt)" \
    --symbols15 "$(cat /tmp/syms_15m-core70.txt)" > "/out/collect-$i.txt" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
cat /out/collect-*.txt | { grep -v "$FILTER" || true; } > /out/collect_logs.txt
rm -f /out/collect-*.txt
[ "$fail" = 0 ] || echo "== часть сбора упала, см. collect_logs.txt"
# 3. отчёт
echo "== отчёт"
{ echo "===== TLINE: линия тренда по закрытиям, пробой с закреплением, ретест, лесенка 3R / 5R ====="
  python -m research.tline report 2>&1 | { grep -v "$FILTER" || true; }; } | tee /out/log.txt
rm -rf /out/parts
