#!/usr/bin/env bash
set -uo pipefail
echo "===== PROBE public.bybit.com ====="
for u in "https://public.bybit.com/" "https://public.bybit.com/premium_index/" "https://public.bybit.com/premium_index/BTCUSDT/" "https://public.bybit.com/kline_for_metatrader4/" "https://public.bybit.com/spot_index/"; do
  echo "--- $u"; curl -s -m 20 "$u" | sed 's/<[^>]*>/ /g' | tr -s ' \n' | head -c 1500; echo
done
echo "--- trades file sample"
curl -s -m 60 https://public.bybit.com/trading/BTCUSDT/BTCUSDT2024-01-02.csv.gz -o /tmp/t.gz; ls -la /tmp/t.gz; zcat /tmp/t.gz | head -3; zcat /tmp/t.gz | wc -l
curl -sI https://public.bybit.com/trading/BTCUSDT/BTCUSDT2024-03-11.csv.gz | grep -i content-length
