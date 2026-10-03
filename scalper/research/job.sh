#!/usr/bin/env bash
set -uo pipefail
echo "===== PROBE tick / orderbook archives ====="
probe() { printf "%-130s " "$1"; curl -s -o /dev/null -I -m 20 -w "%{http_code} %{size_download} len=" "$1"; curl -sI -m 20 "$1" | grep -i content-length | tr -d '\r' | awk '{print $2}'; echo; }
for d in 2023-06-01 2025-01-01 2026-09-01; do
  probe "https://quote-saver.bycsi.com/orderbook/linear/BTCUSDT/${d}_BTCUSDT_ob500.data.zip"
  probe "https://quote-saver.bycsi.com/orderbook/linear/BTCUSDT/${d}_BTCUSDT_ob200.data.zip"
  probe "https://public.bybit.com/trading/BTCUSDT/BTCUSDT${d}.csv.gz"
  probe "https://public.bybit.com/trading/SOLUSDT/SOLUSDT${d}.csv.gz"
  probe "https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-${d}.zip"
  probe "https://data.binance.vision/data/futures/um/daily/bookTicker/BTCUSDT/BTCUSDT-bookTicker-${d}.zip"
  probe "https://data.binance.vision/data/futures/um/daily/bookDepth/BTCUSDT/BTCUSDT-bookDepth-${d}.zip"
done
echo "--- ob500 sample head"
curl -s -m 120 "https://quote-saver.bycsi.com/orderbook/linear/SOLUSDT/2025-01-01_SOLUSDT_ob500.data.zip" -o /tmp/ob.zip && ls -la /tmp/ob.zip && unzip -p /tmp/ob.zip | head -c 700; echo
echo "--- bookDepth sample head"
curl -s -m 60 "https://data.binance.vision/data/futures/um/daily/bookDepth/BTCUSDT/BTCUSDT-bookDepth-2025-01-01.zip" -o /tmp/bd.zip && unzip -p /tmp/bd.zip | head -5
echo "--- speed test (Bybit trades BTC 2025-01-02)"
time curl -s -m 300 "https://public.bybit.com/trading/BTCUSDT/BTCUSDT2025-01-02.csv.gz" -o /tmp/t.gz; ls -la /tmp/t.gz
