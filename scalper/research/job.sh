#!/usr/bin/env bash
# Задание для GitHub Actions (.github/workflows/research.yml); запускается из папки scalper/.
set -uo pipefail
echo "===== PROBE: доступность Bybit из раннера ====="
curl -s https://ipinfo.io/country; echo
for u in "https://api.bybit.com/v5/market/time" \
         "https://api.bytick.com/v5/market/time" \
         "https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval=5&start=1640995200000&limit=3" \
         "https://api.bybit.com/v5/market/open-interest?category=linear&symbol=BTCUSDT&intervalTime=5min&startTime=1640995200000&endTime=1641081600000&limit=3" \
         "https://api.bybit.com/v5/market/account-ratio?category=linear&symbol=BTCUSDT&period=5min&startTime=1640995200000&endTime=1641081600000&limit=3" \
         "https://api.bybit.com/v5/market/funding/history?category=linear&symbol=BTCUSDT&startTime=1640995200000&endTime=1641254400000&limit=3" \
         "https://public.bybit.com/trading/BTCUSDT/" ; do
  echo "--- $u"; curl -s -m 20 -w "\nHTTP %{http_code}\n" "$u" | head -c 600; echo
done
