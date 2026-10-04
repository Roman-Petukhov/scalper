#!/usr/bin/env bash
set -uo pipefail
echo "===== PROBE: источники объявлений и новостей ====="
probe() { echo "--- $1"; curl -s -m 20 -A "Mozilla/5.0" -w "\nHTTP %{http_code}\n" "$1" | head -c 1200; echo; }
probe "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query?type=1&catalogId=48&pageNo=1&pageSize=3"
probe "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query?type=1&catalogId=161&pageNo=1&pageSize=3"
probe "https://api-manager.upbit.com/api/v1/announcements?os=web&page=1&per_page=3&category=trade"
probe "https://api-manager.upbit.com/api/v1/announcements?os=web&page=300&per_page=3&category=trade"
probe "https://api.bybit.com/v5/announcements/index?locale=en-US&limit=3"
probe "https://api.gdeltproject.org/api/v2/doc/doc?query=%22will%20list%22%20binance&mode=artlist&format=json&maxrecords=3&startdatetime=20240101000000&enddatetime=20240201000000"
probe "https://www.okx.com/api/v5/support/announcements?annType=announcements-new-listings&page=1"
probe "https://api.coinbase.com/api/v3/brokerage/market/products?limit=1"
probe "https://api.bithumb.com/v1/notices"
