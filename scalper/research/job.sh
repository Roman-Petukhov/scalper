#!/usr/bin/env bash
# Задание для GitHub Actions (.github/workflows/research.yml); запускается из папки scalper/.
set -euo pipefail
if [ ! -d ~/rs_bybit ]; then
  echo "Данных Bybit ещё нет: запустите на своём ПК python -m research.bybit_data --root data_bybit --publish"
  exit 0
fi
echo "===== BYBIT: репликация кандидатов (данные Bybit, независимые от Binance) ====="
python -m research.deep --root ~/rs_bybit --out ../out/bybit
