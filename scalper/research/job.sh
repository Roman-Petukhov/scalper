#!/usr/bin/env bash
# Задание для GitHub Actions (.github/workflows/research.yml); запускается из папки scalper/.
set -euo pipefail
echo "===== BROAD: все USDT-перпетуалы Binance, хоть раз прошедшие порог ликвидности (список зафиксирован заранее) ====="
python - <<'PY'
import json, pathlib
import research.data as D
D.set_host("cdn")
D.METRICS_FREQ = "1h"
q = json.load(open("research/symbols_qualified.json"))
print(len(q), "монет", flush=True)
D.build(q, "1h", "2022-01", "2026-09", pathlib.Path.home() / "bn", 64, metrics=True)
PY
python -m research.broad --root ~/bn --out ../out --stage eval --host cdn 2>&1 | grep -v "Pandas4Warning\|pd.concat\|port = \|P = pd"
