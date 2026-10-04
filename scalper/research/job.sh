#!/usr/bin/env bash
# Задание для GitHub Actions (.github/workflows/research.yml); запускается из папки scalper/.
set -euo pipefail
echo "===== TREND: пробои и откаты на 4h/1d с фильтрами OI, потока и funding, выходы 1:5 (725 монет Binance) ====="
python - <<'PY'
import json, pathlib
import research.data as D
D.set_host("cdn")
D.METRICS_FREQ = "1h"
q = json.load(open("research/symbols_qualified.json"))
print(len(q), "монет", flush=True)
D.build(q, "1h", "2022-01", "2026-09", pathlib.Path.home() / "bn", 64, metrics=True)
PY
SYMS=$(python -c 'import json; print(",".join(json.load(open("research/symbols_qualified.json"))))')
python -m research.trend --root ~/bn --symbols "$SYMS" 2>&1 | grep -v "Pandas4Warning\|pd.concat"
