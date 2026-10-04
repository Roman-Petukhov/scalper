#!/usr/bin/env bash
# Задание для GitHub Actions (.github/workflows/research.yml); запускается из папки scalper/.
set -euo pipefail
echo "===== STOPS: oi_liq с выносом уровня, стопы 1-5 ATR, рыночный вход и лесенка (725 монет Binance, путь по 1h-барам) ====="
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
python -m research.stops_exam --root ~/bn --symbols "$SYMS" --tf 1h --exam 2>&1 | grep -v "Pandas4Warning\|pd.concat"
