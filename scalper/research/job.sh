#!/usr/bin/env bash
# Задание для GitHub Actions (.github/workflows/research.yml); запускается из папки scalper/.
set -euo pipefail
python - <<'PY'
import json, pathlib
import research.data as D
from research.broad import EXTRA
D.set_host("cdn")
D.METRICS_FREQ = "1h"
root = pathlib.Path.home() / "bn"
q = json.load(open("research/symbols_qualified.json"))
print(len(q), "монет", flush=True)
D.build(q, "1h", "2022-01", "2026-09", root, 64, metrics=True)
s15 = list(dict.fromkeys(D.UNIVERSE + list(EXTRA)))
D.build(s15, "15m", "2022-01", "2026-09", root, 64)
pathlib.Path("/tmp/s15.txt").write_text(",".join(s15))
PY
SYMS=$(python -c 'import json; print(",".join(json.load(open("research/symbols_qualified.json"))))')
echo "===== LUX R:R: LuxAlgo Trendlines with Breaks с выходом 1:3 / 1:5, вход по пробою и на ретесте ====="
python -m research.lux_rr --root ~/bn --symbols15 "$(cat /tmp/s15.txt)" --symbols1h "$SYMS" 2>&1 | grep -v "Pandas4Warning\|pd.concat"
