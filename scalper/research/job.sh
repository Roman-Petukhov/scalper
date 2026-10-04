#!/usr/bin/env bash
# Задание для GitHub Actions (.github/workflows/research.yml); запускается из папки scalper/.
set -euo pipefail
python - <<'PY'
import json, pathlib
import research.data as D
D.set_host("cdn")
D.METRICS_FREQ = "1h"
q = json.load(open("research/symbols_qualified.json"))
print(len(q), "монет", flush=True)
D.build(q, "1h", "2022-01", "2026-09", pathlib.Path.home() / "bn", 64, metrics=True)
PY
python - <<'PY2'
import json, pathlib
import research.data as D
from research.broad import list_symbols
D.set_host("cdn")
q = set(json.load(open("research/symbols_qualified.json")))
allsyms = list_symbols()
extra = [s for s in allsyms if s not in q]
print(len(allsyms), "USDT-перпетуалов в архиве, догружаем свечи и funding для", len(extra), flush=True)
D.build(extra, "1h", "2022-01", "2026-09", pathlib.Path.home() / "bn", 64, metrics=False)
pathlib.Path("/tmp/all_symbols.txt").write_text(",".join(s for s in allsyms if (pathlib.Path.home() / "bn" / f"{s}-1h.parquet").exists()))
PY2
QSYMS=$(python -c 'import json; print(",".join(json.load(open("research/symbols_qualified.json"))))')
echo "===== MOONSHOTS: признаки монет перед +50% / +100% за сутки (все перпетуалы Binance) ====="
pip install -q lightgbm && python -m research.moonshots --root ~/bn --symbols "$(cat /tmp/all_symbols.txt)" --metrics-symbols "$QSYMS" 2>&1 | grep -v "Pandas4Warning\|pd.concat"
