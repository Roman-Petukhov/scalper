"""
Подготовка данных Binance для своей части прогона (research.shard): свечи, funding и метрики только своих монет
(плюс BTCUSDT). Списки монет части пишутся в /tmp/syms_<набор>.txt для job.sh.

    python -m research.prepare 1h-qualified [15m-core70] [1h-all]
Наборы:
    1h-qualified   725 монет из symbols_qualified.json, 1h + OI/LSR-метрики
    15m-core70     core16 + ext54, 15m
    1h-spot        те же 725 монет: спот-свечи Binance 1h (остаются монеты, у которых есть спот)
    1h-all         все USDT-перпетуалы архива (включая делистнутые), 1h без метрик
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from . import data as D
from .broad import list_symbols
from .shard import ALWAYS, mine, shard_id
from .universe import EXTRA

ROOT = Path.home() / "bn"
START, END = "2022-01", "2026-09"
METRICS = os.environ.get("RESEARCH_METRICS", "1") != "0"   # 0 — без OI/LSR: ~1.2 млн дневных архивов, на VPS долго


def build(name: str) -> list[str]:
    if name == "1h-qualified":
        syms = json.loads(Path(__file__).with_name("symbols_qualified.json").read_text())
        part = mine(syms)
        D.METRICS_FREQ = "1h"
        D.build(sorted(set(part) | ALWAYS), "1h", START, END, ROOT, 64, metrics=METRICS)
    elif name == "15m-core70":
        part = mine(list(dict.fromkeys(D.UNIVERSE + list(EXTRA))))
        D.build(sorted(set(part) | ALWAYS), "1h", START, END, ROOT, 64)          # 1h — для фильтра оборота
        D.build(sorted(set(part) | ALWAYS), "15m", START, END, ROOT, 64)
    elif name == "1h-spot":
        syms = json.loads(Path(__file__).with_name("symbols_qualified.json").read_text())
        part = mine(syms)
        D.build(sorted(set(part) | ALWAYS), "1h", START, END, ROOT, 64, spot=True)
        part = [s for s in part if (ROOT / f"{s}-spot-1h.parquet").exists()]
    elif name == "1h-all":
        part = mine(list_symbols())
        D.build(sorted(set(part) | ALWAYS), "1h", START, END, ROOT, 64)
        part = [s for s in part if (ROOT / f"{s}-1h.parquet").exists()]
    else:
        raise SystemExit(f"неизвестный набор {name}")
    Path(f"/tmp/syms_{name}.txt").write_text(",".join(part))
    return part


if __name__ == "__main__":
    D.set_host("cdn")
    for name in sys.argv[1:]:
        p = build(name)
        print(f"часть {shard_id()[0]}/{shard_id()[1]}, набор {name}: монет {len(p)}", flush=True)
