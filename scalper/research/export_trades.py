"""
Выгрузка сделок oi_liq (с фильтром выноса уровня) для моделирования портфеля: время сигнала, монета, доходность на
бюджет сигнала для рыночного входа и лесенки L5x0.5, доля исполненных лимитов. Только монеты с оборотом >= $20 млн.

    python -m research.export_trades --root <binance 1h data with metrics> --symbols ... --out <dir>
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

from .ladder_exam import PRIMARY, signal_rows

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = signal_rows(Path(a.root), a.symbols.split(","))
    r = r[r["sweep"] == "sweep"][["t", "symbol", "group", "market", PRIMARY, f"{PRIMARY}_fill"]]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r.to_csv(out / "oi_liq_trades.csv", index=False)
    print(f"===== EXPORT: сделок oi_liq с выносом уровня {len(r)}, монет {r['symbol'].nunique()} =====")
    print(r.groupby(r["t"].dt.year).agg(n=("symbol", "size"), ladder_bps=(PRIMARY, lambda x: x.mean() * 1e4)).round(1))
