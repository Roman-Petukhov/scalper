"""
Расширение вселенной: проверенные стратегии с ЗАФИКСИРОВАННЫМИ параметрами на монетах, которые не участвовали
ни в подборе, ни в валидации. Это поперечный out-of-sample: для этих монет вся история — новые данные.

В список намеренно включены монеты, которые позже переименовали или сняли с торгов (меньше ошибки выжившего).

    python -m research.universe --print-symbols
    python -m research.universe --root <data> --out <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import strategies as S
from .data import UNIVERSE
from .engine import metrics
from .search import COST_BPS, STRESS_COST_BPS
from .wave2 import Data2, _cut, eval_single, oi_price

EXTRA = ["MATICUSDT", "FILUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "INJUSDT", "SANDUSDT", "MANAUSDT", "AXSUSDT",
         "GALAUSDT", "APEUSDT", "EOSUSDT", "XLMUSDT", "ALGOUSDT", "VETUSDT", "ICPUSDT", "FTMUSDT", "THETAUSDT",
         "EGLDUSDT", "AAVEUSDT", "UNIUSDT", "SUSHIUSDT", "CRVUSDT", "COMPUSDT", "MKRUSDT", "SNXUSDT", "1INCHUSDT",
         "DYDXUSDT", "LDOUSDT", "RUNEUSDT", "KAVAUSDT", "ZECUSDT", "DASHUSDT", "XTZUSDT", "HBARUSDT", "GRTUSDT",
         "CHZUSDT", "ENJUSDT", "ZILUSDT", "IOTAUSDT", "NEOUSDT", "WAVESUSDT", "KSMUSDT", "SUIUSDT", "SEIUSDT",
         "TIAUSDT", "WLDUSDT", "1000PEPEUSDT", "1000SHIBUSDT", "FETUSDT", "ARUSDT", "STXUSDT", "IMXUSDT", "GMTUSDT"]

CANDIDATES = {
    "oi_liq": (oi_price, {"k": 12, "thr": 2.5, "hold": 4, "mode": "liq", "sign": 1}),
    "fe_q99": (S.funding_extreme, {"q": 0.99, "hold": 8, "win_days": 30}),
}


def available(root: Path, symbols: list[str]) -> list[str]:
    return [s for s in symbols if (root / f"{s}-5m.parquet").exists() and (root / f"{s}-funding.parquet").exists()]


def report(data: Data2, label: str) -> pd.DataFrame:
    rows = []
    for name, (fn, p) in CANDIDATES.items():
        for cost in (COST_BPS, STRESS_COST_BPS):
            port, per_sym, trades, gross = eval_single(data, "1h", fn, p, "full", cost)
            for per in ("is", "val", "ho", "oos1"):
                x = _cut(port, per)
                m = metrics(x)
                rows.append({"universe": label, "strategy": name, "cost": cost, "period": per, "sharpe": m["sharpe"],
                             "ann_ret": m["ann_ret"], "max_dd": m["max_dd"],
                             "pos_months": float((x.resample("ME").sum() > 0).mean())})
            if cost == COST_BPS:
                pos_frac = np.mean([v > 0 for v in per_sym.values()])
                print(f"  {label} {name}: сделок всего {trades}, в день {trades / max(len(port) / 24, 1):.2f}, "
                      f"доля монет с Sharpe>0 за весь период {pos_frac:.0%}", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 250)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--out")
    ap.add_argument("--print-symbols", action="store_true")
    ap.add_argument("--all", action="store_true", help="с --print-symbols: исходные 16 + новые")
    a = ap.parse_args()
    if a.print_symbols:
        print(",".join((UNIVERSE if a.all else []) + EXTRA))
        sys.exit(0)
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    extra = available(root, EXTRA)
    print(f"Новые монеты с данными: {len(extra)} из {len(EXTRA)}; нет данных: "
          f"{sorted(set(EXTRA) - set(extra))}")
    res = report(Data2(root, extra), "new_coins")
    res.to_csv(out / "universe.csv", index=False)
    t = res.pivot_table(index=["strategy", "cost"], columns="period", values=["sharpe", "ann_ret", "max_dd"]).round(3)
    print("\nНОВЫЕ МОНЕТЫ (параметры не менялись), равные доли капитала на монету:")
    print(t.to_string())
