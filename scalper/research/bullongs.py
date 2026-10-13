"""Лонги бота 4h только в бычьем режиме: может, в бычьем рынке фильтр агрессоров для лонгов можно смягчить.

Лонги — research/longfilt.py: пробои линий зигзага 3 ATR со всеми фильтрами бота, кроме агрессоров; сделка как у бота
(ретест, 3R, долив 0.5 на +1.5R). Режим — на последнем закрытом дне до сигнала (заданы до запуска):
    приток        предложение долларовых стейблкоинов за 30 дней выросло (research/extreg.py — лучший включатель тренда)
    BTC > SMA200  закрытие BTC выше 200-дневной средней
    оба           оба условия
Фильтр лонгов внутри режима: агрессоры ≥ 55% (бот) / ≥ 52.5% / ≥ 50% / без фильтра.
Принятие пары «режим × фильтр»: R на сделку > 0 во всех периодах, где сделок ≥ 10, и R в месяц больше, чем у лонгов бота
без режима (55%, всегда), в каждом периоде IS / VAL / HO / 2026. Пар 12 — один случайный проход возможен.

    python -m research.longfilt collect ...; python -m research.extreg collect; python -m research.bullongs collect
    python -m research.bullongs report
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .oos import PER_ALL
from .shard import all_parts, part_path, shard_id
from .smc import _cell
from .tline import tf_frame

FILTERS = {"55% (бот)": 0.55, "52.5%": 0.525, "50%": 0.50, "без фильтра": 0.0}
JUDGED = ("is", "val", "ho", "2026")


def collect(root: Path) -> None:
    if shard_id()[0] != 0:
        return
    d = tf_frame(root, "BTCUSDT", "1d")
    if d is None:
        print("  bullongs: нет BTCUSDT", flush=True)
        return
    pd.DataFrame({"day": d.index, "btc": d["close"].to_numpy()}).to_parquet(part_path("bullongs_btc"), index=False)
    print(f"  bullongs: BTC {len(d)} дней", flush=True)


def regimes(stable: pd.Series, btc: pd.Series) -> pd.DataFrame:
    """Режим на закрытии дня D (индекс — дни UTC)."""
    idx = btc.index.union(stable.index)
    st = stable.reindex(idx).ffill(limit=3)
    b = btc.reindex(idx)
    infl = st > st.shift(30)
    up = b > b.rolling(200, min_periods=200).mean()
    return pd.DataFrame({"всегда": True, "приток": infl, "BTC > SMA200": up, "оба": infl & up}, index=idx)


def label(trades: pd.DataFrame, reg: pd.DataFrame) -> pd.DataFrame:
    day = (trades.t + pd.Timedelta(hours=4)).dt.floor("D") - pd.Timedelta(days=1)
    r = reg.reindex(pd.DatetimeIndex(day)).fillna(False).astype(bool)
    r.index = trades.index
    return pd.concat([trades, r], axis=1)


def _per_month(z: pd.DataFrame, a: str, b: str) -> float:
    months = (pd.Timestamp(b) - pd.Timestamp(a)).days / 30.44
    return z.R.sum() / months


def report() -> None:
    lp, ep, bp = all_parts("longfilt"), all_parts("extreg"), all_parts("bullongs_btc")
    print("\n===== BULLONGS: лонги бота 4h в бычьем режиме (приток стейблкоинов / BTC > SMA200) с мягким фильтром "
          "агрессоров; сделка как у бота =====")
    if not lp or not ep or not bp:
        print(f"частей нет (лонги {len(lp)}, внешние {len(ep)}, BTC {len(bp)})")
        return
    tr = pd.concat([pd.read_parquet(p) for p in lp], ignore_index=True)
    tr["t"] = pd.to_datetime(tr["t"], utc=True)
    ext = pd.concat([pd.read_parquet(p) for p in ep], ignore_index=True)
    ext["day"] = pd.to_datetime(ext["day"], utc=True)
    stable = ext.groupby("day").stable.last().dropna()
    bt = pd.concat([pd.read_parquet(p) for p in bp], ignore_index=True)
    bt["day"] = pd.to_datetime(bt["day"], utc=True)
    btc = bt.set_index("day").btc.sort_index()
    df = label(tr.dropna(subset=["R"]), regimes(stable, btc))
    per = {**PER_ALL, "2026": ("2026-01-01", "2026-10-01")}
    print(f"лонгов {len(df)}; дней-сигналов в режиме: " + ", ".join(
        f"{k} {df[k].mean():.0%}" for k in ("приток", "BTC > SMA200", "оба")))
    base = df[df.buy >= 0.55]
    base_pm = {p: _per_month(base[(base.t >= a) & (base.t < b)], a, b) for p, (a, b) in per.items()}
    for reg in ("всегда", "приток", "BTC > SMA200", "оба"):
        print(f"\n  режим «{reg}»")
        for fn, thr in FILTERS.items():
            z = df[df[reg] & (df.buy >= thr)]
            ok, better = [], []
            for p, (a, b) in per.items():
                m = z[(z.t >= a) & (z.t < b)]
                if len(m) >= 10:
                    ok.append(m.R.mean() > 0)
                if p in JUDGED:
                    better.append(_per_month(m, a, b) > base_pm[p])
            tag = "" if (reg == "всегда" and thr == 0.55) else (
                "  → принято" if ok and all(ok) and all(better) else
                f"  → не принято (R > 0 в {sum(ok)} из {len(ok)}, R в месяц выше бота в {sum(better)} из {len(better)})")
            print(f"    {fn}: {len(z)} сделок, всего {z.R.sum():+.1f}R{tag}")
            print("      " + "  ".join(f"{p}: {_cell(z[(z.t >= a) & (z.t < b)])}" for p, (a, b) in per.items()))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    a = ap.parse_args()
    collect(Path(a.root).expanduser()) if a.mode == "collect" else report()
