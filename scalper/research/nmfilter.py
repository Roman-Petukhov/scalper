"""Цена на активный адрес как фильтр шортов бота 4h — продолжение этапа 3 (research/onchainval.py): монеты, дорогие по
капитализации на активный адрес, отставали от BTC во всех периодах с 2022 (за 6 мес. −16%, t −2.0). Если это свойство
монеты, шорты бота по дорогим должны давать больше R, чем по дешёвым.

Сделки — research/crisk.py (4h, правило бота, ретест, 3R; R уже с издержками). Мера на конце месяца перед сигналом:
средние активные адреса за 30 дней / капитализация (CoinMetrics community); трети — по всем монетам CoinMetrics с
перпетуалом в этом месяце (>= 15). Задано до запуска: фильтр полезен, если R на сделку у шортов «дорогой трети» выше,
чем у шортов двух остальных, в каждом периоде 2021 / IS / VAL / HO с >= 20 сделками в обеих группах, и разница по всем
сделкам с t >= 2. Лонги — для справки.

    CRISK_TFS=4h python -m research.crisk collect --symbols <1h-qualified> && python -m research.nmfilter report
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from . import onchainval as O
from .oos import PER_ALL
from .shard import all_parts

MIN_COINS = 15


def load_trades() -> pd.DataFrame | None:
    parts = all_parts("crisk_4h")
    if not parts:
        return None
    z = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    z["t"] = pd.to_datetime(z["t"], utc=True).dt.tz_localize(None)
    return z.dropna(subset=["R"])


def tercile(nm: pd.DataFrame, z: pd.DataFrame) -> pd.Series:
    """Треть монеты сделки по цене на адрес на конце прошлого месяца: 0 — дешёвые, 2 — дорогие; NaN — нет данных."""
    ok = nm.notna().sum(axis=1) >= MIN_COINS
    rk = nm[ok].rank(axis=1, pct=True)                         # больше адресов на доллар — дешевле
    month = (z["t"].dt.to_period("M") - 1).dt.to_timestamp()
    out = []
    for m, s in zip(month, z["symbol"]):
        v = rk.at[m, s] if (m in rk.index and s in rk.columns) else np.nan
        out.append(np.nan if not np.isfinite(v) else (2 if v <= 1 / 3 else 0 if v > 2 / 3 else 1))
    return pd.Series(out, index=z.index)


def diff_t(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 10 or len(b) < 10:
        return float("nan")
    se = np.sqrt(a.var() / len(a) + b.var() / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def report() -> None:
    print("\n===== NMFILTER: шорты бота 4h по цене на активный адрес (CoinMetrics; трети на конце прошлого месяца) =====")
    z = load_trades()
    if z is None:
        print("  nmfilter: нет сделок crisk_4h")
        return
    cm = O.cm_series(O.cm_assets())
    cm["symbol"] = cm["asset"].map(O.binance_symbol)
    cm = cm.dropna(subset=["symbol"])
    adr, cap = O.month_end(cm, "adr", "mean30"), O.month_end(cm, "cap", "last")
    nm = (adr / cap.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    nm = nm[[c for c in nm.columns if c in set(z.symbol)]]
    z["terc"] = tercile(nm, z)
    print(f"  сделок {len(z)}, с данными CoinMetrics {z.terc.notna().sum()} ({z.terc.notna().mean():.0%}), монет в мере "
          f"{nm.shape[1]}")
    names = {0: "дешёвые", 1: "средние", 2: "дорогие"}
    ok, good = [], True
    for side, sname in ((-1, "шорты"), (1, "лонги")):
        s = z[(z.side == side) & z.terc.notna()]
        print(f"\n    {sname}: сделок с мерой {len(s)}")
        for p, (a, b) in PER_ALL.items():
            m = s[(s.t >= a) & (s.t < b)]
            if m.empty:
                continue
            cells = "  ".join(f"{names[k]} {m[m.terc == k].R.mean():+.2f}R ({(m.terc == k).sum()})" for k in (0, 1, 2))
            print(f"      {p}: {cells}")
            if side == -1 and p != "2020":
                dear, rest = m[m.terc == 2], m[m.terc != 2]
                if len(dear) >= 20 and len(rest) >= 20:
                    ok.append(dear.R.mean() > rest.R.mean())
        dear, rest = s[s.terc == 2].R, s[s.terc != 2].R
        t = diff_t(dear, rest)
        print(f"      всё время: дорогие {dear.mean():+.3f}R ({len(dear)}), остальные {rest.mean():+.3f}R ({len(rest)}), "
              f"разница t {t:.1f}")
        if side == -1:
            good = bool(ok) and all(ok) and np.isfinite(t) and t >= 2
    print("\n  ИТОГ: " + ("фильтр «шорты по дорогим» прошёл — кандидат в бумажную проверку" if good else
                         "цена на адрес не отделяет лучшие шорты бота"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
