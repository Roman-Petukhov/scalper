"""
Арбитраж funding на одной бирже: когда ставка на альте взлетает, покупаем спот и шортим перп на ту же сумму —
цена нам безразлична, получаем funding. Проверка на Binance (спот и USDT-перпетуал, часовые свечи); на Bybit механика
та же, наличие спот-пары на Bybit нужно проверять отдельно.

Ставка приводится к 8 ч: r8 = rate x 8 / интервал выплат (бывают 1, 4 и 8 ч).
Вход сразу после выплаты, на которой r8 >= порога входа (0.05 / 0.1 / 0.2 / 0.5%), по close часа перед выплатой
(спот и перп — одинаковый номинал). Держим, получая следующие выплаты; выход сразу после первой выплаты с
r8 < 0.01% или через 14 дней. Одна позиция на монету.
Результат на $1 номинала каждой ноги: funding + перп-нога (шорт) + спот-нога (лонг) − издержки (перп тейкер
2 x 5.5 б.п., спот 2 x 10 б.п.). Капитал на $1 номинала: $1 спот + $0.2 маржи перпа (плечо 5) = $1.2.

    python -m research.fundarb collect --symbols ...   (по частям; спот-свечи готовит research.prepare 1h-spot)
    python -m research.fundarb report
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .shard import all_parts, mine, part_path

THRESH = (0.0005, 0.001, 0.002, 0.005)
EXIT_R8 = 0.0001
MAX_HOLD = pd.Timedelta(days=14)
COST = 2 * 5.5e-4 + 2 * 10e-4
CAP_PER_NOTIONAL = 1.2
PER = {"is": ("2022-01-01", "2024-07-01"), "val": ("2024-07-01", "2025-07-01"), "ho": ("2025-07-01", "2026-10-01")}


def hourly(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    k = pd.read_parquet(path, columns=["open_time", "close"])
    s = pd.Series(k["close"].to_numpy(dtype="float64"), index=pd.to_datetime(k["open_time"], unit="ms", utc=True))
    return s[~s.index.duplicated()].sort_index()


def trades(root: Path, sym: str) -> pd.DataFrame:
    perp, spot = hourly(root / f"{sym}-1h.parquet"), hourly(root / f"{sym}-spot-1h.parquet")
    fp = root / f"{sym}-funding.parquet"
    if perp is None or spot is None or not fp.exists():
        return pd.DataFrame()
    f = pd.read_parquet(fp)
    f = pd.Series(f["rate"].to_numpy(dtype="float64"),
                  index=pd.to_datetime(f["ts"], unit="ms", utc=True).dt.floor("min")).sort_index()
    f = f[~f.index.duplicated()]
    gap_h = pd.Series(f.index, index=f.index).diff().dt.total_seconds().div(3600).clip(1, 8).fillna(8)
    r8 = f * 8 / gap_h
    # цена «на момент выплаты» — close часа, который заканчивается в момент выплаты
    key = f.index.floor("h") - pd.Timedelta(hours=1)
    pp, sp = perp.reindex(key).to_numpy(), spot.reindex(key).to_numpy()
    ts, rate, r8v = f.index, f.to_numpy(), r8.to_numpy()
    rows = []
    for th in THRESH:
        i, n = 0, len(f)
        while i < n:
            if not (r8v[i] >= th and np.isfinite(pp[i]) and np.isfinite(sp[i])):
                i += 1
                continue
            got, j = 0.0, i + 1
            while j < n:
                got += rate[j]
                if r8v[j] < EXIT_R8 or ts[j] - ts[i] >= MAX_HOLD:
                    break
                j += 1
            if j >= n or not (np.isfinite(pp[j]) and np.isfinite(sp[j])):
                break
            perp_leg = -(pp[j] / pp[i] - 1)
            spot_leg = sp[j] / sp[i] - 1
            net = got + perp_leg + spot_leg - COST
            rows.append({"symbol": sym, "thresh": th, "t": ts[i], "hold_h": (ts[j] - ts[i]).total_seconds() / 3600,
                         "r8_in": r8v[i], "funding": got, "basis": perp_leg + spot_leg, "net": net,
                         "roc": net / CAP_PER_NOTIONAL})
            i = j + 1
    return pd.DataFrame(rows)


def report() -> None:
    parts = all_parts("fundarb")
    if not parts:
        print("частей нет")
        return
    tr = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    tr["t"] = pd.to_datetime(tr["t"], utc=True)
    print(f"===== FUND ARB: сделок {len(tr)}, монет {tr.symbol.nunique()} (со спотом на Binance), частей {len(parts)} =====")
    print("на $1 номинала каждой ноги, %; roc — на капитал $1.2; t — кластеризованный по дням входа")
    rows = []
    for (th, p), g in tr.assign(per=pd.cut(tr["t"], [pd.Timestamp(v[0], tz="UTC") for v in PER.values()] +
                                           [pd.Timestamp("2026-10-01", tz="UTC")], labels=list(PER), right=False)
                               ).groupby(["thresh", "per"], observed=True):
        r = g["net"].to_numpy()
        d = g["t"].dt.floor("D").to_numpy()
        gg = pd.Series(r - r.mean()).groupby(d).sum()
        rows.append({"порог r8": f"{th:.2%}", "период": p, "сделок": len(g), "монет": g.symbol.nunique(),
                     "удержание ч (медиана)": g["hold_h"].median(), "funding %": g["funding"].mean() * 100,
                     "базис %": g["basis"].mean() * 100, "итог %": g["net"].mean() * 100,
                     "t": r.sum() / np.sqrt((gg ** 2).sum()) if (gg ** 2).sum() > 0 else np.nan,
                     "прибыльных": (g["net"] > 0).mean(), "худшая %": g["net"].min() * 100,
                     "сделок/мес": len(g) / max(1, g["t"].dt.to_period("M").nunique())})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    for th in THRESH:
        g = tr[tr.thresh == th]
        y = g.groupby(g["t"].dt.year)["net"].agg(["size", "mean"])
        print(f"  порог {th:.2%} по годам (итог %): " + ", ".join(f"{k}: {v['mean'] * 100:+.2f} (n={int(v['size'])})"
                                                          for k, v in y.iterrows()))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 30)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        root = Path(a.root).expanduser()
        out = []
        for s in mine([x for x in a.symbols.split(",") if x]):
            try:
                out.append(trades(root, s))
            except Exception as e:
                print(f"  {s}: пропуск ({e})", flush=True)
        out = [x for x in out if len(x)]
        if out:
            pd.concat(out, ignore_index=True).to_parquet(part_path("fundarb"), index=False)
    else:
        report()
