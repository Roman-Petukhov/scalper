"""Сектора альткоинов: ротация, заражение пампом внутри сектора, опережение одного сектора другим (вопрос владельца
«пампы идут пачками по секторам, один сектор опережает другой»). Раньше сектор проверялся только как фильтр сделок бота
(docs/knowledge.md, раздел 5) — без разницы.

Все USDT-перпетуалы Binance с делистнутыми (1h-all → дневки), 2020-01 … 2026-09. Сектора — разметка research/oos.py
(мемы, AI, L1, L2, DeFi, игры/NFT) и три добавленные здесь (RWA, приватность, DePIN). Индекс сектора — среднее дневных
доходностей участников с оборотом перпа за 30 дней >= $5M; рынок — то же по всем монетам без BTC.
Всё задано до запуска:
    S1 ротация      каждый понедельник — сектор с лучшей доходностью сверх рынка за 7 / 28 дней; держим его корзину
                    7 дней. Справка — худший сектор (разворот).
    S2 заражение    монета сектора выросла за день >= +30% («лидер»); на закрытии дня покупаем других участников сектора,
                    выросших меньше +10% («отстающих»), держим 1 / 3 дня. Плацебо — такие же монеты других секторов
                    в тот же день.
    S3 опережение   корреляция доходности сектора A за день t с сектором B за день t+1: пары с наибольшей |корр.| в IS —
                    держится ли знак в VAL и HO (справка; сделка — только если все три пары сохранят знак).
Издержки taker 11 б.п. за круг. Мерило — сверх рынка того же срока. Принятие (S1, S2): > 0 в каждом периоде IS / VAL / HO
(сделок >= 20), t по дням >= 3, > 0 при издержках ×2; у S2 ещё и лучше плацебо во всех трёх.

    python -m research.sectors collect --symbols <1h-all>
    python -m research.sectors report
"""
from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .oos import PER_ALL, SECTORS
from .ppr import day_t
from .shard import all_parts, mine, part_path
from .tline import tf_frame

EXTRA = {"RWA": ("ONDO", "OM", "POLYX", "PLUME", "CFG", "TRU", "PENDLE", "SYRUP", "USUAL"),
         "приватность": ("XMR", "ZEC", "DASH", "ZEN", "SCRT", "ROSE", "DUSK", "ARRR"),
         "DePIN": ("FIL", "AR", "STORJ", "HNT", "IOTX", "THETA", "GRASS", "ATH", "POWR")}
ADV_MIN = 5e6
COST = 0.0011
LEADER, LAGGARD = 0.30, 0.10
LOOKS = (7, 28)
HS2 = (1, 3)
IS_END = "2024-07-01"


def sector_map() -> dict[str, str]:
    out: dict[str, str] = {}
    for name, coins in list(EXTRA.items()) + list(SECTORS.items()):     # узкие сектора важнее общего L1
        for c in coins:
            out.setdefault(c + "USDT", name)
    return out


def collect(root: Path, syms: list[str]) -> None:
    rows = []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, "1d")
            if d is None or len(d) < 60:
                continue
            rows.append(pd.DataFrame({"day": d.index, "close": d["close"].to_numpy(dtype="float64"),
                                      "qv": d["quote_volume"].to_numpy(dtype="float64"), "symbol": s}))
        except Exception as e:
            print(f"  sectors {s}: пропуск ({e})", flush=True)
    print(f"  sectors: монет {len(rows)}", flush=True)
    if rows:
        pd.concat(rows, ignore_index=True).to_parquet(part_path("daily"), index=False)


def load_daily() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    parts = all_parts("daily")
    if not parts:
        return None
    x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    x["day"] = pd.to_datetime(x["day"], utc=True).dt.tz_localize(None)
    close = x.pivot_table(index="day", columns="symbol", values="close").sort_index()
    qv = x.pivot_table(index="day", columns="symbol", values="qv").reindex_like(close)
    return close, qv


def panel(close: pd.DataFrame, qv: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, pd.Series]]:
    """Дневные доходности (только ликвидные по прошлому обороту), рынок и индексы секторов."""
    r = close.pct_change()
    liquid = qv.rolling(30, min_periods=20).mean().shift(1) >= ADV_MIN
    r = r.where(liquid)
    mkt = r.drop(columns=["BTCUSDT"], errors="ignore").mean(axis=1)
    smap = sector_map()
    idx = {}
    for name in dict.fromkeys(smap.values()):
        cols = [c for c in r.columns if smap.get(c) == name]
        if cols:
            idx[name] = r[cols].mean(axis=1, skipna=True).where(r[cols].notna().sum(axis=1) >= 3)
    return r, mkt, idx


def verdict(z: pd.DataFrame) -> tuple[bool, str]:
    cells, ok = [], []
    for p, (a, b) in PER_ALL.items():
        m = z[(z.t >= a) & (z.t < b)]
        if m.empty:
            continue
        cells.append(f"{p} {m.X.mean():+.2%} ({len(m)})")
        if len(m) >= 20 and p in ("is", "val", "ho"):
            ok.append(m.X.mean() > 0)
    t = day_t(z, "X") if len(z) > 10 else float("nan")
    x2 = (z.X - COST).mean()
    good = bool(ok) and all(ok) and np.isfinite(t) and t >= 3 and x2 > 0
    return good, f"сверх рынка: {'  '.join(cells)}\n        всё время {z.X.mean():+.2%} (t {t:.1f}); ×2 {x2:+.2%}"


def rotation(mkt: pd.Series, idx: dict[str, pd.Series], look: int, best: bool) -> pd.DataFrame:
    sec = pd.DataFrame(idx)
    ex = sec.sub(mkt, axis=0)
    cum = np.log1p(ex.fillna(0.0)).rolling(look, min_periods=look - 2).sum()
    rows = []
    for t in sec.index[sec.index.dayofweek == 0]:
        prev = t - pd.Timedelta(days=1)
        if prev not in cum.index:
            continue
        sc = cum.loc[prev].where(sec.loc[:prev].tail(look).notna().sum() >= look - 2).dropna()
        if len(sc) < 4:
            continue
        pick = sc.idxmax() if best else sc.idxmin()
        fwd = ex[pick].loc[t:t + pd.Timedelta(days=6)]
        if len(fwd) < 5:
            continue
        rows.append({"t": t, "X": float(np.expm1(np.log1p(fwd.fillna(0.0)).sum())) - COST, "sector": pick})
    return pd.DataFrame(rows)


def contagion(close: pd.DataFrame, r: pd.DataFrame, mkt: pd.Series, h: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    smap = sector_map()
    fwd = close.shift(-h) / close - 1
    mfwd = (fwd.where(r.notna())).drop(columns=["BTCUSDT"], errors="ignore").mean(axis=1)
    lag_rows, plc_rows = [], []
    for t in r.index:
        row = r.loc[t]
        leaders = row[row >= LEADER].index
        secs = {smap[c] for c in leaders if c in smap}
        if not secs:
            continue
        lag = [c for c in row.index if smap.get(c) in secs and c not in leaders and row[c] < LAGGARD and np.isfinite(row[c])]
        plc = [c for c in row.index if c in smap and smap[c] not in secs and row[c] < LAGGARD and np.isfinite(row[c])]
        for cols, out in ((lag, lag_rows), (plc, plc_rows)):
            v = fwd.loc[t, cols].dropna()
            if len(v):
                out.append({"t": t, "X": float(v.mean() - mfwd.loc[t]) - COST, "n": len(v)})
    return pd.DataFrame(lag_rows), pd.DataFrame(plc_rows)


def leadlag(idx: dict[str, pd.Series]) -> None:
    sec = pd.DataFrame(idx)
    sec = sec.sub(sec.mean(axis=1), axis=0)                    # сверх среднего по секторам: без общего рынка
    pairs = []
    for a, b in itertools.permutations(sec.columns, 2):
        cs = {}
        for p, (lo, hi) in (("is", ("2022-01-01", IS_END)), ("val", (IS_END, "2025-07-01")), ("ho", ("2025-07-01", "2026-10-01"))):
            m = sec.loc[lo:hi]
            cs[p] = m[a].corr(m[b].shift(-1))
        pairs.append((a, b, cs))
    top = sorted(pairs, key=lambda x: -abs(x[2]["is"] if np.isfinite(x[2]["is"]) else 0))[:3]
    print("\n  --- S3: опережение (сектор A сегодня → сектор B завтра, сверх среднего по секторам) ---")
    keep = []
    for a, b, cs in top:
        same = all(np.sign(cs[p]) == np.sign(cs["is"]) for p in ("val", "ho"))
        keep.append(same)
        print(f"    {a} → {b}: IS {cs['is']:+.3f}, VAL {cs['val']:+.3f}, HO {cs['ho']:+.3f} — знак {'держится' if same else 'меняется'}")
    allc = np.array([cs["is"] for _, _, cs in pairs])
    print(f"    все {len(pairs)} пар в IS: медиана |корр.| {np.nanmedian(np.abs(allc)):.3f}; "
          f"{'все три пары сохранили знак — кандидат' if all(keep) else 'опережения, устойчивого вне IS, нет'}")


def report() -> None:
    print("\n===== SECTORS: ротация секторов, заражение пампом внутри сектора, опережение (перпетуалы Binance, дневки) =====")
    got = load_daily()
    if got is None:
        print("  sectors: нет данных")
        return
    close, qv = got
    r, mkt, idx = panel(close, qv)
    smap = sector_map()
    counts = {k: sum(1 for c in close.columns if smap.get(c) == k) for k in idx}
    print("  монет в секторах: " + ", ".join(f"{k} {v}" for k, v in counts.items()) + f"; всего монет {close.shape[1]}")
    passed = []
    print("\n  --- S1: ротация (лучший сектор недели) ---")
    for look, best in itertools.product(LOOKS, (True, False)):
        z = rotation(mkt, idx, look, best)
        if z.empty:
            continue
        good, txt = verdict(z)
        name = f"{'лучший' if best else 'худший'} сектор за {look} дней"
        tag = ("принято" if good else "нет") if best else "справка"
        top = z.sector.value_counts().head(3)
        print(f"    {name}: {tag}\n        {txt}\n        чаще всего: " + ", ".join(f"{k} {v}" for k, v in top.items()))
        if good and best:
            passed.append(name)
    print("\n  --- S2: заражение пампом внутри сектора ---")
    for h in HS2:
        lag, plc = contagion(close, r, mkt, h)
        if lag.empty:
            continue
        good, txt = verdict(lag)
        _, ptxt = verdict(plc)
        better = []
        for p in ("is", "val", "ho"):
            a, b = PER_ALL[p]
            la, pl = lag[(lag.t >= a) & (lag.t < b)], plc[(plc.t >= a) & (plc.t < b)]
            better.append(len(la) >= 20 and la.X.mean() > pl.X.mean())
        good = good and all(better)
        print(f"    отстающие сектора лидера, {h} д.: {'принято' if good else 'нет'} (событий {len(lag)}, монет в среднем "
              f"{lag.n.mean():.1f})\n        {txt}\n      плацебо (другие сектора):\n        {ptxt}")
        if good:
            passed.append(f"заражение {h} д.")
    leadlag(idx)
    print("\n  ИТОГ: " + ("прошли — " + "; ".join(passed) if passed else "секторные волны не дают сделки, окупающей издержки"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [s for s in a.symbols.split(",") if s])
    else:
        report()
