"""«Топливо сквиза»: открытый интерес перпетуала против реального спотового рынка (вопрос владельца — почему монеты
дают огромные пампы). Громкие пампы 2025–26 (ALPACA: OI $79M при капитализации $15M; MYX, COAI, RAVE — малое обращение)
описываются одной механикой: плечо на перпетуале в разы больше спотовой ликвидности, толпа в шортах — сквиз.

Перпетуалы 1h-qualified с метриками Binance (OI с 2022) и спот Binance той же монеты. На каждый день (00:00 UTC):
    fuel     OI × цена / средний дневной оборот спота за 30 прошлых дней (нет спота на Binance — признак «без спота»)
    fuel_p   то же к обороту самого перпетуала (справка)
    fund3    сумма funding за 3 дня; ret7 — доходность за 7 дней
Часть A — анатомия: дни с ростом >= +50% (закрытие к закрытию) против всех дней; лифт признаков, известных накануне.
Часть B — сделки (заданы до запуска), каждый день по монетам с fuel в верхних 10% дня:
    лонг   fund3 < 0 (толпа в шортах — топливо для сквиза)
    шорт   fund3 > 0 (толпа в лонгах — топливо для обвала)
вход по закрытию дня, выход через 1 / 3 / 7 дней, издержки taker 11 б.п. за круг, funding за удержание — по факту.
Мерило — результат после издержек и сверх среднего по всем монетам того же дня. Принятие: > 0 в каждом периоде IS / VAL /
HO (сделок >= 20), t по дням >= 3, > 0 при издержках ×2.

    python -m research.prepare 1h-qualified 1h-spot && python -m research.squeezefuel collect --symbols <1h-qualified>
    python -m research.squeezefuel report

Прогон 1 (40462018949): лонг 3 / 7 дней прошёл (сверх рынка +0.46% / +0.70%, t 3.6 / 4.1), шорт — нет. Прогон 2 —
устойчивость лонга 7 дней (задана до запуска; принят, если все пункты в плюсе в IS / VAL / HO):
    без перекрытия   по монете не больше одной сделки на 7 дней (первый сигнал эпизода)
    без хвоста       без 1% лучших сделок
    издержки мелочи  60 б.п. за круг, если спотовый оборот за 30 дней < $5M
    соседи           верхние 5% / 20% fuel; funding < −0.05% вместо < 0
    по годам         2022 … 2026 — справка
"""
from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .oos import PER_ALL
from .ppr import day_t
from .shard import all_parts, mine, part_path
from .tline import tf_frame
from .wave2 import metrics_on_bars

HS = (1, 3, 7)
PUMP = 0.5
TOP = 0.10
COST = 0.0011


def spot_daily(root: Path, sym: str) -> pd.Series | None:
    p = root / f"{sym}-spot-1h.parquet"
    if not p.exists():
        return None
    k = pd.read_parquet(p, columns=["open_time", "quote_volume"])
    s = pd.Series(k["quote_volume"].to_numpy(dtype="float64"), index=pd.to_datetime(k["open_time"], unit="ms", utc=True))
    return s[~s.index.duplicated()].sort_index().resample("1D").sum()


def coin_days(d: pd.DataFrame, m: pd.DataFrame, spot: pd.Series | None) -> pd.DataFrame:
    """Дневная таблица монеты: признаки на конце дня t и доходности вперёд от закрытия дня t."""
    day = d.index.floor("D")
    g = pd.DataFrame({"close": d["close"].to_numpy(), "qv": d["quote_volume"].to_numpy(),
                      "fund": d["funding"].fillna(0.0).to_numpy(), "oi": m["oi"].to_numpy(dtype="float64")},
                     index=d.index)
    a = g.groupby(day).agg(close=("close", "last"), qv=("qv", "sum"), fund=("fund", "sum"), oi=("oi", "last"))
    out = pd.DataFrame(index=a.index)
    out["r1"] = a["close"].pct_change()
    oi_val = a["oi"] * a["close"]
    perp_adv = a["qv"].rolling(30, min_periods=20).mean()
    out["fuel_p"] = oi_val / perp_adv
    if spot is not None:
        sadv = spot.reindex(a.index).rolling(30, min_periods=20).mean()
        out["fuel"] = oi_val / sadv.where(sadv > 0)
        out["sadv"] = sadv
        out["spot"] = True
    else:
        out["fuel"] = np.nan
        out["sadv"] = np.nan
        out["spot"] = False
    out["fund3"] = a["fund"].rolling(3).sum()
    out["ret7"] = a["close"].pct_change(7)
    for h in HS:
        out[f"f{h}"] = a["close"].shift(-h) / a["close"] - 1
        out[f"ff{h}"] = a["fund"][::-1].rolling(h, min_periods=h).sum()[::-1].shift(-1)   # funding дней t+1 … t+h
    return out


def collect(root: Path, syms: list[str]) -> None:
    rows = []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, "1h")
            if d is None or len(d) < 24 * 90:
                continue
            m = metrics_on_bars(s, root, d.index, 60)
            if m["oi"].notna().sum() < 24 * 60:
                continue
            x = coin_days(d, m, spot_daily(root, s))
            rows.append(x[x.index >= "2022-01-01"].assign(symbol=s).reset_index(names="day"))
        except Exception as e:
            print(f"  squeezefuel {s}: пропуск ({e})", flush=True)
    print(f"  squeezefuel: монет {len(rows)}, со спотом {sum(bool(r['spot'].iloc[0]) for r in rows)}", flush=True)
    if rows:
        pd.concat(rows, ignore_index=True).to_parquet(part_path("squeezefuel"), index=False)


def anatomy(x: pd.DataFrame) -> None:
    """Лифт признаков накануне дня >= +50%: доля среди пампов / доля среди всех дней."""
    x = x.sort_values(["symbol", "day"]).copy()
    prev = x.groupby("symbol")[["fuel_rk", "fuelp_rk", "fund3", "ret7", "spot"]].shift(1)
    x = x.join(prev.add_suffix("_prev"))
    pumps = x[x.r1 >= PUMP]
    feats = {"fuel в верхних 10% дня": lambda z: z.fuel_rk_prev >= 1 - TOP,
             "fuel к обороту перпа в верхних 10%": lambda z: z.fuelp_rk_prev >= 1 - TOP,
             "funding за 3 дня < 0": lambda z: z.fund3_prev < 0,
             "funding за 3 дня < −0.1%": lambda z: z.fund3_prev < -0.001,
             "нет спота на Binance": lambda z: z.spot_prev == False,               # noqa: E712
             "за 7 дней уже +30%": lambda z: z.ret7_prev >= 0.3,
             "fuel в верхних 10% и funding < 0": lambda z: (z.fuel_rk_prev >= 1 - TOP) & (z.fund3_prev < 0)}
    print(f"\n  --- A: анатомия — дней с ростом >= +{PUMP:.0%}: {len(pumps)} из {len(x)} "
          f"({len(pumps) / len(x):.3%}), монет {pumps.symbol.nunique()} ---")
    for name, f in feats.items():
        base, inp = f(x).mean(), f(pumps).mean()
        print(f"    {name}: у пампов {inp:.0%}, у всех дней {base:.0%}, лифт {inp / base if base > 0 else float('nan'):.1f}")
    after = x.loc[pumps.index]
    for h in HS:
        print(f"    после пампа через {h} д.: средняя {after[f'f{h}'].mean():+.1%}, медиана {after[f'f{h}'].median():+.1%}")


def trades(x: pd.DataFrame, side: int, h: int) -> pd.DataFrame:
    mkt = x.groupby("day")[f"f{h}"].transform("mean")
    sel = x[(x.fuel_rk >= 1 - TOP) & ((x.fund3 < 0) if side > 0 else (x.fund3 > 0))].dropna(subset=[f"f{h}"])
    m = mkt.loc[sel.index]
    raw = side * sel[f"f{h}"] - side * sel[f"ff{h}"].fillna(0.0)
    return pd.DataFrame({"t": sel["day"], "R": raw - COST, "X": raw - side * m - COST, "raw": raw})


def verdict(z: pd.DataFrame) -> tuple[bool, str]:
    cells, ok = [], []
    for p, (a, b) in PER_ALL.items():
        m = z[(z.t >= a) & (z.t < b)]
        if m.empty:
            continue
        cells.append(f"{p} {m.X.mean():+.2%} / {m.R.mean():+.2%} ({len(m)})")
        if len(m) >= 20 and p in ("is", "val", "ho"):
            ok.append(m.X.mean() > 0)
    t = day_t(z, "X") if len(z) > 10 else float("nan")
    x2 = (z.X - COST).mean()
    good = bool(ok) and all(ok) and np.isfinite(t) and t >= 3 and x2 > 0
    return good, (f"сверх рынка / сама: {'  '.join(cells)}\n        всё время сверх рынка {z.X.mean():+.2%} (t {t:.1f}), сама "
                  f"{z.R.mean():+.2%}, в плюсе {(z.R > 0).mean():.0%}; ×2 {x2:+.2%}")


def robust(x: pd.DataFrame) -> None:
    """Прогон 2: устойчивость лонга 7 дней."""
    h = 7
    base = trades(x, 1, h).join(x[["symbol", "fuel", "fund3"]], how="left")
    print("\n  --- C: устойчивость лонга 7 дней ---")

    def show(name: str, z: pd.DataFrame) -> bool:
        good, txt = verdict(z)
        print(f"    {name}: {'держится' if good else 'нет'}\n        {txt}")
        return good

    nz = base.sort_values(["symbol", "t"])
    keep, last = [], {}
    for i, (s, t) in zip(nz.index, zip(nz.symbol, nz.t)):
        if s not in last or (t - last[s]).days >= h:
            keep.append(i)
            last[s] = t
    res = [show("без перекрытия (одна сделка на монету на 7 дней)", base.loc[keep])]
    cut = base.raw.quantile(0.99)
    res.append(show("без 1% лучших сделок", base[base.raw < cut]))
    small = base.index.isin(x.index[(x["sadv"] < 5e6)]) if "sadv" in x else np.zeros(len(base), bool)
    z = base.copy()
    z.loc[small, ["R", "X"]] -= 0.0049
    print(f"      монет со спотовым оборотом < $5M среди сделок: {small.mean():.0%}")
    res.append(show("издержки 60 б.п. у мелочи", z))
    for q, name in ((0.05, "верхние 5% fuel"), (0.20, "верхние 20% fuel")):
        mkt = x.groupby("day")[f"f{h}"].transform("mean")
        sel = x[(x.fuel_rk >= 1 - q) & (x.fund3 < 0)].dropna(subset=[f"f{h}"])
        raw = sel[f"f{h}"] - sel[f"ff{h}"].fillna(0.0)
        res.append(show(name, pd.DataFrame({"t": sel["day"], "R": raw - COST, "X": raw - mkt.loc[sel.index] - COST,
                                            "raw": raw})))
    res.append(show("funding за 3 дня < −0.05%", base[base.fund3 < -0.0005]))
    for y in range(2022, 2027):
        m = base[base.t.dt.year == y]
        if len(m):
            print(f"      {y}: сверх рынка {m.X.mean():+.2%}, сама {m.R.mean():+.2%} ({len(m)}, монет {m.symbol.nunique()})")
    print(f"\n  ИТОГ УСТОЙЧИВОСТИ: {'держится во всех проверках' if all(res) else 'не во всех проверках'} "
          f"({sum(res)} из {len(res)})")


def report() -> None:
    print("\n===== SQUEEZEFUEL: открытый интерес против спотового рынка и перекос толпы — анатомия пампов и сделки =====")
    parts = all_parts("squeezefuel")
    if not parts:
        print("  squeezefuel: нет данных")
        return
    x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    x["day"] = pd.to_datetime(x["day"], utc=True).dt.tz_localize(None)
    x = x.replace([np.inf, -np.inf], np.nan)
    x["fuel_rk"] = x.groupby("day")["fuel"].rank(pct=True)
    x["fuelp_rk"] = x.groupby("day")["fuel_p"].rank(pct=True)
    print(f"  монет {x.symbol.nunique()}, дней {x.day.nunique()}, со спотом {x[x.spot].symbol.nunique()}; медиана fuel "
          f"{x.fuel.median():.2f} (OI в днях спотового оборота), верхние 10% от {x.fuel.quantile(0.9):.1f}")
    anatomy(x)
    passed = []
    print("\n  --- B: сделки по верхним 10% fuel дня ---")
    for side, h in itertools.product((1, -1), HS):
        z = trades(x, side, h)
        good, txt = verdict(z)
        name = f"{'лонг, толпа в шортах' if side > 0 else 'шорт, толпа в лонгах'}, {h} д."
        print(f"    {name}: {'принято' if good else 'нет'}\n        {txt}")
        if good:
            passed.append(name)
    print("\n  ИТОГ: " + ("прошли — " + "; ".join(passed) if passed else "топливо сквиза не даёт сделки, окупающей издержки"))
    robust(x)


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
