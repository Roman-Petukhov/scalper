"""
Моментум — как трендследование делают фонды: не одна монета и не точка входа, а портфель.

1) Межмонетный (cross-sectional), ребалансировка раз в неделю (понедельник 00:00 UTC):
   монеты с оборотом за 30 прошлых дней >= $20M и историей >= 90 дней; сигнал — доходность за L = 7 / 14 / 28 дней,
   сырая или делённая на волатильность 30 дней (vs); лонг — верхние 10%, шорт — нижние 10%, равные веса, неделя.
   Фильтр толпы crowd: не лонговать монеты с funding за 7 дней в верхних 20% рынка, не шортить — в нижних 20%.
   Издержки 6 б.п. на сторону с оборота + funding (лонг платит положительную ставку, шорт получает).
2) По времени (time-series, CTA-стиль), ежедневно: 20 самых ликвидных монет (по обороту 30 дней) или только BTC;
   сигнал — среднее знаков доходности за 7 / 28 / 84 дня; вес = сигнал x (целевая дневная волатильность / вол. 30 д)
   / число монет, не больше 2 на монету; издержки 6 б.п. с изменения веса, funding.
Отчёт: средняя недельная (дневная) доходность, t, Sharpe годовой, макс. просадка — IS / VAL / HOLDOUT.

    python -m research.momentum collect --symbols ...   (по частям: дневные ряды монет)
    python -m research.momentum report
"""
from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .shard import all_parts, mine, part_path

PER = {"is": ("2022-01-01", "2024-07-01"), "val": ("2024-07-01", "2025-07-01"), "ho": ("2025-07-01", "2026-10-01")}
ADV_MIN = 20e6
SIDE_COST = 6e-4
LOOKBACKS = (7, 14, 28)
TOP = 0.10
TARGET_VOL_D = 0.40 / np.sqrt(365)


def daily(root: Path, sym: str) -> pd.DataFrame | None:
    p = root / f"{sym}-1h.parquet"
    if not p.exists():
        return None
    k = pd.read_parquet(p, columns=["open_time", "close", "quote_volume"])
    k.index = pd.to_datetime(k["open_time"], unit="ms", utc=True)
    d = pd.DataFrame({"close": k["close"].resample("1D").last(), "qv": k["quote_volume"].resample("1D").sum()})
    f = root / f"{sym}-funding.parquet"
    if f.exists():
        fr = pd.read_parquet(f)
        fs = pd.Series(fr["rate"].to_numpy(dtype="float64"), index=pd.to_datetime(fr["ts"], unit="ms", utc=True))
        d["fund"] = fs.resample("1D").sum().reindex(d.index).fillna(0.0)
    else:
        d["fund"] = 0.0
    d = d.dropna(subset=["close"])
    d["symbol"] = sym
    return d.rename_axis("date").reset_index()


def panel() -> dict[str, pd.DataFrame]:
    df = pd.concat([pd.read_parquet(p) for p in all_parts("mom_daily")], ignore_index=True)
    out = {c: df.pivot_table(index="date", columns="symbol", values=c, aggfunc="last").sort_index()
           for c in ("close", "qv", "fund")}
    out["adv"] = out["qv"].rolling(30, min_periods=20).mean().shift(1)
    out["age"] = out["close"].notna().cumsum()
    return out


def _ser(pairs: list[tuple[pd.Timestamp, float]]) -> pd.Series:
    return pd.Series([v for _, v in pairs], index=pd.DatetimeIndex([a for a, _ in pairs], tz="UTC"), dtype="float64")


def _stats(r: pd.Series, per_year: int) -> dict:
    r = r.dropna()
    if len(r) < 10:
        return {"n": len(r)}
    eq = (1 + r).cumprod()
    return {"n": len(r), "mean%": r.mean() * 100, "t": r.mean() / (r.std(ddof=1) / np.sqrt(len(r))),
            "sharpe": r.mean() / r.std(ddof=1) * np.sqrt(per_year), "maxDD%": (eq / eq.cummax() - 1).min() * 100}


def cross_sectional(P: dict[str, pd.DataFrame]) -> None:
    close, adv, fund, age = P["close"], P["adv"], P["fund"], P["age"]
    lr = np.log(close).diff()
    vol30 = lr.rolling(30, min_periods=20).std()
    fund7 = fund.rolling(7, min_periods=1).sum()
    mondays = close.index[close.index.dayofweek == 0]
    print("\n=== 1. Межмонетный моментум, неделя: средняя доходность недели %, t, Sharpe, макс. просадка % ===")
    rows = []
    for L, vs, crowd in itertools.product(LOOKBACKS, (False, True), (False, True)):
        sig = np.log(close / close.shift(L))
        if vs:
            sig = sig / (vol30 * np.sqrt(L))
        res = {"ls": [], "long": [], "short": [], "mkt": []}
        prev_w = pd.Series(dtype="float64")
        for a, b in zip(mondays[:-1], mondays[1:]):
            ok = (adv.loc[a] >= ADV_MIN) & (age.loc[a] >= 90) & sig.loc[a].notna() & close.loc[a].notna()
            s = sig.loc[a][ok]
            if len(s) < 30:
                continue
            n = max(int(len(s) * TOP), 3)
            longs, shorts = s.nlargest(n).index, s.nsmallest(n).index
            if crowd:
                f = fund7.loc[a][ok]
                hi, lo = f.quantile(0.8), f.quantile(0.2)
                longs = [x for x in longs if f[x] < hi]
                shorts = [x for x in shorts if f[x] > lo]
            if not len(longs) or not len(shorts):
                continue
            nxt = close.loc[b].reindex(s.index).fillna(close.loc[:b].ffill().iloc[-1].reindex(s.index))
            ret = nxt / close.loc[a][ok] - 1
            fsum = fund.loc[(fund.index > a) & (fund.index <= b)].sum().reindex(s.index).fillna(0.0)
            w = pd.concat([pd.Series(0.5 / len(longs), index=longs), pd.Series(-0.5 / len(shorts), index=shorts)])
            turn = w.subtract(prev_w, fill_value=0.0).abs().sum()
            prev_w = w
            r_l = (ret[longs] - fsum[longs]).mean()
            r_s = (-ret[shorts] + fsum[shorts]).mean()
            res["ls"].append((a, 0.5 * r_l + 0.5 * r_s - turn * SIDE_COST))
            res["long"].append((a, r_l - 2 * SIDE_COST))
            res["short"].append((a, r_s - 2 * SIDE_COST))
            res["mkt"].append((a, ret.mean()))
        for leg in ("ls", "long", "short"):
            r = _ser(res[leg])
            row = {"L": L, "vs": vs, "crowd": crowd, "портфель": leg}
            for p, (pa, pb) in PER.items():
                st = _stats(r[(r.index >= pa) & (r.index < pb)], 52)
                row.update({f"{p}_{k}": v for k, v in st.items() if k != "n"})
            rows.append(row)
    res_df = pd.DataFrame(rows)
    print(res_df.round(2).to_string(index=False))
    m = _ser(res["mkt"])
    print("  для сравнения — рынок (равный вес всех доступных монет), неделя %: " +
          ", ".join(f"{p} {m[(m.index >= a) & (m.index < b)].mean() * 100:+.2f}" for p, (a, b) in PER.items()))
    if not {"is_t", "val_mean%", "val_t"} <= set(res_df.columns):
        print("  мало монет для отбора (пробный запуск?)")
        return
    gate = res_df[(res_df["is_t"] > 2) & (res_df["val_mean%"] > 0) & (res_df["val_t"] > 1.5)]
    print("  ВОРОТА (IS t > 2, VAL > 0 и t > 1.5): " + ("никто" if not len(gate) else
          "; ".join(f"L={g.L} vs={g.vs} crowd={g.crowd} {g['портфель']}: HO {g['ho_mean%']:+.2f}%/нед (t {g['ho_t']:.1f})"
                    for _, g in gate.iterrows())))


def time_series(P: dict[str, pd.DataFrame]) -> None:
    close, adv, fund = P["close"], P["adv"], P["fund"]
    lr = np.log(close).diff()
    vol30 = lr.rolling(30, min_periods=20).std().shift(1)
    sig = sum(np.sign(np.log(close / close.shift(L))) for L in (7, 28, 84)) / 3.0
    ret_next = close.shift(-1) / close - 1
    fund_next = fund.shift(-1).fillna(0.0)
    print("\n=== 2. Трендследование по времени (CTA), день: средняя доходность дня %, t, Sharpe, макс. просадка % ===")
    rows = []
    for name in ("top20", "BTC"):
        if name == "BTC":
            uni = pd.DataFrame(False, index=close.index, columns=close.columns)
            if "BTCUSDT" in uni:
                uni["BTCUSDT"] = True
        else:
            rank = adv.rank(axis=1, ascending=False)
            uni = rank <= 20
        n = uni.sum(axis=1).replace(0, np.nan)
        w = (sig * (TARGET_VOL_D / vol30)).clip(-2, 2).where(uni, 0.0).div(n, axis=0).fillna(0.0)
        pnl = (w * (ret_next.fillna(0.0) - fund_next)).sum(axis=1)
        cost = w.diff().abs().sum(axis=1) * SIDE_COST
        r = (pnl - cost).iloc[90:-1]
        row = {"вселенная": name}
        for p, (pa, pb) in PER.items():
            st = _stats(r[(r.index >= pa) & (r.index < pb)], 365)
            row.update({f"{p}_{k}": v for k, v in st.items() if k not in ("n",)})
        rows.append(row)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 40)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        root = Path(a.root).expanduser()
        parts = [d for s in mine([x for x in a.symbols.split(",") if x]) if (d := daily(root, s)) is not None]
        if parts:
            pd.concat(parts, ignore_index=True).to_parquet(part_path("mom_daily"), index=False)
    else:
        P = panel()
        print(f"===== MOMENTUM: монет {P['close'].shape[1]}, дней {P['close'].shape[0]} =====")
        cross_sectional(P)
        time_series(P)
