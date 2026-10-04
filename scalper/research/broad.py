"""
Проверенные стратегии (oi_liq, fe_q99, параметры НЕ меняются) на всех USDT-перпетуалах Binance.

Без ошибки выжившего: список берётся из архива, где лежат и снятые с торгов монеты.
Без заглядывания в будущее: монета торгуется в момент t, только если её средний дневной оборот за 30 дней ДО t
не ниже ADV_MIN. Данные OI качаются только для монет, которые хоть раз проходили этот порог — остальные стратегия
всё равно никогда бы не торговала.

    python -m research.broad --root <data> --out <dir> --stage list|klines|metrics|eval
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .data import UNIVERSE
from .engine import BAR_MINUTES, metrics, run
from .universe import CANDIDATES, EXTRA
from .wave2 import PERIODS, Data2, _cut

ADV_MIN = 20e6                    # $ в день, средний за 30 дней, только прошлое
TIERS = ((20e6, 50e6), (50e6, 200e6), (200e6, np.inf))
POS_FRACTION = 0.02               # доля капитала на одну позицию (для доходности; Sharpe от неё не зависит)
LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=/&prefix=data/futures/um/monthly/klines/"


def list_symbols() -> list[str]:
    syms, marker = [], ""
    while True:
        url = LIST_URL + (f"&marker={marker}" if marker else "")
        x = urllib.request.urlopen(url, timeout=60).read().decode()
        syms += re.findall(r"<Prefix>data/futures/um/monthly/klines/([^/]+)/</Prefix>", x)
        m = re.search(r"<NextMarker>([^<]+)</NextMarker>", x)
        if not m:
            break
        marker = m.group(1)
    return sorted(s for s in syms if s.endswith("USDT") and "_" not in s)


def adv30(root: Path, sym: str) -> pd.Series:
    """Средний дневной оборот за 30 прошлых дней (сдвиг на день: сегодняшний оборот ещё не известен)."""
    k = pd.read_parquet(root / f"{sym}-1h.parquet", columns=["open_time", "quote_volume"])
    t = pd.to_datetime(k["open_time"], unit="ms", utc=True)
    d = pd.Series(k["quote_volume"].to_numpy(), index=t).resample("1D").sum()
    return d.rolling(30, min_periods=20).mean().shift(1)


def qualified(root: Path, syms: list[str]) -> list[str]:
    out = []
    for s in syms:
        p = root / f"{s}-1h.parquet"
        if p.exists() and (adv30(root, s) >= ADV_MIN).any():
            out.append(s)
    return out


def symbol_runs(data: Data2, root: Path, sym: str, cost: float) -> dict:
    df = data.get(sym, "1h", "full")
    allowed = (adv30(root, sym).reindex(df.index, method="ffill") >= ADV_MIN).to_numpy()
    liq = adv30(root, sym).reindex(df.index, method="ffill").to_numpy()
    res = {}
    for name, (fn, p) in CANDIDATES.items():
        if name == "oi_liq" and df["oi"].notna().sum() < 24 * 60:
            continue
        pos, ex = fn(df, **p)
        pos = np.where(allowed, np.asarray(pos, float), 0.0)
        r = run(df, pos, cost, BAR_MINUTES["1h"], None)
        res[name] = (r.pnl, r.pos, pd.Series(liq, index=df.index))
    return res


def group_of(sym: str) -> str:
    return "core16" if sym in UNIVERSE else "ext54" if sym in EXTRA else "fresh"


def evaluate(root: Path, syms: list[str], out: Path) -> None:
    data = Data2(root, syms)
    rows, tier_rows = [], []
    for cost in (6.0, 10.0, 15.0):
        pnl = {g: {k: [] for k in CANDIDATES} for g in ("core16", "ext54", "fresh")}
        pos_all = {k: [] for k in CANDIDATES}
        per_sym = []
        tier_pnl = {k: {i: [] for i in range(len(TIERS))} for k in CANDIDATES}
        for s in syms:
            try:
                rr = symbol_runs(data, root, s, cost)
            except Exception as e:                       # битые/пустые ряды не должны валить весь прогон
                print(f"  {s}: пропуск ({e})", flush=True)
                continue
            g = group_of(s)
            for k, (pn, ps, liq) in rr.items():
                pnl[g][k].append(pn.rename(s))
                pos_all[k].append(ps.rename(s))
                for i, (lo, hi) in enumerate(TIERS):
                    m = (liq >= lo) & (liq < hi)
                    tier_pnl[k][i].append(pn.where(m, 0.0).rename(s))
                if cost == 6.0:
                    prev = ps.shift(1).fillna(0.0)
                    per_sym.append({"symbol": s, "group": g, "strategy": k,
                                    "trades": int(((ps != 0) & (ps != prev)).sum()),
                                    "total_ret": float(pn.sum()), "sharpe": metrics(pn)["sharpe"]})
        if cost == 6.0:
            ps_df = pd.DataFrame(per_sym)
            ps_df.to_csv(out / "broad_per_symbol.csv", index=False)
            act = ps_df[ps_df["trades"] > 0]
            print("\nДоля монет с положительным итогом (только монеты со сделками):")
            print(act.groupby(["strategy", "group"]).apply(
                lambda x: pd.Series({"монет": len(x), "в плюсе": float((x["total_ret"] > 0).mean())}),
                include_groups=False).round(2).to_string())
            for k in CANDIDATES:
                P = pd.concat(pos_all[k], axis=1).fillna(0.0)
                ent = ((P != 0) & (P != P.shift(1).fillna(0.0))).sum(axis=1).resample("1D").sum()
                conc = (P != 0).sum(axis=1)
                print(f"\n{k}: сделок в день по годам: "
                      + ", ".join(f"{y}: {v:.1f}" for y, v in ent.groupby(ent.index.year).mean().items())
                      + f"; максимум одновременных позиций {int(conc.max())}")
        for g in ("core16", "ext54", "fresh", "all"):
            for k in CANDIDATES:
                parts = sum((pnl[x][k] for x in ("core16", "ext54", "fresh")), []) if g == "all" else pnl[g][k]
                if not parts:
                    continue
                port = pd.concat(parts, axis=1).fillna(0.0).sum(axis=1) * POS_FRACTION
                for per in ("is", "val", "ho"):
                    x = _cut(port, per)
                    m = metrics(x)
                    rows.append({"cost": cost, "group": g, "strategy": k, "period": per, "coins": len(parts),
                                 "sharpe": m["sharpe"], "ann_ret": m["ann_ret"], "max_dd": m["max_dd"],
                                 "pos_months": float((x.resample("ME").sum() > 0).mean())})
        if cost == 6.0:
            for k in CANDIDATES:
                for i, (lo, hi) in enumerate(TIERS):
                    port = pd.concat(tier_pnl[k][i], axis=1).fillna(0.0).sum(axis=1)
                    for per in ("val", "ho"):
                        tier_rows.append({"strategy": k, "tier": f"${lo / 1e6:.0f}M-{hi / 1e6:.0f}M",
                                          "period": per, "sharpe": metrics(_cut(port, per))["sharpe"]})
    res = pd.DataFrame(rows)
    res.to_csv(out / "broad_results.csv", index=False)
    print("\nSharpe по группам монет (fresh — не входили ни в подбор, ни в прошлые проверки):")
    print(res.pivot_table(index=["strategy", "group", "cost"], columns="period", values="sharpe").round(2).to_string())
    print("\nДоходность в год при 2% капитала на позицию (6 б.п.):")
    r6 = res[res["cost"] == 6.0]
    print(r6.pivot_table(index=["strategy", "group"], columns="period", values=["ann_ret", "max_dd"]).round(3).to_string())
    print("\nSharpe по уровню ликвидности монеты (6 б.п.):")
    print(pd.DataFrame(tier_rows).pivot_table(index=["strategy", "tier"], columns="period", values="sharpe")
          .round(2).to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 250)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", required=True, choices=["list", "klines", "metrics", "eval"])
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    D.set_host("s3")
    if a.stage == "list":
        syms = list_symbols()
        (out / "symbols_all.json").write_text(json.dumps(syms))
        print(f"USDT-перпетуалов в архиве: {len(syms)}")
    elif a.stage == "klines":
        syms = json.loads((out / "symbols_all.json").read_text())
        D.build(syms, "1h", "2022-01", "2026-09", root, a.workers, metrics=False)
        q = qualified(root, syms)
        (out / "symbols_qualified.json").write_text(json.dumps(q))
        print(f"монет, хоть раз проходивших порог ${ADV_MIN / 1e6:.0f}M/день: {len(q)} из {len(syms)}")
    elif a.stage == "metrics":
        q = json.loads((out / "symbols_qualified.json").read_text())
        D.METRICS_FREQ = "1h"
        D.build(q, "1h", "2022-01", "2026-09", root, a.workers, metrics=True)
    else:
        q = json.loads((out / "symbols_qualified.json").read_text())
        evaluate(root, q, out)
