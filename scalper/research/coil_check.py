"""
Проверка устойчивости coil (research.trend2) перед тем, как брать его в бота.

coil: реализованная волатильность за окно w (4h-бары) в нижних q_rv своих 90 дней, рост OI за то же окно в верхних
(1 - q_oi) -> на следующем баре вход на первом закрытии за диапазоном окна; толпа (сумма funding за окно):
    against  вверх только при funding < 0, вниз только при funding > 0 (исходная гипотеза)
    none     без условия на funding
    with     наоборот — по толпе (плацебо: если тоже прибыльно, дело не в «топливе» толпы)
Каждая сторона считается отдельной машиной (long / short), итог «both» — их объединение.

Что печатается для каждого варианта (выход tp5, стоп 1.5/2.5 ATR, издержки 2 x 6 б.п. и x2):
    n, средний R, t по сделкам и t, кластеризованный по дням (сделки одного дня — одна ставка), число разных дней,
    средний R без 5 лучших дней, доля R от 10 лучших дней; по годам; монеты вне подбора (ext54 + fresh).

    python -m research.coil_check --root <binance 1h data with metrics> --symbols ...
"""
from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30, group_of
from .trend import COST, EXITS, MAX_HOLD, PER, trade_machine
from .trend2 import D3, _pct, bars4h
from .wave2 import Data2

BASE = (D3, 0.2, 0.8, "against")
VARIANTS = sorted({BASE, *[(D3, r, o, "against") for r in (0.15, 0.2, 0.25) for o in (0.75, 0.8, 0.85)],
                   (12, 0.2, 0.8, "against"), (30, 0.2, 0.8, "against"),
                   (D3, 0.2, 0.8, "none"), (D3, 0.2, 0.8, "with")}, key=str)
STOPS = (1.5, 2.5)
COSTS = {"x1": COST, "x2": 2 * COST}
SIDES = ("long", "short")


def coil_signals(d: pd.DataFrame, w: int, q_rv: float, q_oi: float, crowd: str) -> tuple[np.ndarray, np.ndarray]:
    c = d["close"]
    oi = np.log(d["oi"].replace(0, np.nan))
    rv = np.log(c).diff().rolling(w).std()
    coiled = (_pct(rv) < q_rv) & (_pct(oi.diff(w)) > q_oi)
    hi, lo = d["high"].shift(1).rolling(w).max(), d["low"].shift(1).rolling(w).min()
    armed = coiled.shift(1).fillna(False).astype(bool)
    f = d["funding"].rolling(w).sum()
    if crowd == "against":
        f_up, f_dn = f < 0, f > 0
    elif crowd == "with":
        f_up, f_dn = f > 0, f < 0
    else:
        f_up = f_dn = pd.Series(True, index=d.index)
    up = armed & (c > hi) & f_up
    dn = armed & (c < lo) & f_dn
    return up.fillna(False).to_numpy(bool), dn.fillna(False).to_numpy(bool)


def run(root: Path, syms: list[str]) -> pd.DataFrame:
    data = Data2(root, syms)
    keys = list(itertools.product(VARIANTS, SIDES, STOPS, COSTS))
    kid = {k: i for i, k in enumerate(keys)}
    parts, names = [], []
    for s in syms:
        try:
            h = data.get(s, "1h", "full")
        except Exception as e:
            print(f"  {s}: пропуск ({e})", flush=True)
            continue
        if h["oi"].notna().sum() < 24 * 60:
            continue
        sid = len(names)
        names.append(s)
        d = bars4h(h)
        allowed = (adv30(root, s).reindex(d.index, method="ffill") >= ADV_MIN).to_numpy()
        a = {c: d[c].to_numpy(dtype="float64") for c in ("open", "high", "low", "close", "atr", "funding")}
        ts = d.index.as_unit("ns").asi8
        none = np.zeros(len(d), np.bool_)
        for v in VARIANTS:
            up, dn = coil_signals(d, *v)
            for side, sk, ck in itertools.product(SIDES, STOPS, COSTS):
                L, S = (up & allowed, none) if side == "long" else (none, dn & allowed)
                i, r, _ = trade_machine(a["open"], a["high"], a["low"], a["close"], a["atr"], a["funding"], L, S, sk,
                                        EXITS.index("tp5"), MAX_HOLD, COSTS[ck])
                if len(i):
                    parts.append((kid[(v, side, sk, ck)], sid, ts[i], r.astype(np.float32)))
    out = pd.DataFrame({"k": np.concatenate([np.full(len(p[2]), p[0], np.int16) for p in parts]),
                        "sid": np.concatenate([np.full(len(p[2]), p[1], np.int16) for p in parts]),
                        "t": pd.to_datetime(np.concatenate([p[2] for p in parts]), utc=True),
                        "R": np.concatenate([p[3] for p in parts])})
    meta = pd.DataFrame([(str(v), sd, sk, ck) for v, sd, sk, ck in keys], columns=["variant", "side", "stop", "cost"])
    out = out.join(meta, on="k")
    nm = np.array(names)
    out["symbol"] = nm[out["sid"].to_numpy()]
    out["group"] = [group_of(x) for x in out["symbol"]]
    return out


def stats(x: pd.DataFrame) -> dict:
    if len(x) < 3:
        return {"n": len(x)}
    r = x["R"].to_numpy(dtype="float64")
    day = x["t"].dt.floor("D")
    m = r.mean()
    g = pd.Series(r - m).groupby(day.to_numpy()).sum()               # кластерная дисперсия суммы по дням
    t_cl = r.sum() / np.sqrt((g ** 2).sum()) if (g ** 2).sum() > 0 else np.nan
    by_day = pd.Series(r).groupby(day.to_numpy()).sum().sort_values(ascending=False)
    rest = x[~day.isin(by_day.index[:5])]["R"]
    return {"n": len(r), "days": len(by_day), "win": float((r > 0).mean()), "avgR": m,
            "t": m / (r.std(ddof=1) / np.sqrt(len(r))), "t_day": t_cl,
            "avgR_no5d": rest.mean() if len(rest) else np.nan,
            "top10d_share": by_day.iloc[:10].sum() / r.sum() if r.sum() > 0 else np.nan}


def per_period(x: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p, (a, b) in PER.items():
        rows.append({"period": p, **stats(x[(x.t >= a) & (x.t < b)])})
    return pd.DataFrame(rows).set_index("period")


def report(tr: pd.DataFrame) -> None:
    base = str(BASE)
    print("\n=== 1. Варианты coil, both (long + short), стоп 1.5 ATR, tp5, издержки x1: средний R по периодам "
          "(t по дням) ===")
    rows = []
    for v, g in tr[(tr.stop == 1.5) & (tr.cost == "x1")].groupby("variant"):
        pp = per_period(g)
        rows.append({"variant (w, q_rv, q_oi, crowd)": v,
                     **{f"{p}": f"{pp.loc[p, 'avgR']:+.3f} ({pp.loc[p, 't_day']:.1f}) n={int(pp.loc[p, 'n'])}"
                        for p in PER if "avgR" in pp.columns and not np.isnan(pp.loc[p, "avgR"])}})
    print(pd.DataFrame(rows).to_string(index=False))

    b = tr[tr.variant == base]
    for sk, ck in itertools.product(STOPS, COSTS):
        for side in ("both", *SIDES):
            g = b[(b.stop == sk) & (b.cost == ck) & ((b.side == side) | (side == "both"))]
            print(f"\n=== 2. Базовый coil {base}: стоп {sk} ATR, издержки {ck}, сторона {side} ===")
            print(per_period(g).round(3).to_string())
            if sk == 1.5 and ck == "x1" and side == "both":
                print("  монеты вне подбора (ext54 + fresh):")
                print(per_period(g[g.group.isin(["ext54", "fresh"])]).round(3).to_string())
                y = g.assign(year=g.t.dt.year).groupby("year")["R"].agg(["size", "mean", "sum"])
                q = g.assign(q=g.t.dt.tz_localize(None).dt.to_period("Q").astype(str)).groupby("q")["R"].agg(["size", "mean"])
                print("  по годам (n, средний R, сумма R):\n" + y.round(3).to_string())
                print("  кварталов в плюсе: "
                      f"{int((q['mean'] > 0).sum())} из {len(q)}; худший квартал {q['mean'].min():+.3f} R")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 260)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--symbols", required=True)
    a = ap.parse_args()
    tr = run(Path(a.root), a.symbols.split(","))
    print(f"===== COIL CHECK: сделок {len(tr)}, монет {tr['symbol'].nunique()}, вариантов {len(VARIANTS)} =====")
    report(tr)
