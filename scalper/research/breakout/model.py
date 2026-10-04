"""
Мета-разметка пробоев: градиентный бустинг решает, какие пробои брать.

Протокол (зафиксирован до запуска):
    обучение: 16 исходных монет, IS 2022-01..2024-06; порог отбора — по out-of-fold прогнозам внутри IS
              (5 блоков по времени с зазором 1 день), выбирается доля лучших прогнозов из {2%, 5%, 10%, 20%}
    проверка: 16 монет, VAL 2024-07..2025-06 и HOLDOUT 2025-07..2026-09
    экзамен:  54 новые монеты за все периоды (модель их никогда не видела)
Торговля: по одному пробою на монету одновременно, фиксированная доля капитала на сделку.

    python -m research.breakout.model --events <dir> --out <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..data import UNIVERSE
from ..universe import EXTRA
from ..wave2 import PERIODS
from .dataset import COST_BPS

FRACTION = 0.05                   # доля капитала на сделку
TOP_FRACS = (0.02, 0.05, 0.10, 0.20)
STRESS_EXTRA_BPS = 8.0            # стресс: +4 б.п. на сторону сверх базовых 6
DROP = ("symbol", "ret", "bars", "bar_idx")
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=500, subsample=0.7,
              subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, random_state=7, verbose=-1)


def load(d: Path, symbols: list[str]) -> pd.DataFrame:
    parts = [pd.read_parquet(d / f"{s}.parquet") for s in symbols if (d / f"{s}.parquet").exists()]
    return pd.concat(parts) if parts else pd.DataFrame()


def period(e: pd.DataFrame, name: str) -> pd.DataFrame:
    a, b = PERIODS[name]
    return e[(e.index >= a) & (e.index < b)]


def features(e: pd.DataFrame) -> pd.DataFrame:
    return e.drop(columns=[c for c in DROP if c in e])


def target(e: pd.DataFrame, clip: float) -> np.ndarray:
    return np.clip(e["ret"].to_numpy() * 1e4, -clip, clip)


def simulate(e: pd.DataFrame, take: np.ndarray, extra_bps: float = 0.0) -> dict:
    """Берём отмеченные события; пока по монете открыта сделка (bars баров), новые сигналы по ней пропускаем."""
    rows = []
    e = e.assign(take=take)
    for sym, g in e[e["take"]].groupby("symbol"):
        busy_until = pd.Timestamp.min.tz_localize("UTC")
        for t, r in g.iterrows():
            if t < busy_until:
                continue
            exit_t = t + pd.Timedelta(minutes=5 * int(r["bars"]))
            busy_until = exit_t
            rows.append((exit_t, r["ret"] - extra_bps * 1e-4))
    if not rows:
        return {"trades": 0, "sharpe": np.nan}
    tr = pd.DataFrame(rows, columns=["exit", "ret"]).set_index("exit").sort_index()
    daily = (tr["ret"] * FRACTION).resample("1D").sum()
    days = max((e.index.max() - e.index.min()).days, 1)
    daily = daily.reindex(pd.date_range(e.index.min().floor("1D"), e.index.max().floor("1D"), freq="1D", tz="UTC"),
                          fill_value=0.0)
    eq = daily.cumsum()
    return {"trades": len(tr), "trades_per_day": len(tr) / days, "bps_per_trade": tr["ret"].mean() * 1e4,
            "hit": float((tr["ret"] > 0).mean()), "ann_ret": daily.mean() * 365,
            "sharpe": daily.mean() / daily.std() * np.sqrt(365) if daily.std() > 0 else 0.0,
            "max_dd": float((eq - eq.cummax()).min())}


def oof_predictions(e: pd.DataFrame, clip: float, folds: int = 5) -> np.ndarray:
    t = e.index
    edges = pd.date_range(t.min(), t.max(), periods=folds + 1)
    pred = np.full(len(e), np.nan)
    X, y = features(e), target(e, clip)
    for k in range(folds):
        test = (t >= edges[k]) & (t < edges[k + 1] if k < folds - 1 else t <= edges[k + 1])
        gap = pd.Timedelta("1D")
        train = (t < edges[k] - gap) | (t > (edges[k + 1] + gap))
        m = lgb.LGBMRegressor(**PARAMS).fit(X[train], y[train])
        pred[test] = m.predict(X[test])
    return pred


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 250)
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    core = load(Path(a.events), UNIVERSE)
    new = load(Path(a.events), EXTRA)
    print(f"событий: исходные монеты {len(core)}, новые {len(new)}")

    tr = period(core, "is")
    clip = float(np.nanpercentile(np.abs(tr["ret"]) * 1e4, 99))
    print(f"обучение: {len(tr)} событий IS; средний исход без отбора {tr['ret'].mean() * 1e4:.2f} б.п., "
          f"доля плюсовых {(tr['ret'] > 0).mean():.3f}")

    oof = oof_predictions(tr, clip)
    best, best_q = -np.inf, None
    print("\nOOF внутри IS (выбор доли лучших прогнозов):")
    for q in TOP_FRACS:
        thr = np.nanquantile(oof, 1 - q)
        s = simulate(tr, np.nan_to_num(oof, nan=-1e9) >= thr)
        print(f"  топ {q:.0%}: порог {thr:.2f}, сделок/день {s['trades_per_day']:.1f}, "
              f"{s['bps_per_trade']:.2f} б.п./сделку, Sharpe {s['sharpe']:.2f}")
        if s["sharpe"] > best:
            best, best_q, best_thr = s["sharpe"], q, thr
    print(f"выбрано: топ {best_q:.0%} (порог прогноза {best_thr:.2f} б.п.)")

    model = lgb.LGBMRegressor(**PARAMS).fit(features(tr), target(tr, clip))
    imp = pd.Series(model.booster_.feature_importance("gain"), index=features(tr).columns).sort_values(ascending=False)
    print("\nважность признаков (gain), топ-15:")
    print((imp / imp.sum()).head(15).round(3).to_string())

    rows = []
    for label, e in (("core16", core), ("new54", new)):
        for per in ("is", "val", "ho"):
            x = period(e, per)
            if not len(x):
                continue
            p = model.predict(features(x))
            for name, take in (("модель", p >= best_thr), ("все пробои", np.ones(len(x), bool))):
                for cost_name, extra in (("6 б.п.", 0.0), ("10 б.п.", STRESS_EXTRA_BPS)):
                    s = simulate(x, take, extra)
                    rows.append({"universe": label, "period": per, "rule": name, "cost": cost_name, **s})
    res = pd.DataFrame(rows)
    res.to_csv(out / "breakout_results.csv", index=False)
    print("\nРЕЗУЛЬТАТЫ (IS для core16 — обучающий, остальное — вне выборки):")
    print(res.round(3).to_string(index=False))
