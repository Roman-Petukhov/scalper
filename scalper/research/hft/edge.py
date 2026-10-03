"""
Замер перевеса HFT-сигналов: сколько базисных пунктов движения средней цены (mid) приходится на крайние
значения сигнала на горизонтах 1с…5мин, по сравнению с издержками.

Протокол: пороги крайних квантилей считаются ТОЛЬКО на днях периода IS и затем применяются к VAL/HOLDOUT.
Знак сигнала тоже фиксируется по IS (если на IS высокий сигнал предсказывает падение — торгуем против).

    python -m research.hft.edge --root <папка с SYMBOL-YYYY-MM-DD.parquet> --out <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (1, 5, 30, 60, 300)
QUANTS = (0.01, 0.001)
PERIOD_BOUNDS = (("is", "2022-01-01", "2024-07-01"), ("val", "2024-07-01", "2025-07-01"),
                 ("ho", "2025-07-01", "2026-10-01"))
COST_LINES = {"maker+maker": 4.0, "maker+taker": 7.5, "taker+taker": 11.0}


def period_of(day: str) -> str:
    for name, a, b in PERIOD_BOUNDS:
        if a <= day < b:
            return name
    return "other"


def _imb(a, b):
    s = a + b
    return np.where(s > 0, (a - b) / s, np.nan)


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Признаки на конец секунды t (известны в момент t) и будущие доходности mid в б.п."""
    mid = ((df["bid"] + df["ask"]) / 2).astype("float64")
    lm = np.log(mid)
    out = pd.DataFrame(index=df.index)
    out["spread_bps"] = (df["ask"] - df["bid"]) / mid * 1e4
    # стакан
    out["obi1"] = _imb(df["bid_sz1"], df["ask_sz1"])
    out["obi5"] = _imb(df["bid_dep5"], df["ask_dep5"])
    out["obi20"] = _imb(df["bid_dep20"], df["ask_dep20"])
    out["obi_band10"] = _imb(df["bid_band10"], df["ask_band10"])
    out["obi_band50"] = _imb(df["bid_band50"], df["ask_band50"])
    micro = (df["bid"] * df["ask_sz1"] + df["ask"] * df["bid_sz1"]) / (df["bid_sz1"] + df["ask_sz1"])
    out["micro_dev"] = (micro - mid) / mid * 1e4
    # поток агрессоров Bybit
    for k in (1, 5, 30):
        b = df["buy_vol"].rolling(k, min_periods=1).sum()
        s = df["sell_vol"].rolling(k, min_periods=1).sum()
        out[f"tfi{k}"] = _imb(b, s)
    vol10 = (df["buy_vol"] + df["sell_vol"]).rolling(10, min_periods=1).sum()
    burst = np.log1p(vol10) - np.log1p(vol10).rolling(3600, min_periods=600).mean()
    out["burst_signed"] = burst / np.log1p(vol10).rolling(3600, min_periods=600).std() * \
        np.sign(df["buy_vol"].rolling(10).sum() - df["sell_vol"].rolling(10).sum())
    # прошлое движение (импульс/откат)
    for k in (1, 5, 30, 300):
        out[f"ret{k}"] = (lm - lm.shift(k)) * 1e4
    # Binance: опережение и поток
    if "bn_px" in df:
        lb = np.log(df["bn_px"].astype("float64"))
        for k in (1, 5):
            out[f"bn_lead{k}"] = ((lb - lb.shift(k)) - (lm - lm.shift(k))) * 1e4
        gap = (lb - lm) * 1e4
        out["bn_gap"] = gap - gap.rolling(300, min_periods=60).mean()
        for k in (1, 5):
            out[f"bn_tfi{k}"] = _imb(df["bn_buy"].rolling(k, min_periods=1).sum(),
                                     df["bn_sell"].rolling(k, min_periods=1).sum())
    for h in HORIZONS:
        out[f"fwd{h}"] = (lm.shift(-h) - lm) * 1e4
    return out


FEATS = ["obi1", "obi5", "obi20", "obi_band10", "obi_band50", "micro_dev", "tfi1", "tfi5", "tfi30", "burst_signed",
         "ret1", "ret5", "ret30", "ret300", "bn_lead1", "bn_lead5", "bn_gap", "bn_tfi1", "bn_tfi5"]


def load(root: Path) -> dict[str, pd.DataFrame]:
    by_sym: dict[str, list[pd.DataFrame]] = {}
    for p in sorted(root.glob("*USDT-*.parquet")):
        if p.name.startswith("ev-"):
            continue
        sym, day = p.stem.split("-", 1)
        f = features(pd.read_parquet(p))
        f["day"], f["period"] = day, period_of(day)
        by_sym.setdefault(sym, []).append(f)
    return {s: pd.concat(v) for s, v in by_sym.items()}


def measure(f: pd.DataFrame, sym: str) -> list[dict]:
    rows = []
    is_ = f[f["period"] == "is"]
    n_days = f.groupby("period")["day"].nunique()
    for feat in FEATS:
        if feat not in f or is_[feat].notna().sum() < 1000:
            continue
        for q in QUANTS:
            hi = is_[feat].quantile(1 - q)
            lo = is_[feat].quantile(q)
            if not np.isfinite(hi) or not np.isfinite(lo) or hi <= lo:
                continue
            # знак по IS на горизонте 30с: торгуем в сторону, где на IS был перевес
            m_hi = is_.loc[is_[feat] >= hi, "fwd30"].mean()
            m_lo = is_.loc[is_[feat] <= lo, "fwd30"].mean()
            sign = 1.0 if (m_hi - m_lo) >= 0 else -1.0
            for per, g in f.groupby("period"):
                sig = np.where(g[feat] >= hi, sign, np.where(g[feat] <= lo, -sign, 0.0))
                mask = sig != 0
                # события не чаще раза в 5 секунд, чтобы не считать одно движение много раз
                ev = mask & ~pd.Series(mask, index=g.index).rolling(5, min_periods=1).max().shift(1).fillna(0).astype(bool).to_numpy()
                r = {"symbol": sym, "feature": feat, "q": q, "period": per, "sign": sign,
                     "events_per_day": ev.sum() / max(n_days.get(per, 1), 1),
                     "spread_bps": g["spread_bps"].median()}
                for h in HORIZONS:
                    x = (g[f"fwd{h}"].to_numpy() * sig)[ev]
                    x = x[np.isfinite(x)]
                    r[f"bps{h}"] = x.mean() if len(x) else np.nan
                    r[f"t{h}"] = x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 30 else np.nan
                rows.append(r)
    return rows


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load(Path(a.root))
    rows = []
    for sym, f in data.items():
        print(f"{sym}: {f['day'].nunique()} дней, медианный спред {f['spread_bps'].median():.2f} б.п.", flush=True)
        rows += measure(f, sym)
    res = pd.DataFrame(rows)
    res.to_csv(out / "hft_edge.csv", index=False)
    cols = ["symbol", "feature", "q", "period", "events_per_day"] + [f"bps{h}" for h in HORIZONS] + ["t30", "t300"]
    best = res.assign(best=res[[f"bps{h}" for h in HORIZONS]].max(axis=1))
    print("\nГрубый перевес (б.п., до издержек) на крайних квантилях; издержки круга: "
          + ", ".join(f"{k} {v:g}" for k, v in COST_LINES.items()))
    for per in ("is", "val", "ho"):
        t = best[best["period"] == per].sort_values("best", ascending=False).head(25)
        print(f"\n--- {per}: топ-25 по максимуму по горизонтам ---")
        print(t[cols].round(2).to_string(index=False))
