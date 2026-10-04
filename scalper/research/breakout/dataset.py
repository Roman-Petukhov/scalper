"""
Набор данных пробоев линий тренда для мета-разметки: событие -> признаки (известные на закрытии бара пробоя)
-> исход сделки (тейк/стоп/время, за вычетом издержек).

    python -m research.breakout.dataset --root <data> --out <dir> --symbols BTCUSDT,ETHUSDT,...
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..wave2 import Data2
from .lines import LINE_FEATS, trend_breaks, triple_barrier

SCALES = (6, 12, 24)                 # полуширина фрактала в 5m-барах: 30 мин, 1 ч, 2 ч
CONTEXT_SCALES = (72, 288)           # «старшие» линии: 6 ч и сутки — их положение относительно цены
TP_MULT, SL_MULT, MAX_HOLD, COST_BPS = 2.0, 1.0, 48, 6.0


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def _zs(x: pd.Series, w: int) -> pd.Series:
    return (x - x.rolling(w, min_periods=w // 4).mean()) / x.rolling(w, min_periods=w // 4).std()


def context(df: pd.DataFrame, btc: pd.DataFrame | None) -> pd.DataFrame:
    """Признаки рынка на закрытии каждого 5m-бара (без знака сделки; знак применяется позже)."""
    c = df["close"]
    lc = np.log(c)
    sb = lc.diff().rolling(288, min_periods=144).std()
    atr = _atr(df)
    out = pd.DataFrame(index=df.index)
    out["atr_bps"] = atr / c * 1e4
    for k in (1, 12, 48, 288):
        out[f"ret{k}_z"] = (lc - lc.shift(k)) / (sb * np.sqrt(k))
    hi, lo = df["high"].rolling(288).max(), df["low"].rolling(288).min()
    out["pos24h"] = (c - lo) / (hi - lo)
    out["vol_regime"] = np.log(lc.diff().rolling(48).std() / lc.diff().rolling(288 * 7, min_periods=288).std())
    v = np.log1p(df["volume"])
    out["volume_z"] = _zs(v, 288)
    out["volume12_z"] = _zs(np.log1p(df["volume"].rolling(12).sum()), 288)
    if df["taker_buy_volume"].notna().any():
        b, vol = df["taker_buy_volume"], df["volume"]
        out["tfi1"] = (2 * b - vol) / vol
        out["tfi3"] = (2 * b.rolling(3).sum() - vol.rolling(3).sum()) / vol.rolling(3).sum()
        out["tfi12"] = (2 * b.rolling(12).sum() - vol.rolling(12).sum()) / vol.rolling(12).sum()
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    out["bar_range_atr"] = rng / atr
    out["bar_body"] = (df["close"] - df["open"]) / rng
    out["bar_close_pos"] = (df["close"] - df["low"]) / rng
    oi = df["oi"].replace(0, np.nan) if "oi" in df else pd.Series(np.nan, index=df.index)
    for k in (12, 48):
        out[f"doi{k}_z"] = _zs(np.log(oi / oi.shift(k)), 288 * 30)
    f = df["funding"].replace(0, np.nan).ffill()
    out["funding_z"] = _zs(f, 288 * 30)
    out["funding_bps"] = f * 1e4
    out["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    out["dow"] = df.index.dayofweek
    if btc is not None:
        bl = np.log(btc["close"].reindex(df.index).ffill())
        bsb = bl.diff().rolling(288, min_periods=144).std()
        for k in (1, 12, 48):
            out[f"btc_ret{k}_z"] = (bl - bl.shift(k)) / (bsb * np.sqrt(k))
    # старшие линии: расстояние до них в ATR (NaN, если линии нет)
    a = atr.to_numpy(np.float64)
    hh, ll, cc = df["high"].to_numpy(np.float64), df["low"].to_numpy(np.float64), c.to_numpy(np.float64)
    for n in CONTEXT_SCALES:
        _, _, rl, sl = trend_breaks(hh, ll, cc, a, n)
        out[f"res{n}_dist"] = (rl - cc) / a
        out[f"sup{n}_dist"] = (cc - sl) / a
    return out


SIGNED = ("ret1_z", "ret12_z", "ret48_z", "ret288_z", "tfi1", "tfi3", "tfi12", "bar_body", "funding_z",
          "btc_ret1_z", "btc_ret12_z", "btc_ret48_z", "doi12_z", "doi48_z")


def build_symbol(root: str, sym: str, start: str | None = None) -> pd.DataFrame:
    data = Data2(Path(root), [sym, "BTCUSDT"] if sym != "BTCUSDT" else [sym])
    df = data.get(sym, "5m", "full").copy()
    btc = data.get("BTCUSDT", "5m", "full") if sym != "BTCUSDT" else df
    atr = _atr(df).to_numpy(np.float64)
    o, hh, ll, cc = (df[x].to_numpy(np.float64) for x in ("open", "high", "low", "close"))
    ctx = context(df, btc)
    parts = []
    for n in SCALES:
        ev, lf, _, _ = trend_breaks(hh, ll, cc, atr, n)
        idx = np.flatnonzero(ev != 0)
        if len(idx) == 0:
            continue
        side = ev[idx].astype(np.int64)
        ret, bars = triple_barrier(idx, side, o, hh, ll, cc, atr, TP_MULT, SL_MULT, MAX_HOLD, COST_BPS)
        e = pd.DataFrame(lf[idx], columns=list(LINE_FEATS), index=df.index[idx])
        e["scale"] = n
        e["side"] = side
        e = e.join(ctx.iloc[idx])
        for col in SIGNED:
            if col in e:
                e[col] = e[col] * side
        # для шорта «расстояние до сопротивления/поддержки» меняется местами
        for m in CONTEXT_SCALES:
            r, s = e[f"res{m}_dist"].copy(), e[f"sup{m}_dist"].copy()
            e[f"ahead{m}_dist"] = np.where(side > 0, r, s)       # уровень по ходу сделки
            e[f"behind{m}_dist"] = np.where(side > 0, s, r)      # уровень за спиной
            e = e.drop(columns=[f"res{m}_dist", f"sup{m}_dist"])
        e["pos24h"] = np.where(side > 0, e["pos24h"], 1 - e["pos24h"])
        e["ret"] = ret
        e["bars"] = bars
        e["bar_idx"] = idx
        parts.append(e)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts).sort_index()
    out.insert(0, "symbol", sym)
    f32 = [c for c in out.columns if c not in ("symbol", "ret")]
    out[f32] = out[f32].astype("float32")
    return out


def _job(args):
    root, sym, out = args
    p = Path(out) / f"{sym}.parquet"
    if p.exists():
        return sym, -1
    e = build_symbol(root, sym)
    if len(e):
        e.to_parquet(p)
    return sym, len(e)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    syms = [s for s in a.symbols.split(",") if (Path(a.root) / f"{s}-5m.parquet").exists()]
    with ProcessPoolExecutor(a.workers) as ex:
        for sym, n in ex.map(_job, [(a.root, s, a.out) for s in syms]):
            print(f"  {sym}: {n} событий", flush=True)
