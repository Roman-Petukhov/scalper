"""
«Ловля прострелов» на неликвидных альтах (1m, Binance USDT-перпетуалы, 60 случайных монет с оборотом $5–150 млн).

Каждую минуту, пока позиции нет, стоят лимитки на расстоянии D = m * σ_1h от close прошлой минуты
(σ_1h — std 1m-доходностей за сутки * sqrt(60)): покупка ниже, продажа выше. Исполнение — только если тень прошла
сквозь уровень. После входа: тейк на доле f отката к цене до прострела (лимит), стоп на расстоянии s*D от входа
(рыночный; s = нет / 1), иначе выход по close через T минут. В минуте входа тейк не засчитывается, стоп —
засчитывается (худший порядок); если минута задела обе лимитки — исполняется сторона, против которой закрылась
минута. Комиссии: maker 2 б.п. (вход, тейк), taker 5.5 б.п. (стоп, таймер). Одна позиция на монету.
Протокол: IS 2023-01..2024-06 — отбор по t-статистике, VAL 2024-07..2025-06 — ворота, HOLDOUT 2025-07..2026-09.

    python -m research.spikes --root <1m data> --symbols ...
"""
from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

MS = (2.0, 3.0, 4.0, 6.0)
FS = (0.3, 0.5, 1.0)
TS = (5, 15, 60)
STOPS = (np.inf, 1.0)
MAKER, TAKER = 2e-4, 5.5e-4
PER = {"is": ("2023-01-01", "2024-07-01"), "val": ("2024-07-01", "2025-07-01"), "ho": ("2025-07-01", "2026-10-01")}
MIN_IS = 500


@njit(cache=True)
def spike_machine(o, h, lo, c, sig, m, f, T, stop_s, maker, taker):
    """Возвращает индексы минут входа и чистые доходности сделок (доли)."""
    n = len(c)
    idx = np.empty(n, np.int64)
    ret = np.empty(n)
    k = 0
    side = 0
    entry = tp = sl = 0.0
    t_in = 0
    for t in range(1, n):
        if side == 0:
            s_ = sig[t - 1]
            if not (s_ > 0):
                continue
            d = m * s_
            ref = c[t - 1]
            buy_px, sell_px = ref * (1 - d), ref * (1 + d)
            hb, hs = lo[t] < buy_px, h[t] > sell_px
            if hb and hs:
                if c[t] - lo[t] < h[t] - c[t]:
                    hs = False
                else:
                    hb = False
            if not (hb or hs):
                continue
            side = 1 if hb else -1
            entry = buy_px if hb else sell_px
            tp = entry + f * (ref - entry)
            sl = entry * (1 - side * stop_s * d)
            t_in = t
            if (side > 0 and lo[t] <= sl) or (side < 0 and h[t] >= sl):     # стоп в минуте входа
                idx[k], ret[k] = t, side * (sl / entry - 1) - maker - taker
                k += 1
                side = 0
            continue
        xp, fee = np.nan, 0.0
        if (side > 0 and lo[t] <= sl) or (side < 0 and h[t] >= sl):
            xp = min(sl, o[t]) if side > 0 else max(sl, o[t])
            fee = taker
        elif (side > 0 and h[t] > tp) or (side < 0 and lo[t] < tp):
            xp, fee = tp, maker
        elif t - t_in >= T:
            xp, fee = c[t], taker
        if not np.isnan(xp):
            idx[k], ret[k] = t_in, side * (xp / entry - 1) - maker - fee
            k += 1
            side = 0
    return idx[:k], ret[:k]


def load(root: Path, sym: str) -> pd.DataFrame | None:
    p = root / f"{sym}-1m.parquet"
    if not p.exists():
        return None
    k = pd.read_parquet(p, columns=["open_time", "open", "high", "low", "close"])
    k.index = pd.to_datetime(k["open_time"], unit="ms", utc=True)
    k = k[~k.index.duplicated()].sort_index()
    lr = np.log(k["close"]).diff()
    k["sig"] = lr.rolling(1440, min_periods=720).std() * np.sqrt(60)
    return k


def trades(frames: dict, cfg: dict) -> pd.DataFrame:
    out = []
    for s, k in frames.items():
        i, r = spike_machine(k["open"].to_numpy(), k["high"].to_numpy(), k["low"].to_numpy(), k["close"].to_numpy(),
                             k["sig"].to_numpy(), cfg["m"], cfg["f"], cfg["T"], cfg["stop"], MAKER, TAKER)
        out.append(pd.DataFrame({"symbol": s, "t": k.index[i], "r": r}))
    return pd.concat(out, ignore_index=True)


def stats(tr: pd.DataFrame, n_coins: int) -> dict:
    res = {}
    for p, (a, b) in PER.items():
        x = tr[(tr.t >= a) & (tr.t < b)]["r"]
        days = (pd.Timestamp(b) - pd.Timestamp(a)).days
        res[f"{p}_n"] = len(x)
        res[f"{p}_per_coin_day"] = len(x) / days / n_coins
        res[f"{p}_bps"] = x.mean() * 1e4 if len(x) else np.nan
        res[f"{p}_win"] = float((x > 0).mean()) if len(x) else np.nan
        res[f"{p}_t"] = x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 30 else np.nan
    return res


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 260)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--symbols", required=True)
    a = ap.parse_args()
    frames = {s: f for s in a.symbols.split(",") if (f := load(Path(a.root), s)) is not None}
    print(f"===== SPIKES: ловля прострелов, {len(frames)} монет, 1m =====")
    rows = []
    for m, f, T, st in itertools.product(MS, FS, TS, STOPS):
        cfg = {"m": m, "f": f, "T": T, "stop": st}
        rows.append({**cfg, **stats(trades(frames, cfg), len(frames))})
    res = pd.DataFrame(rows)
    res["stop"] = res["stop"].map(lambda x: "none" if not np.isfinite(x) else f"{x:g}D")
    print(f"конфигураций {len(res)}; б.п. на сделку IS: медиана {res['is_bps'].median():.1f}, "
          f"доля > 0: {(res['is_bps'] > 0).mean():.0%}")
    for col in ("m", "f", "T", "stop"):
        print(f"  {col}: " + ", ".join(f"{k}={v:+.1f}" for k, v in res.groupby(col)["is_bps"].median().items()))
    cols = ["m", "f", "T", "stop"] + [f"{p}_{x}" for p in PER for x in ("per_coin_day", "bps", "win", "t")]
    top = res[res["is_n"] >= MIN_IS].sort_values("is_t", ascending=False).head(10)
    print(f"\nТОП-10 по t-статистике IS (не меньше {MIN_IS} сделок):")
    print(top[cols].round(2).to_string(index=False))
    passed = top[(top["val_bps"] > 0) & (top["val_t"] > 1.5)]
    print("\nВОРОТА VAL (б.п. > 0 и t > 1.5) -> HOLDOUT:")
    print(passed[cols].round(2).to_string(index=False) if len(passed) else "  никто не прошёл")
