"""
LuxAlgo «Trendlines with Breaks» + стакан Bybit: отличает ли состояние стакана и ленты в момент пробоя
настоящий пробой от ложного.

Сигналы — исходные настройки скрипта (length 14, Slope 1, Atr) на 5m-свечах фьючерсов Binance (длинная история
для разгона индикатора, цены совпадают с Bybit до долей б.п.). Признаки в момент закрытия бара сигнала —
по стакану и сделкам Bybit (реплей в research.hft.walls). d = +1 — пробой вверх, -1 — вниз.
    flow_d       перевес агрессоров за 5 мин: (купили - продали) / оборот, умноженный на d
    imb10_d/25_d перекос лимитных заявок в 10/25 б.п. от mid: (bid - ask) / (bid + ask), умноженный на d
    absorbed_against  за 15 мин поглощена стена на стороне, куда идёт пробой (айсберг против: ask для d=+1)
    consumed_with     за 15 мин агрессоры съели стену на пути пробоя (ask для d=+1) — настоящий покупатель
    pulled_support    за 15 мин сняли стену-поддержку пробоя у самой цены (bid для d=+1) — ложная поддержка
Классы (зафиксированы до запуска):
    confirmed  flow_d > 0, imb25_d > 0, absorbed_against == 0          -> торговать по пробою
    fake       absorbed_against > 0 или (flow_d < 0 и imb25_d < 0)      -> торговать против пробоя
    other      остальное
Результат — ret * d в б.п. от закрытия бара сигнала: через 15/60/240 мин и до противоположного сигнала
(выход как в скрипте). Порог полезности — круг taker 11 б.п.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data import fetch_klines
from ..lux import lux_signals, slope_series

LENGTH, MULT, METHOD = 14, 1.0, "atr"
BAR_MS = 5 * 60_000
HORIZONS = {"15m": 3, "60m": 12, "240m": 48}
FLOW_MS = 5 * 60_000
WALL_MS = 15 * 60_000
BANDS_BPS = (10, 25)
TAKER_RT_BPS = 11.0
DAY_MS = 86_400_000


def load_klines(sym: str, days: list[str], cache: Path) -> pd.DataFrame | None:
    """5m-свечи от месяца до первого дня (разгон индикатора) до месяца после последнего (горизонты выхода)."""
    cache.mkdir(parents=True, exist_ok=True)
    first, last = pd.Period(min(days), "M") - 1, pd.Period(max(days), "M") + 1
    parts = []
    for p in pd.period_range(first, last, freq="M"):
        f = fetch_klines(sym, "5m", p.strftime("%Y-%m"), cache)
        if f is not None:
            parts.append(pd.read_parquet(f, columns=["open_time", "open", "high", "low", "close"]))
    if not parts:
        return None
    return pd.concat(parts).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def _next_true(x: np.ndarray) -> np.ndarray:
    """Для каждого i — индекс ближайшего j > i, где x[j] истинно (-1, если такого нет)."""
    out = np.full(len(x), -1, dtype=np.int64)
    nxt = -1
    for i in range(len(x) - 1, -1, -1):
        out[i] = nxt
        if x[i]:
            nxt = i
    return out


def signals(k: pd.DataFrame) -> pd.DataFrame:
    h, lo, c = (k[col].to_numpy(dtype="float64") for col in ("high", "low", "close"))
    up, dn = lux_signals(h, lo, c, slope_series(k, LENGTH, MULT, METHOD), LENGTH)
    nxt_up, nxt_dn = _next_true(up), _next_true(dn)
    idx = np.flatnonzero(up ^ dn)
    d = np.where(up[idx], 1.0, -1.0)
    n = len(c)
    out = {"t_ms": k["open_time"].to_numpy(dtype="int64")[idx] + BAR_MS, "dir": d}
    for name, hb in HORIZONS.items():
        j = idx + hb
        out[f"ret_{name}"] = np.where(j < n, c[np.minimum(j, n - 1)] / c[idx] - 1, np.nan) * 1e4
    j = np.where(d > 0, nxt_dn[idx], nxt_up[idx])
    out["ret_flip"] = np.where(j >= 0, c[np.maximum(j, 0)] / c[idx] - 1, np.nan) * 1e4
    out["hold_bars"] = np.where(j >= 0, j - idx, -1)
    return pd.DataFrame(out)


def probes_by_day(sig: pd.DataFrame, days: list[str]) -> dict[str, pd.DataFrame]:
    """Сигналы внутри дня реплея; первые 15 минут дня пропускаем — окна стен и ленты там неполные."""
    res = {}
    for day in days:
        start = int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)
        m = (sig["t_ms"] >= start + WALL_MS) & (sig["t_ms"] < start + DAY_MS)
        res[day] = sig[m].reset_index(drop=True)
    return res


def _window_sum(ts: np.ndarray, cum: np.ndarray, t: np.ndarray, w: int) -> np.ndarray:
    """Сумма величины по событиям в (t - w, t] через префиксные суммы (ts отсортированы)."""
    hi = np.searchsorted(ts, t, side="right")
    lo = np.searchsorted(ts, t - w, side="right")
    c = np.concatenate([[0.0], cum])
    return c[hi] - c[lo]


def features(sig: pd.DataFrame, snaps: pd.DataFrame, ev: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    out = sig.merge(snaps, on="t_ms", how="left")
    t = out["t_ms"].to_numpy(dtype="int64")
    d = out["dir"].to_numpy(dtype="float64")

    tts = trades["timestamp"].to_numpy(dtype="float64") * 1000.0
    o = np.argsort(tts, kind="stable")
    tts = tts[o]
    sz = trades["size"].to_numpy(dtype="float64")[o]
    sgn = np.where(trades["side"].to_numpy()[o] == "Buy", 1.0, -1.0)
    net = _window_sum(tts, np.cumsum(sz * sgn), t, FLOW_MS)
    tot = _window_sum(tts, np.cumsum(sz), t, FLOW_MS)
    out["flow_d"] = np.where(tot > 0, net / np.where(tot > 0, tot, 1.0), np.nan) * d

    for b in BANDS_BPS:
        bid, ask = out[f"bid{b}"].to_numpy(), out[f"ask{b}"].to_numpy()
        s = bid + ask
        out[f"imb{b}_d"] = np.where(s > 0, (bid - ask) / np.where(s > 0, s, 1.0), np.nan) * d

    def count(kind: str, side_of: np.ndarray) -> np.ndarray:
        """Число событий kind за 15 мин на стороне side_of[i] (1 — bid, -1 — ask) для каждого сигнала."""
        res = np.zeros(len(t))
        if ev is None or not len(ev):
            return res
        for s in (1, -1):
            e = np.sort(ev[(ev["kind"] == kind) & (ev["side"] == s)]["ts"].to_numpy(dtype="float64"))
            c = _window_sum(e, np.cumsum(np.ones(len(e))), t, WALL_MS)
            res = np.where(side_of == s, c, res)
        return res

    out["absorbed_against"] = count("absorbed", -d)
    out["consumed_with"] = count("consumed", -d)
    out["pulled_support"] = count("pulled_near", d)
    return out


def classify(df: pd.DataFrame) -> pd.Series:
    conf = (df["flow_d"] > 0) & (df["imb25_d"] > 0) & (df["absorbed_against"] == 0)
    fake = (df["absorbed_against"] > 0) | ((df["flow_d"] < 0) & (df["imb25_d"] < 0))
    return pd.Series(np.select([fake, conf], ["fake", "confirmed"], "other"), index=df.index)


def _stats(x: pd.Series) -> tuple[int, float, float]:
    x = x.dropna()
    if len(x) < 2:
        return len(x), np.nan, np.nan
    return len(x), float(x.mean()), float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def report(df: pd.DataFrame) -> None:
    df = df.copy()
    df["cls"] = classify(df)
    df["period"] = df["day"].map(lambda s: "is" if s < "2024-07-01" else "val" if s < "2025-07-01" else "ho")
    rets = [f"ret_{h}" for h in HORIZONS] + ["ret_flip"]
    for r in rets:
        df[r + "_d"] = df[r] * df["dir"]
    print(f"\n==== LuxAlgo + стакан: пробоев {len(df)}, монет {df['symbol'].nunique()}, дней {df['day'].nunique()}; "
          f"средний срок до переворота {df.loc[df['hold_bars'] > 0, 'hold_bars'].mean() * 5:.0f} мин")
    print("Доходность ПО пробою (ret*d), б.п.; t — статистика. Круг taker = 11 б.п.: confirmed нужно > +11, "
          "fake нужно < -11 (тогда торговля против пробоя в плюсе)")
    rows = []
    for cls, g in [("ALL", df)] + list(df.groupby("cls")):
        r = {"class": cls, "n": len(g), "share": len(g) / len(df)}
        for col in rets:
            n, m, tt = _stats(g[col + "_d"])
            r[col[4:]], r["t_" + col[4:]] = m, tt
        rows.append(r)
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    print("\nПо периодам (60 мин и до переворота):")
    rows = []
    for (cls, per), g in df.groupby(["cls", "period"]):
        rows.append({"class": cls, "period": per, "n": len(g),
                     "60m": g["ret_60m_d"].mean(), "flip": g["ret_flip_d"].mean()})
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    print("\nПо монетам (класс: среднее ret_60m*d, n):")
    piv = df.pivot_table(index="symbol", columns="cls", values="ret_60m_d", aggfunc=["mean", "count"])
    print(piv.round(1).to_string())

    print("\nРазведка (не гипотеза): ранговая корреляция признаков с ret*d и терцили")
    feats = ["flow_d", "imb10_d", "imb25_d", "absorbed_against", "consumed_with", "pulled_support", "spread_bps"]
    rows = []
    for f in feats:
        if f not in df:
            continue
        r = {"feature": f}
        for col in ("ret_15m_d", "ret_60m_d", "ret_flip_d"):
            ok = df[[f, col]].dropna()
            r[col[4:-2]] = ok[f].rank().corr(ok[col].rank()) if len(ok) > 30 else np.nan
        rows.append(r)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    for f in ("flow_d", "imb25_d"):
        ok = df[[f, "ret_60m_d", "ret_flip_d"]].dropna()
        if len(ok) < 30:
            continue
        q = pd.qcut(ok[f], 3, labels=["низ", "середина", "верх"], duplicates="drop")
        print(f"\n{f} по терцилям:")
        print(ok.groupby(q, observed=True)[["ret_60m_d", "ret_flip_d"]].agg(["mean", "count"]).round(1).to_string())
