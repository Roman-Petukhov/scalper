"""
LuxAlgo «Trendlines with Breaks» (length 14, Slope 1, Atr — как на графике) с выходом R:R 1:5 и входом на ретесте.

Сигнал — собственная реализация логики скрипта без перерисовки: линия на баре t равна upper - slope*length
(именно с ней скрипт сравнивает close; нарисованная на графике линия при включённом Backpainting сдвинута
на length баров назад и выглядит «удобнее», чем была в момент сигнала).

Сетка (зафиксирована до запуска):
    tf       15m (70 монет: core16 + ext54) и 1h (725 монет)
    length   14, 28
    entry    market — по close бара пробоя (тейкер); retest — лимит по цене пробитой линии (линия продолжается
             со своим наклоном), ждём до 8 баров, исполнение только если цена прошла сквозь уровень (мейкер)
    stop     atr1 — 1 ATR(14) от входа; swing — за экстремум последних length баров до пробоя (0.5–3 ATR)
    tp       3R / 5R (лимит, мейкер); иначе выход по close через 4 дня
    filter   none / trend4h (EMA50 на закрытых 4h-барах растёт для лонга, падает для шорта) /
             vol (объём бара пробоя > 1.5 x среднего за 20 баров)
Одна позиция на монету. Комиссии Bybit: тейкер 5.5 б.п., мейкер 2 б.п.; результат — в R после комиссий.
Монеты: оборот за 30 прошлых дней >= $20M. Протокол IS -> VAL -> HOLDOUT, t — кластеризованный по дням.
Плюс разбор примера с графика: SOLUSDT 15m, 29.09–04.10.2026, базовая настройка.

    python -m research.lux_rr --root <binance data> --symbols15 ... --symbols1h ...
"""
from __future__ import annotations

import argparse
import io
import itertools
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from . import data as D
from .broad import ADV_MIN, adv30, group_of
from .lux import _pivot, slope_series

PER = {"is": ("2022-01-01", "2024-07-01"), "val": ("2024-07-01", "2025-07-01"), "ho": ("2025-07-01", "2026-10-01")}
LENGTHS = (14, 28)
ENTRIES = ("market", "retest")
STOPS = ("atr1", "swing")
TPS = (3.0, 5.0)
FILTERS = ("none", "trend4h", "vol")
WAIT = 8
HOLD_H = 96
TAKER, MAKER = 5.5e-4, 2e-4
BARS_PER_H = {"15m": 4, "1h": 1}


@njit(cache=True)
def lux_breaks(high, low, close, slope, length):
    """Пробои с реальным (неперерисованным) значением линии и её наклоном на баре пробоя.
    side: +1 — пробой верхней линии (лонг), -1 — нижней (шорт)."""
    m = len(close)
    side = np.zeros(m, np.int8)
    lv = np.full(m, np.nan)
    ls = np.zeros(m)
    upper, lower = np.nan, np.nan
    s_ph, s_pl = 0.0, 0.0
    upos, dnos = 0, 0
    for t in range(m):
        c = t - length
        ph = _pivot(high, c, length, True)
        pl = _pivot(low, c, length, False)
        if ph:
            s_ph = slope[t]
            upper = high[c]
        elif not np.isnan(upper):
            upper -= s_ph
        if pl:
            s_pl = slope[t]
            lower = low[c]
        elif not np.isnan(lower):
            lower += s_pl
        pu, pd_ = upos, dnos
        if ph:
            upos = 0
        elif not np.isnan(upper) and close[t] > upper - s_ph * length:
            upos = 1
        if pl:
            dnos = 0
        elif not np.isnan(lower) and close[t] < lower + s_pl * length:
            dnos = 1
        if upos > pu and not dnos > pd_:
            side[t], lv[t], ls[t] = 1, upper - s_ph * length, -s_ph
        elif dnos > pd_ and not upos > pu:
            side[t], lv[t], ls[t] = -1, lower + s_pl * length, s_pl
    return side, lv, ls


@njit(cache=True)
def simulate(side, lv, ls, allow, o, h, l, c, atr, length, retest, swing, tp_r, wait, hold, taker, maker):
    """Одна позиция на монету. Возвращает индекс бара сигнала, R после комиссий, сторону."""
    m = len(c)
    out_i = np.empty(m, np.int64)
    out_r = np.empty(m)
    out_s = np.empty(m, np.int8)
    k = 0
    busy = -1
    for t in range(m):
        s = side[t]
        if s == 0 or not allow[t] or t <= busy or not (atr[t] > 0) or t + 2 >= m:
            continue
        # вход
        fill = -1
        entry = 0.0
        fee_in = taker
        if retest:
            for j in range(t + 1, min(t + wait, m - 1) + 1):
                lvj = lv[t] + ls[t] * (j - t)
                if (s > 0 and l[j] < lvj) or (s < 0 and h[j] > lvj):
                    if (s > 0 and o[j] < lvj) or (s < 0 and o[j] > lvj):
                        break                      # гэп сквозь уровень: лимит исполнился бы хуже, сделку пропускаем
                    fill, entry, fee_in = j, lvj, maker
                    break
            if fill < 0:
                continue
        else:
            fill, entry = t, c[t]
        # стоп
        if swing:
            ext = l[t] if s > 0 else h[t]
            for j in range(max(0, t - length), t + 1):
                if s > 0:
                    ext = min(ext, l[j])
                else:
                    ext = max(ext, h[j])
            dist = abs(entry - ext)
            dist = min(max(dist, 0.5 * atr[t]), 3.0 * atr[t])
        else:
            dist = atr[t]
        if dist <= 0:
            continue
        stop = entry - s * dist
        tp = entry + s * tp_r * dist
        exit_px = np.nan
        fee_out = taker
        start = fill + 1 if not retest else fill
        end = min(fill + hold, m - 1)
        j = start
        while j <= end:
            if retest and j == fill:
                # бар исполнения лимита: проверяем только стоп (худший порядок), тейк — со следующего бара
                if (s > 0 and l[j] <= stop) or (s < 0 and h[j] >= stop):
                    exit_px = stop
                    break
                j += 1
                continue
            if s > 0:
                if l[j] <= stop:
                    exit_px = min(stop, o[j])
                    break
                if h[j] >= tp:
                    exit_px, fee_out = tp, maker
                    break
            else:
                if h[j] >= stop:
                    exit_px = max(stop, o[j])
                    break
                if l[j] <= tp:
                    exit_px, fee_out = tp, maker
                    break
            j += 1
        if np.isnan(exit_px):
            j = end
            exit_px = c[end]
        r = (s * (exit_px - entry) - (fee_in + fee_out) * entry) / dist
        out_i[k], out_r[k], out_s[k] = t, r, s
        k += 1
        busy = j
    return out_i[:k], out_r[:k], out_s[:k]


def load(root: Path, sym: str, tf: str) -> pd.DataFrame | None:
    p = root / f"{sym}-{tf}.parquet"
    if not p.exists():
        return None
    k = pd.read_parquet(p, columns=["open_time", "open", "high", "low", "close", "volume"])
    k.index = pd.to_datetime(k["open_time"], unit="ms", utc=True)
    return k.drop(columns="open_time").astype("float64")


def prepare(df: pd.DataFrame, adv: pd.Series) -> dict:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    c4 = df["close"].resample("4h", label="left", closed="left").last().dropna()
    ema = c4.ewm(span=50, adjust=False, min_periods=50).mean()
    trend = np.sign(ema.diff()).shift(1)                                    # только закрытые 4h-бары
    trend = trend.reindex(df.index, method="ffill").fillna(0).to_numpy()
    vol_ok = (df["volume"] > 1.5 * df["volume"].rolling(20).mean().shift(1)).to_numpy()
    liquid = (adv.reindex(df.index, method="ffill") >= ADV_MIN).to_numpy()
    return {"o": df["open"].to_numpy(), "h": df["high"].to_numpy(), "l": df["low"].to_numpy(),
            "c": df["close"].to_numpy(), "atr": atr.to_numpy(), "trend": trend, "vol_ok": vol_ok,
            "liquid": liquid, "ts": df.index}


def run_tf(root: Path, syms: list[str], tf: str) -> pd.DataFrame:
    cfgs = list(itertools.product(LENGTHS, ENTRIES, STOPS, TPS, FILTERS))
    parts = []
    hold = HOLD_H * BARS_PER_H[tf]
    for s in syms:
        df = load(root, s, tf)
        if df is None or len(df) < 500:
            continue
        a = prepare(df, adv30(root, s))
        for L in LENGTHS:
            side, lv, ls = lux_breaks(a["h"], a["l"], a["c"], slope_series(df, L, 1.0, "atr"), L)
            for k, (L2, entry, stop, tp, filt) in enumerate(cfgs):
                if L2 != L:
                    continue
                allow = a["liquid"].copy()
                if filt == "trend4h":
                    allow &= (a["trend"] * side) > 0
                elif filt == "vol":
                    allow &= a["vol_ok"]
                i, r, sd = simulate(side, lv, ls, allow, a["o"], a["h"], a["l"], a["c"], a["atr"], L,
                                    entry == "retest", stop == "swing", tp, WAIT, hold, TAKER, MAKER)
                if len(i):
                    parts.append(pd.DataFrame({"cfg": k, "t": a["ts"][i], "R": r, "side": sd, "symbol": s}))
    out = pd.concat(parts, ignore_index=True)
    meta = pd.DataFrame(cfgs, columns=["length", "entry", "stop", "tp", "filter"])
    return out.join(meta, on="cfg")


def _stat(x: pd.DataFrame) -> dict:
    if len(x) < 10:
        return {"n": len(x)}
    r = x["R"].to_numpy(dtype="float64")
    d = x["t"].dt.floor("D").to_numpy()
    g = pd.Series(r - r.mean()).groupby(d).sum()
    return {"n": len(r), "win": float((r > 0).mean()), "avgR": r.mean(),
            "t_day": r.sum() / np.sqrt((g ** 2).sum()) if (g ** 2).sum() > 0 else np.nan}


def report(tr: pd.DataFrame, tf: str, n_coins: int) -> None:
    keys = ["length", "entry", "stop", "tp", "filter"]
    rows = []
    for _, g in tr.groupby("cfg"):
        row = {k: g[k].iloc[0] for k in keys}
        for p, (a, b) in PER.items():
            st = _stat(g[(g.t >= a) & (g.t < b)])
            row.update({f"{p}_{k}": v for k, v in st.items()})
        days = (pd.Timestamp(PER["is"][1]) - pd.Timestamp(PER["is"][0])).days
        row["сделок/день/монету (IS)"] = row.get("is_n", 0) / days / max(n_coins, 1)
        rows.append(row)
    res = pd.DataFrame(rows).sort_values("is_t_day", ascending=False)
    print(f"\n=== {tf}: все {len(res)} настроек (средний R после комиссий; t по дням), по убыванию t на IS ===")
    print(res.round(3).to_string(index=False))
    gate = res[(res["is_t_day"] > 2) & (res["val_avgR"] > 0) & (res["val_t_day"] > 1.5)]
    print(f"\nВОРОТА {tf}: IS t > 2 и VAL R > 0, t > 1.5 -> HOLDOUT:")
    if not len(gate):
        print("  никто не прошёл")
    for _, f in gate.iterrows():
        g = tr[(tr[keys] == f[keys].values).all(axis=1)]
        new = g[g.symbol.map(group_of) != "core16"]
        print(f"  {dict(zip(keys, f[keys].values))}: HOLDOUT R={f['ho_avgR']:+.3f} (t {f['ho_t_day']:.2f}); "
              f"монеты вне core16 HOLDOUT: {_stat(new[new.t >= PER['ho'][0]])}")
    base = res[(res["length"] == 14) & (res["filter"] == "none")]
    print(f"\nНастройка с графика (length 14, без фильтра), {tf}:")
    print(base[keys + [c for c in res.columns if c.endswith(("_n", "_win", "_avgR", "_t_day"))]].round(3)
          .to_string(index=False))


def _daily(sym: str, tf: str, day: str) -> pd.DataFrame | None:
    blob = D._get(f"{D.DAILY}/klines/{sym}/{tf}/{sym}-{tf}-{day}.zip")
    if blob is None:
        return None
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0]).decode()
    df = pd.read_csv(io.StringIO(raw), header=0 if raw.startswith("open_time") else None, usecols=range(6))
    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
    df.index = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    return df.drop(columns="open_time").astype("float64")


def example(root: Path) -> None:
    """Пример с графика пользователя: SOLUSDT 15m, 29.09–04.10.2026 (время UTC), length 14, Slope 1, Atr."""
    hist = load(root, "SOLUSDT", "15m")
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range("2026-10-01", "2026-10-04")]
    extra = [x for d in days if (x := _daily("SOLUSDT", "15m", d)) is not None]
    if hist is None:
        print("\nПример SOL: нет 15m истории")
        return
    df = pd.concat([hist] + extra)
    df = df[~df.index.duplicated()].sort_index()
    df = df[df.index >= "2026-08-01"]
    side, lv, ls = lux_breaks(df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy(),
                              slope_series(df, 14, 1.0, "atr"), 14)
    a = prepare(df, pd.Series(1e12, index=df.index[:1]))
    print(f"\n=== Пример с графика: SOLUSDT 15m, 29.09–04.10.2026 (UTC), length 14 — каждый сигнал «B» и его исход ===")
    rows = []
    for entry, stop, tp in itertools.product(ENTRIES, STOPS, TPS):
        i, r, sd = simulate(side, lv, ls, np.ones(len(df), np.bool_), a["o"], a["h"], a["l"], a["c"], a["atr"], 14,
                            entry == "retest", stop == "swing", tp, WAIT, HOLD_H * 4, TAKER, MAKER)
        ts = df.index[i]
        m = ts >= "2026-09-29"
        rows.append({"вход": entry, "стоп": stop, "тейк": f"{tp:.0f}R", "сделок": int(m.sum()),
                     "сумма R": float(r[m].sum()), "win": float((r[m] > 0).mean()) if m.any() else np.nan,
                     "сделки (время, сторона, R)": " ".join(f"{t:%d %H:%M}{'L' if x > 0 else 'S'}{v:+.1f}"
                                                         for t, x, v in zip(ts[m], sd[m], r[m]))})
    print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 400)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--symbols15", required=True)
    ap.add_argument("--symbols1h", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    D.set_host("cdn")
    example(root)
    for tf, syms in (("15m", a.symbols15.split(",")), ("1h", a.symbols1h.split(","))):
        tr = run_tf(root, syms, tf)
        print(f"\n===== LUX R:R {tf}: сделок {len(tr):,}, монет {tr.symbol.nunique()} =====", flush=True)
        report(tr, tf, tr.symbol.nunique())
