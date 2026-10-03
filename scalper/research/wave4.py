"""
Волна 4: частые интрадей-входы лимитными ордерами (maker) на 5m/15m барах.

Протокол (зафиксирован до запуска):
    IS 2022-01..2024-06 — перебор; финалисты: 2 лучших конфига на семейство по Sharpe IS (базовые издержки)
                          при не менее MIN_TRADES_PER_DAY сделок в день на портфель
    VAL 2024-07..2025-06 — ворота: Sharpe >= VAL_MIN при базовых издержках и > VAL_MIN_STRESS при стрессовых
    HOLDOUT 2025-07..2026-09 — один прогон только для прошедших ворота
Издержки: база — maker 2 б.п., рыночные выходы 7 б.п.; стресс — 4 и 10 б.п.

    python -m research.wave4 --root <data> --out <dir> [--stage search|all]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import UNIVERSE
from .engine import metrics
from .limit_engine import limit_machine
from .search import grid
from .wave2 import PERIODS, Data2, _z

COSTS = {"base": (2.0, 7.0), "stress": (4.0, 10.0)}
THROUGH = 1e-4                  # лимит исполнен, только если цена прошла сквозь уровень на 1 б.п.
TTL = 3                          # баров жизни лимитного ордера
MIN_TRADES_PER_DAY = 1.5
TOP_PER_FAMILY = 2
VAL_MIN, VAL_MIN_STRESS = 1.0, 0.3
BARS_PER_DAY = {"5m": 288, "15m": 96}


# ---------- признаки ----------

def _sigma_bar(c: pd.Series, win: int) -> pd.Series:
    """Стандартное отклонение доходности одного бара (доли цены) за win баров."""
    return np.log(c).diff().rolling(win, min_periods=win // 2).std()


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype="float64").copy()


def _orders(long_sig, short_sig, close, limit_off, tp_long, tp_short, sl_off):
    """Собирает массивы ордеров: limit_off/sl_off — смещения в ценах (>0), tp_* — абсолютные цены."""
    c = _arr(close)
    ls = np.nan_to_num(_arr(long_sig)).astype(bool)
    ss = np.nan_to_num(_arr(short_sig)).astype(bool) & ~ls
    side = np.where(ls, 1, np.where(ss, -1, 0)).astype(np.int64)
    lo = _arr(limit_off)
    lim = np.where(side > 0, c - lo, np.where(side < 0, c + lo, np.nan))
    tp = np.where(side > 0, _arr(tp_long), np.where(side < 0, _arr(tp_short), np.nan))
    so = _arr(sl_off)
    sl = np.where(side > 0, lim - so, np.where(side < 0, lim + so, np.nan))
    # тейк должен быть по правильную сторону от входа, иначе сигнал отбрасываем
    bad = ((side > 0) & ~(tp > lim)) | ((side < 0) & ~(tp < lim)) | ~np.isfinite(lim) | ~np.isfinite(sl)
    side[bad] = 0
    return side, lim, tp, sl


def mr_ema(df, n=48, thr=2.0, a=0.5, tpm=1.0, slm=3.0, H=48):
    """Перерастяжение от EMA(n) на thr сигм -> лимит глубже рынка на a сигм бара; тейк — доля пути до EMA."""
    c = df["close"]
    sb = _sigma_bar(c, 288)
    u = c * sb
    ema = c.ewm(span=n, adjust=False).mean()
    dev = (c - ema) / (u * np.sqrt(n))
    lim_off = a * u
    tp_l = (c - lim_off) + tpm * (ema - (c - lim_off))
    tp_s = (c + lim_off) - tpm * ((c + lim_off) - ema)
    return _orders(dev < -thr, dev > thr, c, lim_off, tp_l, tp_s, slm * u * np.sqrt(n)) + (TTL, H)


def mr_range(df, n=24, thr=1.0, tz=0.5, a=0.5, slm=2.0, H=24):
    """То же в спокойном рынке: возврат к EMA только если дневной тренд слабый (|z| < tz)."""
    c = df["close"]
    sb = _sigma_bar(c, 288)
    u = c * sb
    ema = c.ewm(span=n, adjust=False).mean()
    dev = (c - ema) / (u * np.sqrt(n))
    trend = np.log(c / c.shift(288)) / (sb * np.sqrt(288))
    calm = trend.abs() < tz
    lim_off = a * u
    return _orders(calm & (dev < -thr), calm & (dev > thr), c, lim_off, ema, ema, slm * u * np.sqrt(n)) + (TTL, H)


def vwap_day(df, thr=2.0, a=0.5, tpm=1.0, slm=3.0, H=48):
    """Отклонение от VWAP с начала суток UTC -> возврат к VWAP лимитным входом."""
    c = df["close"]
    day = df.index.floor("1D")
    qv = df["quote_volume"].groupby(day).cumsum()
    v = df["volume"].groupby(day).cumsum()
    vwap = (qv / v).where(v > 0)
    k = df.groupby(day).cumcount() + 1
    sb = _sigma_bar(c, 288)
    u = c * sb
    dev = (c - vwap) / (u * np.sqrt(k.clip(lower=12)))
    lim_off = a * u
    tp_l = (c - lim_off) + tpm * (vwap - (c - lim_off))
    tp_s = (c + lim_off) - tpm * ((c + lim_off) - vwap)
    return _orders(dev < -thr, dev > thr, c, lim_off, tp_l, tp_s, slm * u * np.sqrt(12)) + (TTL, H)


def sweep(df, N=288, pen=0.0, a=0.0, tp_mode="mid", H=48):
    """Ложный пробой: low ушёл ниже минимума N баров (на pen сигм), а бар закрылся обратно выше -> лонг."""
    c, h, l = df["close"], df["high"], df["low"]
    sb = _sigma_bar(c, 288)
    u = c * sb
    lo_n = l.rolling(N).min().shift(1)
    hi_n = h.rolling(N).max().shift(1)
    long_sig = (l < lo_n - pen * u) & (c > lo_n)
    short_sig = (h > hi_n + pen * u) & (c < hi_n)
    mid = (lo_n + hi_n) / 2
    tp_l = mid if tp_mode == "mid" else hi_n
    tp_s = mid if tp_mode == "mid" else lo_n
    lim_off = a * u
    sl_l_off = (c - lim_off) - (l - u)                   # стоп под минимумом пробойного бара
    sl_s_off = (h + u) - (c + lim_off)
    sl_off = np.where(long_sig.fillna(False), sl_l_off, sl_s_off)
    return _orders(long_sig, short_sig, c, lim_off, tp_l, tp_s, np.maximum(sl_off, 0.5 * _arr(u))) + (TTL, H)


def oi_flush(df, k=4, thr=2.0, othr=2.0, a=0.0, tpm=0.5, H=16):
    """15m: цена резко ушла за k баров, OI резко упал (принудительные закрытия) -> лимит против движения;
    тейк — доля tpm отката к цене k баров назад."""
    c = df["close"]
    sb = _sigma_bar(c, 96 * 3)
    u = c * sb
    dp = np.log(c / c.shift(k)) / (sb * np.sqrt(k))
    oi = df["oi"].replace(0, np.nan)
    doi = _z(np.log(oi / oi.shift(k)), 96 * 30)
    flush = doi < -othr
    lim_off = a * u
    ref = c.shift(k)
    tp_l = (c - lim_off) + tpm * (ref - (c - lim_off))
    tp_s = (c + lim_off) - tpm * ((c + lim_off) - ref)
    return _orders(flush & (dp < -thr), flush & (dp > thr), c, lim_off, tp_l, tp_s, 2.0 * u * np.sqrt(k)) + (TTL, H)


FAMILIES = {
    "mr_ema": ("5m", mr_ema, grid(n=[12, 48, 144], thr=[2.0, 3.0], a=[0.5, 1.5], tpm=[0.5, 1.0], slm=[1.5, 3.0],
                                  H=[12, 48])),
    "mr_range": ("5m", mr_range, grid(n=[24, 96], thr=[1.0, 1.5], tz=[0.5, 1.0], a=[0.5, 1.0], slm=[2.0, 4.0],
                                      H=[24, 72])),
    "vwap_day": ("5m", vwap_day, grid(thr=[1.5, 2.0, 2.5], a=[0.5, 1.5], tpm=[0.5, 1.0], slm=[1.5, 3.0], H=[24, 96])),
    "sweep": ("5m", sweep, grid(N=[48, 288], pen=[0.0, 1.0], a=[0.0, 1.0], tp_mode=["mid", "opp"], H=[24, 96])),
    "oi_flush": ("15m", oi_flush, grid(k=[2, 4, 8], thr=[2.0, 2.5], othr=[1.5, 2.5], a=[0.0, 1.0], tpm=[0.3, 0.6],
                                       H=[8, 16])),
}


# ---------- оценка ----------

def _cut(s: pd.Series, period: str) -> pd.Series:
    a, b = PERIODS[period]
    return s[(s.index >= a) & (s.index < b)]


def run_symbol(df: pd.DataFrame, fn, params: dict, cost: str):
    side, lim, tp, sl, ttl, H = fn(df, **params)
    mk, tk = COSTS[cost]
    ret, pos, trades, kind = limit_machine(side, lim, tp, sl, ttl, H, _arr(df["open"]), _arr(df["high"]),
                                           _arr(df["low"]), _arr(df["close"]), _arr(df["funding"]), mk, tk, THROUGH)
    return pd.Series(ret, index=df.index), pd.Series(pos, index=df.index), pd.Series(kind, index=df.index)


def evaluate(data: Data2, family: str, params: dict, period: str, cost: str = "base"):
    tf, fn, _ = FAMILIES[family]
    rets, entries, kinds = {}, 0, np.zeros(4)
    for s in data.symbols:
        df = data.get(s, tf, "is" if period == "is" else "full")
        r, p, k = run_symbol(df, fn, params, cost)
        r, p, k = _cut(r, period), _cut(p, period), _cut(k, period)
        rets[s] = r
        prev = p.shift(1).fillna(0.0)
        entries += int(((p != 0) & (prev == 0)).sum() + ((k == 2) & (prev == 0)).sum())   # вход+стоп в одном баре
        kinds += np.bincount(k.to_numpy().astype(int), minlength=4)[:4]
    port = pd.DataFrame(rets).fillna(0.0).mean(axis=1)
    m = metrics(port)
    exits = kinds[1:].sum()
    out = {"sharpe": m["sharpe"], "ann_ret": m["ann_ret"], "ann_vol": m["ann_vol"], "max_dd": m["max_dd"],
           "trades": entries, "trades_per_day": entries / max(m["days"], 1),
           "bps_per_trade": float(port.sum() * len(data.symbols) / max(entries, 1) * 1e4),
           "tp_frac": kinds[1] / max(exits, 1), "sl_frac": kinds[2] / max(exits, 1), "time_frac": kinds[3] / max(exits, 1),
           "pos_months": float((port.resample("ME").sum() > 0).mean())}
    return port, out


def search(data: Data2, out: Path) -> pd.DataFrame:
    rows, t0 = [], time.time()
    for fam, (tf, _, g) in FAMILIES.items():
        for p in g:
            _, s = evaluate(data, fam, p, "is", "base")
            rows.append({"family": fam, "tf": tf, "params": json.dumps(p), **s})
        print(f"  {fam}: готово ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "wave4_is.csv", index=False)
    return df


def select(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["trades_per_day"] >= MIN_TRADES_PER_DAY].copy()
    ok["family_median_sharpe"] = ok.groupby("family")["sharpe"].transform("median")
    return ok.sort_values("sharpe", ascending=False).groupby("family").head(TOP_PER_FAMILY)


def gate(data: Data2, fin: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    for _, f in fin.iterrows():
        p = json.loads(f["params"])
        r = {"family": f["family"], "params": f["params"], "is_sharpe": f["sharpe"], "is_tpd": f["trades_per_day"]}
        for cost in COSTS:
            _, s = evaluate(data, f["family"], p, "val", cost)
            r[f"val_sharpe_{cost}"], r[f"val_ret_{cost}"], r[f"val_dd_{cost}"] = s["sharpe"], s["ann_ret"], s["max_dd"]
            r[f"val_tpd_{cost}"], r[f"val_bps_{cost}"] = s["trades_per_day"], s["bps_per_trade"]
        r["passed"] = bool(r["val_sharpe_base"] >= VAL_MIN and r["val_sharpe_stress"] > VAL_MIN_STRESS)
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(out / "wave4_val.csv", index=False)
    return res


def holdout(data: Data2, passed: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    for _, f in passed.iterrows():
        p = json.loads(f["params"])
        for cost in COSTS:
            port, s = evaluate(data, f["family"], p, "ho", cost)
            rows.append({"family": f["family"], "params": f["params"], "cost": cost, **s})
            port.to_frame("pnl").to_csv(out / f"wave4_ho_{f['family']}_{cost}.csv")
    res = pd.DataFrame(rows)
    res.to_csv(out / "wave4_holdout.csv", index=False)
    return res


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 250)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--symbols", default=",".join(UNIVERSE))
    ap.add_argument("--stage", default="all", choices=["search", "all"])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Data2(Path(a.root), a.symbols.split(","))

    res = search(data, out)
    print(f"\nВолна 4, всего конфигураций: {len(res)}")
    print(res.groupby("family")[["sharpe", "trades_per_day"]].describe().loc[:, (slice(None), ["50%", "max"])].round(2)
          .to_string())
    fin = select(res)
    print(f"\nФИНАЛИСТЫ (IS, >= {MIN_TRADES_PER_DAY} сделки/день):")
    print(fin[["family", "params", "sharpe", "ann_ret", "max_dd", "trades_per_day", "bps_per_trade", "tp_frac",
               "sl_frac", "time_frac", "pos_months", "family_median_sharpe"]].round(3).to_string(index=False))
    if a.stage == "all":
        val = gate(data, fin, out)
        print(f"\nVAL 2024-07..2025-06 (ворота: Sharpe >= {VAL_MIN} база и > {VAL_MIN_STRESS} стресс):")
        print(val.round(3).to_string(index=False))
        passed = val[val["passed"]]
        if len(passed):
            print("\nHOLDOUT 2025-07..2026-09 (один прогон):")
            print(holdout(data, passed, out).round(3).to_string(index=False))
        else:
            print("\nНи один финалист не прошёл VAL — holdout не трогаем.")
