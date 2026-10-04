"""
Трендовые стратегии с подтверждением реальными позициями (OI), потоком агрессоров и funding; выходы с R:R до 1:5.

Таймфреймы 4h и 1d (из часовых свечей). Входы (по закрытию бара сигнала):
    donch20 / donch55  close выше максимума (ниже минимума) предыдущих N баров
    pullback           EMA50 > EMA200 (для лонга), за последние 3 бара low был ниже EMA20, close снова выше EMA20
Фильтры (гипотезы зафиксированы до запуска; k = 5 баров):
    none     без фильтра
    oi_up    OI за k баров вырос (z > 0.5 к своей истории) — в движение входят новые позиции
    oi_down  OI за k баров упал (z < -0.5) — диагностика: ожидается хуже (закрытие шортов/ликвидации)
    flow     поток агрессоров фьючерсов за k баров в сторону сделки (z > 0.5)
    oi_flow  oi_up и flow одновременно
    fund_ok  funding (сумма за k баров) не в верхних 20% своей истории в сторону сделки (толпа не перегрета)
Выходы: начальный стоп s * ATR(14); tp5 — тейк 5R и стоп; trail — после +1R стоп в безубыток, далее трейлинг
3 * ATR от экстремума, без тейка. Удержание не дольше 60 баров. Если в баре задеты стоп и тейк — стоп.
Сделка в R: (выход - вход) / (s * ATR) за вычетом издержек 2 * 6 б.п. и funding (в единицах риска).
Протокол: выбор по IS (2022-01..2024-06) на всех монетах по t-статистике среднего R (не меньше 300 сделок),
ворота VAL, затем HOLDOUT; экзамен — монеты вне подбора (ext54 + fresh). Оборот >= $20 млн/день на входе.

    python -m research.trend --root <binance 1h data with metrics> --symbols ...
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

from .broad import ADV_MIN, adv30, group_of
from .wave2 import Data2

TFS = {"4h": "4h", "1d": "1D"}
ENTRIES = ("donch20", "donch55", "pullback")
DIRS = ("long", "both")
FILTERS = ("none", "oi_up", "oi_down", "flow", "oi_flow", "fund_ok")
STOPS = (1.5, 2.5)
EXITS = ("tp5", "trail")
K = 5
MAX_HOLD = 60
COST = 6e-4
PER = {"is": ("2022-01-01", "2024-07-01"), "val": ("2024-07-01", "2025-07-01"), "ho": ("2025-07-01", "2026-10-01")}
MIN_IS = 300
KEYS = ["tf", "entry", "filter", "dir", "stop", "exit"]


@njit(cache=True)
def trade_machine(o, h, lo, c, atr, fund, long_sig, short_sig, stop_k, exit_mode, max_hold, cost):
    """Одна позиция за раз. exit_mode 0 = тейк 5R, 1 = безубыток после +1R и трейлинг 3 ATR.
    Возвращает индексы входов, R-результат сделок (после издержек и funding) и длительность в барах."""
    n = len(c)
    idx = np.empty(n, np.int64)
    rr = np.empty(n)
    dur = np.empty(n, np.int64)
    k = 0
    t = 0
    while t < n - 1:
        side = 1 if long_sig[t] else (-1 if short_sig[t] else 0)
        a = atr[t]
        if side == 0 or not (a > 0):
            t += 1
            continue
        entry = c[t]
        risk = stop_k * a
        stop = entry - side * risk
        tp = entry + side * 5.0 * risk
        ext = entry
        fsum = 0.0
        exit_px = np.nan
        j = t + 1
        end = min(t + max_hold, n - 1)
        while j <= end:
            fsum += fund[j]
            if side > 0:
                if lo[j] <= stop:
                    exit_px = min(stop, o[j])
                    break
                if exit_mode == 0 and h[j] >= tp:
                    exit_px = tp
                    break
                if exit_mode == 1:
                    ext = max(ext, h[j])
                    if ext - entry >= risk:
                        stop = max(stop, entry, ext - 3.0 * atr[j])
            else:
                if h[j] >= stop:
                    exit_px = max(stop, o[j])
                    break
                if exit_mode == 0 and lo[j] <= tp:
                    exit_px = tp
                    break
                if exit_mode == 1:
                    ext = min(ext, lo[j])
                    if entry - ext >= risk:
                        stop = min(stop, entry, ext + 3.0 * atr[j])
            j += 1
        if np.isnan(exit_px):
            j = end
            exit_px = c[end]
        gross = side * (exit_px - entry)
        idx[k] = t
        rr[k] = (gross - 2 * cost * entry - side * fsum * entry) / risk
        dur[k] = j - t
        k += 1
        t = j + 1
    return idx[:k], rr[:k], dur[:k]


def _z(x: pd.Series, w: int) -> pd.Series:
    return (x - x.rolling(w, min_periods=w // 3).mean()) / x.rolling(w, min_periods=w // 3).std()


def bars(h: pd.DataFrame, tf: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
           "taker_buy_volume": "sum", "funding": "sum", "oi": "last"}
    d = h[list(agg)].resample(TFS[tf], label="left", closed="left").agg(agg).dropna(subset=["close"])
    pc = d["close"].shift()
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    w = 180 if tf == "4h" else 120
    d["oi_z"] = _z(np.log(d["oi"].replace(0, np.nan)).diff(K), w)
    flow = (2 * d["taker_buy_volume"] - d["volume"]).rolling(K).sum() / d["volume"].rolling(K).sum()
    d["flow_z"] = _z(flow, w)
    fk = d["funding"].rolling(K).sum()
    d["fund_hi"] = fk > fk.rolling(w, min_periods=w // 3).quantile(0.8)
    d["fund_lo"] = fk < fk.rolling(w, min_periods=w // 3).quantile(0.2)
    return d


def signals(d: pd.DataFrame, entry: str) -> tuple[pd.Series, pd.Series]:
    c = d["close"]
    if entry.startswith("donch"):
        n = int(entry[5:])
        up = c > d["high"].shift(1).rolling(n).max()
        dn = c < d["low"].shift(1).rolling(n).min()
    else:
        e20, e50, e200 = (c.ewm(span=s, adjust=False).mean() for s in (20, 50, 200))
        dipped = (d["low"] < e20).rolling(3).max().astype(bool)
        spiked = (d["high"] > e20).rolling(3).max().astype(bool)
        up = (e50 > e200) & dipped & (c > e20) & (c.shift() <= e20.shift())
        dn = (e50 < e200) & spiked & (c < e20) & (c.shift() >= e20.shift())
    return up.fillna(False), dn.fillna(False)


def apply_filter(d: pd.DataFrame, up: pd.Series, dn: pd.Series, f: str) -> tuple[pd.Series, pd.Series]:
    oi, fl = d["oi_z"], d["flow_z"]
    if f == "none":
        return up, dn
    if f == "oi_up":
        return up & (oi > 0.5), dn & (oi > 0.5)
    if f == "oi_down":
        return up & (oi < -0.5), dn & (oi < -0.5)
    if f == "flow":
        return up & (fl > 0.5), dn & (fl < -0.5)
    if f == "oi_flow":
        return up & (oi > 0.5) & (fl > 0.5), dn & (oi > 0.5) & (fl < -0.5)
    return up & ~d["fund_hi"], dn & ~d["fund_lo"]


def run(root: Path, syms: list[str]) -> pd.DataFrame:
    """Все сделки всех конфигураций; хранится компактно (id конфигурации и монеты), иначе 725 монет не влезут."""
    data = Data2(root, syms)
    cfgs = list(itertools.product(TFS, ENTRIES, FILTERS, DIRS, STOPS, EXITS))
    cfg_id = {c: i for i, c in enumerate(cfgs)}
    parts: list[tuple] = []
    sym_names: list[str] = []
    for s in syms:
        try:
            h = data.get(s, "1h", "full")
        except Exception as e:
            print(f"  {s}: пропуск ({e})", flush=True)
            continue
        if h["oi"].notna().sum() < 24 * 60:
            continue
        adv = adv30(root, s)
        sid = len(sym_names)
        sym_names.append(s)
        for tf in TFS:
            d = bars(h, tf)
            allowed = (adv.reindex(d.index, method="ffill") >= ADV_MIN).to_numpy()
            arr = {c: d[c].to_numpy(dtype="float64") for c in ("open", "high", "low", "close", "atr", "funding")}
            ts = d.index.as_unit("ns").asi8
            for entry in ENTRIES:
                up0, dn0 = signals(d, entry)
                for f in FILTERS:
                    up, dn = apply_filter(d, up0, dn0, f)
                    for dr in DIRS:
                        L = up.to_numpy() & allowed
                        S = (dn.to_numpy() & allowed) if dr == "both" else np.zeros(len(d), np.bool_)
                        for sk, ex in itertools.product(STOPS, EXITS):
                            i, r, du = trade_machine(arr["open"], arr["high"], arr["low"], arr["close"], arr["atr"],
                                                     arr["funding"], L, S, sk, EXITS.index(ex), MAX_HOLD, COST)
                            if len(i):
                                parts.append((cfg_id[(tf, entry, f, dr, sk, ex)], sid, ts[i], r.astype(np.float32), du))
    cid = np.concatenate([np.full(len(p[2]), p[0], np.int16) for p in parts])
    sidv = np.concatenate([np.full(len(p[2]), p[1], np.int16) for p in parts])
    out = pd.DataFrame({"cfg": cid, "sid": sidv, "t": pd.to_datetime(np.concatenate([p[2] for p in parts]), utc=True),
                        "R": np.concatenate([p[3] for p in parts]), "bars": np.concatenate([p[4] for p in parts])})
    cfg_df = pd.DataFrame(cfgs, columns=KEYS)
    out = out.join(cfg_df, on="cfg")
    for k in ("tf", "entry", "filter", "dir", "exit"):
        out[k] = out[k].astype("category")
    names = np.array(sym_names)
    out["symbol"] = pd.Categorical(names[out["sid"].to_numpy()])
    out["group"] = pd.Categorical([group_of(x) for x in names])[out["sid"].to_numpy()]
    return out


def summarize(x: pd.DataFrame) -> dict:
    if not len(x):
        return {"n": 0}
    r = x["R"]
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    return {"n": len(r), "win": float((r > 0).mean()), "avgR": r.mean(),
            "t": r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 2 else np.nan,
            "pf": pos / neg if neg > 0 else np.nan, "big5R": float((r >= 4.5).mean())}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 400)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--symbols", required=True)
    a = ap.parse_args()
    tr = run(Path(a.root), a.symbols.split(","))
    print(f"===== TREND: сделок {len(tr)}, монет {tr['symbol'].nunique()} =====")
    rows = []
    for key, g in tr.groupby("cfg"):
        row = {k: g[k].iloc[0] for k in KEYS}
        for p, (pa, pb) in PER.items():
            st = summarize(g[(g.t >= pa) & (g.t < pb)])
            row.update({f"{p}_{k}": v for k, v in st.items()})
        rows.append(row)
    res = pd.DataFrame(rows)
    print(f"\nконфигураций {len(res)}; средний R на IS: медиана {res['is_avgR'].median():.3f}, "
          f"доля > 0: {(res['is_avgR'] > 0).mean():.0%}")
    for col in KEYS:
        print(f"  {col}: " + ", ".join(f"{k}={v:+.3f}" for k, v in res.groupby(col)["is_avgR"].median().items()))
    print("\nВлияние фильтра (медиана среднего R по всем остальным параметрам), IS / VAL / HO:")
    print(res.groupby("filter")[["is_avgR", "val_avgR", "ho_avgR"]].median().round(3).to_string())
    cols = KEYS + [f"{p}_{k}" for p in PER for k in ("n", "win", "avgR", "t", "pf")]
    top = res[res["is_n"] >= MIN_IS].sort_values("is_t", ascending=False).head(12)
    print(f"\nТОП-12 по t-статистике IS (не меньше {MIN_IS} сделок):")
    print(top[cols].round(3).to_string(index=False))
    passed = top[(top["val_avgR"] > 0) & (top["val_t"] > 1.5)]
    print("\nВОРОТА VAL (средний R > 0 и t > 1.5) -> HOLDOUT; экзамен на монетах вне подбора:")
    if not len(passed):
        print("  никто не прошёл")
    for _, f in passed.iterrows():
        g = tr[(tr[KEYS].astype(object) == f[KEYS].astype(object).values).all(axis=1)]
        new = g[g["group"].isin(["ext54", "fresh"])]
        msg = " | ".join(f"{p}: n={s_['n']}, R={s_.get('avgR', np.nan):+.3f}, win {s_.get('win', np.nan):.0%}, "
                         f"PF {s_.get('pf', np.nan):.2f}"
                         for p, (pa, pb) in PER.items() for s_ in [summarize(new[(new.t >= pa) & (new.t < pb)])])
        print(f"  {dict(zip(KEYS, f[KEYS].values))}\n    новые монеты: {msg}")
        x = g[g.t >= PER["ho"][0]]["R"]
        print(f"    HOLDOUT все монеты: n={len(x)}, R={x.mean():+.3f}, распределение R: "
              f"{np.percentile(x, [5, 25, 50, 75, 95]).round(2).tolist() if len(x) else []}")
