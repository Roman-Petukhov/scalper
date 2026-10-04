"""
Монеты, способные дать +50% / +100% за сутки: есть ли у таких движений признаки ДО начала, и можно ли на этом заработать.

Вселенная — все USDT-перпетуалы архива Binance (включая делистнутые), а не только 725 «хоть раз ликвидных»: иначе
выборка заранее состоит из будущих пампов. В каждый срез — монеты с оборотом за 30 прошлых дней >= ADV_LO.
Срезы — каждые 4 часа (00, 04, ... UTC) по часовым свечам; всё считается по данным, известным на закрытии часа.

Цели (от close среза):  up50 / up100 — максимум high за следующие 24 ч выше на 50% / 100%.
Признаки A (есть у всех монет): доходности 1 ч / 24 ч / 7 д / 30 д, просадка от максимума 90 д, сжатие волатильности,
    объём 24 ч к 30 д, перевес агрессоров 4 ч / 24 ч, funding (последний, сумма 24 ч и 7 д), возраст монеты, оборот,
    крупнейший суточный рост за 30 д («повторные пампы»), BTC за 24 ч / 7 д, «жар рынка» (сколько монет за 7 д
    давали +30% за сутки).
Признаки B (только 725 монет с метриками): OI за 24 ч / 3 д / 7 д, OI к обороту, long/short ratio толпы и топов,
    изменение LSR за 3 д. Модель B обучается только на этих монетах — с оговоркой о смещении выборки.

Анализ:
    1) сколько было событий, у каких монет (возраст, оборот, funding перед стартом), по годам;
    2) лифт признаков: P(up50) в квинтиле (границы по IS) / базовая частота — IS / VAL / HOLDOUT;
    3) LightGBM-классификатор up50 на IS; пороги — верхние 0.1 / 0.5 / 1% внефолдовых прогнозов IS;
       на VAL / HOLDOUT: точность (сколько отмеченных реально дали +50% / +100%) и торговля лонгом:
       вход по open следующего часа, стоп −10% / −20%, тейк +50% / +100%, не дольше 72 ч, издержки 12 б.п., funding;
    4) два правила, зафиксированных до запуска: R1 — шорты платят (funding за 24 ч <= −0.2%); R2 — R1 и OI за 3 д > +20%.

    python -m research.moonshots --root <binance 1h data> --symbols <все монеты> --metrics-symbols <725 монет>
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

from .broad import adv30, group_of
from .wave2 import Data2

ADV_LO = 2e6
STEP_H = 4
COST = 12e-4
HOLD = 72
EXITS = [(s, t) for s in (0.10, 0.20) for t in (0.5, 1.0)]
TOPS = (0.001, 0.005, 0.01)
PER = {"is": ("2022-01-01", "2024-07-01"), "val": ("2024-07-01", "2025-07-01"), "ho": ("2025-07-01", "2026-10-01")}
FEATS_A = ["ret1h", "ret24", "ret7d", "ret30d", "dd90", "rv_ratio", "rv_pct", "vol_ratio", "flow4", "flow24",
           "fund_last", "fund24", "fund7d", "age_d", "adv_log", "max24_30d", "btc24", "btc7d", "heat"]
FEATS_B = FEATS_A + ["oi24", "oi3d", "oi7d", "oi_adv", "lsr", "top_lsr", "lsr_chg3d"]


@njit(cache=True)
def exit_returns(o, h, lo, c, fund, entries, stop, tp, hold, cost):
    """Лонг по open бара e+1; стоп проверяется раньше тейка (консервативно), гэп через стоп — выход по open."""
    n, m = len(c), len(entries)
    out = np.full(m, np.nan)
    for k in range(m):
        e = entries[k] + 1
        if e >= n:
            continue
        px = o[e]
        sp, tpp = px * (1 - stop), px * (1 + tp)
        paid = 0.0
        ex = np.nan
        last = min(e + hold, n) - 1
        for j in range(e, last + 1):
            paid += fund[j]
            if o[j] <= sp and j > e:
                ex = o[j]
            elif lo[j] <= sp:
                ex = sp
            elif o[j] >= tpp and j > e:
                ex = o[j]
            elif h[j] >= tpp:
                ex = tpp
            if not np.isnan(ex):
                break
        if np.isnan(ex):
            if e + hold > n:
                continue                                    # история кончилась раньше выхода
            ex = c[last]
        out[k] = ex / px - 1 - cost - paid
    return out


def _fwd_max(x: pd.Series, k: int) -> pd.Series:
    """max(x[t+1 .. t+k]); NaN, если окно выходит за конец истории."""
    r = x[::-1].rolling(k, min_periods=k).max()[::-1].shift(-1)
    return r


def _pct(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(w, min_periods=w // 3).rank(pct=True)


def coin_rows(h: pd.DataFrame, adv: pd.Series, btc: pd.Series, sym: str, with_metrics: bool) -> pd.DataFrame:
    c, hi, lo, v = h["close"], h["high"], h["low"], h["volume"]
    lc = np.log(c)
    lr = lc.diff()
    f = h["funding"].fillna(0.0)
    flow = (2 * h["taker_buy_volume"] - v)
    rv24 = lr.rolling(24).std()
    feats = pd.DataFrame({
        "ret1h": lr, "ret24": lc.diff(24), "ret7d": lc.diff(168), "ret30d": lc.diff(720),
        "dd90": c / hi.rolling(2160, min_periods=24).max() - 1,
        "rv_ratio": rv24 / lr.rolling(720, min_periods=240).std(),
        "rv_pct": _pct(lr.rolling(72).std(), 2160),
        "vol_ratio": np.log((v.rolling(24).sum() + 1) / (v.rolling(720, min_periods=240).sum() / 30 + 1)),
        "flow4": flow.rolling(4).sum() / v.rolling(4).sum().replace(0, np.nan),
        "flow24": flow.rolling(24).sum() / v.rolling(24).sum().replace(0, np.nan),
        "fund_last": f.replace(0, np.nan).ffill(), "fund24": f.rolling(24).sum(), "fund7d": f.rolling(168).sum(),
        "age_d": (h.index - h.index[0]).days.astype("float64"),
        "adv_log": np.log10(adv.reindex(h.index, method="ffill").clip(lower=1.0)),
        "max24_30d": lc.diff(24).rolling(720, min_periods=24).max(),
        "btc24": np.log(btc).diff(24).reindex(h.index), "btc7d": np.log(btc).diff(168).reindex(h.index),
    }, index=h.index)
    if with_metrics:
        oi = np.log(h["oi"].replace(0, np.nan))
        feats["oi24"], feats["oi3d"], feats["oi7d"] = oi.diff(24), oi.diff(72), oi.diff(168)
        feats["oi_adv"] = np.log10((h["oi"] * c).replace(0, np.nan) / adv.reindex(h.index, method="ffill"))
        feats["lsr"], feats["top_lsr"] = h["lsr"], h["top_lsr"]
        feats["lsr_chg3d"] = np.log(h["lsr"].replace(0, np.nan)).diff(72)
    feats["up24"] = _fwd_max(hi, 24) / c - 1
    feats["full24"] = lc.diff(24)                                   # для «жара рынка»: рост за прошедшие сутки
    snap = (h.index.hour % STEP_H == 0) & (feats["adv_log"] >= np.log10(ADV_LO)).to_numpy() & (np.arange(len(h)) >= 24)
    idx = np.flatnonzero(snap)
    if not len(idx):
        return pd.DataFrame()
    arr = {k: h[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close")}
    for s_, t_ in EXITS:
        feats[f"tr_{int(s_ * 100)}_{int(t_ * 100)}"] = np.nan
        feats.iloc[idx, feats.columns.get_loc(f"tr_{int(s_ * 100)}_{int(t_ * 100)}")] = exit_returns(
            arr["open"], arr["high"], arr["low"], arr["close"], f.to_numpy(dtype="float64"), idx, s_, t_, HOLD, COST)
    out = feats.iloc[idx].copy()
    out = out[out["up24"].notna()]
    out.insert(0, "t", out.index)
    out.insert(0, "symbol", sym)
    return out.reset_index(drop=True).astype({c_: "float32" for c_ in out.columns if c_ not in ("symbol", "t")})


def collect(root: Path, syms: list[str], msyms: set[str]) -> pd.DataFrame:
    data = Data2(root, syms)
    btc = data.get("BTCUSDT", "1h", "full")["close"]
    parts = []
    for k, s in enumerate(syms):
        try:
            h = data.get(s, "1h", "full")
            if len(h) < 24 * 3:
                continue
            ev = coin_rows(h, adv30(root, s), btc, s, s in msyms)
        except Exception as e:
            print(f"  {s}: пропуск ({e})", flush=True)
            continue
        finally:
            data.cache.clear()
        if len(ev):
            parts.append(ev)
        if k % 100 == 0:
            print(f"  {k}/{len(syms)} монет", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    hot = (df["full24"] > np.log(1.3)).groupby(df["t"]).sum()          # монет с +30% за сутки в этот срез
    heat = hot.rolling("7D").sum()
    df["heat"] = df["t"].map(heat).astype("float32")
    df["up50"] = (df["up24"] >= 0.5).astype("int8")
    df["up100"] = (df["up24"] >= 1.0).astype("int8")
    df["metrics"] = df["symbol"].isin(msyms)
    df["group"] = [group_of(x) for x in df["symbol"]]
    return df


def period(df: pd.DataFrame, p: str) -> pd.DataFrame:
    a, b = PER[p]
    return df[(df["t"] >= a) & (df["t"] < b)]


def describe(df: pd.DataFrame) -> None:
    print(f"\n=== 1. События (срезы каждые {STEP_H} ч, оборот >= ${ADV_LO / 1e6:.0f}M/день) ===")
    for p in PER:
        y = period(df, p)
        print(f"  {p}: срезов {len(y):,}, монет {y['symbol'].nunique()}, up50 {y['up50'].mean():.3%} "
              f"({int(y['up50'].sum())}), up100 {y['up100'].mean():.4%} ({int(y['up100'].sum())}); "
              f"монето-дней с up100: {y[y.up100 == 1].groupby(['symbol', y['t'].dt.floor('D')]).ngroups}")
    ev = df[df["up100"] == 1].sort_values("t").drop_duplicates("symbol", keep="first")
    print(f"  монет, хоть раз давших +100% за сутки: {len(ev)}; первые срезы перед стартом (медианы):")
    base = df.sample(min(len(df), 200_000), random_state=0)
    cols = ["age_d", "adv_log", "fund24", "fund7d", "ret7d", "ret30d", "dd90", "max24_30d", "rv_pct", "vol_ratio",
            "oi3d", "lsr"]
    print(pd.DataFrame({"перед +100%": ev[cols].median(), "все срезы": base[cols].median()}).round(4).to_string())
    print("  по годам (монето-дни с up100):")
    u = df[df.up100 == 1]
    print(u.groupby(u["t"].dt.year).apply(lambda g: g.groupby(["symbol", g["t"].dt.floor("D")]).ngroups).to_string())


def lift_table(df: pd.DataFrame, feats: list[str]) -> None:
    is_ = period(df, "is")
    print("\n=== 2. Лифт признаков: P(up50) в квинтиле / базовая частота периода (границы — по IS) ===")
    rows = []
    for f in feats:
        x = is_[f].dropna()
        if x.nunique() < 5:
            continue
        edges = np.unique(np.quantile(x, [0, 0.2, 0.4, 0.6, 0.8, 1]))
        if len(edges) < 3:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        for p in PER:
            y = period(df, p)
            base = y["up50"].mean()
            g = y.groupby(pd.cut(y[f], edges, labels=False), observed=True)["up50"].mean() / base
            rows.append({"feature": f, "period": p, **{f"q{int(k) + 1}": v for k, v in g.items()}})
    print(pd.DataFrame(rows).round(2).to_string(index=False))


def _fit(x: pd.DataFrame, y: pd.Series):
    import lightgbm as lgb
    pos = max(float(y.mean()), 1e-6)
    params = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 500,
              "bagging_fraction": 0.7, "bagging_freq": 1, "feature_fraction": 0.8, "verbose": -1,
              "scale_pos_weight": min(50.0, (1 - pos) / pos)}
    return lgb.train(params, lgb.Dataset(x, y), num_boost_round=400)


def _trade_stats(x: pd.DataFrame, col: str) -> str:
    r = x[col].dropna().to_numpy(dtype="float64")
    if len(r) < 5:
        return f"n={len(r)}"
    day = x.loc[x[col].notna(), "t"].dt.floor("D").to_numpy()
    g = pd.Series(r - r.mean()).groupby(day).sum()
    t_day = r.sum() / np.sqrt((g ** 2).sum()) if (g ** 2).sum() > 0 else np.nan
    return f"n={len(r)} {r.mean():+.1%} (t_day {t_day:.1f}, win {np.mean(r > 0):.0%})"


def model(df: pd.DataFrame, feats: list[str], name: str) -> None:
    is_ = period(df, "is").sort_values("t").reset_index(drop=True)
    folds = np.array_split(np.arange(len(is_)), 4)
    oof = np.full(len(is_), np.nan)
    for k in range(1, 4):
        tr = np.concatenate(folds[:k])
        oof[folds[k]] = _fit(is_[feats].iloc[tr], is_["up50"].iloc[tr]).predict(is_[feats].iloc[folds[k]])
    m = _fit(is_[feats], is_["up50"])
    imp = pd.Series(m.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print(f"\n=== 3{name}. Модель up50: важность (gain) " +
          ", ".join(f"{k} {v / imp.sum():.0%}" for k, v in imp.head(10).items()))
    ok = ~np.isnan(oof)
    thr = {q: np.nanquantile(oof, 1 - q) for q in TOPS}
    rows = []
    for p in PER:
        base = is_[ok] if p == "is" else period(df, p)
        pred = oof[ok] if p == "is" else m.predict(base[feats])
        for q in TOPS:
            sel = base[pred >= thr[q]]
            row = {"period": p + (" OOF" if p == "is" else ""), "top": f"{q:.1%}", "n": len(sel),
                   "coins": sel["symbol"].nunique(), "days": sel["t"].dt.floor("D").nunique(),
                   "P(up50)": sel["up50"].mean(), "lift50": sel["up50"].mean() / base["up50"].mean(),
                   "P(up100)": sel["up100"].mean()}
            for s_, t_ in EXITS:
                row[f"стоп{int(s_ * 100)}/тейк{int(t_ * 100)}"] = _trade_stats(sel, f"tr_{int(s_ * 100)}_{int(t_ * 100)}")
            rows.append(row)
    out = pd.DataFrame(rows)
    print(out.round(4).to_string(index=False))


def rules(df: pd.DataFrame) -> None:
    print("\n=== 4. Правила, зафиксированные до запуска (лонг, одна сделка на монету в сутки) ===")
    r1 = df["fund24"] <= -0.002
    r2 = r1 & (df["oi3d"] > np.log(1.2))
    for name, mask in (("R1 funding 24ч <= -0.2%", r1), ("R2 R1 + OI 3д > +20%", r2)):
        x = df[mask].assign(day=lambda d: d["t"].dt.floor("D")).drop_duplicates(["symbol", "day"])
        for p in PER:
            y = period(x, p)
            base = period(df, p)["up50"].mean()
            print(f"  {name} | {p}: n={len(y)}, P(up50) {y['up50'].mean():.2%} (лифт {y['up50'].mean() / base:.1f}), "
                  f"P(up100) {y['up100'].mean():.2%} | " +
                  " | ".join(f"{s_:.0%}/{t_:.0%}: {_trade_stats(y, f'tr_{int(s_ * 100)}_{int(t_ * 100)}')}"
                             for s_, t_ in EXITS))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 40)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--metrics-symbols", required=True)
    a = ap.parse_args()
    msyms = set(a.metrics_symbols.split(","))
    df = collect(Path(a.root), a.symbols.split(","), msyms)
    print(f"===== MOONSHOTS: срезов {len(df):,}, монет {df['symbol'].nunique()} "
          f"(из них с метриками {df.loc[df.metrics, 'symbol'].nunique()}) =====")
    describe(df)
    lift_table(df, FEATS_B)
    model(df, FEATS_A, "A (все монеты, без OI)")
    model(df[df["metrics"]], FEATS_B, "B (725 монет с OI и LSR — выборка смещена к будущим пампам)")
    rules(df[df["metrics"]])
