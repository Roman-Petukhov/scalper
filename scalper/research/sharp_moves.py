"""
Резкие часовые движения: можно ли в момент закрытия свечи отличить продолжение от разворота.

Событие: часовая свеча с |лог-доходностью| > 3σ (σ — std часовых доходностей монеты за 30 прошлых дней), оборот монеты
>= $20 млн/день, не чаще раза в 4 часа на монету. d — направление свечи. Исходы (от close свечи события):
    fwd1/fwd4/fwd12   доходность в сторону d через 1/4/12 ч, б.п.
    cont              1 — цена прошла ещё |движение| в сторону d раньше, чем откатилась на |движение|; 0 — наоборот
                      (в пределах 12 ч, по часовым high/low; если оба в одном баре — считаем откат)
Признаки (всё известно на закрытии свечи; знаковые — умножены на d):
    move_z, vol_z (лог-объём к 30 дням), oi_z (изменение OI за свечу к истории), oi_up (OI в свече вырос),
    fund_d (последний funding × d), flow_d (перевес агрессоров в свече × d), btc_d (часовая доходность BTC / σ × d),
    breadth (сколько монет дали событие в тот же час), hour, ret24_d (доходность за 24 ч до свечи × d),
    close_pos (где close внутри диапазона свечи в сторону d: 1 — у экстремума), level_break (close за последней
    подтверждённой вершиной/впадиной n=10), age_d (дней с начала торгов), adv_log.
Анализ: 1) базовые частоты; 2) квинтили признаков на IS и их же границы на VAL/HOLDOUT (устойчивость);
3) LightGBM на IS (цель — fwd4, обрезанный по 1-99%), пороги — 10%/90% внефолдовых прогнозов IS (3 фолда по времени);
торговля: верхние 10% — по движению, нижние 10% — против, выход через 4 ч, издержки 12 б.п. на круг.

    python -m research.sharp_moves --root <binance 1h data with metrics> --symbols ...
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30, group_of
from .levels import sr_levels
from .wave2 import Data2

Z_MIN = 3.0
COOLDOWN = 4
COST = 12e-4
PER = {"is": ("2022-01-01", "2024-07-01"), "val": ("2024-07-01", "2025-07-01"), "ho": ("2025-07-01", "2026-10-01")}
FEATS = ["move_z", "vol_z", "oi_z", "oi_up", "fund_d", "flow_d", "btc_d", "breadth", "hour", "ret24_d", "close_pos",
         "level_break", "age_d", "adv_log"]


def coin_events(h: pd.DataFrame, adv: pd.Series, btc_z: pd.Series, sym: str) -> pd.DataFrame:
    c, hi, lo, o = h["close"], h["high"], h["low"], h["open"]
    lr = np.log(c).diff()
    sig = lr.rolling(720, min_periods=240).std().shift(1)
    z = lr / sig
    adv_now = adv.reindex(h.index, method="ffill")
    cand = np.flatnonzero(((z.abs() > Z_MIN) & (adv_now >= ADV_MIN)).to_numpy())
    if not len(cand):
        return pd.DataFrame()
    res, sup = sr_levels(hi.to_numpy(), lo.to_numpy(), 10)
    oi = np.log(h["oi"].replace(0, np.nan))
    doi = oi.diff()
    oi_z = (doi - doi.rolling(720, min_periods=240).mean()) / doi.rolling(720, min_periods=240).std()
    vol = np.log(h["volume"].replace(0, np.nan))
    vol_z = (vol - vol.rolling(720, min_periods=240).mean()) / vol.rolling(720, min_periods=240).std()
    flow = (2 * h["taker_buy_volume"] - h["volume"]) / h["volume"].replace(0, np.nan)
    fund_last = h["funding"].replace(0, np.nan).ffill()
    C, H, Lo, n = c.to_numpy(), hi.to_numpy(), lo.to_numpy(), len(c)
    rows, last = [], -10**9
    for e in cand:
        if e - last < COOLDOWN or e + 12 >= n or e < 24:
            continue
        last = e
        d = 1.0 if z.iloc[e] > 0 else -1.0
        move = abs(C[e] - C[e - 1])
        up_lvl, dn_lvl = C[e] + d * move, C[e] - d * move
        cont = np.nan
        for j in range(e + 1, e + 13):
            ext = (H[j] >= up_lvl) if d > 0 else (Lo[j] <= up_lvl)
            rev = (Lo[j] <= dn_lvl) if d > 0 else (H[j] >= dn_lvl)
            if rev:
                cont = 0.0
                break
            if ext:
                cont = 1.0
                break
        rng = H[e] - Lo[e]
        lvl = res[e] if d > 0 else sup[e]
        rows.append({
            "symbol": sym, "t": h.index[e], "d": d,
            "fwd1": d * (C[e + 1] / C[e] - 1) * 1e4, "fwd4": d * (C[e + 4] / C[e] - 1) * 1e4,
            "fwd12": d * (C[e + 12] / C[e] - 1) * 1e4, "cont": cont,
            "move_z": abs(z.iloc[e]), "vol_z": vol_z.iloc[e], "oi_z": oi_z.iloc[e], "oi_up": float(doi.iloc[e] > 0),
            "fund_d": fund_last.iloc[e] * d * 1e4, "flow_d": flow.iloc[e] * d, "btc_d": btc_z.asof(h.index[e]) * d,
            "hour": h.index[e].hour, "ret24_d": d * (C[e] / C[e - 24] - 1),
            "close_pos": ((C[e] - Lo[e]) / rng if d > 0 else (H[e] - C[e]) / rng) if rng > 0 else np.nan,
            "level_break": float(not np.isnan(lvl) and ((C[e] > lvl) if d > 0 else (C[e] < lvl))),
            "age_d": (h.index[e] - h.index[0]).days, "adv_log": np.log10(max(adv_now.iloc[e], 1.0)),
        })
    return pd.DataFrame(rows)


def collect(root: Path, syms: list[str]) -> pd.DataFrame:
    data = Data2(root, syms)
    b = data.get("BTCUSDT", "1h", "full")["close"]
    blr = np.log(b).diff()
    btc_z = blr / blr.rolling(720, min_periods=240).std().shift(1)
    parts = []
    for s in syms:
        try:
            h = data.get(s, "1h", "full")
        except Exception as e:
            print(f"  {s}: пропуск ({e})", flush=True)
            continue
        if h["oi"].notna().sum() < 24 * 60:
            continue
        ev = coin_events(h, adv30(root, s), btc_z, s)
        if len(ev):
            ev["group"] = group_of(s)
            parts.append(ev)
    ev = pd.concat(parts, ignore_index=True)
    ev["breadth"] = ev.groupby("t")["symbol"].transform("size")
    return ev


def period(df: pd.DataFrame, p: str) -> pd.DataFrame:
    a, b = PER[p]
    return df[(df["t"] >= a) & (df["t"] < b)]


def report_buckets(ev: pd.DataFrame) -> None:
    is_ = period(ev, "is")
    print("\nКвинтили признаков (границы — по IS); в ячейках: доля продолжений cont / средний fwd4 в сторону свечи, б.п.")
    for f in FEATS:
        x = is_[f].dropna()
        if x.nunique() < 3:
            continue
        edges = np.unique(np.quantile(x, [0, 0.2, 0.4, 0.6, 0.8, 1]))
        if len(edges) < 3:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        rows = []
        for p in PER:
            y = period(ev, p)
            q = pd.cut(y[f], edges)
            g = y.groupby(q, observed=True).agg(cont=("cont", "mean"), fwd4=("fwd4", "mean"), n=("fwd4", "size"))
            rows.append(g.apply(lambda r: f"{r['cont']:.0%}/{r['fwd4']:+.0f} ({int(r['n'])})", axis=1).rename(p))
        print(f"\n{f}:")
        print(pd.concat(rows, axis=1).to_string())


def model(ev: pd.DataFrame) -> None:
    import lightgbm as lgb
    params = {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 200,
              "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "verbose": -1}
    is_ = period(ev, "is").sort_values("t")
    lo, hi = is_["fwd4"].quantile([0.01, 0.99])
    y = is_["fwd4"].clip(lo, hi)
    oof = np.full(len(is_), np.nan)
    folds = np.array_split(np.arange(len(is_)), 4)
    for k in range(1, 4):                                       # предсказываем фолд k по фолдам до него
        tr = np.concatenate(folds[:k])
        m = lgb.LGBMRegressor(**params).fit(is_[FEATS].iloc[tr], y.iloc[tr])
        oof[folds[k]] = m.predict(is_[FEATS].iloc[folds[k]])
    q_lo, q_hi = np.nanquantile(oof, [0.1, 0.9])
    m = lgb.LGBMRegressor(**params).fit(is_[FEATS], y)
    imp = pd.Series(m.booster_.feature_importance("gain"), index=FEATS).sort_values(ascending=False)
    print("\nВажность признаков (gain): " + ", ".join(f"{k} {v / imp.sum():.0%}" for k, v in imp.head(8).items()))
    print(f"Пороги по внефолдовым прогнозам IS: нижние 10% < {q_lo:+.1f} б.п., верхние 10% > {q_hi:+.1f} б.п.")
    rows = []
    for p in PER:
        if p == "is":                                           # IS — только внефолдовые прогнозы
            ok = ~np.isnan(oof)
            base, pred = is_[ok], oof[ok]
        else:
            base = period(ev, p)
            pred = m.predict(base[FEATS])
        for name, mask, sign in (("по движению (верх 10%)", pred > q_hi, 1), ("против (низ 10%)", pred < q_lo, -1)):
            for grp, gm in (("все", np.ones(len(base), bool)), ("новые монеты", base["group"].isin(["ext54", "fresh"]).to_numpy())):
                x = sign * base["fwd4"].to_numpy()[mask & gm] - COST * 1e4
                rows.append({"period": p + (" (OOF)" if p == "is" else ""), "trade": name, "coins": grp, "n": len(x),
                             "bps": x.mean() if len(x) else np.nan,
                             "t": x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 30 else np.nan,
                             "win": float((x > 0).mean()) if len(x) else np.nan})
    print("\nТорговля по прогнозу модели (выход через 4 ч, за вычетом 12 б.п.):")
    print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 260)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--symbols", required=True)
    a = ap.parse_args()
    ev = collect(Path(a.root), a.symbols.split(","))
    print(f"===== SHARP MOVES: событий {len(ev)}, монет {ev['symbol'].nunique()} (|движение| > {Z_MIN}σ за час) =====")
    for p in PER:
        y = period(ev, p)
        print(f"  {p}: событий {len(y)}, продолжение {y['cont'].mean():.0%} (разворот {1 - y['cont'].mean():.0%}), "
              f"fwd1 {y['fwd1'].mean():+.1f}, fwd4 {y['fwd4'].mean():+.1f}, fwd12 {y['fwd12'].mean():+.1f} б.п.")
    for side, g in ev.groupby("d"):
        print(f"  {'вверх' if side > 0 else 'вниз '}: событий {len(g)}, продолжение {g['cont'].mean():.0%}, "
              f"fwd4 {g['fwd4'].mean():+.1f} б.п.")
    report_buckets(ev)
    model(ev)
