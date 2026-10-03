"""
Волна 2: гипотезы о позиционировании толпы (funding, открытый интерес, long/short ratio).

Протокол зафиксирован до запуска (волна 1 уже «потратила» 2024-07..2026-09 на своих финалистов):
    IS        2022-01 .. 2024-06   перебор и выбор финалистов (лучшие 2 конфига на семейство)
    VAL       2024-07 .. 2025-06   ворота: Sharpe >= VAL_MIN_SHARPE при 6 б.п. и > 0 при 10 б.п.
    HOLDOUT   2025-07 .. 2026-09   один прогон только для прошедших ворота VAL

Отдельно: проверка устойчивости выжившего в волне 1 funding_extreme (все конфиги, по годам и монетам).

    python -m research.wave2 --root <data> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import strategies as S
from .data import UNIVERSE
from .engine import BAR_MINUTES, _to_utc, metrics, run, state_machine
from .search import COST_BPS, STRESS_COST_BPS, Data, eval_panel, grid

IS_START, IS_END, VAL_END, HO_END = "2022-01-01", "2024-07-01", "2025-07-01", "2026-10-01"
VAL_MIN_SHARPE = 0.5
MIN_TRADES_IS = 200
TOP_PER_FAMILY = 2
METRIC_COLS = {"sum_open_interest": "oi", "sum_toptrader_long_short_ratio": "top_lsr",
               "count_long_short_ratio": "lsr", "sum_taker_long_short_vol_ratio": "taker_lsr"}
PERIODS = {"is": (IS_START, IS_END), "val": (IS_END, VAL_END), "ho": (VAL_END, HO_END),
           "oos1": (IS_END, HO_END), "full": (IS_START, HO_END)}


# ---------- данные ----------

def metrics_on_bars(symbol: str, root: Path, index: pd.DatetimeIndex, bar_min: int) -> pd.DataFrame:
    """Снимок метрик на закрытии бара. Снимок с меткой T считаем известным в T + 5 мин (консервативно);
    берём последний известный не старше часа, иначе NaN."""
    out = pd.DataFrame(index=index, columns=list(METRIC_COLS.values()), dtype="float64")
    p = Path(root) / f"{symbol}-metrics.parquet"
    if not p.exists():
        return out
    m = pd.read_parquet(p)
    known = pd.DataFrame({"t": _to_utc(m["ts"]) + pd.Timedelta(minutes=5)})
    for src, dst in METRIC_COLS.items():
        known[dst] = m[src].to_numpy(dtype="float64")
    known["t"] = known["t"].dt.as_unit("ms")
    known = known.dropna(subset=["t"]).sort_values("t").drop_duplicates("t", keep="last")
    bars = pd.DataFrame({"t": (index + pd.Timedelta(minutes=bar_min)).as_unit("ms")})
    j = pd.merge_asof(bars, known, on="t", direction="backward", tolerance=pd.Timedelta("1h"))
    return pd.DataFrame(j.drop(columns="t").to_numpy(), index=index, columns=list(METRIC_COLS.values()))


class Data2(Data):
    def get(self, sym: str, tf: str, period: str) -> pd.DataFrame:
        key = (sym, tf, "m")
        if key not in self.cache:
            base = super().get(sym, tf, "full")
            self.cache[key] = pd.concat([base, metrics_on_bars(sym, self.root, base.index, BAR_MINUTES[tf])], axis=1)
        df = self.cache[key]
        if period == "full":
            return df
        a, b = PERIODS[period]
        return df[(df.index >= a) & (df.index < b)]


def _cut(s: pd.Series | pd.DataFrame, period: str):
    a, b = PERIODS[period]
    return s[(s.index >= a) & (s.index < b)]


# ---------- гипотезы ----------

def _z(x: pd.Series, win: int) -> pd.Series:
    mu = x.rolling(win, min_periods=win // 3).mean()
    sd = x.rolling(win, min_periods=win // 3).std()
    return (x - mu) / sd


def _price_z(c: pd.Series, k: int, volwin: int = 24 * 30) -> pd.Series:
    return np.log(c / c.shift(k)) / (S._vol(c, volwin) * np.sqrt(k))


def _zeros(n: int) -> np.ndarray:
    return np.zeros(n, np.bool_)


def fe_price(df, q=0.95, hold=8, win_days=30, rz=0.0):
    """funding_extreme + цена уже ушла в сторону толпы за 24 часа (толпа догоняет -> разворот)."""
    f = df["funding"].replace(0, np.nan).ffill()
    w = win_days * 24
    hi = f.rolling(w, min_periods=w // 3).quantile(q).shift(1)
    lo = f.rolling(w, min_periods=w // 3).quantile(1 - q).shift(1)
    paid = df["funding"] != 0
    r24 = _price_z(df["close"], 24)
    es = S._b(paid & (f > hi) & (f > 0) & (r24 > rz))
    el = S._b(paid & (f < lo) & (r24 < -rz))
    return state_machine(el, es, _zeros(len(df)), _zeros(len(df)), hold), None


def _settle_window(df, f_known: pd.Series, qa: float, sign: int, offsets: range) -> np.ndarray:
    """Позиция -sign(f)*sign на барах s+o (o из offsets) вокруг каждой выплаты s.
    Расписание выплат детерминировано и известно заранее; ставка берётся известная на момент входа."""
    n = len(df)
    paid = np.flatnonzero(df["funding"].to_numpy() != 0)
    fa = f_known.abs()
    thr = fa.rolling(24 * 30, min_periods=24 * 10).quantile(qa).shift(1) if qa > 0 else fa * 0
    fk, th = f_known.to_numpy(), thr.to_numpy()
    pos = np.zeros(n)
    first = min(offsets)
    for s in paid:
        t0 = s + first
        if t0 < 0 or not np.isfinite(fk[t0]) or not np.isfinite(th[t0]) or fk[t0] == 0 or abs(fk[t0]) < th[t0]:
            continue
        d = -np.sign(fk[t0]) * sign
        for o in offsets:
            t = s + o
            if 0 <= t < n:
                pos[t] = d
    return pos


def prefund(df, pre_h=2, sign=1, qa=0.0):
    """Дрейф перед выплатой: держим бары, доходности которых приходятся на pre_h часов до выплаты,
    и выходим до неё (funding не платим). sign=+1: против стороны, которая платит."""
    f = df["funding"].replace(0, np.nan).ffill().shift(1)   # последняя ставка, известная до бара
    return _settle_window(df, f, qa, sign, range(-pre_h - 1, -1)), None


def postfund(df, post_h=2, sign=1, qa=0.0):
    """Дрейф после выплаты: держим через выплату и post_h часов после (funding учитывается)."""
    f = df["funding"].replace(0, np.nan).ffill().shift(1)
    return _settle_window(df, f, qa, sign, range(-1, post_h - 1)), None


def oi_price(df, k=4, thr=2.0, hold=12, mode="liq", sign=1, win=24 * 30):
    """liq: резкое движение цены при падении OI (вынужденное закрытие) -> против движения.
    crowd: резкое движение при росте OI (новые позиции догоняют) -> против движения.
    sign=-1 переворачивает в сторону движения."""
    oi = df["oi"].replace(0, np.nan)
    doi = _z(np.log(oi / oi.shift(k)), win)
    dp = _price_z(df["close"], k)
    gate = (doi < -thr) if mode == "liq" else (doi > thr)
    el = (gate & (dp < -thr)) if sign > 0 else (gate & (dp > thr))
    es = (gate & (dp > thr)) if sign > 0 else (gate & (dp < -thr))
    return state_machine(S._b(el), S._b(es), _zeros(len(df)), _zeros(len(df)), hold), None


def lsr_extreme(df, col="top_lsr", thr=2.0, hold=24, sign=1, win=24 * 30):
    """Перекос long/short ratio относительно своей истории. sign=+1: против толпы."""
    x = np.log(df[col].where(df[col] > 0))
    z = _z(x, win) * sign
    return state_machine(S._b(z < -thr), S._b(z > thr), _zeros(len(df)), _zeros(len(df)), hold), None


SINGLE = {
    "fe_price": [("1h", fe_price, grid(q=[0.9, 0.95, 0.99], hold=[4, 8, 16], win_days=[30, 90], rz=[0.0, 1.0]))],
    "prefund": [("1h", prefund, grid(pre_h=[1, 2, 3, 4], sign=[1, -1], qa=[0.0, 0.5, 0.8]))],
    "postfund": [("1h", postfund, grid(post_h=[1, 2, 4], sign=[1, -1], qa=[0.0, 0.5, 0.8]))],
    "oi_price": [("1h", oi_price, grid(k=[1, 4, 12], thr=[1.5, 2.0, 2.5], hold=[4, 12, 24], mode=["liq", "crowd"],
                                       sign=[1, -1]))],
    "lsr_extreme": [("1h", lsr_extreme, grid(col=["top_lsr", "lsr", "taker_lsr"], thr=[1.5, 2.0, 2.5],
                                             hold=[8, 24, 72], sign=[1, -1]))],
}


def fe_xs_weights(funding: pd.DataFrame, k: int, hold: int, win_ev: int) -> pd.DataFrame:
    """Рыночно-нейтральный funding_extreme: в часы 00/08/16 UTC шорт k монет с самым высоким funding
    относительно своей истории (z по последним win_ev выплатам), лонг k с самым низким; держим hold часов."""
    z = {}
    for s in funding.columns:
        ev = funding[s][funding[s] != 0].dropna()
        ze = (ev - ev.rolling(win_ev, min_periods=win_ev // 3).mean().shift(1)) / \
            ev.rolling(win_ev, min_periods=win_ev // 3).std().shift(1)
        z[s] = ze.reindex(funding.index).ffill()
    z = pd.DataFrame(z, index=funding.index)
    reb = np.isin(funding.index.hour, (0, 8, 16))
    rank = z.rank(axis=1, ascending=False)
    n = z.notna().sum(axis=1)
    w = pd.DataFrame(0.0, index=z.index, columns=z.columns)
    w[rank <= k] = -0.5 / k
    w[rank.gt(n - k, axis=0) & z.notna()] = 0.5 / k
    w[n < 2 * k + 2] = 0.0
    w = w.where(pd.Series(reb, index=w.index), np.nan).ffill().fillna(0.0)
    grp = np.cumsum(reb)
    age = pd.Series(grp).groupby(grp).cumcount().to_numpy()   # баров с последней ребалансировки
    w.loc[age >= hold] = 0.0
    return w


def panel_configs():
    for p in grid(k=[1, 2, 3], hold=[4, 8], win_ev=[45, 90, 270]):
        yield "fe_xs", "1h", p


# ---------- оценка ----------

def eval_single(data: Data2, tf: str, fn, params: dict, period: str, cost: float, delay: int = 0):
    """delay: вход/выход на delay баров позже сигнала (проверка чувствительности к исполнению)."""
    pnls, poss = {}, {}
    for s in data.symbols:
        df = data.get(s, tf, "is" if period == "is" else "full")
        pos, ex = fn(df, **params)
        pos = np.asarray(pos, dtype=float)
        if delay:
            pos, ex = np.concatenate([np.zeros(delay), pos[:-delay]]), None
        res = run(df, pos, cost, BAR_MINUTES[tf], ex)
        pnls[s], poss[s] = _cut(res.pnl, period), _cut(res.pos, period)
    port = pd.DataFrame(pnls).fillna(0.0).mean(axis=1)
    per_sym = {s: metrics(p)["sharpe"] for s, p in pnls.items() if len(p)}
    trades = sum(metrics(pnls[s], poss[s]).get("trades", 0) for s in data.symbols if len(pnls[s]))
    gross = pd.DataFrame(poss).abs().fillna(0.0).mean(axis=1)
    return port, per_sym, trades, gross


def eval_panel_period(data: Data2, family: str, tf: str, p: dict, period: str, cost: float):
    full = "is" if period == "is" else "full"
    closes = pd.DataFrame({s: data.get(s, tf, full)["close"] for s in data.symbols})
    fund = pd.DataFrame({s: data.get(s, tf, full)["funding"] for s in data.symbols})
    w = fe_xs_weights(fund, p["k"], p["hold"], p["win_ev"])
    pnl, reb = eval_panel(w, closes, fund, cost)
    return _cut(pnl, period), int((_cut(w.diff().abs().sum(axis=1), period) > 1e-9).sum()), _cut(w.abs().sum(axis=1), period)


def summarize(port: pd.Series, trades: int, per_sym: dict | None, gross: pd.Series) -> dict:
    m = metrics(port)
    out = {k: m[k] for k in ("sharpe", "ann_ret", "ann_vol", "max_dd", "pos_days")}
    out["trades"] = trades
    out["trades_per_day"] = trades / max(m["days"], 1)
    out["avg_gross"] = float(gross.mean())
    out["bps_per_trade"] = float(port.sum() / max(trades, 1) * 1e4 * len(per_sym or [1]))
    if per_sym:
        v = np.array(list(per_sym.values()))
        out["sym_pos_frac"] = float((v > 0).mean())
    return out


def evaluate(data: Data2, family: str, tf: str, params: dict, period: str, cost: float,
             families: dict | None = None, delay: int = 0):
    fams = families if families is not None else SINGLE
    if family in fams:
        fn = next(fn for t, fn, _ in fams[family] if t == tf)
        port, per_sym, trades, gross = eval_single(data, tf, fn, params, period, cost, delay)
    elif family == "funding_extreme":
        port, per_sym, trades, gross = eval_single(data, tf, S.funding_extreme, params, period, cost, delay)
    else:
        port, trades, gross = eval_panel_period(data, family, tf, params, period, cost)
        per_sym = None
    return port, summarize(port, trades, per_sym, gross), per_sym


# ---------- этапы ----------

def search(data: Data2, out: Path, families: dict | None = None, panel=None, tag: str = "wave2") -> pd.DataFrame:
    fams = families if families is not None else SINGLE
    panel = panel if panel is not None else panel_configs
    rows, t0 = [], time.time()
    for fam, specs in fams.items():
        for tf, _, g in specs:
            for p in g:
                _, summ, _ = evaluate(data, fam, tf, p, "is", COST_BPS, fams)
                rows.append({"family": fam, "tf": tf, "params": json.dumps(p), **summ})
        print(f"  {fam}: готово ({time.time() - t0:.0f}s)", flush=True)
    for fam, tf, p in panel():
        _, summ, _ = evaluate(data, fam, tf, p, "is", COST_BPS, fams)
        rows.append({"family": fam, "tf": tf, "params": json.dumps(p), **summ})
    print(f"  панельные: готово ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / f"{tag}_is.csv", index=False)
    return df


def select(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["trades"] >= MIN_TRADES_IS].copy()
    df["family_median_sharpe"] = df.groupby("family")["sharpe"].transform("median")
    best = df.sort_values("sharpe", ascending=False).groupby("family").head(TOP_PER_FAMILY)
    return best.sort_values("sharpe", ascending=False)


def gate(data: Data2, fin: pd.DataFrame, out: Path, families: dict | None = None, tag: str = "wave2") -> pd.DataFrame:
    rows = []
    for _, f in fin.iterrows():
        p = json.loads(f["params"])
        r = {"family": f["family"], "tf": f["tf"], "params": f["params"], "is_sharpe": f["sharpe"]}
        for cost in (COST_BPS, STRESS_COST_BPS):
            _, s, _ = evaluate(data, f["family"], f["tf"], p, "val", cost, families)
            r[f"val_sharpe_{int(cost)}"] = s["sharpe"]
            r[f"val_ret_{int(cost)}"] = s["ann_ret"]
            r[f"val_dd_{int(cost)}"] = s["max_dd"]
        r["passed"] = bool(r["val_sharpe_6"] >= VAL_MIN_SHARPE and r["val_sharpe_10"] > 0)
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(out / f"{tag}_val.csv", index=False)
    return res


def holdout(data: Data2, passed: pd.DataFrame, out: Path, families: dict | None = None,
            tag: str = "wave2") -> pd.DataFrame:
    rows = []
    for _, f in passed.iterrows():
        p = json.loads(f["params"])
        for cost in (COST_BPS, STRESS_COST_BPS):
            port, s, _ = evaluate(data, f["family"], f["tf"], p, "ho", cost, families)
            rows.append({"family": f["family"], "params": f["params"], "cost_bps": cost, **s,
                         "monthly_pos_frac": float((port.resample("ME").sum() > 0).mean())})
    res = pd.DataFrame(rows)
    res.to_csv(out / f"{tag}_holdout.csv", index=False)
    return res


def wave1_robustness(data: Data2, out: Path) -> None:
    """Все конфиги funding_extreme из сетки волны 1: IS и OOS волны 1, по годам и по монетам для лучшего."""
    rows = []
    g = list(grid(q=[0.9, 0.95, 0.99], hold=[8, 24, 72], win_days=[30, 90]))
    for p in g:
        r = {"params": json.dumps(p)}
        for per in ("is", "oos1"):
            for cost in (COST_BPS, STRESS_COST_BPS):
                _, s, _ = evaluate(data, "funding_extreme", "1h", p, per, cost)
                r[f"{per}_sharpe_{int(cost)}"] = s["sharpe"]
                r[f"{per}_ret_{int(cost)}"] = s["ann_ret"]
                if per == "oos1" and cost == COST_BPS:
                    r.update({"oos1_dd_6": s["max_dd"], "oos1_trades": s["trades"],
                              "oos1_avg_gross": s["avg_gross"], "oos1_bps_per_trade": s["bps_per_trade"]})
        rows.append(r)
    rob = pd.DataFrame(rows)
    rob.to_csv(out / "fe_robustness.csv", index=False)
    print("\nFUNDING_EXTREME: все 18 конфигов волны 1 (IS -> OOS 2024-07..2026-09)")
    print(rob.round(3).to_string(index=False))

    best = {"q": 0.99, "hold": 8, "win_days": 30}
    port, s, per_sym = evaluate(data, "funding_extreme", "1h", best, "full", COST_BPS)
    yearly = pd.DataFrame({"ret": port.groupby(port.index.year).sum(),
                           "sharpe": port.groupby(port.index.year).apply(lambda x: metrics(x)["sharpe"])})
    print(f"\nFUNDING_EXTREME {best}: по годам (6 б.п.)\n{yearly.round(3).to_string()}")
    _, _, per_sym_oos = evaluate(data, "funding_extreme", "1h", best, "oos1", COST_BPS)
    print("\nпо монетам, Sharpe OOS: " + ", ".join(f"{k[:-4]} {v:.2f}" for k, v in sorted(per_sym_oos.items(),
                                                                                       key=lambda kv: -kv[1])))
    port.resample("1D").sum().to_frame("pnl").to_csv(out / "fe_best_daily.csv")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--symbols", default=",".join(UNIVERSE))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Data2(Path(a.root), a.symbols.split(","))
    pd.set_option("display.width", 250)

    wave1_robustness(data, out)

    res = search(data, out)
    print(f"\nВолна 2, всего конфигураций: {len(res)}")
    print(res.groupby("family")["sharpe"].describe()[["count", "50%", "max"]].round(2).to_string())
    fin = select(res)
    print("\nФИНАЛИСТЫ (IS):")
    print(fin[["family", "params", "sharpe", "ann_ret", "max_dd", "trades_per_day", "avg_gross",
               "bps_per_trade", "sym_pos_frac", "family_median_sharpe"]].round(3).to_string(index=False))
    val = gate(data, fin, out)
    print(f"\nVAL 2024-07..2025-06 (ворота: Sharpe>={VAL_MIN_SHARPE} при 6 б.п. и >0 при 10 б.п.):")
    print(val.round(3).to_string(index=False))
    passed = val[val["passed"]]
    if len(passed):
        ho = holdout(data, passed, out)
        print("\nHOLDOUT 2025-07..2026-09 (один прогон):")
        print(ho.round(3).to_string(index=False))
    else:
        print("\nНи один финалист не прошёл VAL — holdout не трогаем.")
