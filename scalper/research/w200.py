"""BTC и ETH ниже недельной EMA200 — зона накопления? Наблюдение владельца: «когда BTC / ETH на неделях опускаются под
EMA200, это хорошая скидка для рынка; можно потихоньку присматривать перспективные альты».

Цены BTC и ETH — CoinMetrics community (PriceUSD, дневки: BTC с 2010, ETH с 2015), недельное закрытие — воскресенье;
EMA200 недель считается, когда у монеты >= 200 недель истории (BTC — с ~2014, ETH — с ~2019). Альты — спот Binance с
делистнутыми (части spotrev того же прогона): равновесная корзина монет с оборотом >= $1M и возрастом >= 180 дней,
кроме BTC и ETH; монета умерла — по последнему закрытию минус 2%.

На конце каждого месяца — отношение закрытия к недельной EMA200 (по последней закрытой неделе) и доходность за
следующие 12 и 24 месяца: самой монеты, корзины альтов и корзины сверх BTC. Корзины по отношению: < 1 (ниже EMA200),
1–1.5, 1.5–2.5, > 2.5. Плюс список эпизодов «ниже EMA200»: начало, конец, сколько ещё падала цена после первого
закрытия ниже, доходность через 12 и 24 месяца.

Критерии (заданы до запуска; эпизодов мало, поэтому вердикт с оговоркой о числе эпизодов):
  зона работает для BTC — 12 мес. после месяцев «ниже» больше, чем после остальных, t Ньюи–Уэста (лаг 12) >= 2;
  зона работает для альтов — корзина альтов за 12 мес. после «BTC ниже» больше, чем после остальных, и сверх BTC
  не хуже нуля.

    python -m research.w200 report        (нужны части spotrev и интернет)
"""
from __future__ import annotations

import sys
import time
import urllib.parse
import warnings

import numpy as np
import pandas as pd

from .onchainval import CM, get
from .spotrev import DEAD_HAIRCUT, MIN_AGE, load

SPAN = 200
BINS = (0.0, 1.0, 1.5, 2.5, np.inf)
BIN_NAMES = ("< 1 (ниже)", "1–1.5", "1.5–2.5", "> 2.5")
HS = (12, 24)
ADV_MIN = 1e6


def cm_price(asset: str) -> pd.Series:
    """Дневная цена CoinMetrics community (PriceUSD)."""
    q = urllib.parse.urlencode({"assets": asset, "metrics": "PriceUSD", "frequency": "1d", "start_time": "2010-01-01",
                                "page_size": 10000})
    url, rows = f"{CM}/timeseries/asset-metrics?{q}", []
    for _ in range(20):
        x = get(url)
        if not isinstance(x, dict):
            break
        rows += [(d["time"][:10], d.get("PriceUSD")) for d in x.get("data", [])]
        url = x.get("next_page_url")
        if not url:
            break
        time.sleep(1)
    s = pd.Series({pd.Timestamp(t): float(v) for t, v in rows if v is not None}, dtype="float64")
    return s.sort_index()


def weekly_ratio(daily: pd.Series) -> pd.DataFrame:
    """Недельное закрытие (воскресенье), EMA200 недель (NaN до 200 недель истории) и их отношение."""
    w = daily.resample("W-SUN").last().dropna()
    ema = w.ewm(span=SPAN, adjust=False).mean()
    ema[np.arange(len(w)) < SPAN - 1] = np.nan
    return pd.DataFrame({"close": w, "ema": ema, "ratio": w / ema})


def monthly_state(daily: pd.Series, wk: pd.DataFrame) -> pd.DataFrame:
    """На конце месяца: закрытие и отношение к EMA200 последней закрытой недели (неделя кончилась не позже дня)."""
    me = daily.resample("ME").last().dropna()
    r = wk["ratio"].reindex(wk.index.union(me.index)).ffill().reindex(me.index)
    return pd.DataFrame({"close": me, "ratio": r})


def forward(close: pd.Series, h: int) -> pd.Series:
    return close.shift(-h) / close - 1


def episodes(wk: pd.DataFrame, daily: pd.Series) -> pd.DataFrame:
    """Отрезки недель подряд ниже EMA200: начало, конец, мин. цена после начала до конца, доходность 12 / 24 мес."""
    below = (wk["ratio"] < 1).astype(int)
    rows = []
    start = None
    for t, b in zip(wk.index, below):
        if b and start is None:
            start = t
        if not b and start is not None:
            rows.append((start, prev))
            start = None
        prev = t
    if start is not None:
        rows.append((start, wk.index[-1]))
    out = []
    for a, b in rows:
        p0 = wk.loc[a, "close"]
        seg = daily[(daily.index > a) & (daily.index <= b + pd.Timedelta(days=6))]
        r = {"начало": a.date(), "конец": b.date(), "недель": int(((wk.index >= a) & (wk.index <= b)).sum()),
             "цена": p0, "ещё вниз": (seg.min() / p0 - 1) if len(seg) else 0.0}
        for h in HS:
            t1 = a + pd.DateOffset(months=h)
            r[f"{h} мес."] = daily.asof(t1) / p0 - 1 if t1 <= daily.index[-1] else np.nan
        out.append(r)
    return pd.DataFrame(out)


def alt_forward(c_end: pd.DataFrame, adv: pd.DataFrame, meta: pd.DataFrame, h: int) -> pd.Series:
    """Равновесная корзина альтов: доходность от конца месяца m до конца m+h; умершая монета — последнее закрытие −2%."""
    alts = [c for c in c_end.columns if c not in ("BTCUSDT", "ETHUSDT")]
    px, liq = c_end[alts], adv.reindex_like(c_end)[alts]
    filled = px.ffill()
    out = {}
    months = list(px.index)
    for i, m in enumerate(months[:-h] if h < len(months) else []):
        end = months[i + h]
        age_ok = (m - meta.reindex(alts)["first"]) >= MIN_AGE
        ok = (liq.loc[m] >= ADV_MIN) & px.loc[m].notna() & age_ok.to_numpy()
        if ok.sum() < 20:
            continue
        p0, p1 = px.loc[m, ok], filled.loc[end, ok]
        dead = meta.reindex(p0.index)["last"] < end
        r = p1 / p0 - 1
        r[dead.to_numpy()] = (1 + r[dead.to_numpy()]) * (1 - DEAD_HAIRCUT) - 1
        out[m] = r.mean()
    return pd.Series(out, dtype="float64")


def bucket_table(state: pd.DataFrame, cols: dict[str, pd.Series], h: int) -> pd.DataFrame:
    z = pd.DataFrame({"ratio": state["ratio"], **cols}).dropna(subset=["ratio"])
    z["bin"] = pd.cut(z["ratio"], BINS, labels=BIN_NAMES, right=False)
    rows = []
    for b in BIN_NAMES:
        g = z[z.bin == b]
        row = {"отношение к EMA200": b, "месяцев": len(g)}
        for k in cols:
            v = g[k].dropna()
            row[k] = f"{v.mean():+.0%} / мед. {v.median():+.0%} ({len(v)})" if len(v) else "—"
        rows.append(row)
    return pd.DataFrame(rows)


def diff_t(state: pd.DataFrame, ret: pd.Series, h: int) -> tuple[float, float, int]:
    """Разница средних «ниже» минус «остальные» и t Ньюи–Уэста (регрессия на признак, лаг h)."""
    z = pd.DataFrame({"x": (state["ratio"] < 1).astype(float), "y": ret}).dropna()
    z = z[state["ratio"].reindex(z.index).notna()]
    if z.x.sum() < 3 or len(z) < 24:
        return np.nan, np.nan, int(z.x.sum())
    d = z.y[z.x == 1].mean() - z.y[z.x == 0].mean()
    return float(d), nw_slope_t(z.x.to_numpy(), z.y.to_numpy(), h), int(z.x.sum())


def nw_slope_t(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    """t наклона регрессии y на x с ошибками Ньюи–Уэста (наклон на признак 0/1 = разница средних)."""
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    xx_inv = np.linalg.inv(X.T @ X)
    g = X * e[:, None]
    s = g.T @ g
    for k in range(1, min(lag, len(x) - 1) + 1):
        w = 1 - k / (lag + 1)
        c = g[k:].T @ g[:-k]
        s += w * (c + c.T)
    v = xx_inv @ s @ xx_inv
    return float(beta[1] / np.sqrt(v[1, 1])) if v[1, 1] > 0 else float("nan")


def report() -> None:
    print("\n===== W200: BTC / ETH ниже недельной EMA200 — зона накопления? (CoinMetrics с 2010 / 2015; альты — спот "
          "Binance с делистнутыми) =====")
    prices = {a: cm_price(a) for a in ("btc", "eth")}
    for a, s in prices.items():
        print(f"  {a.upper()}: дней {len(s)}" + (f" ({s.index[0]:%Y-%m-%d} … {s.index[-1]:%Y-%m-%d})" if len(s) else ""))
    if prices["btc"].empty:
        print("  CoinMetrics не ответил")
        return
    got = load()
    alt = {}
    if got is not None:
        c_end, _, adv, meta, _, _ = got
        c_end.index = c_end.index.to_period("M").to_timestamp("M")
        adv.index = c_end.index
        for h in HS:
            alt[h] = alt_forward(c_end, adv, meta, h)
        print(f"  альты: месяцев с корзиной {len(alt[12])} ({alt[12].index.min():%Y-%m} … {alt[12].index.max():%Y-%m})")
    else:
        print("  нет частей spotrev — альтов не будет")
    verdicts = []
    for a, daily in prices.items():
        if daily.empty:
            continue
        wk = weekly_ratio(daily)
        st = monthly_state(daily, wk)
        last = wk.dropna().iloc[-1]
        print(f"\n  ===== {a.upper()} ===== сейчас: неделя {wk.dropna().index[-1]:%Y-%m-%d}, закрытие {last.close:,.0f}, "
              f"EMA200 {last.ema:,.0f}, отношение {last.ratio:.2f}")
        ep = episodes(wk.dropna(), daily)
        print(f"  эпизоды ниже EMA200 (недель подряд; «ещё вниз» — минимум после первого закрытия ниже до конца эпизода):")
        if len(ep):
            fmt = ep.copy()
            for k in ("ещё вниз", *[f"{h} мес." for h in HS]):
                fmt[k] = fmt[k].map(lambda v: f"{v:+.0%}" if np.isfinite(v) else "—")
            fmt["цена"] = fmt["цена"].map(lambda v: f"{v:,.2f}")
            print(fmt.to_string(index=False))
        else:
            print("  не было")
        for h in HS:
            cols = {f"{a.upper()} {h}м": forward(st["close"], h)}
            if h in alt:
                al = alt[h].reindex(st.index)
                cols[f"альты {h}м"] = al
                if a == "btc":
                    cols[f"альты−BTC {h}м"] = al - forward(st["close"], h)
            print(f"\n  --- на конце месяца → доходность за {h} мес. (среднее / медиана (месяцев)) ---")
            print(bucket_table(st, cols, h).to_string(index=False))
            if h == 12:
                d, t, n = diff_t(st, cols[f"{a.upper()} 12м"], 12)
                verdicts.append(f"  {a.upper()}: «ниже» минус «остальные» за 12 мес. {d:+.0%}, t НУ {t:.1f}, месяцев "
                                f"«ниже» {n} → {'работает' if np.isfinite(t) and d > 0 and t >= 2 else 'не подтверждено'}")
                if a == "btc" and 12 in alt:
                    d2, t2, n2 = diff_t(st, alt[12].reindex(st.index), 12)
                    ex = (alt[12].reindex(st.index) - forward(st["close"], 12))[st["ratio"] < 1].dropna()
                    good = np.isfinite(d2) and d2 > 0 and len(ex) and ex.mean() >= 0
                    verdicts.append(f"  альты при BTC ниже: корзина «ниже» минус «остальные» {d2:+.0%} (t {t2:.1f}, "
                                    f"месяцев {n2}), сверх BTC в среднем {ex.mean() if len(ex) else np.nan:+.0%} → "
                                    f"{'работает' if good else 'не подтверждено'}")
    print("\n  Критерии (заданы до запуска):")
    for v in verdicts:
        print(v)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 300)
    report()
