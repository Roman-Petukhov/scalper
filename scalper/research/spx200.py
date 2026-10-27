"""S&P 500 и Nasdaq у недельной EMA200 — зона накопления? Закрепление над EMA50 после неё — начало бычьего рынка?
Наблюдение владельца: «ниже EMA200 — недооценённость рынка, это работает не только на BTC, но и на S&P 500; пробой и
закрепление над EMA50 — рынок переходит в бычий».

Данные:
  W  недельные закрытия Yahoo (^GSPC с 1950, ^IXIC с 1971): EMA200 и EMA50 недель — ровно как у владельца на графике;
  M  месячные средние S&P 500 Роберта Шиллера с 1871 (github.com/datasets/s-and-p-500): EMA46 и EMA12 месяцев
     (≈ 200 и 50 недель) — 155 лет и все большие медвежьи рынки, включая 1929–1932 и 1973–1974.
Yahoo с машин GitHub может не ответить — тогда только M. Цена без дивидендов (у индекса +2…4% в год сверху).

На конце каждого месяца — отношение закрытия к EMA200 и доходность за 12 и 24 мес. Корзины: < 1 (ниже), 1–1.1,
1.1–1.25, > 1.25. Эпизоды «ниже EMA200»: начало, длина, сколько ещё падала цена, доходность через 12 / 24 мес.
Запуск — после закрытия под EMA200 два закрытия подряд над EMA50 (недели у W, месяцы у M).

Критерии (заданы до запуска; зона накопления — про горизонт 2 года, поэтому главный — 24 мес.):
  зона работает — «ниже» минус «остальные» за 24 мес. > 0 с t Ньюи–Уэста (лаг 24) >= 2, и за 12 мес. > 0;
  запуск работает — 12 мес. после запуска больше средней 12 мес. любого месяца, в плюсе >= 75% запусков; запусков
  меньше 10 — справка.

    python -m research.spx200          (нужен интернет)
"""
from __future__ import annotations

import io
import json
import sys
import time
import urllib.request
import warnings

import numpy as np
import pandas as pd

from .w200 import launches, nw_slope_t

SHILLER = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"
YAHOO = "https://query{n}.finance.yahoo.com/v8/finance/chart/{s}?period1=-1325635200&period2={end}&interval=1wk"
BINS = (0.0, 1.0, 1.1, 1.25, np.inf)
BIN_NAMES = ("< 1 (ниже)", "1–1.1", "1.1–1.25", "> 1.25")
HS = (12, 24)
EV_HS = (3, 6, 12, 24)
LAUNCH_K = 2
MODERN = 1950
SPANS = {"W": (200, 50), "M": (46, 12)}


def _get(url: str) -> bytes | None:
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read()
        except Exception as e:
            print(f"  spx200: {url[:80]} → {e}", flush=True)
            time.sleep(2 ** (attempt + 1))
    return None


def parse_shiller(raw: bytes) -> pd.Series:
    df = pd.read_csv(io.BytesIO(raw))
    s = pd.Series(df["SP500"].to_numpy(dtype="float64"), index=pd.to_datetime(df["Date"]) + pd.offsets.MonthEnd(0))
    return s[s > 0].sort_index()


def parse_yahoo(raw: bytes) -> pd.Series:
    r = json.loads(raw)["chart"]["result"][0]
    close = r["indicators"]["quote"][0]["close"]
    idx = pd.to_datetime(r["timestamp"], unit="s").normalize()
    s = pd.Series(close, index=idx, dtype="float64").dropna()
    s = s[s > 0]
    return s[~s.index.duplicated(keep="last")].sort_index()


def yahoo_weekly(sym: str) -> pd.Series:
    end = int(time.time())
    for n in (1, 2):
        raw = _get(YAHOO.format(n=n, s=urllib.request.quote(sym), end=end))
        if raw:
            try:
                s = parse_yahoo(raw)
                if len(s) > 400:
                    return s.iloc[:-1]                 # последняя неделя может быть не закрыта
            except (KeyError, IndexError, TypeError, ValueError) as e:
                print(f"  spx200: Yahoo {sym} не разобрался: {e}", flush=True)
    return pd.Series(dtype="float64")


def frame(close: pd.Series, slow: int, fast: int) -> pd.DataFrame:
    """Закрытие периода, EMA slow / fast (NaN до прогрева) и отношение к медленной."""
    w = pd.DataFrame({"close": close})
    for span, k in ((slow, "ema200"), (fast, "ema50")):
        e = close.ewm(span=span, adjust=False).mean()
        e[np.arange(len(close)) < span - 1] = np.nan
        w[k] = e
    w["ratio"] = w["close"] / w["ema200"]
    return w


def monthly(w: pd.DataFrame) -> pd.DataFrame:
    """На конце месяца: последнее закрытие периода и отношение к EMA200 по нему."""
    return w[["close", "ratio"]].resample("ME").last().dropna(subset=["close"])


def forward(close: pd.Series, h: int) -> pd.Series:
    return close.shift(-h) / close - 1


def at_horizon(close: pd.Series, t: pd.Timestamp, h: int) -> float:
    t1 = t + pd.DateOffset(months=h)
    return float(close.asof(t1) / close.asof(t) - 1) if t1 <= close.index[-1] else np.nan


def episodes(w: pd.DataFrame) -> pd.DataFrame:
    """Отрезки подряд ниже EMA200: начало, длина, минимум после начала до конца отрезка, доходность 12 / 24 мес."""
    z = w.dropna(subset=["ratio"])
    runs, start, prev = [], None, None
    for t, below in zip(z.index, z["ratio"] < 1):
        if below and start is None:
            start = t
        if not below and start is not None:
            runs.append((start, prev))
            start = None
        prev = t
    if start is not None:
        runs.append((start, z.index[-1]))
    out = []
    for a, b in runs:
        p0 = z.loc[a, "close"]
        seg = z.loc[a:b, "close"]
        r = {"начало": a.date(), "периодов": len(seg), "цена": p0, "ещё вниз": seg.min() / p0 - 1}
        for h in HS:
            r[f"{h} мес."] = at_horizon(z["close"], a, h)
        out.append(r)
    return pd.DataFrame(out)


def bucket_table(state: pd.DataFrame, rets: dict[int, pd.Series]) -> pd.DataFrame:
    z = state.dropna(subset=["ratio"]).copy()
    z["bin"] = pd.cut(z["ratio"], BINS, labels=BIN_NAMES, right=False)
    rows = []
    for b in BIN_NAMES:
        g = z[z.bin == b]
        row = {"к EMA200": b, "месяцев": len(g)}
        for h, r in rets.items():
            v = r.reindex(g.index).dropna()
            row[f"{h} мес."] = (f"{v.mean():+.0%} / мед. {v.median():+.0%} / в плюсе {np.mean(v > 0):.0%}"
                                if len(v) else "—")
        rows.append(row)
    return pd.DataFrame(rows)


def diff_t(state: pd.DataFrame, ret: pd.Series, lag: int) -> tuple[float, float, int]:
    z = pd.DataFrame({"x": (state["ratio"] < 1).astype(float), "y": ret}).dropna()
    z = z[state["ratio"].reindex(z.index).notna()]
    if z.x.sum() < 3 or len(z) < 24:
        return np.nan, np.nan, int(z.x.sum())
    d = z.y[z.x == 1].mean() - z.y[z.x == 0].mean()
    return float(d), nw_slope_t(z.x.to_numpy(), z.y.to_numpy(), lag), int(z.x.sum())


def launch_table(w: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in launches(w, LAUNCH_K):
        r = {"запуск": t.date(), "цена": w.loc[t, "close"]}
        for h in EV_HS:
            r[f"{h}м"] = at_horizon(w["close"], t, h)
        rows.append(r)
    return pd.DataFrame(rows)


def study(name: str, close: pd.Series, freq: str) -> list[str]:
    slow, fast = SPANS[freq]
    w = frame(close, slow, fast)
    st = monthly(w)
    rets = {h: forward(st["close"], h) for h in HS}
    unit = "недель" if freq == "W" else "месяцев"
    now = w.dropna().iloc[-1]
    print(f"\n  ===== {name} ===== {close.index[0].date()} … {close.index[-1].date()}, EMA{slow} / EMA{fast} {unit}; "
          f"сейчас {now.close:,.0f}, к EMA200 {now.ratio:.2f}, к EMA50 {now.close / now.ema50:.2f}")
    ep = episodes(w)
    fmt = {"цена": "{:,.2f}".format, "ещё вниз": "{:+.0%}".format, "12 мес.": "{:+.0%}".format,
           "24 мес.": "{:+.0%}".format}
    print(f"  эпизоды ниже EMA200 ({len(ep)}; «периодов» — {unit} подряд; «ещё вниз» — минимум после первого "
          f"закрытия ниже до конца эпизода):")
    print(ep.to_string(index=False, formatters=fmt) if len(ep) else "  не было")
    print("\n  --- на конце месяца → доходность (среднее / медиана / доля в плюсе) ---")
    print(bucket_table(st, rets).to_string(index=False))
    for h in HS:
        r = rets[h].dropna()
        print(f"  фон, любой месяц, {h} мес.: {r.mean():+.0%} / мед. {r.median():+.0%} / в плюсе {np.mean(r > 0):.0%}")
    if st.index[0].year < MODERN:
        m = st[st.index.year >= MODERN]
        print(f"\n  --- то же с {MODERN} (современный рынок) ---")
        print(bucket_table(m, {h: r.reindex(m.index) for h, r in rets.items()}).to_string(index=False))
    lt = launch_table(w)
    print(f"\n  --- запуски: после закрытия под EMA200 {LAUNCH_K} закрытия подряд над EMA50 ({len(lt)}) ---")
    print(lt.to_string(index=False, formatters={"цена": "{:,.2f}".format,
                                                **{f"{h}м": "{:+.0%}".format for h in EV_HS}}) if len(lt) else
          "  не было")

    verdicts = []
    d24, t24, n24 = diff_t(st, rets[24], 24)
    d12, t12, n12 = diff_t(st, rets[12], 12)
    ok = d24 > 0 and t24 >= 2 and d12 > 0
    verdicts.append(f"  {name}: «ниже» минус «остальные» 24 мес. {d24:+.0%} (t НУ {t24:.1f}, месяцев {n24}), 12 мес. "
                    f"{d12:+.0%} (t {t12:.1f}) → {'ПОДТВЕРЖДЕНО' if ok else 'не подтверждено'}")
    if st.index[0].year < MODERN:
        m = st[st.index.year >= MODERN]
        e24, u24, k24 = diff_t(m, rets[24].reindex(m.index), 24)
        verdicts.append(f"  {name}, с {MODERN}: «ниже» минус «остальные» 24 мес. {e24:+.0%} (t {u24:.1f}, месяцев {k24}) — "
                        "справка")
    if len(lt):
        x = lt["12м"].dropna()
        base = rets[12].dropna().mean()
        good = len(x) and x.mean() > base and np.mean(x > 0) >= 0.75
        tag = ("ПОДТВЕРЖДЕНО" if good else "не подтверждено") if len(x) >= 10 else \
            f"{'в пользу' if good else 'не в пользу'}, запусков < 10 — справка"
        verdicts.append(f"  {name}: запуск — 12 мес. {x.mean():+.0%} против фона {base:+.0%}, в плюсе "
                        f"{np.mean(x > 0):.0%} из {len(x)} → {tag}")
    return verdicts


def report() -> None:
    print("\n===== SPX200: S&P 500 / Nasdaq у недельной EMA200 — зона накопления? Запуск над EMA50? =====")
    verdicts: list[str] = []
    raw = _get(SHILLER)
    if raw:
        verdicts += study("S&P 500, месяцы Шиллера", parse_shiller(raw), "M")
    else:
        print("  Шиллер: данные не загрузились")
    for sym, name in (("^GSPC", "S&P 500, недели Yahoo"), ("^IXIC", "Nasdaq Composite, недели Yahoo")):
        s = yahoo_weekly(sym)
        if len(s):
            verdicts += study(name, s, "W")
        else:
            print(f"\n  {name}: Yahoo не ответил — пропуск")
    print("\n  Критерии (заданы до запуска):")
    for v in verdicts:
        print(v)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 300)
    report()
