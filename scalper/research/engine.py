"""
Векторный бэктест бар-стратегий.

Соглашения (без заглядывания в будущее):
- позиция pos[t] решается по данным, известным на закрытии бара t;
- она зарабатывает доходность следующего бара: r[t+1] = close[t+1] / close[t] - 1;
- издержки: cost_bps за каждую единицу изменения позиции (комиссия тейкера + проскальзывание);
- funding: каждое событие funding внутри бара t+1 списывает pos[t] * rate.
Позиция в долях капитала: +1 = лонг на весь капитал без плеча.
"""
from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

# кеш numba в коротком пути: в глубоких папках Windows упирается в лимит 260 символов
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba_cache"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numba import njit  # noqa: E402

BAR_MINUTES = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}


# ---------- данные ----------

def _to_utc(ts: pd.Series) -> pd.DatetimeIndex:
    """Эпоха в мс или мкс (Binance местами перешёл на микросекунды) -> UTC-индекс."""
    v = ts.to_numpy(dtype="int64")
    unit = "us" if len(v) and v.max() > 10**14 else "ms"
    return pd.DatetimeIndex(pd.to_datetime(v, unit=unit, utc=True), name=ts.name)


def load(symbol: str, root: Path, interval: str = "5m") -> pd.DataFrame:
    src = "5m"
    p = Path(root) / f"{symbol}-5m.parquet"
    if not p.exists() and BAR_MINUTES[interval] >= 60 and (Path(root) / f"{symbol}-1h.parquet").exists():
        src, p = "1h", Path(root) / f"{symbol}-1h.parquet"       # для часовых стратегий хватает часовых свечей
    df = pd.read_parquet(p)
    df.index = _to_utc(df["open_time"])
    df = df.drop(columns=["open_time"])
    df = df[~df.index.duplicated()].sort_index()
    if interval != src:
        df = resample(df, interval)
    df["funding"] = funding_on_bars(symbol, root, df.index, BAR_MINUTES[interval])
    return df


def resample(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    rule = {"15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1D"}[interval]
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
           "quote_volume": "sum", "count": "sum", "taker_buy_volume": "sum",
           "taker_buy_quote_volume": "sum"}
    return df.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["close"])


def funding_on_bars(symbol: str, root: Path, index: pd.DatetimeIndex, bar_min: int) -> np.ndarray:
    """Сумма ставок funding, выплаченных внутри каждого бара."""
    p = Path(root) / f"{symbol}-funding.parquet"
    out = np.zeros(len(index))
    if not p.exists():
        return out
    f = pd.read_parquet(p)
    ts = _to_utc(f["ts"]).floor(f"{bar_min}min")
    s = pd.Series(f["rate"].to_numpy(), index=ts).groupby(level=0).sum()
    return s.reindex(index).fillna(0.0).to_numpy()


# ---------- позиции из сигналов ----------

@njit(cache=True)
def state_machine(enter_long, enter_short, exit_long, exit_short, max_hold):
    """Стейтфул-позиция: вход по сигналу, выход по сигналу выхода или по таймеру (0 = без таймера)."""
    n = len(enter_long)
    pos = np.zeros(n)
    cur = 0.0
    held = 0
    for t in range(n):
        if cur > 0:
            held += 1
            if exit_long[t] or (max_hold > 0 and held >= max_hold) or enter_short[t]:
                cur = 0.0
        elif cur < 0:
            held += 1
            if exit_short[t] or (max_hold > 0 and held >= max_hold) or enter_long[t]:
                cur = 0.0
        if cur == 0.0:
            if enter_long[t]:
                cur = 1.0
                held = 0
            elif enter_short[t]:
                cur = -1.0
                held = 0
        pos[t] = cur
    return pos


@njit(cache=True)
def stop_target_machine(enter_long, enter_short, high, low, close, stop_dist, tp_dist, max_hold):
    """Вход по закрытию сигнального бара, выход по стопу/тейку внутри следующих баров или по таймеру.
    Возвращает позицию и цену выхода (для пересчёта доходности бара выхода).
    Если в одном баре задеты и стоп, и тейк, считаем, что сработал стоп (консервативно)."""
    n = len(close)
    pos = np.zeros(n)
    exit_px = np.full(n, np.nan)
    cur = 0.0
    entry = 0.0
    sd = 0.0
    td = 0.0
    held = 0
    for t in range(n):
        if cur != 0.0:
            held += 1
            hit = False
            if cur > 0:
                if low[t] <= entry - sd:
                    exit_px[t] = entry - sd
                    hit = True
                elif high[t] >= entry + td:
                    exit_px[t] = entry + td
                    hit = True
            else:
                if high[t] >= entry + sd:
                    exit_px[t] = entry + sd
                    hit = True
                elif low[t] <= entry - td:
                    exit_px[t] = entry - td
                    hit = True
            if not hit and max_hold > 0 and held >= max_hold:
                exit_px[t] = close[t]
                hit = True
            if hit:
                cur = 0.0
        if cur == 0.0 and np.isnan(exit_px[t]):
            if enter_long[t]:
                cur = 1.0
            elif enter_short[t]:
                cur = -1.0
            if cur != 0.0:
                entry = close[t]
                sd = stop_dist[t]
                td = tp_dist[t]
                held = 0
        pos[t] = cur
    return pos, exit_px


# ---------- результаты ----------

@dataclass
class Result:
    pnl: pd.Series          # доходность на капитал по барам (после издержек и funding)
    pos: pd.Series
    cost_bps: float
    bar_min: int


def run(df: pd.DataFrame, pos: np.ndarray, cost_bps: float, bar_min: int,
        exit_px: np.ndarray | None = None) -> Result:
    close = df["close"].to_numpy()
    r = np.zeros(len(close))
    r[1:] = close[1:] / close[:-1] - 1.0
    prev_pos = np.concatenate([[0.0], pos[:-1]])
    gross = prev_pos * r
    if exit_px is not None:
        # в баре выхода по стопу/тейку доходность считается до цены выхода, а не до close
        m = ~np.isnan(exit_px)
        idx = np.where(m)[0]
        idx = idx[idx > 0]
        gross[idx] = prev_pos[idx] * (exit_px[idx] / close[idx - 1] - 1.0)
    turnover = np.abs(pos - prev_pos)
    funding = prev_pos * df["funding"].to_numpy()
    pnl = gross - turnover * cost_bps / 1e4 - funding
    return Result(pd.Series(pnl, index=df.index), pd.Series(pos, index=df.index), cost_bps, bar_min)


def metrics(pnl: pd.Series, pos: pd.Series | None = None) -> dict:
    if len(pnl) == 0:
        return {}
    daily = pnl.resample("1D").sum()
    daily = daily[daily.index >= pnl.index[0].floor("1D")]
    n_days = max(len(daily), 1)
    ann_ret = daily.mean() * 365
    ann_vol = daily.std() * math.sqrt(365)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    eq = daily.cumsum()
    mdd = (eq - eq.cummax()).min()
    out = {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "max_dd": mdd,
           "total": daily.sum(), "days": n_days, "pos_days": float((daily > 0).mean())}
    if pos is not None:
        p = pos.to_numpy()
        prev = np.concatenate([[0.0], p[:-1]])
        entries = int(((p != 0) & (p != prev)).sum())
        out["trades"] = entries
        out["trades_per_day"] = entries / n_days
        out["exposure"] = float((p != 0).mean())
        out["avg_trade_bps"] = pnl.sum() / entries * 1e4 if entries else 0.0
    return out


def split(s: pd.Series | pd.DataFrame, is_end: str):
    return s[s.index < is_end], s[s.index >= is_end]
