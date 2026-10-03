"""
Библиотека гипотез. Каждая функция возвращает (pos, exit_px | None) по данным одного символа.
Все сигналы используют только прошлое: rolling-окна заканчиваются на текущем баре,
пороги «прошлого» берутся со сдвигом на один бар.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import state_machine, stop_target_machine


def _vol(close: pd.Series, win: int) -> pd.Series:
    return np.log(close).diff().rolling(win, min_periods=win // 2).std()


def _b(x) -> np.ndarray:
    return np.array(x.fillna(False) if hasattr(x, "fillna") else x, dtype=np.bool_)  # копия: pandas 3 отдаёт read-only


# ---------- 1. Тренд: пробой канала Дончиана ----------
def donchian(df, n=50, exit_frac=0.5, long_only=False):
    hh = df["high"].rolling(n).max().shift(1)
    ll = df["low"].rolling(n).min().shift(1)
    m = max(2, int(n * exit_frac))
    ex_ll = df["low"].rolling(m).min().shift(1)
    ex_hh = df["high"].rolling(m).max().shift(1)
    c = df["close"]
    el, es = _b(c > hh), _b(c < ll)
    if long_only:
        es[:] = False
    return state_machine(el, es, _b(c < ex_ll), _b(c > ex_hh), 0), None


# ---------- 2. Тренд: time-series momentum с гистерезисом ----------
def tsmom(df, lookback=72, band=0.5):
    c = df["close"]
    z = np.log(c / c.shift(lookback)) / (_vol(c, 24 * 30) * np.sqrt(lookback))
    return state_machine(_b(z > band), _b(z < -band), _b(z < 0), _b(z > 0), 0), None


# ---------- 3. Возврат к среднему после резкого движения ----------
def meanrev_z(df, k=3, thr=3.0, hold=12, volwin=288):
    c = df["close"]
    z = np.log(c / c.shift(k)) / (_vol(c, volwin) * np.sqrt(k))
    return state_machine(_b(z < -thr), _b(z > thr), _b(z > 0), _b(z < 0), hold), None


# ---------- 4. Возврат к скользящему VWAP ----------
def vwap_rev(df, win=96, thr=2.5, hold=48):
    pv = (df["quote_volume"]).rolling(win).sum()
    v = df["volume"].rolling(win).sum()
    vwap = pv / v
    dev = df["close"] - vwap
    z = dev / dev.rolling(win * 5, min_periods=win).std()
    return state_machine(_b(z < -thr), _b(z > thr), _b(z > 0), _b(z < 0), hold), None


# ---------- 5. Order flow на барах: дисбаланс агрессоров ----------
def _imb_z(df, k, zwin):
    buy = df["taker_buy_volume"].rolling(k).sum()
    vol = df["volume"].rolling(k).sum()
    imb = (2 * buy - vol) / vol
    return (imb - imb.rolling(zwin, min_periods=zwin // 4).mean()) / imb.rolling(zwin, min_periods=zwin // 4).std()


def flow(df, k=12, thr=2.5, hold=12, sign=1, zwin=2000):
    z = _imb_z(df, k, zwin) * sign
    return state_machine(_b(z > thr), _b(z < -thr), _b(z < 0), _b(z > 0), hold), None


# ---------- 6. Поглощение: сильный поток без движения цены -> разворот ----------
def absorption(df, k=6, thr=2.5, move=0.3, hold=12, zwin=2000, volwin=288):
    z = _imb_z(df, k, zwin)
    c = df["close"]
    mv = np.log(c / c.shift(k)) / (_vol(c, volwin) * np.sqrt(k))
    # покупатели давят, а цена не растёт -> продавец поглощает -> шорт
    es = _b((z > thr) & (mv < move))
    el = _b((z < -thr) & (mv > -move))
    return state_machine(el, es, np.zeros(len(c), np.bool_), np.zeros(len(c), np.bool_), hold), None


# ---------- 7. Сезонность по часу суток (обучается только на IS) ----------
def hour_mask_from_is(dfs_is: dict, t_thr=2.0):
    """Пул всех символов: средняя доходность и t-stat по часу UTC. Возвращает {hour: +1/-1}."""
    rows = []
    for df in dfs_is.values():
        r = df["close"].pct_change()
        rows.append(pd.DataFrame({"r": r, "h": df.index.hour}))
    a = pd.concat(rows).dropna()
    g = a.groupby("h")["r"]
    t = g.mean() / (g.std() / np.sqrt(g.count()))
    return {int(h): (1 if v > t_thr else -1 if v < -t_thr else 0) for h, v in t.items()}, t


def seasonal(df, mask: dict):
    # позиция на час h решается в конце часа h-1, поэтому смотрим на час следующего бара
    nxt = (df.index + pd.Timedelta(hours=1)).hour
    return np.array([mask.get(int(h), 0) for h in nxt], dtype=float), None


# ---------- 8. Экстремальный funding: против толпы ----------
def funding_extreme(df, q=0.95, hold=24, win_days=60):
    f = df["funding"].replace(0, np.nan).ffill()       # последняя известная ставка
    bars_per_day = 24 if len(df) and (df.index[1] - df.index[0]) == pd.Timedelta("1h") else 288
    w = win_days * bars_per_day
    hi = f.rolling(w, min_periods=w // 3).quantile(q).shift(1)
    lo = f.rolling(w, min_periods=w // 3).quantile(1 - q).shift(1)
    paid = df["funding"] != 0                           # сигнал только в момент выплаты
    es = _b(paid & (f > hi) & (f > 0))
    el = _b(paid & (f < lo))
    z = np.zeros(len(df), np.bool_)
    return state_machine(el, es, z, z, hold), None


# ---------- 9. Пробой диапазона первых часов дня (UTC) ----------
def opening_range(df, range_hours=2, stop_mult=1.0, rr=2.0, bar_min=15):
    day = df.index.floor("1D")
    in_range = (df.index - day) < pd.Timedelta(hours=range_hours)
    # диапазон известен только после его окончания, поэтому используем его лишь вне диапазона
    hi = df["high"].where(in_range).groupby(day).transform("max").where(~in_range)
    lo = df["low"].where(in_range).groupby(day).transform("min").where(~in_range)
    c = df["close"]
    first_break = ~in_range
    el = _b(first_break & (c > hi) & (c.shift(1) <= hi))
    es = _b(first_break & (c < lo) & (c.shift(1) >= lo))
    rng = (hi - lo).fillna(0).to_numpy()
    bars_left = ((day + pd.Timedelta(days=1) - df.index) / pd.Timedelta(minutes=bar_min)).to_numpy()
    max_hold = int(24 * 60 / bar_min)
    pos, ex = stop_target_machine(el, es, df["high"].to_numpy(), df["low"].to_numpy(), c.to_numpy(),
                                  rng * stop_mult, rng * stop_mult * rr, max_hold)
    return pos, ex


# ---------- 10. Волатильностный пробой от открытия дня ----------
def vol_breakout(df, k=0.5, bar_min=15):
    day = df.index.floor("1D")
    d = df.resample("1D").agg({"open": "first", "high": "max", "low": "min"})
    prev_rng = (d["high"] - d["low"]).shift(1)
    o = d["open"].reindex(day).to_numpy()
    pr = prev_rng.reindex(day).to_numpy()
    c = df["close"].to_numpy()
    el = c > o + k * pr
    es = c < o - k * pr
    last_bar = (df.index + pd.Timedelta(minutes=bar_min)).floor("1D") != day
    xl = np.asarray(last_bar)
    return state_machine(np.nan_to_num(el).astype(np.bool_) & ~xl, np.nan_to_num(es).astype(np.bool_) & ~xl,
                         xl, xl, 0), None


# ---------- 11-13. Кросс-секционные стратегии (панель символов) ----------
def xs_weights(closes: pd.DataFrame, lookback: int, rebalance: int, k: int, sign: int,
               volwin: int = 24 * 14) -> pd.DataFrame:
    """Лонг k лучших / шорт k худших по доходности за lookback (sign=+1 моментум, -1 реверсия).
    Ребалансировка каждые rebalance баров. Вес по ногам 0.5/0.5 капитала, равный внутри ноги."""
    ret = np.log(closes / closes.shift(lookback))
    vol = np.log(closes).diff().rolling(volwin, min_periods=volwin // 3).std()
    score = (ret / (vol * np.sqrt(lookback))) * sign
    rank = score.rank(axis=1, ascending=False)
    n = score.notna().sum(axis=1)
    w = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    w[rank <= k] = 0.5 / k
    w[rank.gt(n - k, axis=0) & score.notna()] = -0.5 / k
    w[n < 2 * k + 2] = 0.0
    mask = np.arange(len(w)) % rebalance == 0
    return w.where(pd.Series(mask, index=w.index), np.nan).ffill().fillna(0.0)


def funding_carry_weights(funding_last: pd.DataFrame, k: int, rebalance: int) -> pd.DataFrame:
    """Шорт k монет с самым высоким funding, лонг k с самым низким: собираем разницу ставок."""
    rank = funding_last.rank(axis=1, ascending=False)
    n = funding_last.notna().sum(axis=1)
    w = pd.DataFrame(0.0, index=funding_last.index, columns=funding_last.columns)
    w[rank <= k] = -0.5 / k
    w[rank.gt(n - k, axis=0) & funding_last.notna()] = 0.5 / k
    w[n < 2 * k + 2] = 0.0
    mask = np.arange(len(w)) % rebalance == 0
    return w.where(pd.Series(mask, index=w.index), np.nan).ffill().fillna(0.0)


def leadlag_weights(closes: pd.DataFrame, k: int, thr: float, hold: int, volwin: int = 288) -> pd.DataFrame:
    """BTC двинулся, альт отстал: позиция в альте в сторону догоняния на hold баров."""
    lr = np.log(closes / closes.shift(k))
    vol = np.log(closes).diff().rolling(volwin, min_periods=volwin // 2).std() * np.sqrt(k)
    btc = lr["BTCUSDT"]
    w = {}
    for s in closes.columns:
        if s == "BTCUSDT":
            continue
        gap = (btc - lr[s]) / vol[s]
        z = np.zeros(len(gap), np.bool_)
        pos = state_machine(_b(gap > thr), _b(gap < -thr), z, z, hold)
        w[s] = pos
    out = pd.DataFrame(w, index=closes.index) / max(len(w), 1)
    out["BTCUSDT"] = 0.0
    return out
