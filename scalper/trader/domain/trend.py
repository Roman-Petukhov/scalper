"""Вторая стратегия: тренд по каналам Дончиана на дневках, бумажный счёт (research/dtrend.py, research/dboost.py).

Ансамбль 9 моделей N = 5…360 дней: модель в лонге с закрытия не ниже максимума закрытий N прошлых дней, выходит на
закрытии не выше середины канала (или, с ATR-выходом, ниже максимума с входа на k × ATR). Сигнал монеты — доля моделей
в лонге. Дальше:
    долив         монета в тренде (сигнал ≥ 0.5): вес × (1 + 0.5 × k), k — новые максимумы закрытий за 20 дней с начала
                  тренда (без дня входа), не больше 3
    сильнейшие    из 20 монет с наибольшим средним оборотом за 30 дней и сигналом > 0 — 7 с лучшим моментумом
                  (доходность за 60 дней / волатильность 90 дней); капитал делится на 7
    размер        вес = сигнал × (40% / годовая волатильность монеты за 90 дней) / 7, общий вес не больше 2
Решение — на закрытии дня, исполнение по этому закрытию; издержки — taker с оборота и funding.
Всё считается заново по истории закрытий каждый день, поэтому состояние стратегии — только бумажный счёт.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

TAKER = 0.00055
LOOKBACKS = (5, 10, 20, 30, 60, 90, 150, 250, 360)
HISTORY_DAYS = 1000                 # дневных свечей на монету: модель 360 дней успевает войти в своё состояние


@dataclass(frozen=True)
class TrendRule:
    key: str
    title: str
    backtest: str                   # что было на бэктесте — для сравнения в панели
    atr_exit: float = 0.0           # 0 — выход на середине канала
    pyramid: bool = True
    strongest: int = 7
    top: int = 20
    min_adv: float = 20e6
    coin_vol: float = 0.40
    vol_days: int = 90
    mom_days: int = 60
    gross_max: float = 2.0
    add_step: float = 0.5
    add_max: int = 3
    add_high: int = 20
    atr_days: int = 14


RULES: tuple[TrendRule, ...] = (
    TrendRule("s7_add", "Сильнейшие 7 + долив",
              "бэктест с 2022: +30% в год, Sharpe 0.91, просадка 44%"),
    TrendRule("s7_add_atr3", "Все три: + выход по 3 ATR",
              "бэктест с 2022: +27% в год, Sharpe 0.94, просадка 28%", atr_exit=3.0),
)


def _mid_models(close: pd.DataFrame, n: int) -> np.ndarray:
    hi = close.rolling(n, min_periods=n).max().shift(1).to_numpy()
    lo = close.rolling(n, min_periods=n).min().shift(1).to_numpy()
    mid = (hi + lo) / 2
    x = close.to_numpy()
    st = np.zeros_like(x)
    cur = np.zeros(x.shape[1])
    for i in range(len(x)):
        cur = np.where((cur > 0) & (x[i] <= mid[i]), 0.0, cur)
        cur = np.where((cur == 0) & (x[i] >= hi[i]), 1.0, cur)
        cur = np.where(np.isnan(x[i]), 0.0, cur)
        st[i] = cur
    return st


def _atr_models(close: pd.DataFrame, n: int, k: float, atr_days: int) -> np.ndarray:
    hi = close.rolling(n, min_periods=n).max().shift(1).to_numpy()
    atr = close.diff().abs().rolling(atr_days, min_periods=atr_days).mean().to_numpy()
    x = close.to_numpy()
    cur = np.zeros(x.shape[1])
    peak = np.full(x.shape[1], np.nan)
    st = np.zeros_like(x)
    for i in range(len(x)):
        peak = np.where(cur > 0, np.fmax(peak, x[i]), peak)
        cur = np.where((cur > 0) & (x[i] <= peak - k * atr[i]), 0.0, cur)
        enter = (cur == 0) & (x[i] >= hi[i])
        cur = np.where(enter, 1.0, cur)
        peak = np.where(enter, x[i], np.where(cur > 0, peak, np.nan))
        cur = np.where(np.isnan(x[i]), 0.0, cur)
        st[i] = cur
    return st


def signal(close: pd.DataFrame, rule: TrendRule) -> pd.DataFrame:
    """Доля моделей ансамбля в лонге (0…1) по каждой монете и дню."""
    models = [_atr_models(close, n, rule.atr_exit, rule.atr_days) if rule.atr_exit else _mid_models(close, n)
              for n in LOOKBACKS]
    return pd.DataFrame(sum(models) / len(LOOKBACKS), index=close.index, columns=close.columns)


def adds(close: pd.DataFrame, sig: pd.DataFrame, rule: TrendRule) -> pd.DataFrame:
    """Сколько доливов сделано: новые максимумы закрытий за add_high дней с начала тренда, не больше add_max."""
    hi = close.rolling(rule.add_high, min_periods=rule.add_high).max().shift(1)
    new_hi = (close >= hi).to_numpy()
    on = (sig >= 0.5).to_numpy()
    k = np.zeros(close.shape[1])
    out = np.zeros(close.shape)
    for i in range(len(close)):
        prev = on[i - 1] if i else np.zeros(close.shape[1], bool)
        k = np.where(on[i], np.where(prev, np.minimum(k + new_hi[i], rule.add_max), 0.0), 0.0)
        out[i] = k
    return pd.DataFrame(out, index=close.index, columns=close.columns)


def adv30(quote_volume: pd.DataFrame) -> pd.DataFrame:
    """Средний дневной оборот за 30 прошлых дней (сегодняшний ещё не известен)."""
    return quote_volume.rolling(30, min_periods=20).mean().shift(1)


@dataclass(frozen=True)
class TrendTarget:
    symbol: str
    weight: float                   # доля капитала стратегии в лонге
    signal: float                   # доля моделей в лонге
    adds: int
    momentum: float


def weights(close: pd.DataFrame, quote_volume: pd.DataFrame, rule: TrendRule) -> pd.DataFrame:
    """Веса по всем дням (как research/dtrend.portfolio с сигналом research/dboost)."""
    sig = signal(close, rule)
    k_adds = adds(close, sig, rule) if rule.pyramid else pd.DataFrame(0.0, index=close.index, columns=close.columns)
    sized = sig * (1 + rule.add_step * k_adds)
    ret = close.pct_change(fill_method=None)
    vol = ret.rolling(rule.vol_days, min_periods=30).std() * np.sqrt(365)
    adv = adv30(quote_volume)
    rk = adv.where(adv >= rule.min_adv).rank(axis=1, ascending=False, method="first")
    member = rk <= rule.top
    k = rule.top
    if rule.strongest:
        mom = (close / close.shift(rule.mom_days) - 1) / vol
        cand = mom.where(member & (sig > 0) & mom.notna())
        keep = cand.rank(axis=1, ascending=False, method="first") <= rule.strongest
        sized = sized.where(keep, 0.0)
        k = rule.strongest
    w = (sized * (rule.coin_vol / vol) / k).where(member & close.notna(), 0.0).fillna(0.0)
    gross = w.abs().sum(axis=1)
    return w.mul(np.minimum(1.0, rule.gross_max / gross.replace(0, np.nan)).fillna(1.0), axis=0)


def targets(close: pd.DataFrame, quote_volume: pd.DataFrame, rule: TrendRule) -> list[TrendTarget]:
    """Портфель на закрытии последнего дня: монеты с весом > 0, крупные первыми."""
    w = weights(close, quote_volume, rule).iloc[-1]
    sig = signal(close, rule)
    k = adds(close, sig, rule).iloc[-1]
    vol = close.pct_change(fill_method=None).rolling(rule.vol_days, min_periods=30).std() * np.sqrt(365)
    mom = ((close / close.shift(rule.mom_days) - 1) / vol).iloc[-1]
    out = [TrendTarget(s, float(w[s]), float(sig.iloc[-1][s]), int(k[s]) if rule.pyramid else 0, float(mom[s]))
           for s in w.index if w[s] > 0]
    return sorted(out, key=lambda t: -t.weight)


@dataclass(frozen=True)
class TrendPosition:
    qty: float                      # монет на капитал стратегии 1.0
    price: float                    # цена последней оценки
    weight: float
    adds: int
    entry_day: date
    entry_price: float


@dataclass(frozen=True)
class TrendDay:
    rule: str
    day: date                       # день, на закрытии которого принято решение
    equity: float                   # капитал стратегии, старт — 1.0
    pnl: float                      # изменение капитала за шаг (с издержками)
    cost: float                     # комиссии и funding за шаг
    positions: dict[str, TrendPosition] = field(default_factory=dict)

    @property
    def gross(self) -> float:
        return sum(p.qty * p.price for p in self.positions.values()) / self.equity if self.equity > 0 else 0.0


def step(prev: TrendDay | None, rule: str, day: date, prices: dict[str, float], funding: dict[str, float],
         tgt: list[TrendTarget]) -> TrendDay:
    """Оценить позиции по новым закрытиям, списать funding (лонг платит при ставке > 0), перейти к целевым весам
    с комиссией taker с оборота. Монета без цены (делистинг, нет свечи) закрывается по последней оценке."""
    equity = prev.equity if prev is not None else 1.0
    held = prev.positions if prev is not None else {}
    pnl = fund = 0.0
    marked: dict[str, float] = {}
    for s, p in held.items():
        px = prices.get(s, p.price)
        pnl += p.qty * (px - p.price)
        fund += p.qty * p.price * funding.get(s, 0.0)
        marked[s] = px
    equity += pnl - fund
    want = {t.symbol: t for t in tgt if t.symbol in prices and prices[t.symbol] > 0}
    turnover = 0.0
    out: dict[str, TrendPosition] = {}
    for s in set(held) | set(want):
        px = marked.get(s) or prices[s]
        old = held[s].qty * px if s in held else 0.0
        new = want[s].weight * equity if s in want else 0.0
        turnover += abs(new - old)
        if s in want:
            t = want[s]
            keep = s in held
            out[s] = TrendPosition(new / px, px, t.weight, t.adds, held[s].entry_day if keep else day,
                                   held[s].entry_price if keep else px)
    cost = TAKER * turnover
    equity -= cost
    return TrendDay(rule, day, equity, equity - (prev.equity if prev is not None else 1.0), fund + cost, out)
