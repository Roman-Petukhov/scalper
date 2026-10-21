"""«Погода рынка»: в каком рынке сейчас торгует бот. Только описание — решений по ней бот не принимает: заранее
угадать удачный для правила режим не удалось ни фильтрами по BTC, волатильности и кривой бота, ни моделью
(docs/knowledge.md: btcline, ddrisk, metalabel, metaroll). Задним числом режим виден: бэктест правила за те же дни
рядом с живым журналом показывает, был ли рынок удачным для правила или бот отработал лучше / хуже него.

Режим: доля ликвидных монет ниже дневной EMA50 (сейчас и неделю назад), медианный ход альтов к BTC за 30 дней,
BTC за 7 и 30 дней. Настроение толпы: индекс страха и жадности (alternative.me), медианный funding ликвидных монет,
соотношение лонгов и шортов по счетам на BTC (Binance)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from statistics import median

import pandas as pd

EMA_SPAN = 50
BTC_FLAT = 0.03         # шорты 4h в бэктесте лучше всего, когда BTC за неделю стоит на месте (−3…+3%)
ALTS_GAP = 0.05         # альты к BTC за 30 дней: заметно слабее / сильнее
BREADTH_DOWN, BREADTH_UP = 0.6, 0.4
FUNDING_HOT = 0.0001    # 0.01% за 8 ч — базовая ставка Binance; заметно выше — толпа в лонгах
FG_FEAR, FG_GREED = 25, 75


@dataclass(frozen=True)
class Mood:
    fear_greed: int | None = None           # 0…100
    fear_greed_label: str | None = None
    fear_greed_week: int | None = None      # неделю назад
    funding_median: float | None = None     # медианная ставка funding за 8 ч, доля
    funding_pos_share: float | None = None  # доля монет, где лонги платят шортам
    btc_long_short: float | None = None     # лонги / шорты по счетам на BTC, последние сутки
    btc_long_short_week: float | None = None


@dataclass(frozen=True)
class ReplayStats:
    tf: str
    since: datetime
    n: int                                  # сделок правила с итогом
    avg_r: float | None
    win_share: float | None
    live_n: int = 0                         # сделок бота с итогом за те же дни
    live_avg_r: float | None = None


@dataclass(frozen=True)
class Weather:
    day: date
    computed_at: datetime
    coins: int
    below_ema: float | None
    below_ema_week: float | None
    alts_vs_btc_30d: float | None
    btc_7d: float | None
    btc_30d: float | None
    mood: Mood = field(default_factory=Mood)
    replay: tuple[ReplayStats, ...] = ()

    def notes(self) -> list[str]:
        """Описание режима простыми словами (что было в бэктесте при похожем рынке) — не прогноз."""
        out = []
        if self.btc_7d is not None:
            if abs(self.btc_7d) <= BTC_FLAT:
                out.append("BTC за неделю на месте — в бэктесте шорты 4h в такие недели лучше всего")
            elif self.btc_7d > BTC_FLAT:
                out.append("BTC за неделю растёт больше 3% — в бэктесте шортам 4h в такие недели хуже")
            else:
                out.append("BTC за неделю падает больше 3%")
        if self.alts_vs_btc_30d is not None:
            if self.alts_vs_btc_30d <= -ALTS_GAP:
                out.append("альты за месяц слабее BTC — рынок пробоев вниз, шорты бота в своей стихии")
            elif self.alts_vs_btc_30d >= ALTS_GAP:
                out.append("альты за месяц сильнее BTC — шортам по альтам тяжелее")
        if self.below_ema is not None:
            if self.below_ema >= BREADTH_DOWN:
                out.append("большинство монет ниже дневной EMA50 — нисходящий рынок")
            elif self.below_ema <= BREADTH_UP:
                out.append("большинство монет выше дневной EMA50 — восходящий рынок")
        m = self.mood
        if m.fear_greed is not None:
            if m.fear_greed <= FG_FEAR:
                out.append("толпа в сильном страхе")
            elif m.fear_greed >= FG_GREED:
                out.append("толпа в жадности")
        if m.funding_median is not None:
            if m.funding_median > FUNDING_HOT:
                out.append("funding выше обычного — толпа в лонгах, шорты получают плату")
            elif m.funding_median < 0:
                out.append("funding отрицательный — толпа в шортах, шортам дороже держать и есть риск выноса вверх")
        return out

    def to_json(self) -> dict:
        d = asdict(self)
        d["day"] = self.day.isoformat()
        d["computed_at"] = self.computed_at.isoformat()
        d["replay"] = [{**asdict(r), "since": r.since.isoformat()} for r in self.replay]
        return d

    @staticmethod
    def from_json(d: dict) -> Weather:
        return Weather(day=date.fromisoformat(d["day"]), computed_at=datetime.fromisoformat(d["computed_at"]),
                       coins=int(d["coins"]), below_ema=d.get("below_ema"), below_ema_week=d.get("below_ema_week"),
                       alts_vs_btc_30d=d.get("alts_vs_btc_30d"), btc_7d=d.get("btc_7d"), btc_30d=d.get("btc_30d"),
                       mood=Mood(**(d.get("mood") or {})),
                       replay=tuple(ReplayStats(**{**r, "since": datetime.fromisoformat(r["since"])})
                                    for r in d.get("replay") or []))


def ret(close: pd.Series, days: int) -> float | None:
    """Ход закрытия за `days` дней (последняя свеча против свечи days назад)."""
    if len(close) <= days or not close.iloc[-1 - days] > 0:
        return None
    return float(close.iloc[-1] / close.iloc[-1 - days] - 1)


def share_below_ema(closes: list[pd.Series], lag: int = 0) -> float | None:
    """Доля монет, у которых закрытие `lag` дней назад ниже EMA50 закрытий на тот же день."""
    flags = []
    for c in closes:
        if len(c) < EMA_SPAN + lag + 1:
            continue
        ema = c.ewm(span=EMA_SPAN, adjust=False).mean()
        flags.append(bool(c.iloc[-1 - lag] < ema.iloc[-1 - lag]))
    return sum(flags) / len(flags) if flags else None


def alts_vs_btc(closes: list[pd.Series], btc: pd.Series, days: int = 30) -> float | None:
    """Медианный ход альтов за `days` дней минус ход BTC."""
    b = ret(btc, days)
    alts = [x for x in (ret(c, days) for c in closes) if x is not None]
    if b is None or not alts:
        return None
    return float(median(alts) - b)


def replay_stats(tf: str, since: datetime, trades: list[float], live: list[float]) -> ReplayStats:
    n = len(trades)
    return ReplayStats(tf=tf, since=since, n=n, avg_r=sum(trades) / n if n else None,
                       win_share=sum(r > 0 for r in trades) / n if n else None, live_n=len(live),
                       live_avg_r=sum(live) / len(live) if live else None)
