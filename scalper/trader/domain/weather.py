"""«Погода рынка»: в каком рынке сейчас торгует бот. Только описание — решений по ней бот не принимает: заранее
угадать удачный для правила режим не удалось ни фильтрами по BTC, волатильности и кривой бота, ни моделью
(docs/knowledge.md: btcline, ddrisk, metalabel, metaroll). Задним числом режим виден: бэктест правила за те же дни
рядом с живым журналом показывает, был ли рынок удачным для правила или бот отработал лучше / хуже него.

Режим: доля ликвидных монет ниже дневной EMA50 (сейчас и неделю назад), медианный ход альтов к BTC за 30 дней,
BTC за 7 и 30 дней, доминация BTC, приток стейблкоинов за 30 дней. Настроение толпы: индекс страха и жадности
(alternative.me), медианный funding и базис к споту ликвидных монет, лонги / шорты всех счетов и крупных трейдеров
на BTC, открытый интерес BTC за неделю (Binance), волатильность опционов BTC DVOL (Deribit). Показатели толпы как
фильтр входа проверялись — устойчивого эффекта нет (docs/research_report.md)."""
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
DOM_SHIFT = 1.0         # доминация BTC за неделю, п. п.
OI_SHIFT = 0.10         # открытый интерес BTC за неделю
DVOL_CALM, DVOL_HOT = 40.0, 60.0


@dataclass(frozen=True)
class Mood:
    fear_greed: int | None = None           # 0…100
    fear_greed_label: str | None = None
    fear_greed_week: int | None = None      # неделю назад
    funding_median: float | None = None     # медианная ставка funding за 8 ч, доля
    funding_pos_share: float | None = None  # доля монет, где лонги платят шортам
    btc_long_short: float | None = None     # лонги / шорты по счетам на BTC, последние сутки
    btc_long_short_week: float | None = None
    top_long_short: float | None = None     # то же по позициям крупных трейдеров
    top_long_short_week: float | None = None
    basis_btc: float | None = None          # фьючерс BTC к споту (mark / index − 1)
    basis_median: float | None = None
    oi_btc_7d: float | None = None          # открытый интерес BTC за неделю
    dominance: float | None = None          # доля BTC в капитализации, %
    dominance_week: float | None = None
    dvol: float | None = None               # волатильность опционов BTC, % годовых
    dvol_week: float | None = None
    stables_30d: float | None = None        # предложение стейблкоинов за 30 дней


@dataclass(frozen=True)
class ReplayStats:
    tf: str
    since: datetime
    n: int                                  # сделок правила с итогом
    avg_r: float | None
    win_share: float | None
    live_n: int = 0                         # сделок бота с итогом за те же дни
    live_avg_r: float | None = None
    long_n: int = 0                         # правило задним числом по сторонам
    long_avg_r: float | None = None
    short_n: int = 0
    short_avg_r: float | None = None
    live_long_n: int = 0                    # бот по сторонам
    live_long_avg_r: float | None = None
    live_short_n: int = 0
    live_short_avg_r: float | None = None


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
        if m.dominance is not None and m.dominance_week is not None:
            if m.dominance - m.dominance_week >= DOM_SHIFT:
                out.append("доминация BTC за неделю растёт — деньги уходят из альтов")
            elif m.dominance_week - m.dominance >= DOM_SHIFT:
                out.append("доминация BTC за неделю падает — деньги идут в альты")
        if m.stables_30d is not None and m.stables_30d < 0:
            out.append("стейблкоинов за месяц стало меньше — новые деньги в крипту не приходят")
        if m.oi_btc_7d is not None and abs(m.oi_btc_7d) >= OI_SHIFT:
            out.append("открытый интерес BTC за неделю " + ("вырос — плеча в рынке больше, резкие выносы вероятнее"
                                                             if m.oi_btc_7d > 0 else "упал — плечо из рынка вышло"))
        if m.dvol is not None:
            if m.dvol >= DVOL_HOT:
                out.append("опционы закладывают сильные движения BTC (DVOL от 60)")
            elif m.dvol <= DVOL_CALM:
                out.append("опционы ждут спокойного рынка (DVOL до 40)")
        if m.top_long_short is not None and m.btc_long_short is not None:
            if m.btc_long_short > 1 > m.top_long_short:
                out.append("толпа в лонгах BTC, крупные трейдеры — в шортах")
            elif m.btc_long_short < 1 < m.top_long_short:
                out.append("толпа в шортах BTC, крупные трейдеры — в лонгах")
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


def replay_stats(tf: str, since: datetime, trades: list[tuple[int, float]], live: list[tuple[int, float]]) -> ReplayStats:
    """trades / live — (сторона +1 / −1, R)."""
    def avg(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    rr, lr = [r for _, r in trades], [r for _, r in live]
    side = lambda xs, sd: [r for s_, r in xs if s_ == sd]
    tl, ts, ll, ls = side(trades, 1), side(trades, -1), side(live, 1), side(live, -1)
    return ReplayStats(tf=tf, since=since, n=len(rr), avg_r=avg(rr),
                       win_share=sum(r > 0 for r in rr) / len(rr) if rr else None, live_n=len(lr), live_avg_r=avg(lr),
                       long_n=len(tl), long_avg_r=avg(tl), short_n=len(ts), short_avg_r=avg(ts),
                       live_long_n=len(ll), live_long_avg_r=avg(ll), live_short_n=len(ls), live_short_avg_r=avg(ls))


R_SD = 1.9                                              # разброс одной сделки −1R / +3R, R
MIN_LIVE = 10


@dataclass(frozen=True)
class Assessment:
    level: str                  # "ok" / "watch" / "warn" — по сделкам бота
    market: str                 # одна строка про рынок
    mood: tuple[str, ...]       # режим и настроение простыми словами (Weather.notes)
    lines: tuple[str, ...]      # выводы по сделкам каждого ТФ

    @property
    def title(self) -> str:
        return {"ok": "всё в норме", "watch": "есть на что посмотреть", "warn": "нужно внимание"}[self.level]

    def text(self) -> str:
        icon = {"ok": "🟢", "watch": "🟡", "warn": "🔴"}[self.level]
        out = [f"{icon} Погода рынка: {self.title}", self.market]
        if self.mood:
            out.append("Настроение: " + "; ".join(self.mood) + ".")
        if self.lines:
            out += ["Сделки:", *(f"• {x}" for x in self.lines)]
        return "\n".join(out)


def market_line(w: Weather) -> str:
    """Рынок для сторон бота по тем признакам, что в бэктесте отличали хорошие недели от плохих, — описание, не прогноз."""
    good = bad = 0
    if w.alts_vs_btc_30d is not None:
        good += w.alts_vs_btc_30d <= -ALTS_GAP
        bad += w.alts_vs_btc_30d >= ALTS_GAP
    if w.btc_7d is not None:
        good += abs(w.btc_7d) <= BTC_FLAT
        bad += w.btc_7d > BTC_FLAT
    if w.below_ema is not None:
        good += w.below_ema >= BREADTH_DOWN
        bad += w.below_ema <= BREADTH_UP
    if good >= 2 and not bad:
        kind = "попутный для шортов, встречный для лонгов"
    elif bad >= 2 and not good:
        kind = "встречный для шортов, попутный для лонгов"
    else:
        kind = "смешанный для обеих сторон"
    return f"Рынок — {kind} (признаков за шорты {good}, за лонги {bad})."


def assess(w: Weather) -> Assessment:
    """Автовывод: рынок и настроение толпы, бот против правила задним числом (с учётом случайности: разница больше
    двух стандартных ошибок) и удачный ли рынок для правила."""
    lines, level = [], "ok"

    def bump(to: str) -> None:
        nonlocal level
        order = ("ok", "watch", "warn")
        level = max(level, to, key=order.index)

    for r in w.replay:
        if r.live_n < MIN_LIVE or r.live_avg_r is None:
            lines.append(f"{r.tf}: сделок бота за окно {r.live_n} — для вывода мало (нужно от {MIN_LIVE})"
                         + (f"; правило задним числом {r.avg_r:+.2f}R на {r.n}" if r.avg_r is not None else ""))
            continue
        if r.avg_r is None:
            lines.append(f"{r.tf}: бот {r.live_avg_r:+.2f}R на {r.live_n}, правило задним числом сделок не нашло")
            bump("watch")
            continue
        se = R_SD / r.live_n ** 0.5
        gap = r.live_avg_r - r.avg_r
        head = f"{r.tf}: бот {r.live_avg_r:+.2f}R на {r.live_n}, правило {r.avg_r:+.2f}R на {r.n}"
        if gap < -2 * se:
            lines.append(f"{head} — бот заметно хуже правила: проверить исполнение, пропуски и ручные действия")
            bump("warn")
        elif gap > 2 * se:
            lines.append(f"{head} — бот заметно лучше правила: скорее везение серии")
            bump("watch")
        elif r.avg_r > 0.3:
            lines.append(f"{head} — рынок удачный для правила, бот в норме")
        elif r.avg_r < 0:
            lines.append(f"{head} — рынок неудачный для правила, бот исправен: это нормальная просадка, не выключать")
            bump("watch")
        else:
            lines.append(f"{head} — бот в пределах правила")
    for r in w.replay:
        if r.long_n and r.short_n:
            part = (f"{r.tf} по сторонам за окно: правило — лонги {r.long_avg_r:+.2f}R на {r.long_n}, "
                    f"шорты {r.short_avg_r:+.2f}R на {r.short_n}")
            if r.live_long_n or r.live_short_n:
                part += "; бот — " + ", ".join(
                    f"{nm} {v:+.2f}R на {n}" for nm, v, n in (("лонги", r.live_long_avg_r, r.live_long_n),
                                                            ("шорты", r.live_short_avg_r, r.live_short_n)) if n)
            lines.append(part)
    return Assessment(level, market_line(w), tuple(w.notes()), tuple(lines))
