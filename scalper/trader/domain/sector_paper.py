"""Бумажная проверка выбора сектора вперёд: каждый понедельник записываем сектор с наибольшим сводным баллом и, для
сравнения, лучший сектор за 7 дней (оба — по данным до конца воскресенья), через неделю — итог сверх рынка.

Зачем: на истории 2020–2026 сводный балл давал +2.0% в неделю сверх рынка (t 2.7, research/posflow.py), но порог
t >= 3 не пройден. Честный ответ даст только время: недели, которых бэктест не видел. Издержки taker 11 б.п. за круг.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from datetime import date, timedelta

import pandas as pd

from .sector_flow import ADV_MIN, SectorFlow

COST = 0.0011
KINDS = {"score": "Сводный балл", "momentum": "Лучший за 7 дней"}


@dataclass(frozen=True)
class PaperWeek:
    start: str                  # понедельник недели удержания (ISO)
    kind: str                   # score / momentum
    sector: str
    coins: tuple[str, ...]      # корзина — ликвидные монеты сектора на воскресенье
    bull: bool | None           # режим BTC на воскресенье
    sector_ret: float | None = None
    market_ret: float | None = None

    @property
    def settled(self) -> bool:
        return self.sector_ret is not None and self.market_ret is not None

    @property
    def excess(self) -> float | None:
        """Сверх рынка после издержек."""
        return self.sector_ret - self.market_ret - COST if self.settled else None   # type: ignore[operator]

    @property
    def end(self) -> str:
        return (date.fromisoformat(self.start) + timedelta(days=6)).isoformat()

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> PaperWeek:
        return PaperWeek(**{**d, "coins": tuple(d["coins"])})


def week_start(flow: SectorFlow) -> str | None:
    """Понедельник, с которого держим выбор: только если последний закрытый день снимка — воскресенье."""
    if not flow.day:
        return None
    d = date.fromisoformat(flow.day)
    return (d + timedelta(days=1)).isoformat() if d.weekday() == 6 else None


def picks(flow: SectorFlow) -> list[PaperWeek]:
    start = week_start(flow)
    if start is None:
        return []
    out = []
    for kind, key in (("score", "score"), ("momentum", "ret_7d")):
        row = flow.best_by(key)
        if row is not None and row.members:
            out.append(PaperWeek(start, kind, row.name, row.members, flow.bull))
    return out


def merge(weeks: list[PaperWeek], new: list[PaperWeek]) -> tuple[list[PaperWeek], list[PaperWeek]]:
    """Добавить новые выборы (одна запись на неделю и вид); вернуть (все, реально добавленные)."""
    have = {(w.start, w.kind) for w in weeks}
    added = [w for w in new if (w.start, w.kind) not in have]
    return sorted(weeks + added, key=lambda w: (w.start, w.kind)), added


def _chain(r: pd.Series) -> float | None:
    r = r.dropna()
    return float((1 + r).prod() - 1) if len(r) >= 5 else None


def settle(weeks: list[PaperWeek], bars: dict[str, pd.DataFrame]) -> tuple[list[PaperWeek], list[PaperWeek]]:
    """Закрыть недели, у которых есть дневки по воскресенье включительно. Сектор — среднее дневных доходностей корзины,
    рынок — среднее ликвидных альтов без BTC (как в исследовании). Вернуть (все, закрытые сейчас)."""
    open_ = [w for w in weeks if not w.settled]
    if not open_ or not bars:
        return weeks, []
    close = pd.DataFrame({s: b["close"] for s, b in bars.items() if len(b)}).sort_index()
    qv = pd.DataFrame({s: b["quote_volume"] for s, b in bars.items() if len(b)}).reindex_like(close)
    close.index = pd.DatetimeIndex(close.index).tz_localize(None) if close.index.tz is not None else close.index
    qv.index = close.index
    if close.empty:
        return weeks, []
    r = close.pct_change()
    liquid = qv.rolling(30, min_periods=20).mean().shift(1) >= ADV_MIN
    alts = [c for c in close.columns if c != "BTCUSDT"]
    mkt = r[alts].where(liquid[alts]).mean(axis=1)
    last = close.index[-1]
    done, out = [], []
    for w in weeks:
        a, b = pd.Timestamp(w.start), pd.Timestamp(w.end)
        if w.settled or last < b:
            out.append(w)
            continue
        cols = [c for c in w.coins if c in r.columns]
        sec = _chain(r.loc[a:b, cols].mean(axis=1)) if cols else None
        m = _chain(mkt.loc[a:b])
        if sec is None or m is None:
            out.append(w)
            continue
        nw = replace(w, sector_ret=sec, market_ret=m)
        out.append(nw)
        done.append(nw)
    return out, done


@dataclass(frozen=True)
class PaperStats:
    weeks: int
    mean: float                 # среднее сверх рынка за неделю
    total: float                # накопленно: произведение (1 + сверх рынка) − 1
    wins: int


def stats(weeks: list[PaperWeek], kind: str, bull: bool | None = None) -> PaperStats | None:
    """По закрытым неделям вида; bull=True — только бычьи, False — только небычьи, None — все."""
    xs = [w.excess for w in weeks if w.kind == kind and w.settled and (bull is None or w.bull is bull)]
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return None
    total = 1.0
    for x in xs:
        total *= 1 + x
    return PaperStats(len(xs), sum(xs) / len(xs), total - 1, sum(1 for x in xs if x > 0))


def week_text(added: list[PaperWeek], closed: list[PaperWeek], flow: SectorFlow) -> str | None:
    """Уведомление раз в неделю: итог прошлой недели и новый выбор."""
    if not added and not closed:
        return None
    lines = []
    for w in closed:
        if w.kind == "score":
            lines.append(f"Итог недели с {w.start}: «{w.sector}» {w.sector_ret:+.1%}, рынок {w.market_ret:+.1%}, "
                         f"сверх рынка {w.excess:+.1%}.")
    pick = next((w for w in added if w.kind == "score"), None)
    if pick is not None:
        row = next((r for r in flow.rows if r.name == pick.sector), None)
        regime = ("BTC выше недельной EMA50 — режим, где перевес был" if pick.bull
                  else "BTC ниже недельной EMA50 — в истории в таком режиме перевеса не было" if pick.bull is False
                  else "режим BTC неизвестен")
        lead = f" Лидеры: {', '.join(row.leaders)}." if row is not None and row.leaders else ""
        lines.append(f"Сектор недели (бумажно): «{pick.sector}».{lead} {regime}. Это проверка вперёд, не сигнал.")
    return "\n".join(lines) if lines else None
