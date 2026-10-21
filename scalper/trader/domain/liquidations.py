"""Ликвидации за сутки по часам: сколько $ лонгов и шортов принудительно закрыла биржа. Источник — живой поток
Binance (!forceOrder@arr), истории у него нет: копится с запуска бота. Поток отдаёт не больше одной ликвидации
на монету в секунду, поэтому суммы занижены — смотреть на соотношение лонгов и шортов, а не на абсолютные $."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .models import Side

WINDOW = timedelta(hours=24)


def _hour(t: datetime) -> str:
    return t.replace(minute=0, second=0, microsecond=0).isoformat()


@dataclass(frozen=True)
class LiqTotals:
    long_usd: float             # ликвидировано лонгов (биржа продавала)
    short_usd: float            # ликвидировано шортов (биржа покупала)
    since: datetime             # с какого момента есть данные в окне
    partial: bool = False       # поток слушается меньше суток

    @property
    def long_share(self) -> float | None:
        total = self.long_usd + self.short_usd
        return self.long_usd / total if total > 0 else None


class LiqBook:
    """Часовые корзины {час: [лонги $, шорты $]} и момент, с которого поток слушается."""

    def __init__(self, buckets: dict[str, list[float]] | None = None, started: datetime | None = None) -> None:
        self.buckets = {k: [float(v[0]), float(v[1])] for k, v in (buckets or {}).items()}
        self.started = started

    def add(self, at: datetime, liquidated: Side, usd: float) -> None:
        if not usd > 0:
            return
        b = self.buckets.setdefault(_hour(at), [0.0, 0.0])
        b[0 if liquidated is Side.LONG else 1] += usd

    def trim(self, now: datetime) -> None:
        edge = _hour(now - WINDOW)
        self.buckets = {k: v for k, v in self.buckets.items() if k >= edge}

    def totals(self, now: datetime) -> LiqTotals | None:
        if self.started is None:
            return None
        edge = now - WINDOW
        keep = [v for k, v in self.buckets.items() if datetime.fromisoformat(k) >= edge.replace(minute=0, second=0,
                                                                                                microsecond=0)]
        return LiqTotals(sum(v[0] for v in keep), sum(v[1] for v in keep), max(self.started, edge),
                         self.started > edge)

    def to_json(self) -> dict:
        return {"buckets": self.buckets, "started": self.started.isoformat() if self.started else None}

    @staticmethod
    def from_json(d: dict) -> LiqBook:
        s = d.get("started")
        return LiqBook(d.get("buckets") or {}, datetime.fromisoformat(s) if s else None)


def parse_force_order(msg: dict) -> tuple[datetime, Side, float] | None:
    """Событие потока Binance: o.S — сторона ордера ликвидации (SELL закрывает лонг), o.z × o.ap — исполнено в $,
    o.T — время сделки, мс."""
    o = msg.get("o") if isinstance(msg, dict) else None
    if not isinstance(o, dict):
        return None
    try:
        usd = float(o.get("z") or o["q"]) * float(o.get("ap") or o["p"])
        at = datetime.fromtimestamp(int(o.get("T") or msg["E"]) / 1000, timezone.utc)
        side = Side.LONG if o["S"] == "SELL" else Side.SHORT
    except (KeyError, TypeError, ValueError):
        return None
    return at, side, usd
