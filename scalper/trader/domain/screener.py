"""Фундаментальный скринер на 1–2 года: недооценённая доля в растущем бизнесе без навеса предложения, плюс режим
рынка по недельным EMA BTC. Чистые правила без сети — данные приносят адаптеры (infrastructure/fundamentals.py).

Откуда правила — docs/knowledge.md, раздел 7, «Венчурная (фундаментальная) оценка монеты»:
  бизнес     — комиссии и выручка протокола или сети за 30 дней и за год (DefiLlama);
  доля       — выручка держателям токена (выкуп, сжигание, раздача) за 30 дней не меньше holders_rev_30d_min;
  навес      — капитализация / FDV не меньше float_min, разлок за 12 месяцев не больше unlock_12m_max
               (нет расписания — пометка, не отсев);
  торгуемость — капитализация, оборот за сутки, история цены не меньше года (свежие листинги в среднем
               отдают рост — event-studies в разделе 7).
Скор — сумма перцентильных рангов дешевизны (FDV / годовая выручка держателям по последним 30 дням) и роста
(выручка держателям за 30 дней к среднему месяцу года). Это список «куда смотреть», не сигнал на покупку:
бэктеста нет (истории FDV и разлоков бесплатно не взять), проверка — бумажный счёт вперёд (first_seen).

Режим рынка (наблюдение владельца, проверка research/w200.py): недельное закрытие BTC под EMA200 — зона
накопления; после неё два закрытия подряд над EMA50 — запуск роста; в бычьем режиме (над EMA200, EMA50 растёт)
касание EMA50 — откат для набора."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, timedelta
from enum import Enum

import pandas as pd

FEES, REVENUE, HOLDERS = "dailyFees", "dailyRevenue", "dailyHoldersRevenue"
DATA_TYPES = (FEES, REVENUE, HOLDERS)
EMA_FAST, EMA_SLOW = 50, 200
LAUNCH_CLOSES = 2           # закрытий подряд над EMA50 после закрытия под EMA200
LAUNCH_WEEKS = 8            # сколько недель после запуска режим зовётся «запуск»
TOUCH, FLOOR = 1.01, 0.97   # откат: минимум недели <= EMA50 × TOUCH, закрытие >= EMA50 × FLOOR
SLOPE_WEEKS = 4


@dataclass(frozen=True)
class Rules:
    holders_rev_30d_min: float = 50_000.0
    mcap_min: float = 20e6
    volume_min: float = 1e6
    float_min: float = 0.5
    unlock_12m_max: float = 0.10
    top: int = 20


@dataclass(frozen=True)
class FeeRow:
    """Строка обзора комиссий DefiLlama: одна версия протокола (Uniswap V3) и один вид данных."""
    group: str               # бизнес: родитель (uniswap) или сам протокол
    slug: str
    name: str
    category: str
    data_type: str
    d30: float
    y1: float


@dataclass(frozen=True)
class Business:
    key: str
    name: str
    category: str
    fees_30d: float = 0.0
    fees_1y: float = 0.0
    rev_30d: float = 0.0
    rev_1y: float = 0.0
    hr_30d: float = 0.0
    hr_1y: float = 0.0


@dataclass(frozen=True)
class Quote:
    gecko_id: str
    symbol: str
    name: str
    price: float | None
    mcap: float | None
    fdv: float | None
    volume: float | None
    chg_1y: float | None     # % за год; None — монете меньше года (или CoinGecko не знает)
    ath_chg: float | None    # % от исторического максимума


@dataclass(frozen=True)
class Candidate:
    gecko_id: str
    symbol: str
    project: str
    category: str
    price: float | None
    mcap: float | None
    fdv: float | None
    volume: float | None
    float_share: float | None
    unlock_12m: float | None
    fees_year: float
    rev_year: float
    hr_year: float
    fdv_to_hr: float | None
    hr_growth: float | None
    chg_1y: float | None
    ath_chg: float | None
    reasons: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    score: float | None = None
    binance_perp: bool = False
    first_seen: date | None = None
    first_price: float | None = None

    @property
    def passed(self) -> bool:
        return not self.reasons

    @property
    def since_seen(self) -> float | None:
        if self.first_price and self.price:
            return self.price / self.first_price - 1
        return None


class Zone(str, Enum):
    ACCUMULATION = "accumulation"
    LAUNCH = "launch"
    PULLBACK = "pullback"
    BULL = "bull"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


ZONE_TEXT = {
    Zone.ACCUMULATION: ("Зона накопления", "BTC закрыл неделю под EMA200 — район прошлых доньев, но не таймер: в 2022 "
                        "цена простояла под линией 40 недель и упала ещё на 41%. Альтам здесь исторически хуже BTC. "
                        "Набирать частями, не всё сразу."),
    Zone.LAUNCH: ("Запуск роста", "После закрытия под EMA200 BTC закрепился над недельной EMA50. Так начинались прошлые "
                  "циклы (4 случая, все в плюсе через год), но случаев мало — это контекст, не сигнал."),
    Zone.PULLBACK: ("Откат к EMA50 в бычьем тренде", "Неделя коснулась EMA50 и закрылась над ней. В истории такие "
                    "откаты не давали больше, чем любая неделя бычьего тренда — добирать можно, особой точки тут нет."),
    Zone.BULL: ("Бычий тренд", "BTC над EMA200, EMA50 растёт. Исторически лучший фон для альтов, пока цена не ушла "
                "дальше 2.5 × EMA200 — там перегрев."),
    Zone.NEUTRAL: ("Без тренда", "BTC над EMA200, но под EMA50 или EMA50 падает — рынок выбирает направление, "
                   "с набором не спешить."),
    Zone.UNKNOWN: ("Нет данных", "Недельные свечи BTC не загрузились."),
}


@dataclass(frozen=True)
class Regime:
    week: date | None
    close: float | None
    ema50: float | None
    ema200: float | None
    zone: Zone
    weeks_in_zone: int = 0

    @property
    def title(self) -> str:
        return ZONE_TEXT[self.zone][0]

    @property
    def text(self) -> str:
        return ZONE_TEXT[self.zone][1]

    @property
    def to_ema200(self) -> float | None:
        return self.close / self.ema200 - 1 if self.close and self.ema200 else None

    @property
    def to_ema50(self) -> float | None:
        return self.close / self.ema50 - 1 if self.close and self.ema50 else None


@dataclass(frozen=True)
class Snapshot:
    computed_at: datetime
    regime: Regime
    candidates: tuple[Candidate, ...]          # прошли отсев, по скору
    rejected: tuple[Candidate, ...]            # крупный бизнес, который правила отсеяли (по выручке)
    businesses: int = 0
    with_holders_rev: int = 0
    priced: int = 0
    unlocks_known: int = 0
    rules: Rules = field(default_factory=Rules)

    def to_json(self) -> dict:
        return {"computed_at": self.computed_at.isoformat(), "regime": _regime_json(self.regime),
                "candidates": [_cand_json(c) for c in self.candidates],
                "rejected": [_cand_json(c) for c in self.rejected], "businesses": self.businesses,
                "with_holders_rev": self.with_holders_rev, "priced": self.priced,
                "unlocks_known": self.unlocks_known, "rules": asdict(self.rules)}

    @staticmethod
    def from_json(d: dict) -> Snapshot:
        r = d["regime"]
        regime = Regime(week=date.fromisoformat(r["week"]) if r.get("week") else None, close=r.get("close"),
                        ema50=r.get("ema50"), ema200=r.get("ema200"), zone=Zone(r["zone"]),
                        weeks_in_zone=int(r.get("weeks_in_zone", 0)))
        return Snapshot(computed_at=datetime.fromisoformat(d["computed_at"]), regime=regime,
                        candidates=tuple(_cand_from(c) for c in d.get("candidates", [])),
                        rejected=tuple(_cand_from(c) for c in d.get("rejected", [])),
                        businesses=int(d.get("businesses", 0)), with_holders_rev=int(d.get("with_holders_rev", 0)),
                        priced=int(d.get("priced", 0)), unlocks_known=int(d.get("unlocks_known", 0)),
                        rules=Rules(**d.get("rules", {})))


def _regime_json(r: Regime) -> dict:
    return {"week": r.week.isoformat() if r.week else None, "close": r.close, "ema50": r.ema50, "ema200": r.ema200,
            "zone": r.zone.value, "weeks_in_zone": r.weeks_in_zone}


def _cand_json(c: Candidate) -> dict:
    d = asdict(c)
    d["reasons"], d["flags"] = list(c.reasons), list(c.flags)
    d["first_seen"] = c.first_seen.isoformat() if c.first_seen else None
    return d


def _cand_from(d: dict) -> Candidate:
    d = dict(d)
    d["reasons"], d["flags"] = tuple(d.get("reasons", ())), tuple(d.get("flags", ()))
    d["first_seen"] = date.fromisoformat(d["first_seen"]) if d.get("first_seen") else None
    return Candidate(**d)


def group_key(slug: str | None, name: str | None, parent: str | None) -> str:
    """Версии одного протокола (Uniswap V2 / V3 / V4) — один бизнес: ключ родителя, иначе свой slug."""
    if isinstance(parent, str) and parent.startswith("parent#"):
        return parent.split("#", 1)[1]
    return str(slug or name or "").strip().lower().replace(" ", "-")


def businesses(rows: list[FeeRow]) -> dict[str, Business]:
    """Суммы по бизнесам; имя и категория — у версии с наибольшими комиссиями за 30 дней."""
    acc: dict[str, dict[str, float]] = {}
    best: dict[str, tuple[float, str, str]] = {}
    for r in rows:
        a = acc.setdefault(r.group, {})
        a[f"{r.data_type}_30"] = a.get(f"{r.data_type}_30", 0.0) + max(r.d30, 0.0)
        a[f"{r.data_type}_1y"] = a.get(f"{r.data_type}_1y", 0.0) + max(r.y1, 0.0)
        if r.data_type == FEES and (r.group not in best or r.d30 > best[r.group][0]):
            best[r.group] = (r.d30, r.name, r.category)
        elif r.group not in best:
            best[r.group] = (-1.0, r.name, r.category)
    return {k: Business(key=k, name=best[k][1], category=best[k][2],
                        fees_30d=a.get(f"{FEES}_30", 0.0), fees_1y=a.get(f"{FEES}_1y", 0.0),
                        rev_30d=a.get(f"{REVENUE}_30", 0.0), rev_1y=a.get(f"{REVENUE}_1y", 0.0),
                        hr_30d=a.get(f"{HOLDERS}_30", 0.0), hr_1y=a.get(f"{HOLDERS}_1y", 0.0))
            for k, a in acc.items()}


def by_token(biz: dict[str, Business], gecko: dict[str, str]) -> dict[str, Business]:
    """Бизнесы с одним токеном (NEAR — сеть и NEAR Intents) складываются; имя — у бизнеса с большей выручкой."""
    out: dict[str, Business] = {}
    for key, b in biz.items():
        g = gecko.get(key)
        if not g:
            continue
        o = out.get(g)
        if o is None:
            out[g] = replace(b, key=g)
            continue
        main = b if b.rev_30d > o.rev_30d else o
        out[g] = Business(key=g, name=main.name, category=main.category, fees_30d=o.fees_30d + b.fees_30d,
                          fees_1y=o.fees_1y + b.fees_1y, rev_30d=o.rev_30d + b.rev_30d, rev_1y=o.rev_1y + b.rev_1y,
                          hr_30d=o.hr_30d + b.hr_30d, hr_1y=o.hr_1y + b.hr_1y)
    return out


def growth(d30: float, y1: float) -> float | None:
    """Последние 30 дней к среднему месяцу за год; None — нет годовой истории."""
    return d30 / (y1 / 12.0) if y1 > 0 else None


def unlock_share(points: list[tuple[date, float]], today: date, days: int = 365) -> float | None:
    """Доля предложения, которая разлочится за days дней, от разлоченного сегодня; points — накопленный разлок."""
    pts = sorted(points)
    if not pts or today < pts[0][0]:
        return None

    def at(d: date) -> float:
        v = pts[0][1]
        for x, y in pts:
            if x > d:
                break
            v = y
        return v

    a = at(today)
    b = at(today + timedelta(days=days))
    return (b - a) / a if a > 0 else None


def _ok(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


def evaluate(b: Business, q: Quote | None, unlock: float | None, rules: Rules, perp: bool = False) -> Candidate:
    """Меры и причины отсева одной монеты (пустые причины — прошла)."""
    mcap = q.mcap if q else None
    fdv = (q.fdv if q and _ok(q.fdv) and q.fdv > 0 else mcap)
    hr_year = b.hr_30d * 365 / 30
    reasons, flags = [], []
    if b.hr_30d < rules.holders_rev_30d_min:
        reasons.append("нет выручки держателям")
    if q is None or not _ok(mcap) or mcap < rules.mcap_min:
        reasons.append("мелкая или нет цены")
    if q is None or not _ok(q.volume) or q.volume < rules.volume_min:
        reasons.append("неликвидна")
    float_share = mcap / fdv if _ok(mcap) and _ok(fdv) and fdv > 0 else None
    if float_share is not None and float_share < rules.float_min:
        reasons.append("мало в обращении")
    if unlock is not None and unlock > rules.unlock_12m_max:
        reasons.append("разлоки")
    if q is not None and q.chg_1y is None:
        reasons.append("моложе года")
    if b.category.lower() == "chain":
        flags.append("эмиссия стейкинга не учтена")
    g = growth(b.hr_30d, b.hr_1y)
    if g is not None and g > 3:
        flags.append("всплеск выручки за месяц")
    return Candidate(gecko_id=b.key, symbol=(q.symbol if q else "").upper(), project=b.name, category=b.category,
                     price=q.price if q else None, mcap=mcap, fdv=fdv, volume=q.volume if q else None,
                     float_share=float_share, unlock_12m=unlock, fees_year=b.fees_30d * 365 / 30,
                     rev_year=b.rev_30d * 365 / 30, hr_year=hr_year,
                     fdv_to_hr=fdv / hr_year if _ok(fdv) and hr_year > 0 else None, hr_growth=g,
                     chg_1y=q.chg_1y if q else None, ath_chg=q.ath_chg if q else None,
                     reasons=tuple(reasons), flags=tuple(flags), binance_perp=perp)


def _pct_rank(values: list[float | None], higher_better: bool) -> list[float]:
    """Перцентильный ранг (0..1]; пропуск — 0.5."""
    have = sorted(v for v in values if _ok(v))
    n = len(have)
    out = []
    for v in values:
        if not _ok(v) or n == 0:
            out.append(0.5)
            continue
        worse = sum(1 for x in have if (x < v if higher_better else x > v))
        same = sum(1 for x in have if x == v)
        out.append((worse + (same + 1) / 2) / n)
    return out


def rank(cands: list[Candidate]) -> list[Candidate]:
    """Скор прошедших: дешевле (меньше FDV / выручка держателям) и быстрее рост — выше."""
    ok = [c for c in cands if c.passed]
    cheap = _pct_rank([c.fdv_to_hr for c in ok], higher_better=False)
    grow = _pct_rank([c.hr_growth for c in ok], higher_better=True)
    scored = [replace(c, score=round(a + b, 4)) for c, a, b in zip(ok, cheap, grow)]
    return sorted(scored, key=lambda c: (-(c.score or 0), -(c.hr_year or 0)))


def screen(biz: dict[str, Business], quotes: dict[str, Quote], unlocks: dict[str, float],
           rules: Rules, perps: set[str] | None = None, rejected_top: int = 15) -> tuple[list[Candidate], list[Candidate]]:
    """(прошли — по скору, отсеянные крупные — по выручке протокола). biz — уже по токенам (by_token)."""
    perps = perps or set()
    cands = [evaluate(b, quotes.get(g), unlocks.get(g), rules,
                      perp=bool(quotes.get(g)) and f"{quotes[g].symbol.upper()}USDT" in perps)
             for g, b in biz.items()]
    passed = rank(cands)
    rejected = sorted((c for c in cands if not c.passed and c.rev_year > 0), key=lambda c: -c.rev_year)
    return passed, rejected[:rejected_top]


def _ema(x: pd.Series, span: int) -> pd.Series:
    e = x.ewm(span=span, adjust=False).mean()
    e.iloc[: span - 1] = float("nan")
    return e


def regime(weekly: pd.DataFrame) -> Regime:
    """Режим по закрытым неделям BTC (столбцы close, low; индекс — неделя по возрастанию)."""
    w = weekly.dropna(subset=["close"])
    if len(w) < EMA_SLOW:
        return Regime(None, None, None, None, Zone.UNKNOWN)
    c, lo = w["close"].astype(float), w["low"].astype(float)
    e50, e200 = _ema(c, EMA_FAST), _ema(c, EMA_SLOW)
    zones: list[Zone] = []
    armed, run, launched_at = False, 0, None
    for i in range(len(w)):
        if not (e200.iloc[i] == e200.iloc[i]):           # NaN до прогрева
            zones.append(Zone.UNKNOWN)
            continue
        close, f, s = c.iloc[i], e50.iloc[i], e200.iloc[i]
        if close < s:
            armed, run, launched_at = True, 0, None
            zones.append(Zone.ACCUMULATION)
            continue
        run = run + 1 if close > f else 0
        if armed and run >= LAUNCH_CLOSES:
            armed, launched_at = False, i
        rising = i >= SLOPE_WEEKS and e50.iloc[i] > e50.iloc[i - SLOPE_WEEKS]
        bull = rising and close > s
        if launched_at is not None and i - launched_at < LAUNCH_WEEKS and close > f:
            zones.append(Zone.LAUNCH)
        elif bull and lo.iloc[i] <= f * TOUCH and close >= f * FLOOR:
            zones.append(Zone.PULLBACK)
        elif bull and close > f:
            zones.append(Zone.BULL)
        else:
            zones.append(Zone.NEUTRAL)
    z = zones[-1]
    n = 0
    for x in reversed(zones):
        if x != z:
            break
        n += 1
    idx = w.index[-1]
    week = idx.date() if hasattr(idx, "date") else None
    return Regime(week=week, close=float(c.iloc[-1]), ema50=float(e50.iloc[-1]), ema200=float(e200.iloc[-1]),
                  zone=z, weeks_in_zone=n)


def new_entries(prev: Snapshot | None, cur: Snapshot) -> list[Candidate]:
    """Монеты, вошедшие в топ, которых не было в прошлом топе (первый снимок — без уведомления о каждой)."""
    if prev is None:
        return []
    top = cur.rules.top
    was = {c.gecko_id for c in prev.candidates[:top]}
    return [c for c in cur.candidates[:top] if c.gecko_id not in was]


def regime_changed(prev: Snapshot | None, cur: Snapshot) -> bool:
    return (prev is not None and cur.regime.zone != Zone.UNKNOWN and prev.regime.zone != cur.regime.zone)


def money(x: float | None) -> str:
    if not _ok(x):
        return "—"
    for k, s in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= k:
            return f"${x / k:.1f}{s}"
    return f"${x:.0f}"


def alert_text(cur: Snapshot, entries: list[Candidate], regime_moved: bool) -> str | None:
    """Текст уведомления: смена режима BTC и новые монеты в топе."""
    lines = []
    if regime_moved:
        r = cur.regime
        lines.append(f"Скринер · режим рынка: {r.title}. {r.text}")
    if entries:
        lines.append("Скринер · новые монеты в топе:")
        for c in entries:
            lines.append(f"• {c.symbol} ({c.project}) — держателям {money(c.hr_year)}/год, FDV {money(c.fdv)} "
                         f"(×{c.fdv_to_hr:.0f} к выручке), рост выручки ×{c.hr_growth:.1f}" if c.fdv_to_hr and c.hr_growth
                         else f"• {c.symbol} ({c.project}) — держателям {money(c.hr_year)}/год, FDV {money(c.fdv)}")
    return "\n".join(lines) if lines else None


def with_first_seen(cands: list[Candidate], seen: dict[str, tuple[str, float | None]], today: date
                    ) -> tuple[list[Candidate], dict[str, tuple[str, float | None]]]:
    """Бумажный счёт вперёд: дата и цена первого попадания в список прошедших хранятся и не перезаписываются."""
    seen = dict(seen)
    out = []
    for c in cands:
        if c.gecko_id not in seen:
            seen[c.gecko_id] = (today.isoformat(), c.price)
        d, p = seen[c.gecko_id]
        out.append(replace(c, first_seen=date.fromisoformat(d), first_price=p))
    return out, seen
