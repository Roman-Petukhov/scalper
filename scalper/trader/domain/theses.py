"""Тезисы «куда пойдут деньги» на 1–3 года: мировой сдвиг → причинная цепочка до крипты → кто получает деньги →
опережающие индикаторы → катализаторы → условие отмены (формат — docs/theses.md). Пишет их ИИ с веб-поиском раз в
месяц (infrastructure/claude_theses.py); здесь — модель данных, каталог измеримых индикаторов, сверка тезиса с
индикаторами и бумажная проверка «с публикации против BTC». Чистые правила без сети."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime
from enum import Enum

UP, DOWN = "up", "down"
GROWTH_OK = 1.05            # индикатор-отношение (30 дней к среднему месяцу года) подтверждает рост от ×1.05
CHANGE_OK = 0.02            # индикатор-уровень (изменение за 30 дней) подтверждает от ±2%
CAPTURE = ("strong", "medium", "weak", "none")
CONFIDENCE = ("high", "medium", "low")
VERDICTS = ("new", "kept", "strengthened", "weakened", "killed")


class Status(str, Enum):
    CONFIRMED = "confirmed"
    AGAINST = "against"
    FLAT = "flat"
    NO_DATA = "no_data"


STATUS_TEXT = {Status.CONFIRMED: "подтверждается", Status.AGAINST: "против тезиса", Status.FLAT: "без изменений",
               Status.NO_DATA: "нет данных"}


@dataclass(frozen=True)
class Reading:
    """Измеримый индикатор: уровень и его изменение; kind ratio — change это отношение (×), level — доля за 30 дней."""
    id: str
    title: str
    value: float | None
    unit: str
    change: float | None
    kind: str = "level"            # "level" | "ratio"
    note: str = ""


@dataclass(frozen=True)
class IndicatorRef:
    id: str
    direction: str                 # up | down — что должно происходить, если тезис верен
    why: str


@dataclass(frozen=True)
class Beneficiary:
    symbol: str
    role: str
    capture: str                   # strong | medium | weak | none — доходит ли выручка до держателя токена
    screener: str = ""             # заполняет сервис: «прошёл, №3», «отсеян: разлоки», «нет в скринере»


@dataclass(frozen=True)
class Source:
    title: str
    url: str


@dataclass(frozen=True)
class Thesis:
    id: str
    title: str
    shift: str
    chain: str
    beneficiaries: tuple[Beneficiary, ...]
    indicators: tuple[IndicatorRef, ...]
    catalysts: tuple[str, ...]
    kill: str
    horizon: str
    confidence: str
    priced_in: str
    verdict: str                   # что стало с тезисом относительно прошлой версии
    sources: tuple[Source, ...]
    risk: bool = False             # риск-тезис: не про покупку, а про угрозу рынку


@dataclass(frozen=True)
class Report:
    created_at: datetime
    model: str
    macro: str
    summary: str
    theses: tuple[Thesis, ...]
    prices: dict[str, float] = field(default_factory=dict)     # цены кандидатов в день публикации (USDT, спот)
    input_tokens: int = 0
    output_tokens: int = 0
    searches: int = 0
    cost_usd: float | None = None

    def to_json(self) -> dict:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        return d

    @staticmethod
    def from_json(d: dict) -> Report:
        theses = []
        for t in d.get("theses", []):
            theses.append(Thesis(
                id=t["id"], title=t["title"], shift=t["shift"], chain=t["chain"],
                beneficiaries=tuple(Beneficiary(**b) for b in t.get("beneficiaries", [])),
                indicators=tuple(IndicatorRef(**i) for i in t.get("indicators", [])),
                catalysts=tuple(t.get("catalysts", [])), kill=t["kill"], horizon=t["horizon"],
                confidence=t["confidence"], priced_in=t["priced_in"], verdict=t.get("verdict", "new"),
                sources=tuple(Source(**s) for s in t.get("sources", [])), risk=bool(t.get("risk", False))))
        return Report(created_at=datetime.fromisoformat(d["created_at"]), model=d.get("model", ""),
                      macro=d.get("macro", ""), summary=d.get("summary", ""), theses=tuple(theses),
                      prices={k: float(v) for k, v in (d.get("prices") or {}).items()},
                      input_tokens=int(d.get("input_tokens", 0)), output_tokens=int(d.get("output_tokens", 0)),
                      searches=int(d.get("searches", 0)), cost_usd=d.get("cost_usd"))

    @property
    def symbols(self) -> list[str]:
        seen: list[str] = []
        for t in self.theses:
            for b in t.beneficiaries:
                if b.symbol and b.symbol not in seen:
                    seen.append(b.symbol)
        return seen


def _ok(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


def indicator_status(ref: IndicatorRef, r: Reading | None) -> Status:
    """Идёт ли индикатор туда, куда велит тезис."""
    if r is None or not _ok(r.change):
        return Status.NO_DATA
    if r.kind == "ratio":
        up, down = r.change >= GROWTH_OK, r.change <= 1 / GROWTH_OK
    else:
        up, down = r.change >= CHANGE_OK, r.change <= -CHANGE_OK
    if not (up or down):
        return Status.FLAT
    good = up if ref.direction == UP else down
    return Status.CONFIRMED if good else Status.AGAINST


def thesis_status(t: Thesis, readings: dict[str, Reading]) -> Status:
    """Итог по тезису: большинство индикаторов с данными за — подтверждается, против — против; иначе без изменений."""
    st = [indicator_status(i, readings.get(i.id)) for i in t.indicators]
    have = [s for s in st if s != Status.NO_DATA]
    if not have:
        return Status.NO_DATA
    pro, con = have.count(Status.CONFIRMED), have.count(Status.AGAINST)
    if pro > con and pro * 2 >= len(have):
        return Status.CONFIRMED
    if con > pro and con * 2 >= len(have):
        return Status.AGAINST
    return Status.FLAT


def performance(report: Report, now_prices: dict[str, float], btc: str = "BTC") -> dict[str, float | None]:
    """Доходность каждого кандидата с публикации минус доходность BTC за то же время (бумажная проверка)."""
    b0, b1 = report.prices.get(btc), now_prices.get(btc)
    base = b1 / b0 - 1 if b0 and b1 else None
    out: dict[str, float | None] = {}
    for s in report.symbols:
        p0, p1 = report.prices.get(s), now_prices.get(s)
        out[s] = (p1 / p0 - 1 - (base or 0.0)) if p0 and p1 and base is not None else None
    return out


def thesis_performance(t: Thesis, perf: dict[str, float | None]) -> float | None:
    """Средняя доходность сверх BTC по кандидатам тезиса с ценой."""
    xs = [perf[b.symbol] for b in t.beneficiaries if perf.get(b.symbol) is not None]
    return sum(xs) / len(xs) if xs else None


def with_screener(report: Report, passed: dict[str, int], rejected: dict[str, str]) -> Report:
    """Пометка скринера у каждого кандидата: место в списке прошедших, причина отсева или «нет в скринере»."""
    theses = []
    for t in report.theses:
        bs = []
        for b in t.beneficiaries:
            s = b.symbol.upper()
            mark = (f"скринер: №{passed[s]}" if s in passed else
                    f"скринер отсеял: {rejected[s]}" if s in rejected else "нет в скринере")
            bs.append(replace(b, screener=mark))
        theses.append(replace(t, beneficiaries=tuple(bs)))
    return replace(report, theses=tuple(theses))


def output_schema(indicator_ids: list[str]) -> dict:
    """JSON-схема ответа модели; id индикаторов — только из каталога (ссылка на несуществующий не пройдёт)."""
    s = {"type": "string"}
    ind_id = {"type": "string", "enum": indicator_ids} if indicator_ids else s
    return {
        "type": "object", "additionalProperties": False,
        "required": ["macro", "summary", "theses"],
        "properties": {
            "macro": s, "summary": s,
            "theses": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["id", "title", "shift", "chain", "beneficiaries", "indicators", "catalysts", "kill",
                             "horizon", "confidence", "priced_in", "verdict", "sources", "risk"],
                "properties": {
                    "id": s, "title": s, "shift": s, "chain": s, "kill": s, "horizon": s, "priced_in": s,
                    "risk": {"type": "boolean"},
                    "confidence": {"type": "string", "enum": list(CONFIDENCE)},
                    "verdict": {"type": "string", "enum": list(VERDICTS)},
                    "catalysts": {"type": "array", "items": s},
                    "beneficiaries": {"type": "array", "items": {
                        "type": "object", "additionalProperties": False, "required": ["symbol", "role", "capture"],
                        "properties": {"symbol": s, "role": s, "capture": {"type": "string", "enum": list(CAPTURE)}}}},
                    "indicators": {"type": "array", "items": {
                        "type": "object", "additionalProperties": False, "required": ["id", "direction", "why"],
                        "properties": {"id": ind_id, "direction": {"type": "string", "enum": [UP, DOWN]}, "why": s}}},
                    "sources": {"type": "array", "items": {
                        "type": "object", "additionalProperties": False, "required": ["title", "url"],
                        "properties": {"title": s, "url": s}}},
                }}},
        },
    }


def parse_output(d: dict, created_at: datetime, model: str, known: set[str]) -> Report:
    """Ответ модели → отчёт. Тезисы без источников отбрасываются, индикаторы вне каталога — тоже."""
    theses = []
    for t in d.get("theses", []):
        sources = tuple(Source(str(x.get("title", "")), str(x["url"])) for x in t.get("sources", [])
                        if str(x.get("url", "")).startswith("http"))
        if not sources:
            continue
        theses.append(Thesis(
            id=str(t["id"]), title=str(t["title"]), shift=str(t["shift"]), chain=str(t["chain"]),
            beneficiaries=tuple(Beneficiary(str(b["symbol"]).upper().strip(), str(b["role"]),
                                            b["capture"] if b["capture"] in CAPTURE else "none")
                                for b in t.get("beneficiaries", []) if str(b.get("symbol", "")).strip()),
            indicators=tuple(IndicatorRef(str(i["id"]), i["direction"] if i["direction"] in (UP, DOWN) else UP,
                                          str(i["why"])) for i in t.get("indicators", []) if i.get("id") in known),
            catalysts=tuple(str(c) for c in t.get("catalysts", [])), kill=str(t["kill"]), horizon=str(t["horizon"]),
            confidence=t["confidence"] if t["confidence"] in CONFIDENCE else "low", priced_in=str(t["priced_in"]),
            verdict=t["verdict"] if t["verdict"] in VERDICTS else "new", sources=sources, risk=bool(t.get("risk"))))
    return Report(created_at=created_at, model=model, macro=str(d.get("macro", "")),
                  summary=str(d.get("summary", "")), theses=tuple(theses))


def catalog_text(readings: list[Reading]) -> str:
    """Каталог индикаторов для модели: id, что измеряет, текущее значение."""
    lines = []
    for r in readings:
        if r.kind == "ratio":
            ch = f"30 дн. к среднему месяцу года ×{r.change:.2f}" if _ok(r.change) else "изменение неизвестно"
        else:
            ch = f"за 30 дн. {r.change:+.1%}" if _ok(r.change) else "изменение пока не накоплено"
        val = f"{r.value:,.0f} {r.unit}".replace(",", " ") if _ok(r.value) else "—"
        lines.append(f"- {r.id}: {r.title}; сейчас {val}; {ch}")
    return "\n".join(lines)


def alert_text(r: Report, panel_url: str = "") -> str:
    head = f"Тезисы обновлены ({r.created_at:%d.%m.%Y}): {len(r.theses)} шт."
    lines = [head, r.summary.strip()] if r.summary.strip() else [head]
    for t in r.theses[:8]:
        mark = {"new": "новый", "kept": "без изменений", "strengthened": "усилен", "weakened": "ослаб",
                "killed": "снят"}.get(t.verdict, t.verdict)
        coins = ", ".join(b.symbol for b in t.beneficiaries[:4])
        lines.append(f"• {t.title} — {mark}{f' · {coins}' if coins else ''}")
    if panel_url:
        lines.append(f"{panel_url.rstrip('/')}/#theses")
    return "\n".join(lines)


def level_change(history: dict[date, float], today: date, days: int = 30) -> float | None:
    """Изменение уровня за days дней по собственной истории (ближайшая точка не позже today − days)."""
    past = [d for d in history if (today - d).days >= days]
    if not past or today not in history:
        return None
    d0 = max(past)
    v0 = history[d0]
    return history[today] / v0 - 1 if v0 else None


def slug(text: str) -> str:
    return "-".join(str(text).lower().replace("/", " ").split())


def fee_readings(rows: list, min_fees_30d: float = 1e6) -> list[Reading]:
    """Комиссии по категориям протоколов (DefiLlama): 30 дней к среднему месяцу года. rows — FeeRow скринера."""
    acc: dict[str, list[float]] = {}
    for r in rows:
        if r.data_type != "dailyFees":
            continue
        a = acc.setdefault(r.category, [0.0, 0.0])
        a[0] += max(r.d30, 0.0)
        a[1] += max(r.y1, 0.0)
    out = []
    for cat, (d30, y1) in sorted(acc.items(), key=lambda kv: -kv[1][0]):
        if d30 < min_fees_30d or cat in ("—", ""):
            continue
        out.append(Reading(id=f"fees:{slug(cat)}", title=f"Комиссии категории «{cat}» за 30 дней", value=d30,
                           unit="$", change=d30 / (y1 / 12) if y1 > 0 else None, kind="ratio"))
    return out


def tvl_readings(today: dict[str, float], history: dict[str, dict[date, float]], day: date,
                 min_tvl: float = 100e6) -> list[Reading]:
    """TVL по категориям: уровень сегодня и изменение за 30 дней по собственной истории (копится с запуска)."""
    out = []
    for cat, v in sorted(today.items(), key=lambda kv: -kv[1]):
        if v < min_tvl or cat in ("—", ""):
            continue
        h = dict(history.get(cat, {}))
        h[day] = v
        out.append(Reading(id=f"tvl:{slug(cat)}", title=f"TVL категории «{cat}»", value=v, unit="$",
                           change=level_change(h, day), kind="level"))
    return out


def stablecoin_reading(series: dict[date, float], day: date) -> Reading | None:
    if not series:
        return None
    last = max(d for d in series if d <= day) if any(d <= day for d in series) else None
    if last is None:
        return None
    return Reading(id="stablecoins", title="Все долларовые стейблкоины в обращении", value=series[last], unit="$",
                   change=level_change(series, last), kind="level")
