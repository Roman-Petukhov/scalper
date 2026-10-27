"""Тезисы (domain/theses.py) в фоне: индикаторы — раз в сутки (бесплатно: DefiLlama), полный пересчёт тезисов ИИ —
раз в 30 дней, недельная сверка новостей по катализаторам и условиям отмены — раз в 7 дней. Без ключа ИИ индикаторы
всё равно считаются. Снимки и история — в kv; цены кандидатов в день публикации — для проверки «с публикации против BTC»."""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone

from ..domain.screener import Snapshot, money
from ..domain.theses import (Reading, Report, alert_text, catalog_text, fee_readings, output_schema, parse_output,
                             performance, stablecoin_reading, thesis_status, tvl_readings, with_screener, STATUS_TEXT)
from .ports import CategoryTvl, FundamentalsData, KeyValueStore, Notifier, SpotPrices, StablecoinData, ThesesWriter

log = logging.getLogger(__name__)
KEY = "theses"
CHECK_KEY = "theses:check"
READ_KEY = "theses:readings"
TVL_KEY = "theses:tvl"              # своя история TVL по категориям: {категория: {день: $}}
PRICES_KEY = "theses:prices_now"
REPORT_EVERY = timedelta(days=30)
CHECK_EVERY = timedelta(days=7)
READ_EVERY = timedelta(hours=24)
RETRY = timedelta(hours=1)
TVL_KEEP_DAYS = 400


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _load(raw: str | None, default):
    if not raw:
        return default
    try:
        return json.loads(raw)
    except ValueError:
        return default


def _reading_json(r: Reading) -> dict:
    return {"id": r.id, "title": r.title, "value": r.value, "unit": r.unit, "change": r.change, "kind": r.kind,
            "note": r.note}


class ThesesService:
    def __init__(self, writer: ThesesWriter | None, fundamentals: FundamentalsData, tvl: CategoryTvl,
                 stables: StablecoinData, prices: SpotPrices, store: KeyValueStore,
                 screener: Callable[[], Snapshot | None], notifier: Notifier | None = None, panel_url: str = "",
                 model: str = "", clock: Callable[[], datetime] = _utcnow) -> None:
        self.writer, self.fundamentals, self.tvl, self.stables, self.prices_src = writer, fundamentals, tvl, stables, prices
        self.store, self.screener, self.notifier, self.panel_url, self.model = store, screener, notifier, panel_url, model
        self.clock = clock
        self._task: asyncio.Task | None = None
        self._tried: dict[str, datetime] = {}
        self.last_error: str | None = None
        self.job: str | None = None

    # --- чтение ---
    def current(self) -> Report | None:
        raw = self.store.kv_get(KEY)
        if not raw:
            return None
        try:
            return Report.from_json(json.loads(raw))
        except (ValueError, KeyError, TypeError):
            log.exception("тезисы: отчёт не читается")
            return None

    def readings(self) -> tuple[datetime | None, dict[str, Reading]]:
        d = _load(self.store.kv_get(READ_KEY), {})
        at = datetime.fromisoformat(d["at"]) if d.get("at") else None
        return at, {r["id"]: Reading(**r) for r in d.get("items", [])}

    def last_check(self) -> dict | None:
        return _load(self.store.kv_get(CHECK_KEY), None)

    def prices_now(self) -> tuple[datetime | None, dict[str, float]]:
        d = _load(self.store.kv_get(PRICES_KEY), {})
        return (datetime.fromisoformat(d["at"]) if d.get("at") else None), d.get("prices", {})

    def view(self) -> dict:
        """Всё для вкладки: отчёт, статус каждого тезиса по индикаторам, доходность с публикации против BTC."""
        rep, (at, rd), chk = self.current(), self.readings(), self.last_check()
        _, now_prices = self.prices_now()
        perf = performance(rep, now_prices) if rep else {}
        rows = []
        for t in rep.theses if rep else ():
            st = thesis_status(t, rd)
            rows.append({"t": t, "status": st, "status_text": STATUS_TEXT[st],
                         "check": next((x for x in (chk or {}).get("items", []) if x.get("id") == t.id), None)})
        return {"report": rep, "rows": rows, "readings": rd, "readings_at": at, "check": chk, "perf": perf,
                "has_ai": self.writer is not None, "running": self.running, "job": self.job,
                "error": self.last_error}

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    # --- расписание ---
    def _due(self, kind: str, now: datetime) -> bool:
        tried = self._tried.get(kind)
        if tried is not None and now - tried < RETRY:
            return False
        if kind == "readings":
            at, _ = self.readings()
            return at is None or now - at >= READ_EVERY
        if self.writer is None:
            return False
        rep = self.current()
        if kind == "report":
            return rep is None or now - rep.created_at >= REPORT_EVERY
        chk = self.last_check()
        last = datetime.fromisoformat(chk["at"]) if chk and chk.get("at") else None
        return rep is not None and now - rep.created_at >= CHECK_EVERY and (last is None or now - last >= CHECK_EVERY)

    async def maybe_run(self) -> None:
        if self.running:
            return
        now = self.clock()
        for kind in ("readings", "report", "check"):
            if self._due(kind, now):
                self.start(kind, now)
                return

    def start(self, kind: str = "report", now: datetime | None = None) -> bool:
        if self.running or (kind != "readings" and self.writer is None):
            return False
        self._tried[kind] = now or self.clock()
        self.job = kind
        self._task = asyncio.create_task(self._guarded(kind))
        return True

    async def _guarded(self, kind: str) -> None:
        try:
            if kind == "readings":
                await self.refresh_readings()
            elif kind == "report":
                await self.run_report()
            else:
                await self.run_check()
            self.last_error = None
        except Exception as e:
            self.last_error = f"{kind}: {str(e)[:200]}"
            log.exception("тезисы: %s", kind)
        finally:
            self.job = None

    # --- работа ---
    async def refresh_readings(self) -> dict[str, Reading]:
        now = self.clock()
        day = now.date()
        items: list[Reading] = []
        try:
            s = await self.stables.supply()
            r = stablecoin_reading({d.date(): float(v) for d, v in s.items()}, day)
            if r:
                items.append(r)
        except Exception as e:
            log.warning("тезисы: стейблкоины — %s", e)
        try:
            items += fee_readings(await self.fundamentals.fee_rows())
        except Exception as e:
            log.warning("тезисы: комиссии — %s", e)
        try:
            tvl_now = await self.tvl.tvl_by_category()
            raw = _load(self.store.kv_get(TVL_KEY), {})
            hist = {c: {date.fromisoformat(d): float(v) for d, v in h.items()} for c, h in raw.items()}
            items += tvl_readings(tvl_now, hist, day)
            for c, v in tvl_now.items():
                hist.setdefault(c, {})[day] = v
            cut = day - timedelta(days=TVL_KEEP_DAYS)
            self.store.kv_set(TVL_KEY, json.dumps({c: {d.isoformat(): v for d, v in h.items() if d >= cut}
                                                   for c, h in hist.items()}))
        except Exception as e:
            log.warning("тезисы: TVL — %s", e)
        if not items:
            raise RuntimeError("ни один индикатор не загрузился")
        self.store.kv_set(READ_KEY, json.dumps({"at": now.isoformat(), "items": [_reading_json(r) for r in items]}))
        rep = self.current()
        if rep is not None:
            try:
                p = await self.prices_src.prices(rep.symbols + ["BTC"])
                self.store.kv_set(PRICES_KEY, json.dumps({"at": now.isoformat(), "prices": p}))
            except Exception as e:
                log.warning("тезисы: цены — %s", e)
        return {r.id: r for r in items}

    def _screener_lines(self) -> tuple[str, dict[str, int], dict[str, str]]:
        snap = self.screener()
        if snap is None:
            return "Скринер ещё не считался.", {}, {}
        passed = {c.symbol: i + 1 for i, c in enumerate(snap.candidates) if c.symbol}
        rejected = {c.symbol: ", ".join(c.reasons) for c in snap.rejected if c.symbol}
        r = snap.regime
        lines = [f"Режим BTC на неделях: {r.title} (закрытие {r.close or 0:,.0f}, к EMA200 "
                 f"{(r.to_ema200 or 0):+.0%}, к EMA50 {(r.to_ema50 or 0):+.0%}).".replace(",", " "),
                 "Прошли фундаментальный отбор (выручка держателям, нет навеса, торгуется), по скору:"]
        for i, c in enumerate(snap.candidates[:30]):
            lines.append(f"{i + 1}. {c.symbol} — {c.project} ({c.category}): держателям {money(c.hr_year)}/год, "
                         f"FDV {money(c.fdv)}, рост выручки ×{c.hr_growth or 0:.1f}")
        if snap.rejected:
            lines.append("Крупный бизнес, отсеянный правилами:")
            lines += [f"- {c.symbol} — {c.project}: {', '.join(c.reasons)}" for c in snap.rejected]
        return "\n".join(lines), passed, rejected

    def _context(self, readings: dict[str, Reading], screener_text: str) -> str:
        now = self.clock()
        prev = self.current()
        parts = [f"Сегодня {now:%d.%m.%Y}. Сделай 5–8 тезисов (включая 1–2 риск-тезиса)."]
        if prev is not None:
            parts.append(f"Прошлая версия от {prev.created_at:%d.%m.%Y}:")
            for t in prev.theses:
                st = STATUS_TEXT[thesis_status(t, readings)]
                coins = ", ".join(b.symbol for b in t.beneficiaries)
                parts.append(f"- [{t.id}] {t.title}: {t.chain} | кандидаты: {coins} | индикаторы: {st} | "
                             f"отмена: {t.kill}")
        else:
            parts.append("Прошлой версии нет — это первая.")
        parts.append("Каталог индикаторов (в тезисах ссылайся только на эти id):\n" +
                     catalog_text(list(readings.values())))
        parts.append(screener_text)
        return "\n\n".join(parts)

    async def run_report(self) -> Report:
        assert self.writer is not None
        readings = await self.refresh_readings()
        text, passed, rejected = self._screener_lines()
        ids = sorted(readings)
        data, usage = await self.writer.write(self._context(readings, text), output_schema(ids))
        now = self.clock()
        rep = parse_output(data, now, self.model, set(ids))
        if not rep.theses:
            raise RuntimeError("модель не вернула ни одного тезиса с источниками")
        rep = with_screener(rep, passed, rejected)
        try:
            prices = await self.prices_src.prices(rep.symbols + ["BTC"])
        except Exception as e:
            log.warning("тезисы: цены на день публикации — %s", e)
            prices = {}
        rep = Report(created_at=rep.created_at, model=rep.model, macro=rep.macro, summary=rep.summary,
                     theses=rep.theses, prices=prices, input_tokens=getattr(usage, "input_tokens", 0),
                     output_tokens=getattr(usage, "output_tokens", 0), searches=getattr(usage, "searches", 0),
                     cost_usd=getattr(usage, "cost_usd", None))
        raw = json.dumps(rep.to_json())
        self.store.kv_set(KEY, raw)
        self.store.kv_set(f"{KEY}:{now.date().isoformat()}", raw)
        self.store.kv_set(PRICES_KEY, json.dumps({"at": now.isoformat(), "prices": prices}))
        if self.notifier is not None:
            try:
                await self.notifier.text(alert_text(rep, self.panel_url))
            except Exception:
                log.exception("тезисы: уведомление")
        return rep

    async def run_check(self) -> dict:
        assert self.writer is not None
        rep = self.current()
        if rep is None:
            raise RuntimeError("тезисов ещё нет")
        ctx = "\n".join(f"[{t.id}] {t.title}. Катализаторы: {'; '.join(t.catalysts)}. Отмена: {t.kill}"
                        for t in rep.theses)
        data, usage = await self.writer.check(f"Сегодня {self.clock():%d.%m.%Y}.\n{ctx}")
        known = {t.id for t in rep.theses}
        items = [x for x in data.get("items", []) if x.get("id") in known]
        out = {"at": self.clock().isoformat(), "items": items, "cost_usd": getattr(usage, "cost_usd", None)}
        self.store.kv_set(CHECK_KEY, json.dumps(out))
        moved = [x for x in items if x.get("change") in ("against", "kill", "support")]
        if moved and self.notifier is not None:
            titles = {t.id: t.title for t in rep.theses}
            mark = {"support": "за", "against": "против", "kill": "условие отмены"}
            lines = ["Тезисы · новости недели:"] + [f"• {titles[x['id']]} — {mark[x['change']]}: {x['note']}"
                                                   for x in moved]
            try:
                await self.notifier.text("\n".join(lines))
            except Exception:
                log.exception("тезисы: уведомление сверки")
        return out
