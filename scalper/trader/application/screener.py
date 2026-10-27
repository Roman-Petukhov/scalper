"""Фундаментальный скринер (domain/screener.py) в фоне: полный пересчёт раз в EVERY (комиссии DefiLlama обновляются
раз в сутки, цены — постоянно), снимок и бумажный счёт «с момента попадания» — в kv, уведомление о новых монетах
в топе и о смене режима BTC. Кнопка в панели запускает пересчёт сразу."""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone

from ..domain.screener import (Regime, Rules, Snapshot, Zone, alert_text, businesses, by_token, new_entries, regime,
                               regime_changed, screen, unlock_share, with_first_seen)
from .ports import FundamentalsData, KeyValueStore, MarketData, MarketQuotes, Notifier, WeeklyBars

log = logging.getLogger(__name__)
KEY = "screener"
SEEN_KEY = "screener:seen"          # gecko_id → [день, цена] первого попадания в список прошедших
IDS_KEY = "screener:ids"            # найденные поштучно id CoinGecko (родители протоколов, сети)
UNLOCK_KEY = "screener:unlocks"     # доли разлока за 12 мес. и день расчёта: расписания меняются редко
EVERY = timedelta(hours=6)
RETRY = timedelta(minutes=30)
LOOKUP_MAX = 40                     # поштучных запросов id за пересчёт
BTC = "BTCUSDT"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _load(raw: str | None, default):
    if not raw:
        return default
    try:
        return json.loads(raw)
    except ValueError:
        return default


class ScreenerService:
    def __init__(self, source: FundamentalsData, quotes: MarketQuotes, weekly: WeeklyBars, store: KeyValueStore,
                 notifier: Notifier | None = None, market: MarketData | None = None, rules: Rules = Rules(),
                 clock: Callable[[], datetime] = _utcnow, every: timedelta = EVERY) -> None:
        self.source, self.quotes_src, self.weekly_src, self.store = source, quotes, weekly, store
        self.notifier, self.market, self.rules, self.clock, self.every = notifier, market, rules, clock, every
        self._task: asyncio.Task | None = None
        self._tried: datetime | None = None
        self.last_error: str | None = None

    def current(self) -> Snapshot | None:
        raw = self.store.kv_get(KEY)
        if not raw:
            return None
        try:
            return Snapshot.from_json(json.loads(raw))
        except (ValueError, KeyError, TypeError):
            log.exception("скринер: снимок не читается")
            return None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def due(self, now: datetime) -> bool:
        if self.running or (self._tried is not None and now - self._tried < RETRY):
            return False
        s = self.current()
        return s is None or now - s.computed_at >= self.every

    async def maybe_run(self) -> None:
        now = self.clock()
        if self.due(now):
            self.start(now)

    def start(self, now: datetime | None = None) -> bool:
        """Пересчёт в фоне; False — уже идёт."""
        if self.running:
            return False
        self._tried = now or self.clock()
        self._task = asyncio.create_task(self._guarded())
        return True

    async def _guarded(self) -> None:
        try:
            prev = self.current()
            snap = await self.run()
            self.store.kv_set(KEY, json.dumps(snap.to_json()))
            self.store.kv_set(f"{KEY}:{snap.computed_at.date().isoformat()}", json.dumps(snap.to_json()))
            self.last_error = None
            text = alert_text(snap, new_entries(prev, snap), regime_changed(prev, snap))
            if text and self.notifier is not None:
                try:
                    await self.notifier.text(text)
                except Exception:
                    log.exception("скринер: уведомление")
        except Exception as e:
            self.last_error = str(e)[:200]
            log.exception("скринер")

    async def _ids(self, keys: list[str], known: dict[str, str]) -> dict[str, str]:
        """id CoinGecko для бизнесов, которых нет в общем списке; найденные запоминаются, ненайденные — тоже ("")."""
        cache: dict[str, str] = _load(self.store.kv_get(IDS_KEY), {})
        asked = 0
        for k in keys:
            if k in known or k in cache:
                continue
            if asked >= LOOKUP_MAX:
                break
            asked += 1
            try:
                cache[k] = await self.source.token_id(k) or ""
            except Exception as e:
                log.warning("скринер: id для %s — %s", k, e)
        self.store.kv_set(IDS_KEY, json.dumps(cache))
        return {**known, **{k: v for k, v in cache.items() if v}}

    async def _unlocks(self, keys_by_token: dict[str, set[str]], today: date) -> dict[str, float]:
        cached = _load(self.store.kv_get(UNLOCK_KEY), {})
        if cached.get("day") == today.isoformat() and set(cached.get("asked", [])) >= set(keys_by_token):
            return cached.get("share", {})
        keys = set().union(*keys_by_token.values()) if keys_by_token else set()
        sched = await self.source.unlock_schedules(keys)
        share = {}
        for g, ks in keys_by_token.items():
            for k in ks:
                if k in sched:
                    v = unlock_share(sched[k], today)
                    if v is not None:
                        share[g] = v
                        break
        self.store.kv_set(UNLOCK_KEY, json.dumps({"day": today.isoformat(), "asked": sorted(keys_by_token),
                                                  "share": share}))
        return share

    async def run(self) -> Snapshot:
        now = self.clock()
        rows = await self.source.fee_rows()
        biz = businesses(rows)
        slugs: dict[str, set[str]] = {}
        for r in rows:
            slugs.setdefault(r.group, set()).update({r.group, r.slug.lower()})
        known = await self.source.token_ids()
        need = [k for k, b in sorted(biz.items(), key=lambda kv: -kv[1].hr_30d)
                if b.hr_30d >= self.rules.holders_rev_30d_min or b.rev_30d >= self.rules.holders_rev_30d_min]
        ids = await self._ids(need, known)
        gecko = {k: ids[k] for k in biz if k in ids}
        tok = by_token(biz, gecko)
        quotes = await self.quotes_src.quotes(list(tok))
        keys_by_token: dict[str, set[str]] = {}
        for k, g in gecko.items():
            if g in tok and tok[g].hr_30d >= self.rules.holders_rev_30d_min:
                keys_by_token.setdefault(g, set()).update(slugs.get(k, {k}))
        unlocks = await self._unlocks(keys_by_token, now.date())
        perps: set[str] = set()
        if self.market is not None:
            try:
                perps = set(await self.market.universe(0.0))
            except Exception as e:
                log.warning("скринер: список перпетуалов — %s", e)
        passed, rejected = screen(tok, quotes, unlocks, self.rules, perps)
        seen = {k: tuple(v) for k, v in _load(self.store.kv_get(SEEN_KEY), {}).items()}
        passed, seen = with_first_seen(passed, seen, now.date())
        self.store.kv_set(SEEN_KEY, json.dumps(seen))
        try:
            reg = regime(await self.weekly_src.weekly(BTC))
        except Exception as e:
            log.warning("скринер: недели BTC — %s", e)
            reg = Regime(None, None, None, None, Zone.UNKNOWN)
        return Snapshot(computed_at=now, regime=reg, candidates=tuple(passed), rejected=tuple(rejected),
                        businesses=len(biz), with_holders_rev=sum(1 for b in tok.values()
                                                                  if b.hr_30d >= self.rules.holders_rev_30d_min),
                        priced=len(quotes), unlocks_known=len(unlocks), rules=self.rules)
