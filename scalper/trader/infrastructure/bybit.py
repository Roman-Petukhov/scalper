"""Bybit через ccxt: USDT-перпетуалы, единый торговый аккаунт, режим одной позиции на монету.
Демо-счёт (api-demo.bybit.com) включается enable_demo_trading — ключи для него создаются в демо-режиме Bybit."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import ccxt.async_support as ccxt

from ..domain.execution import Account, ClosedPnl, Instrument, OrderRequest, Position
from ..domain.models import EntryKind, Side

log = logging.getLogger(__name__)
KV_KEY = "bybit"


def _closed(i: dict) -> ClosedPnl:
    """Запись closed-pnl Bybit: closedPnl — итог с комиссиями; side — сторона закрывающего ордера (Sell закрывает
    лонг)."""
    return ClosedPnl(side=Side.LONG if i.get("side") == "Sell" else Side.SHORT, qty=_f(i.get("closedSize")),
                     entry=_f(i.get("avgEntryPrice")), exit=_f(i.get("avgExitPrice")), pnl=_f(i.get("closedPnl")),
                     closed_at=datetime.fromtimestamp(int(_f(i.get("updatedTime"))) / 1000, timezone.utc))


def _f(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if v == v else default          # NaN → default


_MULT_SUFFIX = re.compile(r"^([A-Z]+?)(10{2,})USDT$")
_SYMBOL = re.compile(r"^[A-Z0-9]{1,30}USDT$")
ALIASES_KEY = "bybit_aliases"
ALIAS_MAX_GAP = 0.02            # ручное соответствие принимается, если цены Binance и Bybit расходятся не больше чем на 2%


def parse_symbol(raw: str) -> str:
    s = raw.strip().upper()
    if not _SYMBOL.match(s):
        raise ValueError(f"«{raw.strip()}» — не похоже на USDT-перпетуал (пример: PUMPUSDT)")
    return s


def load_aliases(kv: Any) -> dict[str, str]:
    """Ручные соответствия {имя Binance: имя Bybit} из панели."""
    try:
        d = json.loads(kv.kv_get(ALIASES_KEY) or "{}")
    except ValueError:
        return {}
    return {str(k): str(v) for k, v in d.items()} if isinstance(d, dict) else {}


def save_aliases(kv: Any, aliases: dict[str, str]) -> None:
    kv.kv_set(ALIASES_KEY, json.dumps(dict(sorted(aliases.items()))))


def binance_aliases(bybit_ids: set[str]) -> dict[str, str]:
    """Имена Binance для перпетуалов, которые Bybit называет иначе: множитель на Binance — приставка (1000SHIBUSDT),
    на Bybit — суффикс (SHIB1000USDT). Цена и размер контракта у обоих — за 1000 монет. Вернёт {Binance: Bybit}."""
    out = {}
    for i in bybit_ids:
        m = _MULT_SUFFIX.match(i)
        if m and (alias := f"{m[2]}{m[1]}USDT") not in bybit_ids:
            out[alias] = i
    return out


@dataclass(frozen=True)
class BybitCredentials:
    api_key: str
    secret: str
    network: str                              # "demo" / "live"

    def __post_init__(self) -> None:
        if self.network not in ("demo", "live"):
            raise ValueError("сеть — demo или live")
        if len(self.api_key) < 10 or len(self.secret) < 10:
            raise ValueError("ключ и секрет Bybit выглядят неполными")

    @property
    def masked_key(self) -> str:
        return f"{self.api_key[:4]}…{self.api_key[-4:]}"

    def dumps(self) -> str:
        return json.dumps({"api_key": self.api_key, "secret": self.secret, "network": self.network})

    @staticmethod
    def loads(raw: str | None) -> BybitCredentials | None:
        if not raw:
            return None
        try:
            d = json.loads(raw)
            return BybitCredentials(d["api_key"], d["secret"], d["network"])
        except (ValueError, KeyError, TypeError):
            return None


MARKETS_MAX_AGE = 6 * 3600       # список инструментов перечитывается не реже (плечо, шаг цены, делистинги)
MARKETS_MISS_RETRY = 15 * 60     # монеты нет в списке — перечитать не чаще (новый листинг на Bybit)


class BybitBroker:
    def __init__(self, creds: BybitCredentials, exchange: Any | None = None,
                 clock: Any = time.monotonic, manual: Callable[[], dict[str, str]] = dict) -> None:
        self.network = creds.network
        self.ex = exchange or ccxt.bybit({"apiKey": creds.api_key, "secret": creds.secret, "enableRateLimit": True,
                                          "options": {"defaultType": "swap"}})
        if exchange is None and creds.network == "demo":
            self.ex.enable_demo_trading(True)
        self._by_id: dict[str, dict] | None = None
        self._local: dict[str, str] = {}           # id Bybit → имя Binance, под которым монету знает бот
        self._markets_lock = asyncio.Lock()
        self._clock = clock
        self._loaded_at = 0.0
        self._manual = manual                      # ручные соответствия из панели: читаются при каждом обращении

    async def _markets(self, missing: str | None = None) -> dict[str, dict]:
        async with self._markets_lock:
            age = self._clock() - self._loaded_at
            stale = self._by_id is None or age > MARKETS_MAX_AGE or (
                missing is not None and missing not in self._by_id and age > MARKETS_MISS_RETRY)
            if stale:
                markets = await self.ex.load_markets(self._by_id is not None)
                by_id = {m["id"]: m for m in markets.values()
                         if m.get("swap") and m.get("linear") and m.get("settle") == "USDT"}
                aliases = binance_aliases(set(by_id))
                self._local = {v: k for k, v in aliases.items()}
                self._by_id = by_id | {k: by_id[v] for k, v in aliases.items()}
                self._loaded_at = self._clock()
        return self._by_id

    async def _market(self, symbol: str) -> dict | None:
        """Рынок ccxt по имени Binance (ETHUSDT, 1000SHIBUSDT): только линейные USDT-перпетуалы."""
        target = self._manual().get(symbol, symbol)
        return (await self._markets(target)).get(target)

    def _to_local(self, bybit_id: str) -> str:
        back = {v: k for k, v in self._manual().items()}
        return back.get(bybit_id) or self._local.get(bybit_id, bybit_id)

    async def bybit_price(self, bybit_id: str) -> float:
        """Последняя цена по имени Bybit (для проверки ручного соответствия)."""
        m = (await self._markets(bybit_id)).get(bybit_id)
        if m is None or m.get("id") != bybit_id:
            raise ValueError(f"на Bybit нет перпетуала {bybit_id}")
        t = await self.ex.fetch_ticker(m["symbol"])
        return _f(t.get("last"))

    async def instrument(self, symbol: str) -> Instrument | None:
        m = await self._market(symbol)
        if m is None or not m.get("active", True):
            return None
        lim = m.get("limits") or {}
        prec = m.get("precision") or {}
        return Instrument(symbol=symbol, qty_step=_f(prec.get("amount"), 0.0), min_qty=_f((lim.get("amount") or {}).get("min")),
                          tick=_f(prec.get("price"), 0.0), max_leverage=_f((lim.get("leverage") or {}).get("max"), 1.0),
                          min_notional=_f((lim.get("cost") or {}).get("min"), 5.0) or 5.0)

    async def price(self, symbol: str) -> float:
        m = await self._market(symbol)
        if m is None:
            raise ValueError(f"{symbol} не торгуется на Bybit")
        t = await self.ex.fetch_ticker(m["symbol"])
        return _f(t.get("last"))

    async def account(self) -> Account:
        await self._markets()
        bal, positions, orders = await asyncio.gather(
            self.ex.fetch_balance(), self.ex.fetch_positions(None, {"settleCoin": "USDT"}),
            self.ex.fetch_open_orders(None, None, None, {"settleCoin": "USDT"}))
        info = ((bal.get("info") or {}).get("result") or {}).get("list") or [{}]
        equity = _f(info[0].get("totalEquity"), _f((bal.get("total") or {}).get("USDT")))
        available = _f(info[0].get("totalAvailableBalance"), _f((bal.get("free") or {}).get("USDT")))
        pos = []
        for p in positions:
            qty = _f(p.get("contracts"))
            if qty <= 0:
                continue
            raw = self._to_local((p.get("info") or {}).get("symbol") or p.get("symbol", ""))
            pos.append(Position(symbol=raw, side=Side.LONG if p.get("side") == "long" else Side.SHORT, qty=qty,
                                entry=_f(p.get("entryPrice")), mark=_f(p.get("markPrice")),
                                upnl=_f(p.get("unrealizedPnl")),
                                stop=_f(p.get("stopLossPrice")) or None, target=_f(p.get("takeProfitPrice")) or None))
        pending = frozenset(self._to_local((o.get("info") or {}).get("symbol", "")) for o in orders
                            if not o.get("reduceOnly") and not (o.get("info") or {}).get("reduceOnly"))
        return Account(equity=equity, available=available, upnl=sum(p.upnl for p in pos),
                       positions=tuple(sorted(pos, key=lambda x: -abs(x.upnl))), pending_symbols=pending)

    async def _set_leverage(self, leverage: int, sym: str) -> None:
        try:
            await self.ex.set_leverage(leverage, sym)
        except ccxt.BadRequest as e:                           # 110043: плечо уже такое
            if "110043" not in str(e) and "not modified" not in str(e).lower():
                raise

    async def _create(self, sym: str, *args: Any) -> dict:
        """create_order; если монета на Bybit в режиме хеджирования (две позиции), переключить её в режим одной позиции
        (бот работает в нём) и повторить. Bybit разрешает это, только пока по монете нет позиции и ордеров."""
        try:
            return await self.ex.create_order(sym, *args)
        except ccxt.ExchangeError as e:
            if "position idx not match position mode" not in str(e):
                raise
        log.warning("%s в режиме хеджирования на Bybit — переключаю в режим одной позиции", sym)
        try:
            await self.ex.set_position_mode(False, sym)
        except ccxt.ExchangeError as e:
            raise ValueError(f"{sym.replace('/USDT:USDT', 'USDT')} на Bybit в режиме хеджирования, переключить не вышло (есть позиция или ордера). "
                             f"Bybit → режим позиции → One-Way") from e
        return await self.ex.create_order(sym, *args)

    async def place(self, order: OrderRequest) -> str:
        m = await self._market(order.symbol)
        if m is None:
            raise ValueError(f"{order.symbol} не торгуется на Bybit")
        sym = m["symbol"]
        await self._set_leverage(order.leverage, sym)
        side = "buy" if order.side is Side.LONG else "sell"
        params = {"stopLoss": {"triggerPrice": order.stop}, "takeProfit": {"triggerPrice": order.target},
                  "positionIdx": 0, "clientOrderId": order.client_id}
        if order.kind is EntryKind.RETEST:
            res = await self._create(sym, "limit", side, order.qty, order.price, {**params, "timeInForce": "GTC"})
        else:
            res = await self._create(sym, "market", side, order.qty, None, params)
        return str(res.get("id") or res.get("clientOrderId") or order.client_id)

    async def open_order_ids(self) -> set[str]:
        orders = await self.ex.fetch_open_orders(None, None, None, {"settleCoin": "USDT"})
        return {str(o.get("id")) for o in orders}

    async def cancel(self, symbol: str, order_id: str) -> None:
        m = await self._market(symbol)
        if m is None:
            raise ValueError(f"{symbol} не торгуется на Bybit")
        await self.ex.cancel_order(order_id, m["symbol"])

    async def adjust(self, symbol: str, qty: float, leverage: int) -> None:
        m = await self._market(symbol)
        if m is None:
            raise ValueError(f"{symbol} не торгуется на Bybit")
        await self._set_leverage(leverage, m["symbol"])
        await self._create(m["symbol"], "market", "buy" if qty > 0 else "sell", abs(qty), None, {"positionIdx": 0})

    async def close_position(self, symbol: str) -> None:
        m = await self._market(symbol)
        if m is None:
            raise ValueError(f"{symbol} не торгуется на Bybit")
        for p in await self.ex.fetch_positions([m["symbol"]]):
            qty = _f(p.get("contracts"))
            if qty > 0:
                side = "sell" if p.get("side") == "long" else "buy"
                await self.ex.create_order(m["symbol"], "market", side, qty, None,
                                           {"reduceOnly": True, "positionIdx": 0})

    async def closed_pnl(self, symbol: str, since: datetime, until: datetime) -> list[ClosedPnl]:
        """/v5/position/closed-pnl по одной монете."""
        m = await self._market(symbol)
        if m is None:
            return []
        rows = await self.ex.fetch_positions_history([m["symbol"]], int(since.timestamp() * 1000), 100,
                                                     {"until": int(until.timestamp() * 1000)})
        return [_closed(p.get("info") or {}) for p in rows]

    async def closed_pnl_all(self, since: datetime, until: datetime) -> list[ClosedPnl]:
        """/v5/position/closed-pnl по всем USDT-перпетуалам, постранично (по 100 записей)."""
        out: list[ClosedPnl] = []
        cursor = None
        while True:
            req = {"category": "linear", "startTime": int(since.timestamp() * 1000),
                   "endTime": int(until.timestamp() * 1000), "limit": 100}
            if cursor:
                req["cursor"] = cursor
            res = (await self.ex.privateGetV5PositionClosedPnl(req)).get("result") or {}
            rows = res.get("list") or []
            out += [_closed(i) for i in rows]
            cursor = res.get("nextPageCursor")
            if not cursor or not rows:
                return out

    async def close(self) -> None:
        await self.ex.close()


class BrokerHolder:
    """Текущее подключение к бирже по ключам из базы; пересоздаётся, когда ключи меняют в панели."""

    def __init__(self, kv: Any) -> None:
        self.kv = kv
        self._broker: BybitBroker | None = None
        self._creds: BybitCredentials | None = None

    @property
    def credentials(self) -> BybitCredentials | None:
        return BybitCredentials.loads(self.kv.kv_get(KV_KEY))

    def __call__(self) -> BybitBroker | None:
        creds = self.credentials
        if creds != self._creds:
            old, self._creds = self._broker, creds
            self._broker = BybitBroker(creds, manual=lambda: load_aliases(self.kv)) if creds else None
            if old is not None:
                try:
                    asyncio.get_running_loop().create_task(old.close())   # закрыть HTTP-сессию старых ключей
                except RuntimeError:
                    pass                                                  # вне цикла сессии ещё не открывались
        return self._broker

    def save(self, creds: BybitCredentials) -> None:
        self.kv.kv_set(KV_KEY, creds.dumps())

    def forget(self) -> None:
        self.kv.kv_delete(KV_KEY)

    async def close(self) -> None:
        if self._broker is not None:
            await self._broker.close()
