"""Цена к активным адресам: сколько долларов капитализации приходится на один активный адрес сети в день.

Откуда: research/onchainval.py (docs/knowledge.md, «Покупка недооценённых альтов», этап 3). Монеты, дорогие по этой
мере, отставали от BTC во всех периодах с 2022 года (за 6 месяцев −16%, t −2.0); дешёвые BTC не обгоняли устойчиво
(дешёвые минус дорогие +12.6% за 3 месяца, t 1.8). Поэтому это красный флаг для дорогой трети, а не повод покупать
дешёвую. Мера грубая: адрес в сети с UTXO, в сети со счетами и у токена ERC-20 — разные вещи.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime

MIN_COINS = 15              # меньше — трети не считаем: одна монета сдвигает границу
MIN_DAYS = 20               # дней с данными за окно
MIN_ADDRESSES = 100.0       # средних активных адресов в день: меньше — сеть почти не используется, мера — шум
EXPENSIVE, CHEAP, MIDDLE = "expensive", "cheap", "middle"
TITLES = {EXPENSIVE: "дорогая", CHEAP: "дешёвая", MIDDLE: "середина"}


@dataclass(frozen=True)
class Reading:
    """Сырые данные по монете за окно: средние активные адреса в день и последняя капитализация."""
    asset: str                  # тикер источника (btc, uni)
    addresses: float
    cap: float
    days: int


@dataclass(frozen=True)
class AddressValue:
    symbol: str                 # BTC, UNI
    cap: float
    addresses: float
    cap_per_address: float      # долларов капитализации на активный адрес в день
    group: str                  # expensive / middle / cheap
    perp: bool = False

    @property
    def title(self) -> str:
        return TITLES[self.group]


@dataclass(frozen=True)
class OnchainSnapshot:
    computed_at: datetime
    rows: tuple[AddressValue, ...]          # от самой дорогой к самой дешёвой
    assets: int = 0                         # монет в источнике

    @property
    def expensive(self) -> tuple[AddressValue, ...]:
        return tuple(r for r in self.rows if r.group == EXPENSIVE)

    @property
    def cheap(self) -> tuple[AddressValue, ...]:
        return tuple(r for r in reversed(self.rows) if r.group == CHEAP)

    def to_json(self) -> dict:
        return {"computed_at": self.computed_at.isoformat(), "rows": [asdict(r) for r in self.rows], "assets": self.assets}

    @staticmethod
    def from_json(d: dict) -> OnchainSnapshot:
        return OnchainSnapshot(datetime.fromisoformat(d["computed_at"]), tuple(AddressValue(**r) for r in d.get("rows", [])),
                               int(d.get("assets", 0)))


def symbol_of(asset: str, alias: dict[str, str] | None = None) -> str | None:
    """Тикер монеты по имени источника; обёртки и версии на чужих сетях (usdt_eth, wbtc) — не монеты, None."""
    alias = alias or {}
    if asset in alias:
        return alias[asset]
    return None if "_" in asset else asset.upper()


def rank(readings: list[Reading], perps: set[str] | None = None, alias: dict[str, str] | None = None,
         skip: frozenset[str] = frozenset()) -> list[AddressValue]:
    """Трети по цене активного адреса: верхняя — дорогие, нижняя — дешёвые. Монет меньше MIN_COINS — пусто."""
    perps = perps or set()
    vals = []
    for r in readings:
        sym = symbol_of(r.asset, alias)
        if sym is None or sym in skip or r.days < MIN_DAYS:
            continue
        if not (math.isfinite(r.cap) and math.isfinite(r.addresses)) or r.cap <= 0 or r.addresses < MIN_ADDRESSES:
            continue
        vals.append((sym, r.cap, r.addresses, r.cap / r.addresses))
    best: dict[str, tuple] = {}
    for v in vals:                                          # одна монета под двумя именами — берём с большей сетью
        if v[0] not in best or v[2] > best[v[0]][2]:
            best[v[0]] = v
    rows = sorted(best.values(), key=lambda v: -v[3])
    if len(rows) < MIN_COINS:
        return []
    third = len(rows) // 3
    out = []
    for i, (sym, cap, adr, per) in enumerate(rows):
        group = EXPENSIVE if i < third else CHEAP if i >= len(rows) - third else MIDDLE
        out.append(AddressValue(sym, cap, adr, per, group, perp=f"{sym}USDT" in perps))
    return out
