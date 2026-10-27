"""Куда заходят деньги по секторам альтов: 7 дней сверх рынка и рост оборота сектора к его 90-дневному среднему.

Исследование research/secorder.py (прогон 40518559504): «сектор, у которого одновременно растут оборот и цена» — лучший
из проверенных признаков следующего сектора (+1.2% в неделю сверх рынка, t 1.8, минус в 2024–2025) — не принят как
сигнал. Здесь — только ориентир «где сейчас деньги», не сигнал на покупку. Состав секторов — как в исследовании.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime

import pandas as pd

SECTORS: dict[str, tuple[str, ...]] = {
    "RWA": (
        "ONDO", "OM", "POLYX", "PLUME", "CFG", "TRU", "PENDLE", "SYRUP", "USUAL"),
    "приватность": (
        "XMR", "ZEC", "DASH", "ZEN", "SCRT", "ROSE", "DUSK", "ARRR"),
    "DePIN": (
        "FIL", "AR", "STORJ", "HNT", "IOTX", "THETA", "GRASS", "ATH", "POWR"),
    "мемы": (
        "DOGE", "SHIB", "PEPE", "FLOKI", "BONK", "WIF", "BOME", "MEME", "PEOPLE", "TURBO", "NEIRO", "POPCAT",
        "MOODENG", "PNUT", "BRETT", "MEW", "DOGS", "TRUMP", "FARTCOIN", "SPX", "GOAT", "ACT", "CHILLGUY",
        "BABYDOGE", "SATS", "ORDI", "MYRO", "PONKE", "BAN", "HIPPO", "WHY", "CAT", "MOG", "PENGU", "PUMP",
        "BANANAS31"),
    "AI": (
        "FET", "AGIX", "OCEAN", "RNDR", "RENDER", "TAO", "WLD", "ARKM", "AI16Z", "VIRTUAL", "NFP", "AI", "IO",
        "AIXBT", "GRIFFAIN", "ZEREBRO", "COOKIE", "PHB", "CTXC", "NMR", "ALT", "AKT", "GLM", "VANA", "KAITO",
        "SAHARA"),
    "L1": (
        "BTC", "ETH", "SOL", "ADA", "AVAX", "DOT", "NEAR", "APT", "SUI", "SEI", "TIA", "ATOM", "TRX", "TON",
        "ALGO", "ICP", "FTM", "S", "EGLD", "HBAR", "XLM", "XRP", "LTC", "BCH", "ETC", "KAS", "INJ", "BNB",
        "EOS", "NEO", "XTZ", "FIL", "KAVA", "ONE", "ZIL", "IOTA", "VET", "QTUM", "WAVES", "FLOW", "MINA",
        "ROSE", "CFX", "KSM", "XMR", "ZEC", "DASH", "THETA", "BERA", "HYPE", "MOVE", "BSV", "LUNA", "LUNA2",
        "CELO", "ASTR", "SXT", "IP"),
    "L2": (
        "ARB", "OP", "MATIC", "POL", "STRK", "ZK", "MANTA", "METIS", "IMX", "BLAST", "SCR", "ZRO", "SKL", "LRC",
        "CELR", "MNT", "STX", "TAIKO", "OMNI", "SAGA", "DYM", "ALT", "AEVO", "ZETA", "W", "ME"),
    "DeFi": (
        "UNI", "AAVE", "MKR", "CRV", "COMP", "SNX", "LDO", "PENDLE", "DYDX", "GMX", "SUSHI", "1INCH", "JUP",
        "RUNE", "ENA", "ETHFI", "CAKE", "BAL", "YFI", "LQTY", "RDNT", "CVX", "FXS", "SSV", "RPL", "JTO", "ONDO",
        "MORPHO", "AERO", "RAY", "ORCA", "EIGEN", "LISTA", "BAKE", "KNC", "ZRX", "BAND", "UMA", "API3", "LINK",
        "PYTH", "TRB"),
    "игры/NFT": (
        "AXS", "SAND", "MANA", "GALA", "APE", "ILV", "PIXEL", "YGG", "BEAMX", "ENJ", "CHZ", "MAGIC", "GMT",
        "PORTAL", "XAI", "ACE", "BIGTIME", "SUPER", "ALICE", "TLM", "HIGH", "VOXEL", "NOT", "CATI", "HMSTR",
        "BLUR", "LOOKS", "FLOW"),
}
SHORT, LONG = 7, 90
ADV_MIN = 5e6                   # ликвидная монета: средний оборот за 30 дней >= $5M, как в исследовании
MIN_COINS = 3


def base_asset(symbol: str) -> str:
    """1000PEPEUSDT → PEPE, 1MBABYDOGEUSDT → BABYDOGE."""
    s = symbol.removesuffix("USDT")
    for p in ("1000000", "1000", "1M"):
        if s.startswith(p) and len(s) > len(p):
            return s[len(p):]
    return s


def sector_of(symbol: str) -> str | None:
    b = base_asset(symbol)
    for name, coins in SECTORS.items():
        if b in coins:
            return name
    return None


@dataclass(frozen=True)
class SectorRow:
    name: str
    coins: int                  # ликвидных монет с данными
    ret_7d: float               # доходность сектора за 7 дней сверх рынка (доля)
    turnover: float             # оборот за 7 дней к среднему за 90, деленный на то же у рынка (×)
    leaders: tuple[str, ...]    # три лучшие монеты сектора за 7 дней
    inflow: bool = False        # растут и цена, и оборот
    top: bool = False           # среди inflow — с наибольшим ростом оборота


@dataclass(frozen=True)
class SectorFlow:
    computed_at: datetime
    day: str                    # последний закрытый день UTC
    rows: tuple[SectorRow, ...]
    market_7d: float
    coins: int = 0
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = asdict(self)
        d["computed_at"] = self.computed_at.isoformat()
        return d

    @staticmethod
    def from_json(d: dict) -> SectorFlow:
        return SectorFlow(computed_at=datetime.fromisoformat(d["computed_at"]), day=d["day"],
                          rows=tuple(SectorRow(**{**r, "leaders": tuple(r["leaders"])}) for r in d["rows"]),
                          market_7d=float(d["market_7d"]), coins=int(d.get("coins", 0)), extra=d.get("extra", {}))

    @property
    def top(self) -> SectorRow | None:
        return next((r for r in self.rows if r.top), None)


def _chain(r: pd.Series) -> float:
    r = r.dropna()
    return float((1 + r).prod() - 1) if len(r) else float("nan")


def compute(bars: dict[str, pd.DataFrame], now: datetime) -> SectorFlow:
    """bars — дневные свечи перпетуалов (close, quote_volume), индекс — день UTC. Рынок — все альты без BTC."""
    close = pd.DataFrame({s: b["close"] for s, b in bars.items() if len(b)}).sort_index()
    qv = pd.DataFrame({s: b["quote_volume"] for s, b in bars.items() if len(b)}).reindex_like(close)
    if close.empty or len(close) < SHORT + 1:
        return SectorFlow(now, "", (), float("nan"))
    r = close.pct_change()
    liquid = qv.rolling(30, min_periods=20).mean() >= ADV_MIN
    last7 = r.tail(SHORT).where(liquid.tail(SHORT))
    alts = [c for c in close.columns if c != "BTCUSDT"]
    mkt_daily = last7[alts].mean(axis=1)
    mkt7 = _chain(mkt_daily)
    tot = qv[alts].sum(axis=1)
    m_ratio = tot.tail(SHORT).mean() / tot.tail(LONG).mean() if tot.tail(LONG).mean() > 0 else float("nan")
    rows = []
    for name in SECTORS:
        cols = [c for c in alts if sector_of(c) == name and bool(liquid[c].iloc[-1])]
        if len(cols) < MIN_COINS:
            continue
        sec7 = _chain(last7[cols].mean(axis=1))
        v = qv[cols].sum(axis=1)
        ratio = (v.tail(SHORT).mean() / v.tail(LONG).mean()) / m_ratio if v.tail(LONG).mean() > 0 else float("nan")
        coin7 = ((1 + r[cols].tail(SHORT)).prod() - 1).sort_values(ascending=False)
        lead = tuple(base_asset(s) for s in coin7.index[:3])
        ex = sec7 - mkt7
        rows.append(SectorRow(name, len(cols), float(ex), float(ratio), lead,
                              inflow=bool(math.isfinite(ex) and math.isfinite(ratio) and ex > 0 and ratio > 1)))
    rows.sort(key=lambda x: (not x.inflow, -(x.turnover if math.isfinite(x.turnover) else -1e9)))
    if rows and rows[0].inflow:
        rows[0] = replace(rows[0], top=True)
    return SectorFlow(now, str(close.index[-1].date()), tuple(rows), mkt7, coins=len(alts))


def flow_changed(prev: SectorFlow | None, cur: SectorFlow) -> str | None:
    """Текст уведомления, если сменился сектор, куда заходят деньги (или он появился)."""
    a, b = (prev.top if prev else None), cur.top
    if b is None or (a is not None and a.name == b.name):
        return None
    lead = ", ".join(b.leaders)
    return (f"Деньги заходят в сектор «{b.name}»: {b.ret_7d:+.1%} сверх рынка за 7 дней, оборот ×{b.turnover:.1f} "
            f"к обычному. Лидеры: {lead}. Это ориентир, не сигнал: в истории такой признак давал +1.2% в неделю "
            f"и не прошёл проверку.")
