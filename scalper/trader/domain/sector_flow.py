"""Куда заходят деньги по секторам альтов: сводный балл сектора и режим BTC.

Сводный балл — как в research/posflow.py (прогон 40542971766): среднее z-оценок по секторам четырёх признаков за 7 дней —
доходность сверх рынка, оборот к своим 90 дням (сверх рынка), доля рыночных покупок к своим 90 дням, рост открытого
интереса. Сектор с наибольшим баллом следующую неделю был лучше рынка на +2.0% (t 2.7; в неделях, когда BTC выше
недельной EMA50, +2.95%, вне их — около нуля), но порог t >= 3 не пройден и VAL слабый. Поэтому здесь — ориентир и
бумажная проверка вперёд (domain/sector_paper.py), а не сигнал. Состав секторов — как в исследовании.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime

import numpy as np
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
STABLE_DAYS = 28                # окно роста предложения стейблкоинов
STABLE_MIN = 0.01               # приток: предложение за окно выросло больше чем на 1% (research/rotlead.py, G1)
MIN_PARTS = 2                   # сводный балл — хотя бы по двум признакам
EMA_WEEKS = 50


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


NAN = float("nan")


@dataclass(frozen=True)
class SectorRow:
    name: str
    coins: int                  # ликвидных монет с данными
    ret_7d: float               # доходность сектора за 7 дней сверх рынка (доля)
    turnover: float             # оборот за 7 дней к среднему за 90, деленный на то же у рынка (×)
    leaders: tuple[str, ...]    # три лучшие монеты сектора за 7 дней
    inflow: bool = False        # растут и цена, и оборот
    top: bool = False           # наибольший сводный балл
    taker: float = NAN          # доля рыночных покупок за 7 дней к своим 90 сверх секторных монет (log-odds)
    oi_7d: float = NAN          # рост открытого интереса за 7 дней сверх секторных монет (log)
    score: float = NAN          # сводный балл: среднее z-оценок признаков по секторам
    members: tuple[str, ...] = ()   # ликвидные монеты сектора — корзина для бумажной проверки


OPTIONAL = ("taker", "oi_7d", "score")


@dataclass(frozen=True)
class SectorFlow:
    computed_at: datetime
    day: str                    # последний закрытый день UTC
    rows: tuple[SectorRow, ...]
    market_7d: float
    coins: int = 0
    extra: dict = field(default_factory=dict)
    bull: bool | None = None    # BTC выше недельной EMA50 (по закрытой неделе); None — мало истории
    btc_ema: float | None = None    # закрытие BTC к недельной EMA50
    stable_28d: float | None = None     # рост предложения долларовых стейблкоинов за 28 дней; None — нет данных

    @property
    def inflow(self) -> bool | None:
        """В крипту заходят новые доллары: предложение стейблкоинов за 28 дней выросло больше чем на STABLE_MIN."""
        return None if self.stable_28d is None else self.stable_28d > STABLE_MIN

    def to_json(self) -> dict:
        d = asdict(self)
        d["computed_at"] = self.computed_at.isoformat()
        if d["stable_28d"] is not None and not math.isfinite(d["stable_28d"]):
            d["stable_28d"] = None
        for r in d["rows"]:                             # NaN в JSON не по стандарту — храним null
            for k in OPTIONAL:
                if not math.isfinite(r[k]):
                    r[k] = None
        return d

    @staticmethod
    def from_json(d: dict) -> SectorFlow:
        rows = tuple(SectorRow(**{**r, "leaders": tuple(r["leaders"]), "members": tuple(r.get("members", ())),
                                  **{k: NAN if r.get(k) is None else float(r[k]) for k in OPTIONAL}})
                     for r in d["rows"])
        return SectorFlow(computed_at=datetime.fromisoformat(d["computed_at"]), day=d["day"], rows=rows,
                          market_7d=float(d["market_7d"]), coins=int(d.get("coins", 0)), extra=d.get("extra", {}),
                          bull=d.get("bull"), btc_ema=d.get("btc_ema"), stable_28d=d.get("stable_28d"))

    @property
    def top(self) -> SectorRow | None:
        return next((r for r in self.rows if r.top), None)

    def best_by(self, key: str) -> SectorRow | None:
        """Сектор с наибольшим значением признака (score / ret_7d); NaN не считается."""
        ok = [r for r in self.rows if math.isfinite(getattr(r, key))]
        return max(ok, key=lambda r: getattr(r, key)) if ok else None


def _chain(r: pd.Series) -> float:
    r = r.dropna()
    return float((1 + r).prod() - 1) if len(r) else NAN


def stable_growth(supply: pd.Series, day: str, days: int = STABLE_DAYS) -> float | None:
    """Рост предложения стейблкоинов за days дней на конец дня day; None — нет ряда или он короче окна."""
    if supply is None or not len(supply) or not day:
        return None
    s = supply.sort_index()
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    s = s[s.index <= pd.Timestamp(day)].dropna()
    if not len(s):
        return None
    before = s[s.index <= s.index[-1] - pd.Timedelta(days=days)]
    if not len(before) or before.iloc[-1] <= 0 or s.iloc[-1] <= 0 or (s.index[-1] - before.index[-1]).days > days + 3:
        return None
    return float(s.iloc[-1] / before.iloc[-1] - 1)


def btc_regime(close: pd.Series) -> tuple[bool | None, float | None]:
    """BTC выше недельной EMA50 по последней ЗАКРЫТОЙ неделе (воскресенье UTC). Нужно >= 30 недель."""
    c = close.dropna()
    if c.empty:
        return None, None
    w = c.resample("W-SUN").last()
    if c.index[-1].day_of_week != 6:                      # неделя не закрыта — её не берём
        w = w.iloc[:-1]
    ema = w.ewm(span=EMA_WEEKS, adjust=False, min_periods=30).mean()
    if ema.empty or not math.isfinite(float(ema.iloc[-1])):
        return None, None
    ratio = float(w.iloc[-1] / ema.iloc[-1])
    return ratio > 1, ratio


def _relative(coin: pd.Series, groups: dict[str, list[str]]) -> dict[str, float]:
    """Среднее признака монет сектора минус среднее по всем секторным монетам; < MIN_COINS значений — NaN."""
    every = [c for cols in groups.values() for c in cols]
    vals = coin.reindex(every).dropna()
    base = float(vals.mean()) if len(vals) else NAN
    out = {}
    for name, cols in groups.items():
        v = coin.reindex(cols).dropna()
        out[name] = float(v.mean() - base) if len(v) >= MIN_COINS else NAN
    return out


def composite(features: dict[str, dict[str, float]]) -> dict[str, float]:
    """features: признак → {сектор: значение}. Балл — среднее z-оценок по секторам, если признаков >= MIN_PARTS."""
    names = sorted({s for f in features.values() for s in f})
    zs: dict[str, list[float]] = {s: [] for s in names}
    for f in features.values():
        v = pd.Series(f, dtype="float64").reindex(names)
        mu, sd = v.mean(), v.std()
        if not (math.isfinite(sd) and sd > 0):
            continue
        for s, x in ((v - mu) / sd).items():
            if math.isfinite(x):
                zs[s].append(float(x))
    return {s: (sum(z) / len(z) if len(z) >= MIN_PARTS else NAN) for s, z in zs.items()}


def _taker_tilt(bars: dict[str, pd.DataFrame], cols: list[str]) -> pd.Series:
    out = {}
    for c in cols:
        b = bars[c]
        if "taker_buy_volume" not in b or "volume" not in b:
            continue
        share = (b["taker_buy_volume"] / b["volume"].where(b["volume"] > 0)).clip(0.01, 0.99)
        lg = np.log(share / (1 - share)).dropna()
        if len(lg) >= 60:
            out[c] = float(lg.tail(SHORT).mean() - lg.tail(LONG).mean())
    return pd.Series(out, dtype="float64")


def _oi_growth(oi: dict[str, pd.Series], cols: list[str]) -> pd.Series:
    out = {}
    for c in cols:
        s = oi.get(c)
        if s is None:
            continue
        s = s.dropna()
        s = s[s > 0]
        if len(s) >= SHORT + 1:
            out[c] = float(np.log(s.iloc[-1] / s.iloc[-1 - SHORT]))
    return pd.Series(out, dtype="float64")


def compute(bars: dict[str, pd.DataFrame], now: datetime, oi: dict[str, pd.Series] | None = None,
            bull: bool | None = None, btc_ema: float | None = None, stables: pd.Series | None = None) -> SectorFlow:
    """bars — дневные свечи перпетуалов (close, quote_volume; для агрессоров — volume, taker_buy_volume), индекс — день
    UTC; oi — открытый интерес ($) по дням. Рынок — все альты без BTC."""
    close = pd.DataFrame({s: b["close"] for s, b in bars.items() if len(b)}).sort_index()
    qv = pd.DataFrame({s: b["quote_volume"] for s, b in bars.items() if len(b)}).reindex_like(close)
    if close.empty or len(close) < SHORT + 1:
        return SectorFlow(now, "", (), NAN, bull=bull, btc_ema=btc_ema)
    r = close.pct_change()
    liquid = qv.rolling(30, min_periods=20).mean() >= ADV_MIN
    last7 = r.tail(SHORT).where(liquid.tail(SHORT))
    alts = [c for c in close.columns if c != "BTCUSDT"]
    mkt_daily = last7[alts].mean(axis=1)
    mkt7 = _chain(mkt_daily)
    tot = qv[alts].sum(axis=1)
    m_ratio = tot.tail(SHORT).mean() / tot.tail(LONG).mean() if tot.tail(LONG).mean() > 0 else NAN
    groups = {name: [c for c in alts if sector_of(c) == name and bool(liquid[c].iloc[-1])] for name in SECTORS}
    groups = {k: v for k, v in groups.items() if len(v) >= MIN_COINS}
    every = [c for cols in groups.values() for c in cols]
    taker = _relative(_taker_tilt(bars, every), groups)
    oi_g = _relative(_oi_growth(oi or {}, every), groups)
    base: dict[str, dict] = {}
    for name, cols in groups.items():
        sec_daily = last7[cols].mean(axis=1)
        sec7 = _chain(sec_daily)
        v = qv[cols].sum(axis=1)
        ratio = (v.tail(SHORT).mean() / v.tail(LONG).mean()) / m_ratio if v.tail(LONG).mean() > 0 else NAN
        coin7 = ((1 + r[cols].tail(SHORT)).prod() - 1).sort_values(ascending=False)
        mom = float(np.log1p((sec_daily - mkt_daily).dropna()).sum())
        base[name] = {"cols": cols, "ex": float(sec7 - mkt7), "ratio": float(ratio), "mom": mom,
                      "lead": tuple(base_asset(s) for s in coin7.index[:3])}
    feats = {"mom": {k: b["mom"] for k, b in base.items()},
             "att": {k: (math.log(b["ratio"]) if math.isfinite(b["ratio"]) and b["ratio"] > 0 else NAN)
                     for k, b in base.items()},
             "taker": taker, "oi": oi_g}
    score = composite(feats)
    rows = []
    for name, b in base.items():
        ex, ratio = b["ex"], b["ratio"]
        rows.append(SectorRow(name, len(b["cols"]), ex, ratio, b["lead"],
                              inflow=bool(math.isfinite(ex) and math.isfinite(ratio) and ex > 0 and ratio > 1),
                              taker=taker.get(name, NAN), oi_7d=oi_g.get(name, NAN), score=score.get(name, NAN),
                              members=tuple(b["cols"])))
    rows.sort(key=lambda x: -(x.score if math.isfinite(x.score) else -1e9))
    if rows and math.isfinite(rows[0].score):
        rows[0] = replace(rows[0], top=True)
    day = str(close.index[-1].date())
    return SectorFlow(now, day, tuple(rows), mkt7, coins=len(alts), bull=bull, btc_ema=btc_ema,
                      stable_28d=stable_growth(stables, day) if stables is not None else None)


HISTORY_DAYS = 90
BARS_FOR_HISTORY = LONG + HISTORY_DAYS + 20     # 90 дней графика + 90 на норму оборота и покупок


def _none(v: float) -> float | None:
    return round(float(v), 6) if v is not None and math.isfinite(v) else None


def _zscore_mean(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Сводный балл по дням: среднее z-оценок по секторам, если признаков >= MIN_PARTS (как composite())."""
    zs = []
    for p in parts:
        mu, sd = p.mean(axis=1), p.std(axis=1)
        zs.append(p.sub(mu, axis=0).div(sd.where(sd > 0), axis=0).to_numpy(dtype="float64"))
    stack = np.stack(zs)
    cnt = np.isfinite(stack).sum(axis=0)
    total = np.nansum(stack, axis=0)
    return pd.DataFrame(np.where(cnt >= MIN_PARTS, total / np.maximum(cnt, 1), np.nan),
                        index=parts[0].index, columns=parts[0].columns)


def _relative_frame(coin: pd.DataFrame, groups: dict[str, list[str]], live: pd.DataFrame) -> pd.DataFrame:
    """По дням: среднее признака ликвидных монет сектора минус среднее по всем секторным; < MIN_COINS — NaN."""
    every = sorted({c for cols in groups.values() for c in cols})
    v = coin.reindex(columns=every).where(live.reindex(columns=every, fill_value=False))
    base = v.mean(axis=1)
    out = {}
    for name, cols in groups.items():
        n = v[cols].notna().sum(axis=1)
        out[name] = (v[cols].mean(axis=1) - base).where(n >= MIN_COINS)
    return pd.DataFrame(out, index=coin.index)


def history(bars: dict[str, pd.DataFrame], oi: dict[str, pd.Series] | None = None,
            btc_close: pd.Series | None = None, days: int = HISTORY_DAYS) -> dict:
    """Ряды для графиков по секторам за последние `days` дней (те же определения, что в compute()):
    index — накопленная доходность сектора сверх рынка альтов с начала окна, turnover — оборот 7/90 сверх рынка,
    taker — доля рыночных покупок 7/90 сверх секторов, oi — рост OI за 7 дней сверх секторов (только последние
    ~3 недели: Binance отдаёт историю OI за 30 дней), score — сводный балл; bull — режим BTC по дням."""
    close = pd.DataFrame({s: b["close"] for s, b in bars.items() if len(b)}).sort_index()
    if close.empty:
        return {"days": [], "sectors": {}, "bull": []}
    qv = pd.DataFrame({s: b["quote_volume"] for s, b in bars.items() if len(b)}).reindex_like(close)
    r = close.pct_change()
    liquid = qv.rolling(30, min_periods=20).mean() >= ADV_MIN
    alts = [c for c in close.columns if c != "BTCUSDT"]
    rl = r.where(liquid)
    mkt = rl[alts].mean(axis=1)
    groups = {name: [c for c in alts if sector_of(c) == name] for name in SECTORS}
    groups = {k: v for k, v in groups.items() if v}
    tot = qv[alts].sum(axis=1)
    m_ratio = tot.rolling(SHORT, min_periods=6).mean() / tot.rolling(LONG, min_periods=60).mean()
    ex, turn = {}, {}
    for name, cols in groups.items():
        n = rl[cols].notna().sum(axis=1)
        ex[name] = (rl[cols].mean(axis=1) - mkt).where(n >= MIN_COINS)
        v = qv[cols].where(liquid[cols]).sum(axis=1, min_count=1)
        turn[name] = (v.rolling(SHORT, min_periods=6).mean() / v.rolling(LONG, min_periods=60).mean() / m_ratio) \
            .where(n >= MIN_COINS)
    ex_df, turn_df = pd.DataFrame(ex), pd.DataFrame(turn)
    mom = np.log1p(ex_df).rolling(SHORT, min_periods=6).sum().where(ex_df.notna())
    taker_coin = {}
    for c in [c for cols in groups.values() for c in cols]:
        b = bars[c]
        if "taker_buy_volume" in b and "volume" in b:
            share = (b["taker_buy_volume"] / b["volume"].where(b["volume"] > 0)).clip(0.01, 0.99)
            lg = np.log(share / (1 - share))
            taker_coin[c] = lg.rolling(SHORT, min_periods=6).mean() - lg.rolling(LONG, min_periods=60).mean()
    taker = _relative_frame(pd.DataFrame(taker_coin).reindex(close.index), groups, liquid) if taker_coin else \
        pd.DataFrame(np.nan, index=close.index, columns=list(groups))
    oi_coin = {}
    for c, s in (oi or {}).items():
        s = s.dropna()
        s = s[s > 0]
        if len(s) > SHORT:
            oi_coin[c] = np.log(s / s.shift(SHORT))
    oi_df = _relative_frame(pd.DataFrame(oi_coin).reindex(close.index), groups, liquid) if oi_coin else \
        pd.DataFrame(np.nan, index=close.index, columns=list(groups))
    att = np.log(turn_df.where(turn_df > 0))
    score = _zscore_mean([mom, att, taker.reindex_like(mom), oi_df.reindex_like(mom)])
    win = close.index[-days:]
    bull = pd.Series(np.nan, index=close.index)
    if btc_close is not None and len(btc_close.dropna()):
        c = btc_close.dropna()
        c.index = pd.DatetimeIndex(c.index)
        w = c.resample("W-SUN").last()
        if c.index[-1].day_of_week != 6:
            w = w.iloc[:-1]
        ema = w.ewm(span=EMA_WEEKS, adjust=False, min_periods=30).mean()
        flag = (w > ema).astype("float64").where(ema.notna())
        bull = flag.reindex(close.index.union(flag.index)).ffill().reindex(close.index)
    out_sectors = {}
    for name in groups:
        e = ex_df[name].reindex(win)
        if e.notna().sum() < 10:
            continue
        idx = (1 + e.fillna(0.0)).cumprod() - 1
        out_sectors[name] = {
            "index": [_none(x) for x in idx.where(idx.index >= e.first_valid_index())],
            "score": [_none(x) for x in score[name].reindex(win)] if name in score else [],
            "turnover": [_none(x) for x in turn_df[name].reindex(win)],
            "taker": [_none(x) for x in taker[name].reindex(win)] if name in taker else [],
            "oi": [_none(x) for x in oi_df[name].reindex(win)] if name in oi_df else [],
        }
    return {"days": [d.strftime("%Y-%m-%d") for d in win], "sectors": out_sectors,
            "bull": [None if not math.isfinite(x) else bool(x > 0.5) for x in bull.reindex(win)]}
