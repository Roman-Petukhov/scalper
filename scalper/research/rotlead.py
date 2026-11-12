"""Что ещё может угадывать ротацию секторов (продолжение research/posflow.py и research/social.py: лучший результат —
сводный балл «цена + оборот + агрессоры + OI», +2.0% в неделю при t 2.7, и только в бычьем режиме BTC).

Шесть направлений, которых по секторам ещё не проверяли:
    N1 стейблкоины по сетям   (Artemis 2026, фактор «Stablecoins»: сети с притоком стейблкоинов обгоняют остальные,
                              Sharpe 1.67 по их бэктесту). DefiLlama: предложение стейблкоинов в сети по дням.
                              a: три базовых токена с наибольшим ростом предложения за 28 дней (рост > 0) → неделя;
                              b: то же за 7 дней; c: сеть-лидер за 28 дней → корзина её экосистемы на неделю.
    N2 без беты               перевес сводного балла в бычьих неделях может быть просто бетой: в растущем рынке
                              высокобетовые сектора обгоняют среднее. Остаток сектора = его доходность − бета(90 дн.,
                              на вчера) × рынок. a: сектор с наибольшим остаточным моментумом (сумма остатков за 7 дней /
                              их σ за 90) → неделя, сверх рынка; b: то же, исход — остаток; c (справка): сводный балл и
                              «лучший за 7 дней» с исходом-остатком, все недели и бычьи.
    N3 внутри сектора         что держать в секторе-лидере по сводному баллу: a: три лидера сектора за 7 дней;
                              b: три отстающих (догоняют?) — оба против рынка, все недели и бычьи. Корзина сектора
                              целиком — это сам сводный балл.
    N4 пробой силы            сектор на 90-дневном максимуме накопленной доходности сверх рынка. a: все такие сектора
                              поровну → неделя; b: только свежий пробой (первый за 28 дней).
    N5 режимы                 когда ротация работает — те же отборы (сводный, лучший за 7 дней) в неделях:
                              G1 предложение стейблкоинов за 28 дней выросло больше чем на 1% (DefiLlama);
                              G2 больше половины ликвидных альтов выше своей EMA50;
                              G3 альты за 28 дней обогнали BTC (доминация падает);
                              G4 разброс 7-дневного моментума между секторами выше своей медианы за год;
                              и R1 (BTC выше недельной EMA50) вместе с каждым из них — для сводного.
    N6 Coinbase               американский спот: доля сектора в долларовом обороте Coinbase за 7 дней к доле за 90.
                              a: наибольший рост → неделя; b: то же, сектор в плюсе за 7 дней; c: рост отношения
                              оборота сектора на Coinbase к обороту его перпетуалов на Binance (спот ведёт, а не плечи).
Цены и признаки — части research/posflow.py (перпетуалы Binance с делистнутыми). Составы экосистем, сети DefiLlama и
рынки Coinbase — сегодняшние (ошибка выжившего). Издержки taker 11 б.п. за круг, мерило — сверх рынка (среднее альтов
без BTC) той же недели.
Принятие (задано до запуска): > 0 в каждом периоде IS / VAL / HO (недель >= 20; в режимных отборах — >= 10), t >= 3,
> 0 при издержках ×2. Здесь около 30 проверок: один-два результата с t около 2 ожидаемы случайно, порог не снижаем.

    python -m research.prepare 1h-all sect-metrics && python -m research.posflow collect --symbols <1h-all>
    python -m research.rotlead
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings

import numpy as np
import pandas as pd

from .flows import BASES, baskets, judge, load_ecosystems
from .flows import _get as get_json
from .posflow import (LONG, SHORT, btc_bull, composite, gated_by, in_regime, load, momentum, oi_growth, sector_score,
                      taker_tilt, turnover_score, upbit_share_score)
from .secorder import mondays, pick_weekly, sector_turnover, week_ret
from .ppr import day_t
from .sectors import COST, panel, sector_map
from .shard import smoke

STABLES = "https://stablecoins.llama.fi"
COINBASE = "https://api.exchange.coinbase.com"
CB_SINCE = pd.Timestamp("2020-06-01")
CB_PAGE = 300                                   # Coinbase отдаёт не больше 300 свечей за запрос
STABLE_MIN = 5e7                                # сеть с предложением меньше $50M: рост в процентах — шум
STABLE_LOOKS = (28, 7)
TOP_CHAINS = 3
MIN_CHAINS = 6
BETA_WIN = 90
RS_WIN = 90
FRESH = 28
INSIDE = 3                                      # лидеров / отстающих внутри сектора
STABLE_GROWTH = 0.01
# сети сверх research/flows.py BASES: базовый токен → имя сети в DefiLlama (нет такой сети — пропуск)
MORE_CHAINS = {"MNTUSDT": "Mantle", "CELOUSDT": "Celo", "ALGOUSDT": "Algorand", "XLMUSDT": "Stellar", "XRPUSDT": "XRPL",
               "HBARUSDT": "Hedera", "KAVAUSDT": "Kava", "STRKUSDT": "Starknet", "ZKUSDT": "ZKsync Era",
               "LINEAUSDT": "Linea", "MOVEUSDT": "Movement", "KAIAUSDT": "Kaia", "FLOWUSDT": "Flow",
               "EGLDUSDT": "MultiversX", "XTZUSDT": "Tezos", "ZILUSDT": "Zilliqa", "ROSEUSDT": "Oasis",
               "CFXUSDT": "Conflux", "METISUSDT": "Metis", "MANTAUSDT": "Manta", "BLASTUSDT": "Blast",
               "SCRUSDT": "Scroll", "TAIKOUSDT": "Taiko", "MODEUSDT": "Mode", "CROUSDT": "Cronos",
               "GLMRUSDT": "Moonbeam", "ASTRUSDT": "Astar", "ICPUSDT": "ICP", "EOSUSDT": "EOS"}
ALIASES = {"op mainnet": ("optimism",), "hyperliquid l1": ("hyperliquid",), "zksync era": ("era", "zksync"),
           "xrpl": ("ripple", "xrp ledger"), "ton": ("the open network",), "bsc": ("binance", "bnb chain")}


# --- N1: стейблкоины по сетям ---

def chain_names() -> dict[str, str]:
    """Базовый токен → имя сети, как в research/flows.py, плюс MORE_CHAINS."""
    return {**{b: ll for b, (_, ll) in BASES.items()}, **MORE_CHAINS}


def match_chains(want: dict[str, str], listed: object) -> dict[str, str]:
    """Имена сетей из /stablecoinchains для наших базовых токенов (без учёта регистра, с ALIASES)."""
    have = {str(c.get("name", "")).lower(): str(c.get("name")) for c in listed if isinstance(c, dict)} \
        if isinstance(listed, list) else {}
    out = {}
    for base, name in want.items():
        for key in (name.lower(), *ALIASES.get(name.lower(), ())):
            if key in have:
                out[base] = have[key]
                break
    return out


def parse_supply(raw: object) -> pd.Series:
    """Ответ /stablecoincharts/<сеть> → предложение всех стейблкоинов сети в долларах по дням."""
    rows = {}
    for d in raw if isinstance(raw, list) else []:
        if not isinstance(d, dict) or "date" not in d:
            continue
        tot = d.get("totalCirculatingUSD")
        if isinstance(tot, dict):
            v = sum(float(x) for x in tot.values() if isinstance(x, (int, float)))
            rows[pd.Timestamp(int(d["date"]), unit="s").floor("D")] = v
    s = pd.Series(rows, dtype="float64").sort_index()
    return s[~s.index.duplicated(keep="last")]


def stable_supply(index: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.Series]:
    """Предложение стейблкоинов: по сетям (столбцы — базовые токены) и всего."""
    chains = match_chains(chain_names(), get_json(f"{STABLES}/stablecoinchains"))
    cols = {}
    for base, name in list(chains.items())[:4 if smoke() else None]:
        s = parse_supply(get_json(f"{STABLES}/stablecoincharts/{urllib.parse.quote(name)}"))
        if len(s) > 60:
            cols[base] = s
        time.sleep(0.4)
    total = parse_supply(get_json(f"{STABLES}/stablecoincharts/all"))
    print(f"  N1: сетей DefiLlama сопоставлено {len(chains)} из {len(chain_names())}, с историей {len(cols)}; "
          f"всего стейблкоинов — дней {len(total)}", flush=True)
    return pd.DataFrame(cols).reindex(index).ffill(limit=3), total.reindex(index).ffill(limit=3)


def supply_growth(supply: pd.DataFrame, look: int) -> pd.DataFrame:
    """log(предложение сегодня / look дней назад); база меньше STABLE_MIN — NaN."""
    base = supply.shift(look)
    return np.log(supply.where(supply > 0) / base.where(base >= STABLE_MIN))


def pick_top(score: pd.DataFrame, r: pd.DataFrame, mkt: pd.Series, n: int = TOP_CHAINS,
             min_have: int = MIN_CHAINS) -> pd.DataFrame:
    """Каждый понедельник — n монет с наибольшим score на вчера (score > 0), поровну на неделю, сверх рынка."""
    rows = []
    for t in mondays(score.index):
        prev = t - pd.Timedelta(days=1)
        if prev not in score.index or t not in r.index:
            continue
        sc = score.loc[prev].dropna()
        sc = sc[[c for c in sc.index if c in r and np.isfinite(r.at[t, c])]]
        if len(sc) < min_have:
            continue
        top = sc[sc > 0].nlargest(n)
        if len(top) < n:
            continue
        x, m = week_ret(r[list(top.index)].mean(axis=1), t), week_ret(mkt, t)
        if x is not None and m is not None:
            rows.append({"t": t, "X": x - m - COST, "sector": top.index[0].removesuffix("USDT")})
    return pd.DataFrame(rows)


# --- N2: без беты ---

def residual(idx: pd.DataFrame, mkt: pd.Series, win: int = BETA_WIN) -> pd.DataFrame:
    """Дневной остаток сектора: доходность − бета × рынок; бета — за win дней по вчера включительно."""
    var = mkt.rolling(win, min_periods=60).var()
    beta = idx.apply(lambda s: s.rolling(win, min_periods=60).cov(mkt)).div(var.where(var > 0), axis=0)
    return idx - beta.shift(1).mul(mkt, axis=0)


def residual_momentum(res: pd.DataFrame) -> pd.DataFrame:
    sd = res.rolling(LONG, min_periods=60).std()
    return res.rolling(SHORT, min_periods=6).sum() / sd.where(sd > 0)


# --- N3: внутри сектора ---

def inside(z: pd.DataFrame, close: pd.DataFrame, r: pd.DataFrame, mkt: pd.Series, lead: bool,
           n: int = INSIDE) -> pd.DataFrame:
    """Для каждой недели отбора z — n лидеров (lead) или отстающих сектора за 7 дней; нужно >= 2n ликвидных монет."""
    smap = sector_map()
    past7 = close / close.shift(SHORT) - 1
    rows = []
    for t, sec in zip(z.t, z.sector) if len(z) else []:
        prev = t - pd.Timedelta(days=1)
        if prev not in past7.index or t not in r.index:
            continue
        cols = [c for c in r.columns if smap.get(c) == sec and np.isfinite(r.at[t, c])]
        p = past7.loc[prev, cols].dropna()
        if len(p) < 2 * n:
            continue
        pick = list((p.nlargest(n) if lead else p.nsmallest(n)).index)
        x, m = week_ret(r[pick].mean(axis=1), t), week_ret(mkt, t)
        if x is not None and m is not None:
            rows.append({"t": t, "X": x - m - COST, "sector": sec})
    return pd.DataFrame(rows)


# --- N4: пробой относительной силы ---

def rs_high(ex: pd.DataFrame, win: int = RS_WIN) -> pd.DataFrame:
    """True — накопленная доходность сектора сверх рынка на максимуме за win дней."""
    line = np.log1p(ex.fillna(0.0)).cumsum().where(ex.notna())
    return (line >= line.rolling(win, min_periods=60).max() - 1e-12) & line.notna()


def fresh_high(high: pd.DataFrame, quiet: int = FRESH) -> pd.DataFrame:
    """Пробой, перед которым quiet дней максимума не было."""
    before = high.astype("float64").shift(1).rolling(quiet, min_periods=quiet).max()
    return high & (before == 0)


def pick_flagged(flag: pd.DataFrame, ex: pd.DataFrame) -> pd.DataFrame:
    """Каждый понедельник — все сектора с флагом на вчера, поровну на неделю."""
    rows = []
    for t in mondays(flag.index):
        prev = t - pd.Timedelta(days=1)
        if prev not in flag.index:
            continue
        on = [s for s in flag.columns if bool(flag.at[prev, s])]
        xs = [x for x in (week_ret(ex[s], t) for s in on) if x is not None]
        if xs:
            rows.append({"t": t, "X": float(np.mean(xs)) - COST, "sector": on[0] if len(on) == 1 else f"{len(on)} сект."})
    return pd.DataFrame(rows)


# --- N5: режимы ---

def alt_breadth(close: pd.DataFrame, live: pd.DataFrame) -> pd.Series:
    """Доля ликвидных альтов выше своей дневной EMA50."""
    c = close.drop(columns=["BTCUSDT"], errors="ignore")
    ema = c.ewm(span=50, adjust=False, min_periods=30).mean()
    ok = live.reindex(columns=c.columns).fillna(False) & ema.notna()
    n = ok.sum(axis=1)
    return ((c > ema) & ok).sum(axis=1) / n.where(n >= 20)


def alts_beat_btc(mkt: pd.Series, btc: pd.Series, look: int = 28) -> pd.Series:
    a = np.log1p(mkt.fillna(0.0)).rolling(look, min_periods=look - 2).sum()
    b = np.log(btc / btc.shift(look))
    return (a - b) > 0


def dispersion_high(mom: pd.DataFrame, win: int = 365) -> pd.Series:
    d = mom.std(axis=1).where(mom.notna().sum(axis=1) >= 4)
    return d > d.rolling(win, min_periods=120).median().shift(1)


# --- N6: Coinbase ---

def parse_candles(raw: object) -> pd.Series:
    """Свечи Coinbase [[time, low, high, open, close, volume], …] → долларовый оборот по дням."""
    rows = {}
    for c in raw if isinstance(raw, list) else []:
        if isinstance(c, list) and len(c) >= 6:
            rows[pd.Timestamp(int(c[0]), unit="s").floor("D")] = float(c[4]) * float(c[5])
    return pd.Series(rows, dtype="float64").sort_index()


def _coinbase(url: str) -> object | None:
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            time.sleep(0.12)                                    # публичный лимит Coinbase ~10 запросов/с
            return data
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(2 + 2 * attempt)
                continue
            if 400 <= e.code < 500:
                return None
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return None


def coinbase_turnover(until: pd.Timestamp, want: set[str]) -> pd.DataFrame:
    """Оборот USD-рынков Coinbase по дням, столбцы — <BASE>USDT; только монеты из want."""
    products = _coinbase(f"{COINBASE}/products")
    ids = sorted({str(p["id"]) for p in products if isinstance(p, dict) and p.get("quote_currency") == "USD"
                  and str(p.get("base_currency", "")) + "USDT" in want}) if isinstance(products, list) else []
    cols = {}
    for pid in ids[:6 if smoke() else None]:
        parts, end = [], until
        while end > CB_SINCE:
            start = max(end - pd.Timedelta(days=CB_PAGE - 1), CB_SINCE)
            q = urllib.parse.urlencode({"granularity": 86400, "start": start.strftime("%Y-%m-%dT00:00:00Z"),
                                        "end": end.strftime("%Y-%m-%dT00:00:00Z")})
            s = parse_candles(_coinbase(f"{COINBASE}/products/{pid}/candles?{q}"))
            if s.empty:
                break
            parts.append(s)
            end = start - pd.Timedelta(days=1)
        if parts:
            s = pd.concat(parts).sort_index()
            cols[pid.split("-", 1)[0] + "USDT"] = s[~s.index.duplicated(keep="last")]
    print(f"  N6: USD-рынков Coinbase с нашими монетами {len(ids)}, с историей {len(cols)}", flush=True)
    return pd.DataFrame(cols).sort_index()


def spot_vs_perp(cb: pd.DataFrame, qv: pd.DataFrame, sectors: list[str]) -> pd.DataFrame:
    """log(отношение оборота сектора на Coinbase к обороту его перпетуалов на Binance: 7 дней к 90)."""
    smap = sector_map()
    perp = pd.DataFrame(sector_turnover(qv))
    out = {}
    for name in sectors:
        cols = [c for c in cb.columns if smap.get(c) == name]
        if len(cols) < 2 or name not in perp:
            continue
        ratio = cb[cols].sum(axis=1, min_count=1).reindex(qv.index) / perp[name].where(perp[name] > 0)
        out[name] = np.log(ratio.rolling(SHORT, min_periods=6).mean() / ratio.rolling(LONG, min_periods=60).mean())
    return pd.DataFrame(out).reindex(qv.index)


# --- N7: устойчивость режима G1 (прогон 43155792562: сводный | G1 прошёл с t 3.1 среди ~30 проверок) ---

def stable_flag(total: pd.Series, look: int, thr: float) -> pd.Series:
    return (np.log(total / total.shift(look)) > thr).fillna(False).astype(bool)


def placebo_p(z: pd.DataFrame, flag: pd.Series, n: int = 2000, seed: int = 7) -> tuple[float, float]:
    """Доля случайных наборов недель того же размера, чьё среднее не хуже отбора по флагу; и среднее отбора."""
    if z.empty:
        return float("nan"), float("nan")
    picked = in_regime(z, flag)
    k = len(picked)
    if k < 10:
        return float("nan"), float("nan")
    x = z.X.to_numpy(dtype="float64")
    rng = np.random.default_rng(seed)
    means = np.array([x[rng.choice(len(x), k, replace=False)].mean() for _ in range(n)])
    m = float(picked.X.mean())
    return float((means >= m).mean()), m


def robust_g1(z_c: pd.DataFrame, z_m: pd.DataFrame, total: pd.Series, r1: pd.Series) -> None:
    print("\n  --- N7: устойчивость G1 (рост предложения стейблкоинов) ---")
    if z_c.empty:
        print("    сделок нет")
        return
    print("    сетка (окно × порог): среднее сверх рынка в неделю, t, недель; без 2020–2021 — то же")
    for look in (14, 28, 56):
        cells = []
        for thr in (0.0, 0.005, 0.01, 0.02, 0.03):
            z = in_regime(z_c, stable_flag(total, look, thr))
            if z.empty:
                cells.append(f"{thr:.1%}: —")
                continue
            late = z[z.t >= pd.Timestamp("2022-01-01")]
            t_all = day_t(z, "X") if len(z) > 10 else float("nan")
            t_late = day_t(late, "X") if len(late) > 10 else float("nan")
            cells.append(f"{thr:.1%}: {z.X.mean():+.2%} t {t_all:.1f} ({len(z)}) | {late.X.mean():+.2%} t {t_late:.1f} "
                         f"({len(late)})" if len(z) else f"{thr:.1%}: —")
        print(f"    {look} дн.  " + "\n             ".join(cells))
    flag = stable_flag(total, 28, STABLE_GROWTH)
    out = in_regime(z_c, ~flag & total.shift(28).notna())
    print(f"    вне G1 (для сравнения): {out.X.mean():+.2%} ({len(out)})" if len(out) else "    вне G1: —")
    for name, z in (("сводный", z_c), ("лучший за 7 дн.", z_m)):
        p, m = placebo_p(z, flag)
        print(f"    плацебо — случайные недели того же числа, {name}: отбор {m:+.2%}, p = {p:.3f}")
    late = z_c[z_c.t >= pd.Timestamp("2022-01-01")]
    judge("сводный | G1, только 2022+", in_regime(late, flag), [], min_n=10)
    judge("сводный | R1 и G1, только 2022+", in_regime(late, flag & r1), [], min_n=10)


# --- отчёт ---

def report() -> None:
    print("\n===== ROTLEAD: стейблкоины по сетям, остаток без беты, внутри сектора, пробой силы, режимы, Coinbase =====")
    got = load()
    if got is None:
        print("  rotlead: нет данных")
        return
    close, qv = got["close"], got["qv"]
    r, mkt, idx_d = panel(close, qv)
    live = r.notna()
    sectors = list(idx_d)
    idx = pd.DataFrame(idx_d)
    ex = idx.sub(mkt, axis=0)
    mom = momentum(ex)
    print(f"  монет {close.shape[1]}, секторов {len(sectors)}, дни {close.index[0].date()} … {close.index[-1].date()}")
    passed: list[str] = []
    comp = composite([mom, turnover_score(qv, sectors), sector_score(taker_tilt(got["tb"], qv), live, sectors),
                      sector_score(oi_growth(got["oi"]), live, sectors)]) if sectors else mom
    z_c, z_m = pick_weekly(comp, ex), pick_weekly(mom, ex)
    bull = btc_bull(close["BTCUSDT"]) if "BTCUSDT" in close else pd.Series(dtype="float64")
    r1 = bull.fillna(0.0) > 0.5
    judge("опора: сводный балл → неделя (как в posflow)", z_c, [])
    judge("опора: сводный балл | R1 BTC > EMA50 недельной", in_regime(z_c, r1), [], min_n=10)

    print("\n  --- N1: приток стейблкоинов в сеть (DefiLlama) ---")
    supply, total = stable_supply(close.index)
    if supply.empty:
        print("    DefiLlama недоступен")
    else:
        for look in STABLE_LOOKS:
            judge(f"{'a' if look == 28 else 'b'}: {TOP_CHAINS} базовых токена с наибольшим ростом за {look} дн. → неделя",
                  pick_top(supply_growth(supply, look), r, mkt), passed)
        if smoke():
            print("    c: пробный запуск — экосистемы CoinGecko не загружаем")
        else:
            eco = baskets(r, load_ecosystems(set(close.columns)))
            score = supply_growth(supply, 28).reindex(columns=eco.columns)
            judge("c: сеть-лидер по росту за 28 дн. → корзина её экосистемы на неделю",
                  pick_weekly(score, eco.sub(mkt, axis=0), gate=score), passed)

    print("\n  --- N2: остаток без беты ---")
    res = residual(idx, mkt)
    rmom = residual_momentum(res)
    judge("a: наибольший остаточный моментум → неделя, сверх рынка", pick_weekly(rmom, ex), passed)
    judge("b: наибольший остаточный моментум → неделя, исход — остаток", pick_weekly(rmom, res), passed)
    for name, score in (("сводный балл", comp), ("лучший за 7 дн.", mom)):
        z = pick_weekly(score, res)
        judge(f"c (справка): {name}, исход — остаток, все недели", z, [])
        judge(f"c (справка): {name}, исход — остаток | R1", in_regime(z, r1), [], min_n=10)

    print("\n  --- N3: что держать внутри сектора-лидера по сводному баллу ---")
    for lead, name in ((True, f"a: {INSIDE} лидера сектора за 7 дн."), (False, f"b: {INSIDE} отстающих сектора за 7 дн.")):
        z = inside(z_c, close, r, mkt, lead)
        judge(f"{name} → неделя", z, passed)
        judge(f"{name} | R1", in_regime(z, r1), passed, min_n=10)

    print("\n  --- N4: пробой относительной силы (90-дневный максимум сверх рынка) ---")
    high = rs_high(ex)
    judge("a: все сектора на максимуме → неделя", pick_flagged(high, ex), passed)
    judge(f"b: свежий пробой (первый за {FRESH} дн.) → неделя", pick_flagged(fresh_high(high), ex), passed)

    print("\n  --- N5: режимы ---")
    gates: dict[str, pd.Series] = {}
    if total.notna().any():
        gates[f"G1 стейблкоины за 28 дн. > {STABLE_GROWTH:.0%}"] = np.log(total / total.shift(28)) > STABLE_GROWTH
    gates["G2 больше половины альтов выше EMA50"] = alt_breadth(close, live) > 0.5
    if "BTCUSDT" in close:
        gates["G3 альты обогнали BTC за 28 дн."] = alts_beat_btc(mkt, close["BTCUSDT"])
    gates["G4 разброс секторов выше годовой медианы"] = dispersion_high(mom)
    for gname, flag in gates.items():
        flag = flag.fillna(False).astype(bool)
        print(f"\n    [{gname}]  дней в режиме: {flag.mean():.0%}")
        judge(f"сводный балл | {gname}", in_regime(z_c, flag), passed, min_n=10)
        judge(f"сводный балл | R1 и {gname}", in_regime(z_c, flag & r1), passed, min_n=10)
        judge(f"лучший за 7 дн. | {gname}", in_regime(z_m, flag), passed, min_n=10)

    if total.notna().any():
        robust_g1(z_c, z_m, total, r1)

    print("\n  --- N6: американский спот (Coinbase USD) ---")
    cb = coinbase_turnover(close.index[-1] + pd.Timedelta(days=1), set(sector_map()))
    if cb.empty:
        print("    Coinbase недоступен")
    else:
        share = upbit_share_score(cb, sectors, close.index)
        judge("a: наибольший рост доли сектора в обороте Coinbase → неделя", pick_weekly(share, ex), passed)
        judge("b: рост доли + сектор в плюсе за 7 дн. → неделя", pick_weekly(gated_by(share, mom), ex), passed)
        judge("c: рост оборота Coinbase к перпетуалам Binance → неделя", pick_weekly(spot_vs_perp(cb, qv, sectors), ex),
              passed)
    print("\n  ИТОГ: " + ("прошли — " + "; ".join(passed) if passed else
                          "ни один из новых признаков не даёт отбора сектора с запасом на издержки"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 300)
    report()
