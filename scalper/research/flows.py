"""Куда пойдут деньги внутри альтов — три источника, которые могут сказать о следующем секторе раньше цены (продолжение
research/secorder.py: по цене и обороту очередь не угадывается).

    F1 экосистемы    (Wintermute 2025, «L1 wealth spillover»: рост базового токена сети перетекает в токены её
                     экосистемы, на Solana — около двух месяцев). Состав экосистем — категории CoinGecko «… Ecosystem»
                     и сети протоколов DefiLlama (протокол только на одной сети); токены, что есть в >= 3 экосистемах
                     (мультичейн: USDT, LINK…), и сам базовый токен — не в корзине.
       a: каждый понедельник — сеть, чей базовый токен лучше всех обогнал рынок за 28 дней (и в плюсе), → корзина её
          экосистемы на неделю;
       b: событие — базовый токен обогнал рынок на >= 30% за 28 дней → корзина экосистемы на 28 дней (одно событие на
          сеть за 28 дней); плацебо — корзины других экосистем с той же даты.
    F2 внимание      просмотры английской Википедии (Kristoufek 2013: Google Trends и Википедия ведут цену BTC; Maître и
                     др. 2025 — аномальное внимание → доходность). Сектор — статьи о его монетах и о самом явлении;
                     балл — средний по статьям log(просмотры за 7 дней / за 90 дней) минус то же по всем статьям.
       a: сектор с наибольшим баллом → неделя; b: то же, но только если сектор сам в плюсе сверх рынка за 7 дней.
    F3 комиссии      DefiLlama /overview/fees с разбивкой по протоколам за каждый день: комиссии категории (Dexs,
                     Lending, Derivatives…) за 30 дней к среднему месяцу предыдущих 90; корзина категории — токены её
                     протоколов, у которых есть перпетуал.
       a: каждый понедельник — категория с наибольшим ростом комиссий → неделя; b: то же → 4 недели (раз в 4 недели).
Цены — дневки перпетуалов Binance с делистнутыми (части research/sectors.py того же прогона). Состав экосистем и
категорий — сегодняшний (ошибка выжившего: умершие токены могут выпасть из категорий CoinGecko). Издержки taker
11 б.п. за круг. Мерило — сверх рынка (среднее альтов без BTC) того же срока.
Принятие (задано до запуска): > 0 в каждом периоде IS / VAL / HO (сделок >= 20; у F1b и F3b — >= 10), t >= 3,
> 0 при издержках ×2; у F1b ещё и лучше плацебо в IS / VAL / HO.

    python -m research.sectors collect --symbols <1h-all> && python -m research.flows
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from .oos import PER_ALL
from .ppr import day_t
from .secorder import past, pick_weekly
from .sectors import COST, load_daily, panel

UA = {"User-Agent": "scalper-research/1.0 (github.com/Roman-Petukhov/scalper)", "Accept": "application/json"}
CG = "https://api.coingecko.com/api/v3"
LLAMA = "https://api.llama.fi"
WIKI = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/"
        "{title}/daily/20190901/{end}")

# базовый токен → (как зовут сеть в названии категории CoinGecko, как зовут в DefiLlama)
BASES = {
    "SOLUSDT": ("Solana", "Solana"), "BNBUSDT": ("BNB Chain", "BSC"), "AVAXUSDT": ("Avalanche", "Avalanche"),
    "SUIUSDT": ("Sui", "Sui"), "APTUSDT": ("Aptos", "Aptos"), "NEARUSDT": ("Near Protocol", "Near"),
    "TRXUSDT": ("Tron", "Tron"), "TONUSDT": ("TON", "TON"), "ARBUSDT": ("Arbitrum", "Arbitrum"),
    "OPUSDT": ("Optimism", "OP Mainnet"), "POLUSDT": ("Polygon", "Polygon"), "MATICUSDT": ("Polygon", "Polygon"),
    "FTMUSDT": ("Fantom", "Fantom"), "SUSDT": ("Sonic", "Sonic"), "ATOMUSDT": ("Cosmos", "Cosmos"),
    "SEIUSDT": ("Sei", "Sei"), "INJUSDT": ("Injective", "Injective"), "HYPEUSDT": ("Hyperliquid", "Hyperliquid L1"),
    "ETHUSDT": ("Ethereum", "Ethereum"), "ADAUSDT": ("Cardano", "Cardano"), "DOTUSDT": ("Polkadot", "Polkadot"),
    "BERAUSDT": ("Berachain", "Berachain"), "STXUSDT": ("Stacks", "Stacks"), "KASUSDT": ("Kaspa", "Kaspa"),
}
MULTI = 3
SPILL = 0.30
HOLD = 28
MIN_BASKET = 3

WIKI_PAGES = {
    "мемы": ("Dogecoin", "Shiba_Inu_(cryptocurrency)", "Meme_coin", "Pepe_(cryptocurrency)", "Pump.fun", "Bonk_(cryptocurrency)"),
    "AI": ("Bittensor", "Worldcoin", "SingularityNET", "Fetch.ai", "Render_Network", "Virtuals_Protocol"),
    "L1": ("Solana_(blockchain_platform)", "Cardano_(blockchain_platform)", "Avalanche_(blockchain_platform)",
           "Polkadot_(cryptocurrency)", "Tron_(cryptocurrency)", "Near_Protocol", "Sui_(blockchain)", "Aptos_(blockchain)"),
    "L2": ("Polygon_(blockchain)", "Arbitrum", "Optimism_(blockchain)", "Layer_2_(blockchain)", "Base_(blockchain)"),
    "DeFi": ("Decentralized_finance", "Uniswap", "Aave", "MakerDAO", "Curve_Finance", "Lido_Finance", "Compound_(company)"),
    "игры/NFT": ("Non-fungible_token", "Axie_Infinity", "The_Sandbox", "Decentraland", "OpenSea", "GameFi"),
    "RWA": ("Tokenization_(finance)", "Ondo_Finance", "Real-world_asset_tokenization", "Pendle_Finance"),
    "приватность": ("Monero", "Zcash", "Dash_(cryptocurrency)", "Privacy_coin", "Secret_Network"),
    "DePIN": ("Filecoin", "Helium_Network", "Arweave", "Theta_Network", "Decentralized_physical_infrastructure_network"),
}
ATT_SHORT, ATT_LONG = 7, 90
FEE_SHORT, FEE_BASE = 30, 90
FEE_MIN = 1e6
FEE_SKIP = {"CEX", "Chain", "Rollup", "Stablecoin Issuer", "Services", "Wallets", "Developer Tools", "Telegram Bot"}


def _get(url: str, tries: int = 4, pause: float = 2.0) -> object | None:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            wait = float(e.headers.get("Retry-After") or pause * 2 ** attempt) if e.code == 429 else pause * 2 ** attempt
            print(f"  flows: {url[:90]} → HTTP {e.code}, жду {wait:.0f} с", flush=True)
            time.sleep(min(wait, 90))
        except Exception as e:
            print(f"  flows: {url[:90]} → {e}", flush=True)
            time.sleep(pause * 2 ** attempt)
    return None


# --- F1: экосистемы ---

def eco_category_ids(cats: object) -> dict[str, str]:
    """id категории CoinGecko для каждого базового токена: «<сеть> Ecosystem» без учёта регистра."""
    out = {}
    names = {str(c.get("name", "")).lower(): c.get("category_id") for c in cats if isinstance(c, dict)} \
        if isinstance(cats, list) else {}
    for base, (cg, _) in BASES.items():
        cid = names.get(f"{cg} ecosystem".lower())
        if cid:
            out[base] = str(cid)
    return out


def symbols_of(markets: object) -> set[str]:
    return {str(m.get("symbol", "")).upper() + "USDT" for m in markets if isinstance(m, dict) and m.get("symbol")} \
        if isinstance(markets, list) else set()


def llama_chains(protocols: object) -> dict[str, set[str]]:
    """Сеть DefiLlama → перпетуалы токенов протоколов, живущих только на этой сети."""
    out: dict[str, set[str]] = {}
    for p in protocols if isinstance(protocols, list) else []:
        if not isinstance(p, dict) or not p.get("symbol") or p.get("symbol") == "-":
            continue
        chains = p.get("chains") or []
        if len(chains) == 1:
            out.setdefault(str(chains[0]), set()).add(str(p["symbol"]).upper() + "USDT")
    return out


def ecosystems(cg_members: dict[str, set[str]], llama: dict[str, set[str]], have: set[str]) -> dict[str, list[str]]:
    """Корзины экосистем: CoinGecko ∪ DefiLlama, без базовых токенов и мультичейн-токенов, только с перпетуалом."""
    raw: dict[str, set[str]] = {}
    for base, (_, ll) in BASES.items():
        raw[base] = set(cg_members.get(base, set())) | set(llama.get(ll, set()))
    count = Counter(s for v in raw.values() for s in v)
    out = {}
    for base, v in raw.items():
        keep = sorted(s for s in v if s in have and s not in BASES and s != "BTCUSDT" and count[s] < MULTI)
        if len(keep) >= MIN_BASKET and base in have:
            out[base] = keep
    return out


def load_ecosystems(have: set[str]) -> dict[str, list[str]]:
    members: dict[str, set[str]] = {}
    ids = eco_category_ids(_get(f"{CG}/coins/categories/list"))
    print(f"  F1: категорий CoinGecko найдено {len(ids)} из {len(BASES)}", flush=True)
    for base, cid in ids.items():
        got: set[str] = set()
        for page in (1, 2):
            q = urllib.parse.urlencode({"vs_currency": "usd", "category": cid, "per_page": 250, "page": page})
            m = _get(f"{CG}/coins/markets?{q}", pause=8.0)
            got |= symbols_of(m)
            time.sleep(7)
            if not isinstance(m, list) or len(m) < 250:
                break
        members[base] = got
    llama = llama_chains(_get(f"{LLAMA}/protocols"))
    return ecosystems(members, llama, have)


def baskets(r: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    """Дневная доходность корзины (среднее ликвидных участников; меньше MIN_BASKET — пропуск)."""
    out = {}
    for k, cols in groups.items():
        cols = [c for c in cols if c in r]
        if cols:
            out[k] = r[cols].mean(axis=1).where(r[cols].notna().sum(axis=1) >= MIN_BASKET)
    return pd.DataFrame(out, index=r.index)


def spill_weekly(r: pd.DataFrame, mkt: pd.Series, eco: pd.DataFrame) -> pd.DataFrame:
    ex_base = r[[b for b in eco.columns if b in r]].sub(mkt, axis=0)
    mom = np.log1p(ex_base.fillna(0.0)).rolling(HOLD, min_periods=HOLD - 2).sum()
    return pick_weekly(mom, eco.sub(mkt, axis=0), gate=mom)


def spill_events(r: pd.DataFrame, mkt: pd.Series, eco: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ex = eco.sub(mkt, axis=0)
    rows, plc = [], []
    last: dict[str, pd.Timestamp] = {}
    for t in r.index[HOLD:]:
        fired = []
        for b in eco.columns:
            if b not in r or (b in last and (t - last[b]).days < HOLD):
                continue
            p = past(r[b] - mkt, t, HOLD)
            if p is not None and p >= SPILL and eco[b].loc[:t].tail(5).notna().any():
                fired.append(b)
        for b in fired:
            last[b] = t
            x = window(ex[b], t, HOLD)
            if x is not None:
                rows.append({"t": t, "X": x - COST, "sector": b})
            others = [window(ex[o], t, HOLD) for o in ex.columns if o not in fired]
            others = [o for o in others if o is not None]
            if others:
                plc.append({"t": t, "X": float(np.mean(others)) - COST})
    return pd.DataFrame(rows), pd.DataFrame(plc)


def window(s: pd.Series, t: pd.Timestamp, days: int) -> float | None:
    w = s.loc[t:t + pd.Timedelta(days=days - 1)].dropna()
    return float(np.expm1(np.log1p(w).sum())) if len(w) >= days * 0.7 else None


# --- F2: внимание по Википедии ---

def parse_views(raw: object) -> pd.Series:
    items = raw.get("items", []) if isinstance(raw, dict) else []
    s = pd.Series({pd.Timestamp(str(x["timestamp"])[:8]): float(x.get("views", 0)) for x in items
                   if isinstance(x, dict) and x.get("timestamp")}, dtype="float64")
    return s.sort_index()


def wiki_views() -> dict[str, pd.Series]:
    end = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d")
    out = {}
    for pages in WIKI_PAGES.values():
        for title in pages:
            s = parse_views(_get(WIKI.format(title=urllib.parse.quote(title, safe="()_.,"), end=end), tries=3))
            if len(s) > 200:
                out[title] = s
            time.sleep(0.2)
    return out


def attention_score(views: dict[str, pd.Series]) -> pd.DataFrame:
    """Балл сектора на день: средний по статьям log(7 дн. / 90 дн.) минус средний по всем статьям."""
    ratios = {}
    for title, s in views.items():
        s = s.asfreq("D")
        ratios[title] = np.log((s.rolling(ATT_SHORT, min_periods=6).mean() + 1) /
                               (s.rolling(ATT_LONG, min_periods=60).mean() + 1))
    allr = pd.DataFrame(ratios)
    market = allr.mean(axis=1)
    out = {}
    for sector, pages in WIKI_PAGES.items():
        cols = [p for p in pages if p in allr]
        if len(cols) >= 2:
            out[sector] = allr[cols].mean(axis=1).sub(market)
    return pd.DataFrame(out)


# --- F3: комиссии по категориям ---

def fee_frame(raw: object) -> tuple[pd.DataFrame, dict[str, str]]:
    """Дневные комиссии по протоколам и категория каждого протокола из ответа /overview/fees."""
    if not isinstance(raw, dict):
        return pd.DataFrame(), {}
    cat: dict[str, str] = {}
    for p in raw.get("protocols", []) or []:
        if isinstance(p, dict) and p.get("category"):
            for k in (p.get("name"), p.get("displayName"), p.get("module")):
                if k:
                    cat[str(k)] = str(p["category"])
    rows = {}
    for item in raw.get("totalDataChartBreakdown", []) or []:
        if isinstance(item, list) and len(item) == 2 and isinstance(item[1], dict):
            rows[pd.Timestamp(int(item[0]), unit="s").normalize()] = {k: float(v) for k, v in item[1].items()
                                                                       if isinstance(v, (int, float))}
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    return df, cat


def category_fees(df: pd.DataFrame, cat: dict[str, str]) -> pd.DataFrame:
    cols: dict[str, list[str]] = {}
    for c in df.columns:
        k = cat.get(c)
        if k and k not in FEE_SKIP:
            cols.setdefault(k, []).append(c)
    return pd.DataFrame({k: df[v].sum(axis=1, min_count=1) for k, v in cols.items()}, index=df.index).fillna(0.0)


def fee_growth(cf: pd.DataFrame) -> pd.DataFrame:
    """Комиссии за 30 дней к среднему 30-дневному отрезку предыдущих 90 дней (log); мелкие категории — NaN."""
    s30 = cf.rolling(FEE_SHORT, min_periods=FEE_SHORT).sum()
    prev = cf.shift(FEE_SHORT).rolling(FEE_BASE, min_periods=FEE_BASE).sum() * FEE_SHORT / FEE_BASE
    g = np.log((s30 + 1) / (prev + 1))
    return g.where(s30 >= FEE_MIN)


def category_tokens(protocols: object, have: set[str]) -> dict[str, list[str]]:
    out: dict[str, set[str]] = {}
    for p in protocols if isinstance(protocols, list) else []:
        if isinstance(p, dict) and p.get("category") and p.get("symbol") not in (None, "", "-"):
            s = str(p["symbol"]).upper() + "USDT"
            if s in have and s != "BTCUSDT":
                out.setdefault(str(p["category"]), set()).add(s)
    return {k: sorted(v) for k, v in out.items() if k not in FEE_SKIP}


def pick_every(score: pd.DataFrame, ex: pd.DataFrame, hold: int) -> pd.DataFrame:
    """Раз в hold дней (по понедельникам) — лучшая по баллу корзина на hold дней."""
    rows, nxt = [], None
    for t in score.index[score.index.dayofweek == 0]:
        if nxt is not None and t < nxt:
            continue
        prev = t - pd.Timedelta(days=1)
        if prev not in score.index:
            continue
        sc = score.loc[prev].dropna()
        sc = sc[[c for c in sc.index if c in ex and ex[c].loc[:prev].tail(5).notna().any()]]
        if len(sc) < 3:
            continue
        pick = sc.idxmax()
        x = window(ex[pick], t, hold)
        if x is not None:
            rows.append({"t": t, "X": x - COST, "sector": pick})
            nxt = t + pd.Timedelta(days=hold)
    return pd.DataFrame(rows)


# --- отчёт ---

def judge(name: str, z: pd.DataFrame, passed: list[str], min_n: int = 20, plc: pd.DataFrame | None = None) -> None:
    if z.empty:
        print(f"    {name}: сделок нет")
        return
    cells, ok = [], []
    for p, (a, b) in PER_ALL.items():
        m = z[(z.t >= a) & (z.t < b)]
        if m.empty:
            continue
        cells.append(f"{p} {m.X.mean():+.2%} ({len(m)})")
        if p in ("is", "val", "ho"):
            good_p = len(m) >= min_n and m.X.mean() > 0
            if plc is not None:
                q = plc[(plc.t >= a) & (plc.t < b)]
                good_p = good_p and len(q) > 0 and m.X.mean() > q.X.mean()
            ok.append(good_p)
    t = day_t(z, "X") if len(z) > 10 else float("nan")
    x2 = (z.X - COST).mean()
    good = len(ok) == 3 and all(ok) and np.isfinite(t) and t >= 3 and x2 > 0
    print(f"    {name}: {'ПРИНЯТО' if good else 'нет'} (сделок {len(z)})\n        сверх рынка: {'  '.join(cells)}\n"
          f"        всё время {z.X.mean():+.2%} (t {t:.1f}); ×2 {x2:+.2%}")
    if "sector" in z:
        print("        чаще всего: " + ", ".join(f"{k} {v}" for k, v in z.sector.value_counts().head(4).items()))
    if plc is not None and len(plc):
        print(f"        плацебо (другие экосистемы с той же даты): {plc.X.mean():+.2%} ({len(plc)})")
    if good:
        passed.append(name)


def report() -> None:
    print("\n===== FLOWS: экосистемы сетей, внимание по Википедии, комиссии по категориям — куда пойдут деньги =====")
    got = load_daily()
    if got is None:
        print("  flows: нет данных")
        return
    close, qv = got
    r, mkt, idx = panel(close, qv)
    have = set(close.columns)
    passed: list[str] = []

    print("\n  --- F1: экосистемы (рост базового токена → токены его сети) ---")
    eco_groups = load_ecosystems(have)
    print("  корзины: " + (", ".join(f"{b[:-4]} {len(v)}" for b, v in eco_groups.items()) or "нет"))
    if eco_groups:
        eco = baskets(r, eco_groups)
        judge("a: лучший базовый токен за 28 дн. → его экосистема, неделя", spill_weekly(r, mkt, eco), passed)
        ev, plc = spill_events(r, mkt, eco)
        judge(f"b: базовый токен обогнал рынок на >= {SPILL:.0%} за 28 дн. → экосистема на {HOLD} дн.", ev, passed,
              min_n=10, plc=plc)

    print("\n  --- F2: внимание (просмотры Википедии) ---")
    views = wiki_views()
    print(f"  статей с данными: {len(views)} из {sum(len(v) for v in WIKI_PAGES.values())}")
    att = attention_score(views)
    if att.shape[1] >= 4:
        att = att.reindex(r.index)
        ex = pd.DataFrame(idx).sub(mkt, axis=0)
        mom = np.log1p(ex.fillna(0.0)).rolling(7, min_periods=6).sum()
        judge("a: сектор с наибольшим ростом просмотров → неделя", pick_weekly(att, ex), passed)
        judge("b: рост просмотров + сектор в плюсе за 7 дн. → неделя", pick_weekly(att, ex, gate=mom), passed)
    else:
        print("    секторов с данными меньше 4 — пропуск")

    print("\n  --- F3: комиссии по категориям DefiLlama ---")
    raw = _get(f"{LLAMA}/overview/fees?excludeTotalDataChart=true&excludeTotalDataChartBreakdown=false"
               "&dataType=dailyFees")
    df, cat = fee_frame(raw)
    toks = category_tokens(_get(f"{LLAMA}/protocols"), have)
    if df.empty or not toks:
        print("    данные DefiLlama не загрузились — пропуск")
    else:
        cf = category_fees(df, cat)
        growth = fee_growth(cf).reindex(r.index)
        bk = baskets(r, {k: v for k, v in toks.items() if k in cf.columns})
        ex = bk.sub(mkt, axis=0)
        growth = growth[[c for c in growth.columns if c in ex]]
        print(f"  протоколов {df.shape[1]}, категорий с комиссиями {cf.shape[1]}, с корзиной токенов {growth.shape[1]}; "
              f"дни {df.index[0].date()} … {df.index[-1].date()}")
        judge("a: наибольший рост комиссий категории → неделя", pick_weekly(growth, ex), passed)
        judge("b: наибольший рост комиссий категории → 4 недели", pick_every(growth, ex, HOLD), passed, min_n=10)

    print("\n  ИТОГ: " + ("прошли — " + "; ".join(passed) if passed else "ни один источник не предсказывает следующий "
                                                                       "сектор с запасом на издержки"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 300)
    report()
