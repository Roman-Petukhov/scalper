"""Внимание соцсетей по секторам — бесплатный аналог social volume / social dominance (Santiment) и mindshare (Kaito,
LunarCrush) на данных с историей (продолжение research/posflow.py: сводный балл цена + оборот + агрессоры + OI —
+2.0% в неделю, t 2.7, не принят).

Источники:
    Reddit    архив Arctic Shift (arctic-shift.photon-reddit.com): заголовки ВСЕХ постов сабреддитов SUBS с 2020-06;
    Telegram  публичные каналы CHANNELS через веб-просмотр t.me/s/<канал> (без входа), вглубь до 2020-06.
Упоминание сектора — в заголовке (Reddit) или тексте поста (Telegram) есть тикер монеты сектора (заглавными, как слово,
можно с $; тикеры-слова вроде AI, ONE, NOT не считаются), её имя или слово темы (meme coin, AI agent, RWA…). Пост
может упоминать несколько секторов. Доля сектора за день — постов с сектором / всех постов дня (social dominance).
Балл — log(доля за 7 дней / доля за 90 дней): рост внимания к сектору сверх его нормы и сверх роста всей болтовни.

Проверки (задано до запуска; издержки taker 11 б.п. за круг, мерило — сверх рынка альтов той же недели):
    S1 a: наибольший рост доли (Reddit + Telegram вместе) → неделя; b: то же, только если сектор в плюсе за 7 дней;
       c: (знак) наибольший рост доли → против рынка — Santiment пишет, что всплеск толпы чаще отмечает вершину.
    S2 то же по Reddit и по Telegram отдельно (справка: какой источник несёт сигнал).
    S3 опережение: корреляция балла соцсетей на воскресенье с ростом оборота сектора за СЛЕДУЮЩУЮ неделю против
       корреляции с прошлой — ведут ли соцсети оборот, или наоборот (справка).
    S4 сводный балл posflow + соцсети пятым признаком → неделя; и он же только в неделях BTC выше недельной EMA50.
Принятие: > 0 в каждом периоде IS / VAL / HO (недель >= 20; в бычьих — >= 10), t >= 3, > 0 при издержках ×2.

    python -m research.social collect      (части: сабреддит × год, каналы)
    python -m research.social report       (нужны части posflow того же прогона)
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from .flows import _get as get_json
from .flows import judge
from .posflow import (FNG, GREED, LONG, SHORT, btc_bull, composite, fear_greed, gated_by, in_regime, load, momentum,
                      oi_growth, sector_score, taker_tilt, turnover_score)
from .secorder import pick_weekly
from .sectors import panel, sector_map
from .shard import all_parts, mine, part_path, smoke

ARCTIC = "https://arctic-shift.photon-reddit.com/api/posts/search"
SUBS = ("CryptoCurrency", "CryptoMarkets", "altcoin", "SatoshiStreetBets", "CryptoMoonShots")
CHANNELS = ("WatcherGuru", "cointelegraph", "CoinDeskGlobal", "bitcoinmagazine", "binance_announcements",
            "crypto_insider_news", "whale_alert_io")
SINCE = pd.Timestamp("2020-06-01", tz="UTC")
UNTIL = pd.Timestamp("2026-10-01", tz="UTC")
PART = "social"
UA = {"User-Agent": "scalper-research/1.0 (github.com/Roman-Petukhov/scalper)", "Accept": "application/json"}
SMOKE_PAGES = 3

# тикеры, которые в заголовках чаще обычные слова или буквы
AMBIGUOUS = {"AI", "S", "W", "ME", "IO", "ALT", "CAT", "ACT", "NOT", "PEOPLE", "SUPER", "HIGH", "ONE", "FLOW", "BAN",
             "WHY", "MEME", "API3", "GAS", "OM", "TRU", "ATH", "AR", "IP", "CELO", "OP", "ZK", "W", "ENA", "JUP", "API",
             "BAND", "BAL", "ACE", "GMT", "MAGIC", "PIXEL", "PORTAL", "SAND", "MANA", "APE", "SAFE", "LINK", "UNI",
             "COMP", "SUSHI", "RUNE", "CAKE", "TAO", "FET", "DOGS", "PUMP", "SPX", "GOAT", "TRUMP", "OMNI", "SAGA",
             "SCR", "BLUR", "HMSTR", "CATI", "XAI", "ICP", "TON", "NEAR", "ATOM", "DOT", "SOL", "S", "ONDO"}
# имена и слова темы (без учёта регистра, как слово)
NAMES = {
    "мемы": ("meme coin", "memecoin", "meme coins", "memecoins", "dogecoin", "shiba", "pepe", "floki", "bonk",
             "dogwifhat", "popcat", "fartcoin", "pump.fun", "pumpfun", "pengu", "trump coin", "moodeng", "brett"),
    "AI": ("ai agent", "ai agents", "ai coin", "ai coins", "ai token", "ai tokens", "bittensor", "fetch.ai",
           "singularitynet", "render network", "worldcoin", "virtuals", "ai16z", "aixbt", "ocean protocol"),
    "L1": ("solana", "cardano", "avalanche", "polkadot", "near protocol", "aptos", "cosmos", "tron", "toncoin",
           "algorand", "hedera", "ripple", "litecoin", "kaspa", "hyperliquid", "berachain", "sui network"),
    "L2": ("layer 2", "layer-2", "layer2", "rollup", "rollups", "arbitrum", "optimism", "polygon", "starknet", "zksync",
           "base chain", "mantle", "immutable x", "blast l2", "scroll l2", "linea"),
    "DeFi": ("defi", "uniswap", "aave", "makerdao", "curve finance", "compound finance", "lido", "pendle", "dydx",
             "gmx", "1inch", "jupiter", "thorchain", "ethena", "eigenlayer", "restaking", "chainlink", "pyth", "dex"),
    "игры/NFT": ("nft", "nfts", "gamefi", "play to earn", "play-to-earn", "p2e", "axie", "sandbox", "decentraland",
                 "gala games", "apecoin", "illuvium", "opensea", "metaverse", "notcoin", "hamster kombat"),
    "RWA": ("rwa", "real world asset", "real-world asset", "real world assets", "real-world assets", "tokenization",
            "tokenized treasury", "tokenized treasuries", "ondo finance", "plume", "centrifuge", "polymesh"),
    "приватность": ("privacy coin", "privacy coins", "monero", "zcash", "dash", "horizen", "secret network",
                    "oasis network", "tornado cash", "zero-knowledge privacy"),
    "DePIN": ("depin", "filecoin", "arweave", "helium", "storj", "iotex", "theta network",
              "akash", "io.net", "decentralized physical infrastructure"),
}


def sector_patterns() -> dict[str, tuple[re.Pattern, re.Pattern | None]]:
    """Сектор → (регэксп тикеров с учётом регистра, регэксп имён без учёта регистра)."""
    by: dict[str, set[str]] = {}
    for sym, name in sector_map().items():
        t = sym.removesuffix("USDT")
        for p in ("1000000", "1000", "1M"):
            if t.startswith(p) and len(t) > len(p):
                t = t[len(p):]
        if t not in AMBIGUOUS and len(t) >= 3:
            by.setdefault(name, set()).add(t)
    out = {}
    for name in set(by) | set(NAMES):
        ticks = sorted(by.get(name, ()), key=len, reverse=True)
        tick_re = re.compile(r"(?<![A-Za-z0-9])\$?(?:" + "|".join(map(re.escape, ticks)) + r")(?![A-Za-z0-9])") \
            if ticks else re.compile(r"(?!x)x")
        names = sorted(NAMES.get(name, ()), key=len, reverse=True)
        name_re = re.compile(r"(?<![a-z0-9])(?:" + "|".join(map(re.escape, names)) + r")(?![a-z0-9])", re.I) \
            if names else None
        out[name] = (tick_re, name_re)
    return out


def tag(text: str, pats: dict[str, tuple[re.Pattern, re.Pattern | None]]) -> list[str]:
    hits = []
    for name, (tick, nm) in pats.items():
        if tick.search(text) or (nm is not None and nm.search(text)):
            hits.append(name)
    return hits


def daily_counts(items: list[tuple[pd.Timestamp, str]], source: str) -> pd.DataFrame:
    """(время, текст) → строки day, source, sector, n; sector '__all__' — все посты дня."""
    pats = sector_patterns()
    c: Counter = Counter()
    for t, text in items:
        day = t.tz_convert("UTC").floor("D").tz_localize(None)
        c[(day, "__all__")] += 1
        for s in tag(text, pats):
            c[(day, s)] += 1
    rows = [{"day": d, "source": source, "sector": s, "n": n} for (d, s), n in c.items()]
    return pd.DataFrame(rows, columns=["day", "source", "sector", "n"])


# --- Reddit (Arctic Shift) ---

def _fetch(url: str, headers: dict, tries: int = 6) -> bytes | None:
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            wait = float(e.headers.get("X-RateLimit-Reset") or e.headers.get("Retry-After") or 2 ** attempt) \
                if e.code == 429 else 2 ** attempt
            print(f"  social: {url[:110]} → HTTP {e.code}, жду {wait:.0f} с", flush=True)
            time.sleep(min(max(wait, 1.0), 120))
        except Exception as e:
            print(f"  social: {url[:110]} → {e}", flush=True)
            time.sleep(2 ** attempt)
    return None


def parse_arctic(raw: bytes | None) -> list[dict]:
    if not raw:
        return []
    try:
        d = json.loads(raw)
    except ValueError:
        return []
    rows = d.get("data") if isinstance(d, dict) else d
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


def reddit_year(sub: str, year: int) -> list[tuple[pd.Timestamp, str]]:
    a = max(pd.Timestamp(f"{year}-01-01", tz="UTC"), SINCE)
    b = min(pd.Timestamp(f"{year + 1}-01-01", tz="UTC"), UNTIL)
    after, seen, out, pages = int(a.timestamp()), set(), [], 0
    while after < b.timestamp():
        q = urllib.parse.urlencode({"subreddit": sub, "after": after, "before": int(b.timestamp()), "sort": "asc",
                                    "limit": "auto", "fields": "id,created_utc,title"})
        rows = parse_arctic(_fetch(f"{ARCTIC}?{q}", UA))
        if not rows:
            q = q.replace("limit=auto", "limit=100")
            rows = parse_arctic(_fetch(f"{ARCTIC}?{q}", UA))
        if not rows:
            break
        last = after
        for r in rows:
            try:
                ts = int(float(r["created_utc"]))
            except (KeyError, TypeError, ValueError):
                continue
            last = max(last, ts)
            if r.get("id") in seen:
                continue
            seen.add(r.get("id"))
            out.append((pd.Timestamp(ts, unit="s", tz="UTC"), str(r.get("title") or "")))
        pages += 1
        if last <= after:                                       # страница целиком из одной секунды
            last = after + 1
        after = last
        if smoke() and pages >= SMOKE_PAGES:
            break
        time.sleep(0.3)
    print(f"  social: r/{sub} {year}: постов {len(out)}, страниц {pages}", flush=True)
    return out


# --- Telegram (t.me/s) ---

MSG_RE = re.compile(r'data-post="[^"/]+/(\d+)"(.*?)(?=data-post="|\Z)', re.S)
TIME_RE = re.compile(r'<time[^>]*datetime="([^"]+)"')
TEXT_RE = re.compile(r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', re.S)
TAG_RE = re.compile(r"<[^>]+>")


def parse_tme(page: str) -> list[tuple[int, pd.Timestamp, str]]:
    out = []
    for m in MSG_RE.finditer(page):
        mid, body = int(m.group(1)), m.group(2)
        t = TIME_RE.search(body)
        if not t:
            continue
        txt = TEXT_RE.search(body)
        text = html.unescape(TAG_RE.sub(" ", txt.group(1))) if txt else ""
        try:
            out.append((mid, pd.Timestamp(t.group(1)).tz_convert("UTC"), text))
        except (ValueError, TypeError):
            continue
    return out


def telegram_channel(ch: str) -> list[tuple[pd.Timestamp, str]]:
    out, before, pages, seen = [], None, 0, set()
    head = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
    while True:
        url = f"https://t.me/s/{ch}" + (f"?before={before}" if before else "")
        raw = _fetch(url, head, tries=4)
        msgs = parse_tme(raw.decode("utf-8", "replace")) if raw else []
        msgs = [x for x in msgs if x[0] not in seen]
        if not msgs:
            break
        for mid, t, text in msgs:
            seen.add(mid)
            if SINCE <= t < UNTIL:
                out.append((t, text))
        pages += 1
        before = min(x[0] for x in msgs)
        if min(x[1] for x in msgs) < SINCE or before <= 1 or (smoke() and pages >= SMOKE_PAGES):
            break
        time.sleep(0.5)
    print(f"  social: t.me/{ch}: постов {len(out)}, страниц {pages}", flush=True)
    return out


def collect() -> None:
    jobs = [("reddit", s, y) for s in SUBS for y in range(SINCE.year, UNTIL.year + 1)] + \
           [("telegram", c, 0) for c in CHANNELS]
    own = mine(jobs, key=lambda j: f"{j[0]}:{j[1]}:{j[2]}")
    frames = []
    for src, key, year in own:
        try:
            items = reddit_year(key, year) if src == "reddit" else telegram_channel(key)
            if items:
                frames.append(daily_counts(items, f"{src}:{key}"))
        except Exception as e:
            print(f"  social: {src} {key} {year}: пропуск ({e})", flush=True)
    if frames:
        pd.concat(frames, ignore_index=True).to_parquet(part_path(PART), index=False)


# --- отчёт ---

def load_counts() -> pd.DataFrame | None:
    parts = all_parts(PART)
    if not parts:
        return None
    x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    x["day"] = pd.to_datetime(x["day"])
    return x.groupby(["day", "source", "sector"], as_index=False)["n"].sum()


def share_score(counts: pd.DataFrame, sectors: list[str], index: pd.DatetimeIndex) -> pd.DataFrame:
    """log(доля постов с сектором за 7 дней / за 90 дней) по сумме источников."""
    tab = counts.pivot_table(index="day", columns="sector", values="n", aggfunc="sum").sort_index()
    tab = tab.reindex(pd.date_range(tab.index.min(), tab.index.max(), freq="D")).fillna(0.0)
    tot = tab.get("__all__")
    if tot is None:
        return pd.DataFrame(index=index, columns=sectors, dtype="float64")
    out = {}
    for s in sectors:
        if s not in tab:
            continue
        s7 = tab[s].rolling(SHORT, min_periods=SHORT).sum() / tot.rolling(SHORT, min_periods=SHORT).sum()
        s90 = tab[s].rolling(LONG, min_periods=60).sum() / tot.rolling(LONG, min_periods=60).sum()
        out[s] = np.log(s7.where(s7 > 0) / s90.where(s90 > 0))
    return pd.DataFrame(out).reindex(index=index, columns=sectors)


def lead_lag(score: pd.DataFrame, turn: pd.DataFrame) -> tuple[float, float]:
    """Корреляция балла соцсетей на воскресенье с изменением балла оборота за следующую и за прошлую неделю."""
    sun = score.index[score.index.day_of_week == 6]
    nxt = (turn.shift(-7) - turn).reindex(sun)
    prv = (turn - turn.shift(7)).reindex(sun)
    s = score.reindex(sun)
    a = pd.concat([s.stack(), nxt.stack()], axis=1, join="inner").dropna()
    b = pd.concat([s.stack(), prv.stack()], axis=1, join="inner").dropna()
    ca = float(a.corr().iloc[0, 1]) if len(a) > 30 else float("nan")
    cb = float(b.corr().iloc[0, 1]) if len(b) > 30 else float("nan")
    return ca, cb


def report() -> None:
    print("\n===== SOCIAL: внимание Reddit и Telegram по секторам (аналог social dominance / mindshare) =====")
    counts = load_counts()
    got = load()
    if counts is None or got is None:
        print("  social: нет данных" + (" соцсетей" if counts is None else " цен (части posflow)"))
        return
    for src, g in counts[counts.sector == "__all__"].groupby("source"):
        print(f"  {src}: постов {int(g.n.sum())}, дни {g.day.min().date()} … {g.day.max().date()}")
    close, qv = got["close"], got["qv"]
    r, mkt, idx = panel(close, qv)
    live = r.notna()
    sectors = list(idx)
    ex = pd.DataFrame(idx).sub(mkt, axis=0)
    mom = momentum(ex)
    hits = counts[counts.sector != "__all__"].groupby("sector").n.sum().sort_values(ascending=False)
    print("  упоминаний по секторам: " + ", ".join(f"{k} {int(v)}" for k, v in hits.items()))
    passed: list[str] = []
    allsc = share_score(counts, sectors, close.index)

    print("\n  --- S1: рост доли сектора в постах (Reddit + Telegram) ---")
    judge("a: наибольший рост доли → неделя", pick_weekly(allsc, ex), passed)
    judge("b: рост доли + сектор в плюсе за 7 дн. → неделя", pick_weekly(gated_by(allsc, mom), ex), passed)
    judge("c: (знак) наибольший рост доли → против рынка, неделя", pick_weekly(allsc, -ex), [])

    print("\n  --- S2: по источникам (справка) ---")
    for fam in ("reddit", "telegram"):
        sub = counts[counts.source.str.startswith(fam + ":")]
        if sub.empty:
            print(f"    {fam}: данных нет")
            continue
        sc = share_score(sub, sectors, close.index)
        judge(f"{fam}: наибольший рост доли → неделя", pick_weekly(sc, ex), [])
        judge(f"{fam}: рост доли + цена → неделя", pick_weekly(gated_by(sc, mom), ex), [])

    print("\n  --- S3: кто кого ведёт (справка) ---")
    att = turnover_score(qv, sectors)
    ca, cb = lead_lag(allsc, att)
    print(f"    корреляция балла соцсетей с ростом оборота сектора: следующая неделя {ca:+.3f}, прошлая {cb:+.3f} — "
          f"{'соцсети опережают оборот' if ca > cb + 0.02 else 'соцсети не опережают оборот'}")

    print("\n  --- S4: сводный балл posflow + соцсети ---")
    p1 = sector_score(oi_growth(got["oi"]), live, sectors)
    p4 = sector_score(taker_tilt(got["tb"], qv), live, sectors)
    z4 = pick_weekly(composite([mom, att, p4, p1]), ex)
    z5 = pick_weekly(composite([mom, att, p4, p1, allsc]), ex)
    judge("сводный (4 признака, как в posflow) → неделя", z4, [])
    judge("сводный + соцсети (5 признаков) → неделя", z5, passed)
    bull = btc_bull(close["BTCUSDT"]) if "BTCUSDT" in close else pd.Series(dtype="float64")
    r1 = bull.fillna(0.0) > 0.5
    judge("сводный + соцсети | BTC > EMA50 недельной", in_regime(z5, r1), passed, min_n=10)
    fng = fear_greed(get_json(FNG))
    if len(fng):
        greed = fng.rolling(SHORT, min_periods=5).mean().reindex(close.index, method="ffill") >= GREED
        judge(f"сводный + соцсети | R1 и жадность >= {GREED}", in_regime(z5, r1 & greed.fillna(False)), passed,
              min_n=10)
    print("\n  ИТОГ: " + ("прошли — " + "; ".join(passed) if passed else
                          "внимание в соцсетях не даёт отбора сектора с запасом на издержки"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 300)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    a = ap.parse_args()
    collect() if a.mode == "collect" else report()
