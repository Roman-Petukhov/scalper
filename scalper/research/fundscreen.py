"""Фундаментальный скринер монет на 1–2 года: недооценённая доля в растущем бизнесе без навеса предложения.

Логика — docs/knowledge.md, раздел 7, «Венчурная (фундаментальная) оценка монеты». Правила заданы до запуска:
  1. Бизнес: комиссии и выручка протокола или сети (DefiLlama /overview/fees, 30 дней и год).
  2. Доля токена: выручка держателям (dailyHoldersRevenue — выкуп, сжигание, раздача) за 30 дней >= HR_MIN.
  3. Навес: капитализация / FDV >= FLOAT_MIN; разлок за 12 месяцев <= UNLOCK_MAX от разлоченного сейчас
     (DefiLlama emissions; нет расписания — пометка «?», не отсев).
  4. Торгуемость: капитализация от MCAP_MIN, оборот за сутки от VOL_MIN (CoinGecko /coins/markets).
Скор — сумма рангов: дёшево (FDV / годовая выручка держателям по последним 30 дням) и рост (выручка держателям
за 30 дней к среднему месяцу за год). Это список для бумажного портфеля вперёд, не бэктест: истории FDV и разлоков
бесплатно нет.

    python -m research.fundscreen report      (нужен интернет; сырые ответы сохраняются в $FS_OUT/fundscreen)
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .unlocks import _schedules

FEES_URL = ("https://api.llama.fi/overview/fees?excludeTotalDataChart=true&excludeTotalDataChartBreakdown=true"
            "&dataType={dt}")
DATA_TYPES = ("dailyFees", "dailyRevenue", "dailyHoldersRevenue")
PROTOCOLS_URL = "https://api.llama.fi/protocols"
CHAINS_URL = "https://api.llama.fi/v2/chains"
SUMMARY_URL = "https://api.llama.fi/summary/fees/{slug}?dataType=dailyHoldersRevenue"
CG_URL = ("https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids={ids}&per_page=250&page=1"
          "&price_change_percentage=30d,1y")
HR_MIN = 50_000.0            # выручка держателям за 30 дней, $
FLOAT_MIN = 0.5
UNLOCK_MAX = 0.10
MCAP_MIN = 20e6
VOL_MIN = 1e6
TOP = 30


def fetch(url: str, retries: int = 5, pause: float = 0.0) -> object | None:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                data = json.loads(r.read())
            if pause:
                time.sleep(pause)
            return data
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 403, 404):
                print(f"  fundscreen: {e.code} {url[:120]}", flush=True)
                return None
            time.sleep(min(60, 5 * 2 ** attempt))
        except Exception:
            time.sleep(min(60, 5 * 2 ** attempt))
    print(f"  fundscreen: не ответил {url[:120]}", flush=True)
    return None


def group_key(row: dict) -> str:
    """Протоколы с общим родителем (Uniswap V2 / V3 / V4) — один бизнес и один токен."""
    parent = row.get("parentProtocol")
    if isinstance(parent, str) and parent.startswith("parent#"):
        return parent.split("#", 1)[1]
    return str(row.get("slug") or row.get("name") or "").lower().replace(" ", "-")


def fees_table(overviews: dict[str, list[dict]]) -> pd.DataFrame:
    """Сумма по группам: 30 дней и год для комиссий, выручки и выручки держателям; категория — самая крупная часть."""
    rows = []
    for dt, protos in overviews.items():
        for p in protos:
            rows.append({"group": group_key(p), "name": p.get("displayName") or p.get("name"), "dt": dt,
                         "category": p.get("category"), "d30": float(p.get("total30d") or 0),
                         "y1": float(p.get("total1y") or 0), "chains": len(p.get("chains") or [])})
    if not rows:
        return pd.DataFrame()
    z = pd.DataFrame(rows)
    piv = z.pivot_table(index="group", columns="dt", values=["d30", "y1"], aggfunc="sum").fillna(0.0)
    piv.columns = [f"{dt}_{w}" for w, dt in piv.columns]
    top = z.sort_values("d30", ascending=False).drop_duplicates("group").set_index("group")
    out = piv.join(top[["category"]])
    names = z[z.dt == "dailyFees"].sort_values("d30", ascending=False).drop_duplicates("group").set_index("group")
    out["name"] = names["name"].reindex(out.index).fillna(pd.Series(out.index, index=out.index))
    for dt in DATA_TYPES:
        for w in ("d30", "y1"):
            if f"{dt}_{w}" not in out:
                out[f"{dt}_{w}"] = 0.0
    return out.reset_index()


def growth(d30: float, y1: float) -> float:
    """Последние 30 дней к среднему месяцу за год; nan — нет годовой истории."""
    return d30 / (y1 / 12.0) if y1 > 0 else np.nan


def unlock_share(cum: pd.Series, now: pd.Timestamp, days: int = 365) -> float:
    """Доля предложения, которая разлочится за days дней, от разлоченного сейчас."""
    if cum.empty or now < cum.index[0]:
        return np.nan
    a = float(cum.asof(now))
    b = float(cum.asof(min(now + pd.Timedelta(days=days), cum.index[-1])))
    return (b - a) / a if a > 0 else np.nan


def rank_score(df: pd.DataFrame) -> pd.Series:
    """Сумма перцентильных рангов: дешевле (меньше FDV / выручка держателям) и быстрее рост — выше."""
    cheap = (-df["fdv_hr"]).rank(pct=True)
    grow = df["hr_growth"].rank(pct=True).fillna(0.5)
    return cheap + grow


def screen(fees: pd.DataFrame, gecko: dict[str, str], mkt: pd.DataFrame, unl: dict[str, float]) -> pd.DataFrame:
    """Таблица кандидатов со всеми мерами и причиной отсева (пусто — прошёл)."""
    z = fees.copy()
    z["gecko_id"] = z["group"].map(gecko)
    z = z[z.gecko_id.notna()].merge(mkt, on="gecko_id", how="left")
    z["hr_y"] = z["dailyHoldersRevenue_d30"] * 365 / 30
    z["fdv"] = z["fdv"].where(z["fdv"] > 0, z["mcap"])
    z["float"] = z["mcap"] / z["fdv"]
    z["fdv_hr"] = z["fdv"] / z["hr_y"].where(z["hr_y"] > 0)
    z["hr_growth"] = [growth(a, b) for a, b in zip(z["dailyHoldersRevenue_d30"], z["dailyHoldersRevenue_y1"])]
    z["fee_growth"] = [growth(a, b) for a, b in zip(z["dailyFees_d30"], z["dailyFees_y1"])]
    z["unlock12"] = z["gecko_id"].map(unl)
    why = []
    for r in z.itertuples():
        w = []
        if not (r.dailyHoldersRevenue_d30 >= HR_MIN):
            w.append("нет выручки держателям")
        if not (r.mcap >= MCAP_MIN):
            w.append("мелкая/нет цены")
        if not (r.volume >= VOL_MIN):
            w.append("неликвидна")
        if not (r.float >= FLOAT_MIN):
            w.append("мало в обращении")
        if np.isfinite(r.unlock12) and r.unlock12 > UNLOCK_MAX:
            w.append("разлоки")
        why.append(", ".join(w))
    z["why"] = why
    ok = z["why"] == ""
    z["score"] = np.nan
    z.loc[ok, "score"] = rank_score(z[ok])
    return z.sort_values(["score", "hr_y"], ascending=[False, False])


def _money(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    for k, s in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= k:
            return f"{x / k:.1f}{s}"
    return f"{x:.0f}"


def show(z: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "монета": z["symbol"].str.upper().fillna("?"), "проект": z["name"].astype(str).str[:22],
        "категория": z["category"].astype(str).str[:14], "кап.": z["mcap"].map(_money), "FDV": z["fdv"].map(_money),
        "в обр.": z["float"].map(lambda v: f"{v:.0%}" if np.isfinite(v) else "—"),
        "разлок 12м": z["unlock12"].map(lambda v: f"{v:.0%}" if np.isfinite(v) else "?"),
        "выручка/год": (z["dailyRevenue_d30"] * 365 / 30).map(_money), "держателям/год": z["hr_y"].map(_money),
        "FDV/держ.": z["fdv_hr"].map(lambda v: f"{v:.0f}x" if np.isfinite(v) else "—"),
        "рост держ.": z["hr_growth"].map(lambda v: f"{v:.2f}" if np.isfinite(v) else "—"),
        "цена 1г": z["chg_1y"].map(lambda v: f"{v:+.0f}%" if np.isfinite(v) else "—"),
        "от ATH": z["ath_chg"].map(lambda v: f"{v:+.0f}%" if np.isfinite(v) else "—"),
        "оборот/д": z["volume"].map(_money),
    })


def report() -> None:
    out = Path(os.environ.get("FS_OUT", "../out")) / "fundscreen"
    out.mkdir(parents=True, exist_ok=True)
    print("\n===== FUNDSCREEN: недооценённая доля в растущем бизнесе без навеса предложения (DefiLlama + CoinGecko, "
          f"на {pd.Timestamp.now(tz='UTC'):%Y-%m-%d}) =====")
    overviews = {}
    for dt in DATA_TYPES:
        d = fetch(FEES_URL.format(dt=dt))
        overviews[dt] = (d or {}).get("protocols") or []
        print(f"  {dt}: протоколов {len(overviews[dt])}", flush=True)
    if not overviews["dailyFees"]:
        print("  DefiLlama не ответил — нечего считать")
        return
    if overviews["dailyFees"]:
        print(f"  поля протокола в обзоре: {sorted(overviews['dailyFees'][0].keys())}")
    fees = fees_table(overviews)
    protos = fetch(PROTOCOLS_URL) or []
    chains = fetch(CHAINS_URL) or []
    gecko: dict[str, str] = {}
    for p in protos:
        g = p.get("gecko_id")
        if g:
            gecko.setdefault(group_key(p), g)
            gecko.setdefault(str(p.get("slug") or "").lower(), g)
    for c in chains:
        if c.get("gecko_id"):
            gecko.setdefault(str(c.get("name") or "").lower().replace(" ", "-"), c["gecko_id"])
    need = fees[(fees.dailyHoldersRevenue_d30 >= HR_MIN) & ~fees.group.isin(gecko.keys())].group.tolist()
    for g in need:
        d = fetch(SUMMARY_URL.format(slug=g), retries=2, pause=0.3)
        if isinstance(d, dict) and d.get("gecko_id"):
            gecko[g] = d["gecko_id"]
    print(f"  групп-бизнесов {len(fees)}, с выручкой держателям >= ${HR_MIN / 1e3:.0f}K за 30 дн. "
          f"{int((fees.dailyHoldersRevenue_d30 >= HR_MIN).sum())}; с id CoinGecko {fees.group.isin(gecko.keys()).sum()}")
    ids = sorted({gecko[g] for g in fees.group if g in gecko})
    mrows = []
    for i in range(0, len(ids), 200):
        d = fetch(CG_URL.format(ids=",".join(ids[i:i + 200])), pause=8.0) or []
        for m in d:
            mrows.append({"gecko_id": m.get("id"), "symbol": m.get("symbol"), "mcap": m.get("market_cap"),
                          "fdv": m.get("fully_diluted_valuation"), "volume": m.get("total_volume"),
                          "chg_1y": m.get("price_change_percentage_1y_in_currency"),
                          "ath_chg": m.get("ath_change_percentage")})
    mkt = pd.DataFrame(mrows, columns=["gecko_id", "symbol", "mcap", "fdv", "volume", "chg_1y", "ath_chg"])
    for c in ("mcap", "fdv", "volume", "chg_1y", "ath_chg"):
        mkt[c] = pd.to_numeric(mkt[c], errors="coerce")
    print(f"  CoinGecko: монет с ценой {len(mkt)} из {len(ids)}")
    unl: dict[str, float] = {}
    try:
        cache = out / "llama_cache"
        cache.mkdir(exist_ok=True)
        sched, _ = _schedules(str(cache))
        now = pd.Timestamp.now(tz="UTC").floor("D")
        for _, token, cum in sched:
            if isinstance(token, str) and token.startswith("coingecko:"):
                unl[token.split(":", 1)[1]] = unlock_share(cum, now)
        print(f"  расписаний разлоков с id CoinGecko: {len(unl)}")
    except Exception as e:
        print(f"  разлоки не получены ({e})")
    z = screen(fees, gecko, mkt, unl)
    z.to_csv(out / "fundscreen.csv", index=False)
    for name, obj in (("overviews", overviews), ("markets", mrows)):
        with gzip.open(out / f"{name}.json.gz", "wt") as f:
            json.dump(obj, f)
    passed = z[z.score.notna()]
    print(f"\n  Правила: выручка держателям >= ${HR_MIN / 1e3:.0f}K / 30 дн., кап. >= ${MCAP_MIN / 1e6:.0f}M, оборот >= "
          f"${VOL_MIN / 1e6:.0f}M / день, в обращении >= {FLOAT_MIN:.0%} FDV, разлок 12 мес. <= {UNLOCK_MAX:.0%}. "
          f"Прошли {len(passed)}. Скор — дешевизна (FDV / выручка держателям) + рост (30 дн. к месяцу за год).")
    print(f"\n  --- прошли отсев, лучшие {TOP} по скору ---")
    print(show(passed.head(TOP)).to_string(index=False))
    print("\n  --- крупный бизнес, отсеянный правилами (топ-25 по выручке протокола) ---")
    rej = z[z.score.isna()].sort_values("dailyRevenue_d30", ascending=False).head(25)
    print(pd.concat([show(rej).reset_index(drop=True), rej["why"].reset_index(drop=True).rename("почему")],
                    axis=1).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 400)
    report()
