"""Внешние данные как фильтр режима: эмиссия стейблкоинов (DefiLlama) и подразумеваемая волатильность BTC (Deribit DVOL).

Вопрос: говорят ли деньги, приходящие в крипту, и страх на опционах, когда лонги / дневной тренд Дончиана работают.
Данные (бесплатные, известны на закрытии дня, решение — на следующий день):
    стейблкоины   суммарное предложение долларовых стейблкоинов, stablecoins.llama.fi/stablecoincharts/all
    DVOL          индекс подразумеваемой волатильности BTC на 30 дней, Deribit public/get_volatility_index_data
                  (с марта 2021 — раньше включатели на DVOL считаются выключенными, сравнение — на тех же днях)
Включатели (заданы до запуска):
    приток        предложение стейблкоинов за 30 дней выросло
    приток 2%     выросло больше чем на 2%
    спокойно      DVOL ниже своей медианы за 365 дней
    страх уходит  DVOL за 30 дней снизился
    приток + BTC  приток и BTC выше SMA200 (сравнить с BTC > SMA200 из research/dtregime.py)
Отчёт: дневной тренд (research/dtrend.py: лонг, Дончиан 5–360, топ-20) с каждым включателем по периодам; рынок
(равный вес топ-20 и BTC) — средняя доходность следующих 30 дней при включателе «да» и «нет».
Принятие: включатель лучше «всегда» по Sharpe тренда во всех периодах, где он хотя бы иногда выключен.

    python -m research.trendfam collect --symbols <1h-qualified>     (дневные цены монет)
    python -m research.extreg collect                                 (внешние ряды — только часть 0)
    python -m research.extreg report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import warnings

import numpy as np
import pandas as pd

from .broad import ADV_MIN
from .dtrend import _stats, portfolio
from .oos import PER_ALL
from .shard import all_parts, part_path, shard_id

STABLE_URL = "https://stablecoins.llama.fi/stablecoincharts/all"
DVOL_URL = ("https://www.deribit.com/api/v2/public/get_volatility_index_data?currency=BTC&start_timestamp={a}"
            "&end_timestamp={b}&resolution=43200")
DVOL_START = pd.Timestamp("2021-01-01", tz="UTC")
TOP = 20


def _get(url: str) -> object | None:
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            print(f"  extreg: {url[:80]} → HTTP {e.code}", flush=True)
            if e.code in (400, 403, 404):
                return None
        except Exception as e:
            print(f"  extreg: {url[:80]} → {e}", flush=True)
        time.sleep(2 ** attempt)
    return None


def parse_stable(raw: object) -> pd.Series:
    """Дневной ряд суммарного предложения долларовых стейблкоинов, $."""
    out = {}
    for row in raw if isinstance(raw, list) else []:
        try:
            day = pd.Timestamp(int(row["date"]), unit="s", tz="UTC").normalize()
            v = (row.get("totalCirculatingUSD") or row.get("totalCirculating") or {}).get("peggedUSD")
            if v is not None:
                out[day] = float(v)
        except (KeyError, TypeError, ValueError):
            continue
    return pd.Series(out, dtype=float).sort_index()


def parse_dvol(pages: list[object]) -> pd.Series:
    """Дневное закрытие DVOL из свечей [ts_ms, open, high, low, close]: последнее значение дня (UTC)."""
    pts = {}
    for p in pages:
        data = (p or {}).get("result", {}).get("data", []) if isinstance(p, dict) else []
        for row in data:
            try:
                pts[pd.Timestamp(int(row[0]), unit="ms", tz="UTC")] = float(row[4])
            except (IndexError, TypeError, ValueError):
                continue
    s = pd.Series(pts, dtype=float).sort_index()
    return s.groupby(s.index.normalize()).last() if len(s) else s


def fetch_dvol(now: pd.Timestamp) -> pd.Series:
    pages, end = [], int(now.timestamp() * 1000)
    start = int(DVOL_START.timestamp() * 1000)
    for _ in range(200):
        p = _get(DVOL_URL.format(a=start, b=end))
        if not isinstance(p, dict) or "result" not in p:
            if pages == []:
                print(f"  extreg: DVOL — неожиданный ответ: {str(p)[:300]}", flush=True)
            break
        pages.append(p)
        cont = p["result"].get("continuation")
        if not cont or not p["result"].get("data") or int(cont) <= start:
            break
        end = int(cont)
    return parse_dvol(pages)


def collect() -> None:
    if shard_id()[0] != 0:
        return
    raw = _get(STABLE_URL)
    st = parse_stable(raw)
    if not len(st):
        print(f"  extreg: стейблкоины — не разобрал ответ: {str(raw)[:300]}", flush=True)
    dv = fetch_dvol(pd.Timestamp.now(tz="UTC"))
    print(f"  extreg: стейблкоины {len(st)} дней ({st.index.min()} … {st.index.max()}), "
          f"DVOL {len(dv)} дней ({dv.index.min()} … {dv.index.max()})", flush=True)
    df = pd.DataFrame({"stable": st, "dvol": dv})
    df.index.name = "day"
    if len(df):
        df.reset_index().to_parquet(part_path("extreg"), index=False)


def gates(ext: pd.DataFrame, btc: pd.Series, idx: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """Включатели на закрытии дня (индекс — дни без пояса, как в dtrend)."""
    e = ext.reindex(idx).ffill(limit=3)
    st, dv = e["stable"], e["dvol"]
    ch = st / st.shift(30) - 1
    med = dv.rolling(365, min_periods=180).median()
    btc_up = btc > btc.rolling(200, min_periods=200).mean()
    infl = ch > 0
    return {"всегда": pd.Series(True, index=idx), "приток": infl, "приток 2%": ch > 0.02,
            "спокойно": dv < med, "страх уходит": dv < dv.shift(30), "BTC > SMA200": btc_up,
            "приток + BTC": infl & btc_up}


def report() -> None:
    parts, eparts = all_parts("trendfam"), all_parts("extreg")
    print("\n===== EXTREG: стейблкоины (DefiLlama) и DVOL (Deribit) как включатели дневного тренда и режима рынка =====")
    if not parts or not eparts:
        print(f"частей нет (монеты {len(parts)}, внешние {len(eparts)})")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["day"] = pd.to_datetime(df["day"], utc=True).dt.tz_localize(None)
    close = df.pivot_table(index="day", columns="symbol", values="close")
    fund = df.pivot_table(index="day", columns="symbol", values="funding").reindex_like(close)
    adv = df.pivot_table(index="day", columns="symbol", values="adv").reindex_like(close)
    ext = pd.concat([pd.read_parquet(p) for p in eparts], ignore_index=True)
    ext["day"] = pd.to_datetime(ext["day"], utc=True).dt.tz_localize(None)
    ext = ext.groupby("day")[["stable", "dvol"]].last()
    print(f"стейблкоины: {ext.stable.notna().sum()} дней с {ext.stable.first_valid_index()}; DVOL: "
          f"{ext.dvol.notna().sum()} дней с {ext.dvol.first_valid_index()}")
    btc = close["BTCUSDT"] if "BTCUSDT" in close else pd.Series(np.nan, index=close.index)
    g = gates(ext, btc, close.index)
    dv_ok = ext.dvol.reindex(close.index).ffill(limit=3).notna()

    ret = close.pct_change(fill_method=None)
    rk = adv.where(adv >= ADV_MIN).rank(axis=1, ascending=False, method="first")
    mkt = ret.where(rk.shift(1) <= TOP).mean(axis=1)                         # равный вес топ-20 вчерашнего дня
    fwd = {"топ-20": (1 + mkt).rolling(30).apply(np.prod, raw=True).shift(-30) - 1,
           "BTC": btc.shift(-30) / btc - 1}

    base = portfolio(close, fund, adv)
    verdicts = {}
    for nm, gt in g.items():
        r = base if nm == "всегда" else portfolio(close, fund, adv, gate=gt)
        on = gt.reindex(r.index).fillna(False).astype(bool)
        dv_gate = nm in ("спокойно", "страх уходит")
        print(f"\n  {nm}: включён {on[r.index >= '2020-01-01'].mean():.0%} дней" +
              (" (DVOL с 2021-03: до этого выключен; сравнение «всегда» — на днях с DVOL)" if dv_gate else ""))
        ok = []
        for p, (a, b) in PER_ALL.items():
            sel = (r.index >= a) & (r.index < b)
            if dv_gate:
                sel &= dv_ok.reindex(r.index).fillna(False).to_numpy()
            line = f"    {p:>4} (вкл. {on[sel].mean():.0%}): {_stats(r[sel])}"
            if nm != "всегда" and sel.sum() > 30:
                line += f"  | всегда: {_stats(base[sel])}"
                if 0 < on[sel].mean() < 1 and r[sel].std() > 0 and base[sel].std() > 0:
                    ok.append(r[sel].mean() / r[sel].std() > base[sel].mean() / base[sel].std())
            print(line)
            parts_ = []
            for mn, fr in fwd.items():
                f_ = fr.reindex(r.index)[sel]
                parts_.append(f"{mn} +30д: да {f_[on[sel]].mean():+.1%} / нет {f_[~on[sel]].mean():+.1%}")
            print("          рынок — " + "; ".join(parts_))
        if nm != "всегда":
            verdicts[nm] = "принято" if ok and all(ok) else f"не принято ({sum(ok)} из {len(ok)})"
    print("\n  итог по Sharpe тренда против «всегда»: " + "; ".join(f"{k} — {v}" for k, v in verdicts.items()))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    a = ap.parse_args()
    collect() if a.mode == "collect" else report()
