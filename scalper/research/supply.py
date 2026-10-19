"""Навес будущего предложения (доля выпущенных токенов, MC / FDV) и сделки бота 4h — по вопросу владельца про
микро-фундаментал.

Данные — бесплатный CoinGecko без ключа: снимок /coins/markets (общее предложение сейчас, id по тикеру — монета с
наибольшей капитализацией) и /coins/{id}/market_chart за 365 дней (капитализация и цена по дням). Доля выпущенного на
день = (капитализация / цена) / общее предложение сейчас. Глубже 365 дней бесплатно не отдают, поэтому проверка — только
на сделках последнего года (это почти весь HOLDOUT); общее предложение берётся сегодняшнее (сжигания и допечатки за год
считаются малыми). Сделки — из research/macrocal.py (правило бота 4h без изменений).

Признак и направление заданы до запуска: доля выпущенного на день сигнала < 50% («большой навес»); ждём, что шорты
с большим навесом лучше, лонги — хуже. Справочно — терцили доли. Принятие: разница «навес − остальные» у шортов с
t ≥ 2 при ≥ 20 сделках в каждой группе. Один период — поэтому даже прошедший признак только кандидат: проверять
вперёд, по мере накопления снимков.

    python -m research.supply report        (после python -m research.macrocal collect)
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
import warnings

import numpy as np
import pandas as pd

from .footprint import diff_t
from .shard import all_parts
from .smc import _cell

CG = "https://api.coingecko.com/api/v3"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
PAUSE, PAGES, OVERHANG = 7.0, 4, 0.5


def _get(url: str) -> object | None:
    for k in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            print(f"  supply: {url.split('?')[0]} — {e}", flush=True)
            time.sleep(PAUSE * 2 ** k)
    return None


def ids_by_ticker() -> dict[str, tuple[str, float]]:
    """Тикер → (id CoinGecko, общее предложение) по крупнейшей монете с этим тикером среди первых PAGES×250."""
    out: dict[str, tuple[str, float, float]] = {}
    for p in range(1, PAGES + 1):
        rows = _get(f"{CG}/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=250&page={p}") or []
        for r in rows if isinstance(rows, list) else []:
            sym, ts, mc = str(r.get("symbol", "")).upper(), r.get("total_supply"), r.get("market_cap") or 0
            if ts and ts > 0 and (sym not in out or mc > out[sym][2]):
                out[sym] = (r["id"], float(ts), float(mc))
        time.sleep(PAUSE)
    return {k: (v[0], v[1]) for k, v in out.items()}


def base(sym: str) -> str:
    s = sym[:-4] if sym.endswith("USDT") else sym
    for p in ("1000000", "10000", "1000"):
        if s.startswith(p):
            return s[len(p):]
    return s


def issued(cg_id: str, total: float) -> pd.Series | None:
    """Доля выпущенного по дням (индекс — день UTC)."""
    x = _get(f"{CG}/coins/{cg_id}/market_chart?vs_currency=usd&days=365&interval=daily")
    time.sleep(PAUSE)
    if not isinstance(x, dict) or not x.get("market_caps") or not x.get("prices"):
        return None
    mc = pd.Series({pd.Timestamp(t, unit="ms").normalize(): v for t, v in x["market_caps"]})
    px = pd.Series({pd.Timestamp(t, unit="ms").normalize(): v for t, v in x["prices"]})
    r = (mc / px / total).replace([np.inf, -np.inf], np.nan).dropna()
    return r[(r > 0) & (r <= 1.05)]


def report() -> None:
    parts = all_parts("macrocal")
    if not parts:
        print("  supply: нет сделок (research/macrocal.py)")
        return
    x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
    x = x[x.t >= x.t.max() - pd.Timedelta(days=360)]
    print("===== SUPPLY: доля выпущенных токенов (MC / FDV) и сделки бота 4h за последний год (CoinGecko) =====")
    ids = ids_by_ticker()
    print(f"  supply: тикеров в снимке CoinGecko {len(ids)}; сделок за год {len(x)}, монет {x.symbol.nunique()}", flush=True)
    share = {}
    for sym in sorted(x.symbol.unique()):
        hit = ids.get(base(sym))
        if hit:
            s = issued(*hit)
            if s is not None and len(s):
                share[sym] = s
    x["issued"] = [float(share[s].asof(t.normalize() - pd.Timedelta(days=1))) if s in share else np.nan
                   for s, t in zip(x.symbol, x.t)]
    z = x.dropna(subset=["issued"])
    print(f"     с долей выпущенного: {len(z)} из {len(x)} сделок, монет {z.symbol.nunique()}; "
          f"медиана доли {z.issued.median():.0%}")
    if not len(z):
        return
    q = z.issued.quantile([1 / 3, 2 / 3]).to_numpy()
    for side, nm in ((-1, "шорты"), (1, "лонги")):
        m = z[z.side == side]
        if not len(m):
            continue
        big, rest = m[m.issued < OVERHANG], m[m.issued >= OVERHANG]
        print(f"  {nm}: навес (< {OVERHANG:.0%} выпущено) {_cell(big)};  остальные {_cell(rest)};  "
              f"разница t {diff_t(big.R.to_numpy(), rest.R.to_numpy()):.1f}")
        for lo, hi, lab in ((0, q[0], f"< {q[0]:.0%}"), (q[0], q[1], f"{q[0]:.0%}–{q[1]:.0%}"), (q[1], 1.1, f"≥ {q[1]:.0%}")):
            print(f"     терциль {lab}: {_cell(m[(m.issued >= lo) & (m.issued < hi)])}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    report()
