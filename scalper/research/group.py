"""Подтверждение группой (Ливермор: входить, только когда идёт и сектор): пробой 4h бота против тренда соседей.

Сигналы — пробои линий 4h с фильтрами бота (research/topcrowd.py coin_rows: оборот, дневной тренд монеты, закрытие
в своей половине, стоп за свингом, ретест, 3R и долив), отдельно — с агрессорами ≥ 55% (как торгует бот) и все.
Тренд соседа — дневное закрытие выше EMA50 (лонг) / ниже (шорт) на последнем закрытом дне до закрытия свечи пробоя;
соседи — монеты с оборотом от $20M в этот день, кроме самой монеты, BTC и ETH. Доля соседей «за нас» по группам:
    сектор        та же группа разметки research/oos.py SECTORS (мемы, AI, L1, L2, DeFi, игры), нужно ≥ 3 соседа
    двойники      10 монет с наибольшей корреляцией дневных доходностей за 90 дней до начала месяца сигнала (от 60
                  общих дней), из них — с оборотом от $20M в день сигнала (нужно ≥ 3)
    весь рынок    все соседи — контроль: если он отбирает так же, это погода рынка, а не сектор
Решение (задано до запуска): группа помогает, если сделки с долей соседей «за нас» ≥ 50% лучше, чем < 50%, во всех
периодах 2020 / 2021 / IS / VAL / HO / 2026 (в обеих частях ≥ 10 сделок), а отсекаемые с 2022 в сумме в минусе
(фильтр добавляет R, а не только режет сделки). Порог 2/3 — для справки.

    python -m research.group collect --symbols <1h-qualified>
    python -m research.group report
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30
from .oos import PER_ALL, sector
from .shard import all_parts, mine, part_path
from .smc import _cell
from .tline import tf_frame
from .topcrowd import coin_rows

TF_HOURS = 4
EMA = 50
TWINS, CORR_DAYS, CORR_MIN = 10, 90, 60
MIN_PEERS = 3
SKIP = ("BTCUSDT", "ETHUSDT")
SINCE = "2022-01-01"
PER = {**PER_ALL, "2026": ("2026-01-01", "2026-10-01")}


def known_day(t: pd.Timestamp, hours: int = TF_HOURS) -> pd.Timestamp:
    """Последний дневной бар, закрытый к закрытию свечи, открытой в t."""
    return (t + pd.Timedelta(hours=hours)).floor("D") - pd.Timedelta(days=1)


def trend_state(close: pd.DataFrame) -> pd.DataFrame:
    """+1 — закрытие выше EMA50, −1 — ниже, NaN — истории меньше 50 дней."""
    ema = close.ewm(span=EMA, adjust=False, min_periods=EMA).mean()
    st = np.sign(close - ema)
    return st.where(ema.notna() & close.notna())


def share(state: pd.Series, peers: list[str], side: int) -> float:
    """Доля соседей с трендом в сторону сделки; NaN, если соседей с известным трендом меньше MIN_PEERS."""
    s = state.reindex(peers).dropna()
    return float((s == side).mean()) if len(s) >= MIN_PEERS else np.nan


def twins(ret: pd.DataFrame, month: pd.Timestamp, pool: list[str], k: int = TWINS) -> dict[str, list[str]]:
    """Для каждой монеты — k монет из pool с наибольшей корреляцией дневных доходностей за CORR_DAYS дней до начала
    месяца month (весь месяц живёт с одним списком — без заглядывания вперёд)."""
    end = month - pd.Timedelta(days=1)
    w = ret.loc[end - pd.Timedelta(days=CORR_DAYS - 1):end]
    w = w.loc[:, w.notna().sum() >= CORR_MIN]
    if w.shape[1] < 2:
        return {}
    m = w.corr(min_periods=CORR_MIN).to_numpy(copy=True)
    np.fill_diagonal(m, np.nan)
    cm = pd.DataFrame(m, index=w.columns, columns=w.columns)
    cols = cm[[s for s in pool if s in cm]]
    return {s: list(cols.loc[s].dropna().nlargest(k).index) for s in cm.index}


def peer_shares(sig: pd.DataFrame, close: pd.DataFrame, adv: pd.DataFrame) -> pd.DataFrame:
    """К каждому сигналу — доля соседей «за нас» в секторе, у двойников и по всему рынку."""
    state = trend_state(close)
    ret = close.pct_change(fill_method=None)
    sec = {s: sector(s) for s in close.columns}
    every = [s for s in close.columns if s not in SKIP]
    cache: dict[pd.Timestamp, dict[str, list[str]]] = {}
    out = sig.copy()
    cols: dict[str, list[float]] = {"sector": [], "twins": [], "market": []}
    for x in sig.itertuples(index=False):
        day = known_day(x.t)
        if day not in state.index:
            for v in cols.values():
                v.append(np.nan)
            continue
        month = day.replace(day=1)
        if month not in cache:
            cache[month] = twins(ret, month, every)
        live = adv.loc[day]
        pool = {s for s in live.index[live.to_numpy() >= ADV_MIN] if s != x.symbol and s not in SKIP}
        st = state.loc[day]
        own = sec.get(x.symbol, "другое")
        same = [s for s in pool if sec[s] == own] if own != "другое" else []
        tw = [s for s in cache[month].get(x.symbol, []) if s in pool]
        cols["sector"].append(share(st, same, x.side))
        cols["twins"].append(share(st, tw, x.side))
        cols["market"].append(share(st, list(pool), x.side))
    for k, v in cols.items():
        out[k] = v
    return out


def collect(root: Path, syms: list[str]) -> None:
    rows, days = [], []
    for s in mine(syms):
        try:
            a = adv30(root, s)
            dd = tf_frame(root, s, "1d")
            if dd is not None and len(dd):
                days.append(pd.DataFrame({"symbol": s, "day": dd.index, "close": dd["close"].to_numpy(),
                                          "adv": a.reindex(dd.index).to_numpy()}))
            if s in SKIP:
                continue
            d = tf_frame(root, s, "4h")
            if d is None or len(d) < 500:
                continue
            liq = a.reindex(d.index.floor("D")).to_numpy()
            rows += coin_rows(d, liq, s, "4h", np.zeros(len(d)))
        except Exception as e:
            print(f"  group {s}: пропуск ({e})", flush=True)
    print(f"  group: пробоев {len(rows)}, монет с дневками {len(days)}", flush=True)
    if rows:
        pd.DataFrame(rows).drop(columns=["dev", "tf"]).to_parquet(part_path("group"), index=False)
    if days:
        pd.concat(days, ignore_index=True).to_parquet(part_path("group_daily"), index=False)


def _verdict(z: pd.DataFrame, col: str, thr: float) -> tuple[bool, list[str]]:
    ok, cells = [], []
    for p, (a, b) in PER.items():
        m = z[(z.t >= a) & (z.t < b)]
        hi, lo = m[m[col] >= thr], m[m[col] < thr]
        if len(hi) >= 10 and len(lo) >= 10:
            ok.append(hi.R.mean() > lo.R.mean())
        cells.append(f"{p}: {_cell(hi)} | {_cell(lo)}")
    cut = z[(z.t >= SINCE) & (z[col] < thr)].R.sum()
    passed = bool(ok) and all(ok) and len(ok) == len(PER) and cut < 0
    cells.append(f"→ лучше в {sum(ok)} из {len(ok)} периодов, отсекаемые с 2022 {cut:+.1f}R "
                 f"→ {'ПОМОГАЕТ' if passed else 'не помогает'}")
    return passed, cells


def report() -> None:
    sp, dp = all_parts("group"), all_parts("group_daily")
    print("\n===== GROUP: пробой 4h бота и тренд соседей (сектор / двойники по корреляции / весь рынок) =====")
    if not sp or not dp:
        print("частей нет")
        return
    sig = pd.concat([pd.read_parquet(p) for p in sp], ignore_index=True)
    sig["t"] = pd.to_datetime(sig["t"], utc=True)
    dd = pd.concat([pd.read_parquet(p) for p in dp], ignore_index=True)
    dd["day"] = pd.to_datetime(dd["day"], utc=True)
    close = dd.pivot_table(index="day", columns="symbol", values="close").sort_index()
    adv = dd.pivot_table(index="day", columns="symbol", values="adv").reindex_like(close)
    z = peer_shares(sig.dropna(subset=["R"]).sort_values("t").reset_index(drop=True), close, adv)
    print(f"пробоев {len(z)} (с агрессорами {int(z.aggr.sum())}), монет {z.symbol.nunique()}, "
          f"с сектором {z.sector.notna().mean():.0%}, с двойниками {z.twins.notna().mean():.0%}")
    for nm, sub in (("сделки бота (агрессоры ≥ 55%)", z[z.aggr]), ("все пробои (без агрессоров)", z)):
        print(f"\n  {nm}: всего {_cell(sub)}")
        for col, title in (("sector", "сектор"), ("twins", "двойники"), ("market", "весь рынок (контроль)")):
            g = sub.dropna(subset=[col])
            print(f"    {title}: {len(g)} сделок, медиана доли «за нас» {g[col].median():.0%}")
            for thr in (0.5, 2 / 3):
                _, cells = _verdict(g, col, thr)
                print(f"      порог {thr:.0%}  (≥ | <)")
                for c in cells:
                    print(f"        {c}")
        both = sub.dropna(subset=["sector", "market"])
        if len(both):
            print("    сектор при «рынок за нас ≥ 50%» — есть ли что-то сверх погоды рынка:")
            _, cells = _verdict(both[both.market >= 0.5], "sector", 0.5)
            for c in cells:
                print(f"        {c}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x])
    else:
        report()
