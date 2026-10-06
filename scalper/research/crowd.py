"""Два фильтра для шортов бота (4h): переполненность шортов по funding и ближайший разлок токена.

Правило бота: линии по значимым точкам, агрессоры >= 55%, тренд старшего ТФ, закрытие в верхней половине свечи,
оборот от $20M, всё на 3R. К каждой сделке на момент закрытия свечи пробоя (только прошлое):
    fund_last   последняя выплаченная ставка funding, % за 8 ч
    fund_3d     средняя ставка за 3 дня (9 выплат)
    fund_pct    где fund_3d среди ставок этой монеты за 90 дней до сигнала (0 — самые отрицательные)
    unlock_in   через сколько дней ближайший крупный разлок (DefiLlama, >= 1% разлоченного предложения за день,
                скачок, а не линейный вестинг — как в research/unlocks.py); NaN — в ближайшие 60 дней нет
    unlock_size размер этого разлока, доля разлоченного предложения
Гипотезы: шорт в толпу (funding сильно отрицательный) хуже — риск шорт-сквиза; шорт перед крупным разлоком лучше.
Порог funding для правила берётся по IS (нижние 20% fund_3d шортов), VAL и HO его не видят.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN
from .engine import _to_utc
from .shard import all_parts, mine, part_path
from .smc import _cell
from .tline import BAR_MIN, PER, TAKER, _bot_base, _per_month, _years, coin_trades, market_context
from .unlocks import unlock_events

COLS = ["symbol", "t", "side", "line", "entry", "confirm", "aggr", "with_trend", "close_loc", "adv", "risk_pct", "R3"]
UNLOCK_AHEAD_D = 60


def _funding(root: Path, sym: str) -> pd.Series | None:
    p = root / f"{sym}-funding.parquet"
    if not p.exists():
        return None
    f = pd.read_parquet(p)
    s = pd.Series(f["rate"].to_numpy(dtype="float64"), index=_to_utc(f["ts"])).sort_index()
    return s[~s.index.duplicated()]


def _fund_features(fund: pd.Series | None, at: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=at.index, columns=["fund_last", "fund_3d", "fund_pct"], dtype="float64")
    if fund is None or fund.empty:
        return out
    m3 = fund.rolling("3D").mean()
    for i, t in at.items():
        k = fund.index.searchsorted(t, side="right") - 1                   # последняя выплата не позже закрытия свечи
        if k < 0:
            continue
        out.at[i, "fund_last"] = fund.iloc[k] * 100
        out.at[i, "fund_3d"] = m3.iloc[k] * 100
        hist = m3.iloc[:k + 1]
        hist = hist[hist.index > t - pd.Timedelta(days=90)]
        if len(hist) >= 60:
            out.at[i, "fund_pct"] = (hist < m3.iloc[k]).mean()
    return out


def collect(root: Path, syms: list[str]) -> None:
    ctx = market_context(root)
    cache = root / "cache" / "llama"
    cache.mkdir(parents=True, exist_ok=True)
    try:
        ev = unlock_events(cache)
    except Exception as e:
        print(f"  crowd: разлоки не загрузились ({e})", flush=True)
        ev = pd.DataFrame(columns=["ticker", "t", "size"])
    have = set(syms)
    tick = {}
    for tk in ev["ticker"].unique() if len(ev) else []:
        for s in (f"{tk}USDT", f"1000{tk}USDT"):
            if s in have:
                tick[tk] = s
                break
    ev = ev.assign(symbol=ev["ticker"].map(tick)).dropna(subset=["symbol"]) if len(ev) else ev
    print(f"  crowd: разлоков с монетой Binance {len(ev)}", flush=True)
    parts = []
    for s in mine(syms):
        try:
            x = coin_trades(root, s, "4h", ctx)
            if not len(x):
                continue
            x = x[x.line == "zz"][COLS].copy()
            close_t = pd.to_datetime(x["t"], utc=True) + pd.Timedelta(minutes=BAR_MIN["4h"])
            x = x.join(_fund_features(_funding(root, s), close_t))
            x["unlock_in"], x["unlock_size"] = np.nan, np.nan
            e = ev[ev.symbol == s].sort_values("t") if len(ev) else ev
            if len(e):
                et = pd.DatetimeIndex(pd.to_datetime(e["t"], utc=True))
                for i, t in close_t.items():
                    k = et.searchsorted(t, side="right")
                    if k < len(et):
                        days = (et[k] - t) / pd.Timedelta(days=1)
                        if days <= UNLOCK_AHEAD_D:
                            x.at[i, "unlock_in"], x.at[i, "unlock_size"] = days, e["size"].iloc[k]
            parts.append(x)
        except Exception as ex:
            print(f"  crowd {s}: пропуск ({ex})", flush=True)
    print(f"  crowd: монет со сделками {len(parts)}", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path("crowd"), index=False)


def _table(items: list[tuple[str, pd.DataFrame]]) -> None:
    rows = []
    for nm, z in items:
        z = z.assign(R=z["R3"])
        rows.append({"вариант": nm, **{p: _cell(z[z.per == p]) for p in PER}, "R/мес IS / VAL / HO": _per_month(z, "R3")})
    print(pd.DataFrame(rows).to_string(index=False))


def report() -> None:
    parts = all_parts("crowd")
    print("\n===== CROWD: шорты бота 4h — funding на момент сигнала и ближайший разлок; ячейка — R на сделку (t по дням, "
          "прибыльных, сделок в месяц), всё на 3R; R/мес — сумма R за месяц =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["per"] = ""
    for p, (a, b) in PER.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    base = _bot_base(df[df.adv.isna() | (df.adv >= ADV_MIN)])
    print(f"сделок правила бота {len(base)}, с funding {base.fund_3d.notna().mean():.0%}, "
          f"с разлоком в ближайшие {UNLOCK_AHEAD_D} дней {base.unlock_in.notna().mean():.0%}")
    for entry in ("retest", "market"):
        g = base[base.entry == entry]
        sh, lg = g[g.side == -1], g[g.side == 1]
        en = "ретест" if entry == "retest" else "рынок"
        print(f"\n  --- {en}: шорты по funding (средняя ставка за 3 дня, % за 8 ч; стандарт Binance +0.01) ---")
        _table([("все шорты", sh),
                ("funding < -0.02 (толпа в шортах)", sh[sh.fund_3d < -0.02]),
                ("-0.02 … 0", sh[(sh.fund_3d >= -0.02) & (sh.fund_3d < 0)]),
                ("0 … 0.01", sh[(sh.fund_3d >= 0) & (sh.fund_3d < 0.0101)]),
                ("> 0.01 (толпа в лонгах)", sh[sh.fund_3d >= 0.0101]),
                ("последняя ставка < -0.05", sh[sh.fund_last < -0.05]),
                ("ниже всего за 90 дней (нижние 10%)", sh[sh.fund_pct <= 0.1]),
                ("выше медианы за 90 дней", sh[sh.fund_pct > 0.5])])
        is_sh = sh[(sh.per == "is") & sh.fund_3d.notna()]
        if len(is_sh) >= 30:
            q = float(is_sh.fund_3d.quantile(0.2))
            keep = sh[~(sh.fund_3d < q)]
            print(f"  правило: пропускать шорт, если funding за 3 дня ниже {q:+.4f}% (нижние 20% по IS):")
            _table([("шорты с правилом", keep), ("отброшенные", sh[sh.fund_3d < q]),
                    ("с правилом + лонги (весь бот)", pd.concat([keep, lg]))])
            q_ = sh.fund_3d.quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
            tab = sh.assign(q=np.digitize(sh.fund_3d, q_)).groupby(["q", "per"])["R3"].mean().unstack() \
                .reindex(columns=list(PER))
            print("  средний R по квинтилям funding (0 — самые отрицательные):")
            print("  " + tab.round(3).to_string().replace("\n", "\n  "))
        print(f"  лонги по funding: " + "; ".join(
            f"{nm}: {z.R3.mean():+.2f} ({len(z)})" for nm, z in (
                ("funding > 0.03", lg[lg.fund_3d > 0.03]), ("остальные", lg[~(lg.fund_3d > 0.03)])) if len(z)))

        print(f"\n  --- {en}: шорты и ближайший крупный разлок ---")
        big = sh.unlock_size >= 0.02
        _table([("все шорты", sh),
                ("разлок через 0–7 дней", sh[sh.unlock_in <= 7]),
                ("разлок через 7–14 дней", sh[(sh.unlock_in > 7) & (sh.unlock_in <= 14)]),
                ("разлок через 14–30 дней", sh[(sh.unlock_in > 14) & (sh.unlock_in <= 30)]),
                ("разлок через 0–14 дней, >= 2% предложения", sh[(sh.unlock_in <= 14) & big]),
                ("без разлока в ближайшие 30 дней", sh[~(sh.unlock_in <= 30)])])
        z = sh[sh.unlock_in <= 14]
        if len(z):
            print(f"  шорты с разлоком через 0–14 дней по годам: {_years(z)}")
            print(f"  они же при издержках x2: {z.R3.sub(2 * TAKER / z.risk_pct).mean():+.3f}R на сделку")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 300)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x])
    else:
        report()
