"""Разворот на среднем сроке на споте Binance против удержания BTC — этап 2 проверки «покупать недооценённые альты на
месяцы» (docs/knowledge.md, раздел 7, обзор 2026-10). Литература: моментум до 2 недель, разворот с 4–6 недель до
2 лет (Dobrynskaya); премия мелочи неторгуема (Ballis, Ioannidis 2026), поэтому только монеты с оборотом.

Спот Binance, все USDT-пары с делистнутыми, дневки с 2017-08. Раз в месяц, на конце месяца m — сигнал, вход по
закрытию первого дня месяца m+1, выход по закрытию первого дня месяца m+1+H. Монета умерла раньше (данных нет) —
выход по последнему закрытию минус 2%. Отбор (известен на конце m): средний оборот за 30 дней >= $1M (главная полоса;
$5M — контроль), торгуется >= 180 дней (свежие листинги исключены), отобранных >= 40.
Корзина равными весами:
    проигравшие   10% худших по доходности за L месяцев (разворот — главная гипотеза)
    победители    10% лучших (справка: моментум)
    все           все отобранные (справка: «средний альт»)
Режим: всегда / только при BTC выше SMA200 на конце m (иначе кэш). Издержки за круг 0.8% (taker 0.1% × 2 +
проскальзывание мелочи 0.3% × 2), ×2 — 1.6%.
Мерило — доходность сверх BTC за тот же срок (у режима — сверх BTC с тем же фильтром, чтобы отделить выбор монет от
ухода в кэш). Варианты главной полосы: L 1 / 3 / 6 / 12 × H 1 / 3 / 6 × режим 2 = 24 у проигравших.
Принятие: средняя доходность сверх BTC > 0 в каждом периоде 2017–19 / 2020–21 / IS / VAL / HO (по дате входа, где
когорт >= 6), t Ньюи–Уэста (лаг H) >= 3, и > 0 при издержках ×2.

    python -m research.spotrev collect          (сам качает дневки спота)
    python -m research.spotrev report
"""
from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .htftrend import list_spot, load_daily
from .shard import all_parts, mine, part_path, shard_id

LS = (1, 3, 6, 12)
HS = (1, 3, 6)
BANDS = {"от $1M": 1e6, "от $5M": 5e6}
MIN_AGE = pd.Timedelta(days=180)
MIN_ELIG = 40
DECILE = 0.10
DEAD_HAIRCUT = 0.02
COST = 0.008
REGIMES = ("всегда", "BTC > SMA200")
BASKETS = ("проигравшие", "победители", "все")
PERIODS = {"2017–19": ("2017-08-01", "2020-01-01"), "2020–21": ("2020-01-01", "2022-01-01"),
           "is": ("2022-01-01", "2024-07-01"), "val": ("2024-07-01", "2025-07-01"), "ho": ("2025-07-01", "2026-10-01")}
MIN_COHORTS = 6
# Этап 2б (по обзору, до запуска): низкая волатильность за 6 мес. (Kaya, Mostowfi, FRL 2022; Leong и др., JEF 2023) и
# MAX — лучший день прошлого месяца (Özdamar и др., Financial Innovation 2021: плюс у высоких MAX); пара — справка
CHAR_BASKETS = {"vol6": ("низкая вола", "высокая вола"), "mx": ("MAX низ", "MAX верх")}
CHAR_MAIN = ("низкая вола", "MAX верх")


def monthly(d: pd.DataFrame) -> pd.DataFrame:
    """Помесячно: закрытие последнего дня и первого дня месяца, средний оборот за 30 дней на конце месяца, дисперсия
    дневных доходностей за месяц и лучший день месяца (MAX)."""
    c = d["close"].astype("float64")
    adv = d["quote_volume"].astype("float64").rolling(30, min_periods=20).mean()
    m = c.index.tz_localize(None).to_period("M") if c.index.tz is not None else c.index.to_period("M")
    g = pd.DataFrame({"c": c, "adv": adv, "m": m, "r": c.pct_change()})
    end = g.groupby("m").agg(c_end=("c", "last"), adv=("adv", "last"), var=("r", "var"), mx=("r", "max"))
    d1 = g[g.index.day == 1].groupby("m")["c"].first().rename("c_d1")
    out = end.join(d1, how="outer")
    out.index = out.index.to_timestamp()
    return out


def collect(root: Path, end: str) -> None:
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    syms = mine(list_spot())
    with ThreadPoolExecutor(16) as pool:
        frames = dict(zip(syms, pool.map(lambda s: load_daily(s, cache, end), syms)))
    rows, meta = [], []
    for s, d in frames.items():
        try:
            if d is None or len(d) < 60:
                continue
            m = monthly(d)
            rows.append(m.assign(symbol=s).reset_index(names="month"))
            meta.append({"symbol": s, "first": d.index[0], "last": d.index[-1], "last_close": float(d["close"].iloc[-1])})
        except Exception as e:
            print(f"  spotrev {s}: пропуск ({e})", flush=True)
    print(f"  spotrev: монет {len(rows)} из {len(syms)}", flush=True)
    if rows:
        pd.concat(rows, ignore_index=True).to_parquet(part_path("spotrev"), index=False)
        pd.DataFrame(meta).to_parquet(part_path("spotrev_meta"), index=False)
    if shard_id()[0] == 0:
        btc = load_daily("BTCUSDT", cache, end)
        if btc is not None:
            pd.DataFrame({"day": btc.index, "close": btc["close"].astype("float64").to_numpy()}) \
                .to_parquet(part_path("spotrev_btc"), index=False)


def btc_regime(btc: pd.Series) -> pd.Series:
    """BTC выше SMA200 на последнем дне месяца — по месяцам (метка — начало месяца)."""
    on = btc > btc.rolling(200, min_periods=200).mean()
    s = on.groupby(on.index.to_period("M")).last()
    s.index = s.index.to_timestamp()
    return s


def exit_prices(c_d1: pd.DataFrame, c_end: pd.DataFrame, meta: pd.DataFrame, exit_m: pd.Timestamp) -> pd.Series:
    """Цена выхода на первом дне месяца exit_m; умершая раньше монета — последнее закрытие минус DEAD_HAIRCUT."""
    px = c_d1.loc[exit_m] if exit_m in c_d1.index else pd.Series(np.nan, index=c_d1.columns)
    dead = meta["last"] < exit_m
    px = px.where(~dead.reindex(px.index, fill_value=False), meta["last_close"].reindex(px.index) * (1 - DEAD_HAIRCUT))
    prev = exit_m - pd.offsets.MonthBegin(1)
    if prev in c_end.index:
        px = px.fillna(c_end.loc[prev])                      # пропуск первого дня у живой монеты
    return px


def cohorts(c_end: pd.DataFrame, c_d1: pd.DataFrame, adv: pd.DataFrame, meta: pd.DataFrame, regime: pd.Series,
            btc_sym: str, L: int, H: int, band: float, chars: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Когорты: дата входа, доходности корзин (до издержек), BTC, режим и число отобранных. chars — признаки на
    конце месяца (vol6 — дисперсия дневных доходностей за 6 мес., mx — лучший день месяца) для корзин этапа 2б."""
    months = c_end.index
    first = meta["first"].reindex(c_end.columns)
    out = []
    for i in range(L, len(months) - 1):
        m = months[i]
        entry_m, exit_m = months[i] + pd.offsets.MonthBegin(1), months[i] + pd.offsets.MonthBegin(1 + H)
        if entry_m not in c_d1.index or exit_m > months[-1]:
            continue
        past = c_end.loc[m] / c_end.loc[months[i - L]] - 1
        p0 = c_d1.loc[entry_m]
        age_ok = (first + MIN_AGE) <= (m + pd.offsets.MonthEnd(0))
        ok = past.notna() & p0.notna() & (adv.loc[m] >= band) & age_ok.reindex(past.index, fill_value=False)
        ok[btc_sym] = False
        if ok.sum() < MIN_ELIG:
            continue
        p1 = exit_prices(c_d1, c_end, meta, exit_m)
        r = (p1 / p0 - 1)[ok].dropna()
        sig = past[r.index]
        rk = sig.rank(pct=True)
        b0, b1 = c_d1[btc_sym].get(entry_m, np.nan), exit_prices(c_d1, c_end, meta, exit_m).get(btc_sym, np.nan)
        row = {"t": entry_m, "проигравшие": r[rk <= DECILE].mean(), "победители": r[rk > 1 - DECILE].mean(),
               "все": r.mean(), "btc": b1 / b0 - 1, "on": bool(regime.get(m, False)), "n": len(r)}
        for key, (lo_name, hi_name) in CHAR_BASKETS.items():
            if chars and key in chars and m in chars[key].index:
                ck = chars[key].loc[m].reindex(r.index).rank(pct=True)
                row[lo_name], row[hi_name] = r[ck <= DECILE].mean(), r[ck > 1 - DECILE].mean()
        out.append(row)
    return pd.DataFrame(out)


def nw_t(x: np.ndarray, lag: int) -> float:
    """t среднего с поправкой Ньюи–Уэста на перекрытие когорт."""
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return float("nan")
    e = x - x.mean()
    s = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else float("nan")


def excess(z: pd.DataFrame, basket: str, regime: str, cost: float) -> pd.Series:
    r = z[basket] - cost
    b = z["btc"]
    if regime != "всегда":
        r, b = r.where(z["on"], 0.0), b.where(z["on"], 0.0)
    return pd.Series((r - b).to_numpy(), index=z["t"])


def verdict(z: pd.DataFrame, basket: str, regime: str, H: int) -> tuple[bool, str]:
    x1, x2 = excess(z, basket, regime, COST), excess(z, basket, regime, 2 * COST)
    cells, ok = [], []
    for p, (a, b) in PERIODS.items():
        m = x1[(x1.index >= a) & (x1.index < b)]
        if len(m) == 0:
            continue
        cells.append(f"{p} {m.mean():+.1%} ({len(m)})")
        if len(m) >= MIN_COHORTS:
            ok.append(m.mean() > 0)
    t = nw_t(x1.to_numpy(), H)
    good = bool(ok) and all(ok) and np.isfinite(t) and t >= 3 and x2.mean() > 0
    raw = z[basket] - COST
    if regime != "всегда":
        raw = raw.where(z["on"], 0.0)
    txt = (f"{'  '.join(cells)}\n        всё время сверх BTC {x1.mean():+.1%} за {H} мес. (t {t:.1f}, лучше BTC "
           f"{(x1 > 0).mean():.0%} когорт); ×2 {x2.mean():+.1%}; сама корзина {raw.mean():+.1%}, медиана "
           f"{raw.median():+.1%}, хуже −50% — {(raw < -0.5).mean():.0%} когорт")
    return good, txt


def compound(z: pd.DataFrame, basket: str, regime: str) -> str:
    """Счёт при H = 1 (когорты не перекрываются): в год и просадка у корзины и у BTC."""
    r = z.set_index("t")[basket] - COST
    b = z.set_index("t")["btc"]
    if regime != "всегда":
        r, b = r.where(z.set_index("t")["on"], 0.0), b.where(z.set_index("t")["on"], 0.0)

    def st(x: pd.Series) -> str:
        eq = (1 + x.fillna(0.0)).cumprod()
        yrs = len(x) / 12
        return f"{eq.iloc[-1] ** (1 / yrs) - 1:+.0%} в год, просадка {(1 - eq / eq.cummax()).max():.0%}"

    return f"корзина {st(r)}; BTC {st(b)}"


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, dict] | None:
    parts, mparts, bparts = all_parts("spotrev"), all_parts("spotrev_meta"), all_parts("spotrev_btc")
    if not parts or not mparts or not bparts:
        return None
    x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    x["month"] = pd.to_datetime(x["month"]).dt.tz_localize(None)
    meta = pd.concat([pd.read_parquet(p) for p in mparts], ignore_index=True).drop_duplicates("symbol").set_index("symbol")
    for k in ("first", "last"):
        meta[k] = pd.to_datetime(meta[k], utc=True).dt.tz_localize(None).dt.normalize()
    c_end, c_d1, adv = (x.pivot_table(index="month", columns="symbol", values=k) for k in ("c_end", "c_d1", "adv"))
    c_d1, adv = c_d1.reindex(index=c_end.index, columns=c_end.columns), adv.reindex_like(c_end)
    chars = {}
    if "var" in x and "mx" in x:
        var = x.pivot_table(index="month", columns="symbol", values="var").reindex_like(c_end)
        chars = {"vol6": var.rolling(6, min_periods=5).mean(),
                 "mx": x.pivot_table(index="month", columns="symbol", values="mx").reindex_like(c_end)}
    b = pd.concat([pd.read_parquet(p) for p in bparts]).drop_duplicates("day")
    btc = pd.Series(b["close"].to_numpy(), index=pd.to_datetime(b["day"], utc=True).dt.tz_localize(None)).sort_index()
    return c_end, c_d1, adv, meta, btc_regime(btc), chars


def report() -> None:
    print("\n===== SPOTREV: разворот 1–12 мес. на споте Binance (с делистнутыми) против удержания BTC; доходность "
          f"когорты за срок H, издержки {COST:.1%} за круг =====")
    got = load()
    if got is None:
        print("  spotrev: нет данных")
        return
    c_end, c_d1, adv, meta, regime, chars = got
    btc_sym = "BTCUSDT"
    if btc_sym not in c_end.columns:
        print("  spotrev: нет BTCUSDT в монетах")
        return
    dead = int((meta["last"] < meta["last"].max() - pd.Timedelta(days=7)).sum())
    print(f"  монет {len(meta)}, из них перестали торговаться {dead}; месяцев {len(c_end)} "
          f"({c_end.index.min():%Y-%m} … {c_end.index.max():%Y-%m})")
    passed = []
    for bname, band in BANDS.items():
        main = bname == "от $1M"
        print(f"\n  ===== полоса {bname}{'' if main else ' (контроль)'} =====")
        for L, H in itertools.product(LS, HS):
            z = cohorts(c_end, c_d1, adv, meta, regime, btc_sym, L, H, band)
            if z.empty:
                print(f"    L {L}, H {H}: когорт нет")
                continue
            print(f"\n    L {L} мес., H {H} мес.: когорт {len(z)}, отобранных в медиане {z.n.median():.0f}")
            for basket, reg in itertools.product(BASKETS, REGIMES):
                good, txt = verdict(z, basket, reg, H)
                tag = ("принято" if good else "нет") if (main and basket == "проигравшие") else "справка"
                print(f"      {basket}, {reg}: {tag}\n        {txt}")
                if H == 1 and basket != "победители":
                    print(f"        счёт: {compound(z, basket, reg)}")
                if good and main and basket == "проигравшие":
                    passed.append(f"L {L} H {H} {reg}")
    print("\n  ИТОГ: " + ("прошли — " + "; ".join(passed) if passed else
                         "покупка упавших альтов на месяцы не обгоняет удержание BTC"))
    if chars:
        char_report(c_end, c_d1, adv, meta, regime, chars, btc_sym)


def char_report(c_end: pd.DataFrame, c_d1: pd.DataFrame, adv: pd.DataFrame, meta: pd.DataFrame, regime: pd.Series,
                chars: dict, btc_sym: str) -> None:
    """Этап 2б: низкая волатильность и MAX, полоса от $1M, H 1 / 3 / 6, те же правила принятия."""
    print("\n  ===== этап 2б: низкая волатильность за 6 мес. и MAX (лучший день месяца), полоса от $1M =====")
    passed = []
    for H in HS:
        z = cohorts(c_end, c_d1, adv, meta, regime, btc_sym, 1, H, BANDS["от $1M"], chars)
        if z.empty or "низкая вола" not in z:
            print(f"    H {H}: когорт нет")
            continue
        print(f"\n    H {H} мес.: когорт {len(z)}")
        for basket, reg in itertools.product([b for pair in CHAR_BASKETS.values() for b in pair], REGIMES):
            good, txt = verdict(z.dropna(subset=[basket]), basket, reg, H)
            tag = ("принято" if good else "нет") if basket in CHAR_MAIN else "справка"
            print(f"      {basket}, {reg}: {tag}\n        {txt}")
            if H == 1 and basket in CHAR_MAIN:
                print(f"        счёт: {compound(z.dropna(subset=[basket]), basket, reg)}")
            if good and basket in CHAR_MAIN:
                passed.append(f"{basket} H {H} {reg}")
    print("\n  ИТОГ 2б: " + ("прошли — " + "; ".join(passed) if passed else
                            "ни низкая волатильность, ни MAX не обгоняют удержание BTC"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--end", default=pd.Timestamp.now(tz="UTC").strftime("%Y-%m"))
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), a.end)
    else:
        report()
