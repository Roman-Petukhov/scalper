"""Пробои линий на золоте, серебре и нефти: работает ли правило бота вне крипты.

Bybit торгует их как TradFi-перпетуалы в USDT (с апреля 2026) — их история слишком короткая для проверки на трёх
периодах, поэтому берётся длинная история самих активов:
    Dukascopy   часовые свечи XAUUSD, XAGUSD, WTI (LIGHTCMDUSD), Brent (BRENTCMDUSD) с 2018 года; объём тиковый,
                агрессоров нет — фильтр агрессоров не применяется
    Binance     спот PAXGUSDT (токен золота) с 2020 года — есть доля агрессоров, правило бота целиком
Свечи складываются в тот же формат, что у крипты, и проходят через tline.coin_trades — линии, вход, стоп, цели и
комиссии те же, что у бота (funding — 0: у TradFi-перпетуалов своя ставка, в истории её нет).

    python -m tools.tradfi --root ~/tradfi       — скачать, посчитать, напечатать отчёт
"""
from __future__ import annotations

import argparse
import io
import lzma
import struct
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from research import data as D
from research.smc import _cell
from research.tline import PER, _per_month, _years, coin_trades

DUKA = "https://datafeed.dukascopy.com/datafeed"
# инструмент: (название в отчёте, ожидаемый диапазон цены — по нему выбирается масштаб целых чисел Dukascopy)
DUKA_SYMS = {"XAUUSD": ("золото", (250.0, 10000.0)), "XAGUSD": ("серебро", (3.0, 200.0)),
             "LIGHTCMDUSD": ("нефть WTI", (5.0, 300.0)), "BRENTCMDUSD": ("нефть Brent", (5.0, 300.0))}
PAXG = "PAXGUSDT"
START_YEAR = 2018
TFS = ("4h", "12h", "1d")
DUMMY_TURNOVER = 1e12          # coin_trades отсекает неликвидные монеты по обороту; у этих рынков он не ограничение


def _duka_month(sym: str, year: int, month0: int) -> pd.DataFrame | None:
    """Часовые свечи одного месяца: файл LZMA, записи по 24 байта (секунды от начала месяца, open, close, low,
    high — целые в пунктах, объём — float), big-endian."""
    blob = D._get(f"{DUKA}/{sym}/{year}/{month0:02d}/BID_candles_hour_1.bi5")
    if not blob:
        return None
    try:
        raw = lzma.decompress(blob)
    except lzma.LZMAError:
        return None
    n = len(raw) // 24
    if n == 0:
        return None
    rec = np.array(struct.unpack(">" + "IIIIIf" * n, raw[: n * 24]), dtype="float64").reshape(n, 6)
    t0 = pd.Timestamp(year=year, month=month0 + 1, day=1, tz="UTC")
    idx = t0 + pd.to_timedelta(rec[:, 0], unit="s")
    return pd.DataFrame({"open": rec[:, 1], "close": rec[:, 2], "low": rec[:, 3], "high": rec[:, 4],
                         "volume": rec[:, 5]}, index=idx)


def _scale(df: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    """Целые Dukascopy -> цена: делитель 10^k, при котором медиана попадает в ожидаемый диапазон."""
    med = float(df["close"].median())
    for k in range(0, 7):
        if lo <= med / 10 ** k <= hi:
            px = df[["open", "high", "low", "close"]] / 10 ** k
            return df.assign(**{c: px[c] for c in px.columns})
    raise ValueError(f"масштаб цены не найден, медиана {med}")


def fetch_duka(sym: str, end: pd.Timestamp) -> pd.DataFrame | None:
    jobs = [(y, m) for y in range(START_YEAR, end.year + 1) for m in range(12)
            if pd.Timestamp(year=y, month=m + 1, day=1, tz="UTC") < end]
    with ThreadPoolExecutor(8) as ex:
        parts = [p for p in ex.map(lambda ym: _duka_month(sym, *ym), jobs) if p is not None and len(p)]
    if not parts:
        return None
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated()]
    df = df[(df["high"] >= df["low"]) & (df["close"] > 0)]
    return _scale(df, *DUKA_SYMS[sym][1])


def fetch_paxg(end: pd.Timestamp) -> pd.DataFrame | None:
    parts = []
    for m in pd.period_range("2020-09", end.strftime("%Y-%m"), freq="M"):
        blob = D._get(f"{D.HOSTS['cdn']}/data/spot/monthly/klines/{PAXG}/1h/{PAXG}-1h-{m}.zip")
        if blob is None:
            continue
        k = D._read_zip_csv(blob, D.KCOLS)
        ot = pd.to_numeric(k["open_time"])
        ot = np.where(ot > 1e14, ot // 1000, ot)                          # спот-архив с 2025 — в микросекундах
        k.index = pd.to_datetime(ot, unit="ms", utc=True)
        parts.append(k[["open", "high", "low", "close", "volume", "taker_buy_volume"]].astype("float64"))
    if not parts:
        return None
    df = pd.concat(parts).sort_index()
    return df[~df.index.duplicated()]


def save_as_coin(root: Path, sym: str, df: pd.DataFrame) -> None:
    """Формат свечей крипты (research/data.py), который читает tline.tf_frame."""
    out = pd.DataFrame({"open_time": (df.index.as_unit("ms").asi8).astype("int64"),
                        "open": df["open"], "high": df["high"], "low": df["low"], "close": df["close"],
                        "volume": df["volume"],
                        "quote_volume": DUMMY_TURNOVER / 24,
                        "count": 1.0,
                        "taker_buy_volume": df["taker_buy_volume"] if "taker_buy_volume" in df else np.nan,
                        "taker_buy_quote_volume": np.nan})
    out.reset_index(drop=True).to_parquet(root / f"{sym}-1h.parquet", index=False)


def collect(root: Path) -> pd.DataFrame:
    root.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp.now(tz="UTC").normalize()
    names = {}
    for sym, (name, _) in DUKA_SYMS.items():
        df = fetch_duka(sym, end)
        if df is None:
            print(f"  {sym}: данных нет (Dukascopy не ответил)", flush=True)
            continue
        print(f"  {sym} ({name}): {len(df):,} часовых свечей, {df.index[0]:%Y-%m-%d} — {df.index[-1]:%Y-%m-%d}, "
              f"цена {df['close'].iloc[0]:.2f} → {df['close'].iloc[-1]:.2f}", flush=True)
        save_as_coin(root, sym, df)
        names[sym] = name
    px = fetch_paxg(end)
    if px is not None:
        print(f"  {PAXG} (золото, спот Binance): {len(px):,} часовых свечей, {px.index[0]:%Y-%m-%d} — "
              f"{px.index[-1]:%Y-%m-%d}", flush=True)
        save_as_coin(root, PAXG, px)
        names[PAXG] = "золото PAXG (с агрессорами)"
    rows = []
    for sym in names:
        for tf in TFS:
            try:
                x = coin_trades(root, sym, tf)
            except Exception as e:                                           # отчёт по остальным всё равно нужен
                print(f"  {sym} {tf}: пропуск ({e})", flush=True)
                continue
            if len(x):
                rows.append(x.assign(name=names[sym]))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def report(df: pd.DataFrame) -> None:
    if df.empty:
        print("сделок нет")
        return
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["per"] = ""
    for p, (a, b) in PER.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    base = df[~df.confirm & (df.line == "zz")]
    print("\n=== TRADFI: правило бота на золоте, серебре и нефти (линии по значимым точкам, пробой, стоп за свингом, "
          "всё на 3R); ячейка — R на сделку (t, прибыльных, сделок в месяц) ===")
    print("фильтр бота «агрессоры >= 55%» — только у PAXG; у Dukascopy объём тиковый, агрессоров нет")

    def table(items: list[tuple[str, pd.DataFrame]]) -> None:
        rows = []
        for nm, z in items:
            z = z.assign(R=z["R3"])
            rows.append({"вариант": nm, **{p: _cell(z[z.per == p]) for p in PER},
                         "R/мес IS / VAL / HO": _per_month(z, "R3")})
        print(pd.DataFrame(rows).to_string(index=False))

    for tf in TFS:
        for entry in ("retest", "market"):
            g = base[(base.tf == tf) & (base.entry == entry)]
            if g.empty:
                continue
            rule = g[g.with_trend & (g.close_loc >= 0.5)]
            duka = rule[rule.symbol != PAXG]
            items = [("все пробои, без фильтров", g[g.symbol != PAXG]),
                     ("тренд + закрытие в верхней половине (как у бота, без агрессоров)", duka),
                     ("  лонги", duka[duka.side == 1]), ("  шорты", duka[duka.side == -1])]
            items += [(f"  {n}", duka[duka.name == n]) for n in sorted(duka.name.unique())]
            px = rule[rule.symbol == PAXG]
            if len(px):
                items += [("PAXG: тренд + верхняя половина", px), ("PAXG: + агрессоры >= 55% (правило бота)",
                                                                   px[px.aggr >= 0.55])]
            print(f"\n  --- {tf}, {'ретест' if entry == 'retest' else 'рынок'} ---")
            table(items)
            print(f"  по годам (без PAXG, правило без агрессоров): {_years(duka)}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 20)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/tradfi")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    df = collect(Path(a.root).expanduser())
    if a.out and not df.empty:
        df.to_parquet(Path(a.out) / "tradfi_trades.parquet", index=False)
    report(df)


if __name__ == "__main__":
    main()
