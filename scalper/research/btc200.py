"""BTC против EMA200 на дневках и неделях как фильтр стороны для сделок бота — наблюдение владельца: «EMA200 на
биткоине, на дневном и недельном, хорошо отделяет тренд». Раньше 200-дневную BTC смотрели только как режим для лонгов (research/regime.py, bullongs.py) и для
трендовой книги (dtregime.py), и то SMA; шорты бота и 15m с ней не проверяли.

Сделки — правило бота без фильтра тренда (research/ema200.py: 4h ретест, 15m по рынку с наклоном <= 0.04), к ним
навешиваются: coin — тренд бота (закрытие старшего ТФ монеты против EMA50: дневка для 4h, час для 15m);
d200 — закрытие последней закрытой дневки BTC против дневной EMA200; w200 — закрытие последней закрытой недели BTC
против недельной EMA200; справочно d_sma — против SMA200 дневок. Всё известно на закрытии свечи сигнала. BTC — спот
Binance с 2017-08 (недельной EMA200 нужно 200 недель: прогрета с ~2021-06; на перпе — только с 2023), если спота нет —
перп. Знак «+1» — по стороне сделки (шорт при BTC ниже EMA200). Сделки до прогрева средней в её вариантах не берём.
Варианты, заданы до запуска, для d200 и w200 отдельно (решают 4h обе стороны и 15m шорты, как в живых настройках):
  над ботом      — сделки бота (coin = +1), из которых убираем те, где BTC против стороны. Принятие — как у надстроек
                   (research/spec.cut_verdicts): убранные убыточны в каждом периоде, где их >= 20, t по дням <= −3,
                   R в месяц с фильтром не ниже;
  вместо EMA50   — сторону решает только BTC (coin не смотрим). Принятие — R в месяц не ниже бота в каждом периоде
                   2020 / 2021 / IS / VAL / HO и t по дням >= 3 (research/ema200.replace_verdict).
Справочно — R на сделку по сторонам при BTC выше / ниже EMA200 и доля времени BTC выше EMA200 по годам.

    python -m research.btc200 collect --symbols <монеты>      # B200_TFS=4h или 15m
    python -m research.btc200 report
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import adv30
from .htftrend import load_daily
from .ema200 import coin_rows as e200_rows, replace_verdict
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _cell
from .spec import cut_verdicts
from .tline import _htf_bars, _to_bars, tf_frame

TFS = tuple(os.environ.get("B200_TFS", "4h,15m").split(","))
BASE = "BTCUSDT"
SPAN = 200
REGS = {"d200": ("1D", "дневка BTC против EMA200"), "w200": ("1W", "неделя BTC против EMA200"),
        "d_sma": ("1D", "дневка BTC против SMA200 (справочно)")}


def _sign(c: pd.Series, sma: bool = False) -> pd.Series:
    avg = c.rolling(SPAN).mean() if sma else c.ewm(span=SPAN, adjust=False).mean()
    return np.sign(c - avg).where(np.arange(len(c)) >= SPAN - 1)


def btc_regimes(day: pd.DataFrame) -> dict[str, pd.Series]:
    """Знаки закрытия BTC против средних 200: по дням (индекс — открытие дня) и по неделям (открытие недели, пн)."""
    week = _htf_bars(day, "1W")
    return {"d200": _sign(day["close"]), "w200": _sign(week["close"]), "d_sma": _sign(day["close"], sma=True)}


def coin_rows(d: pd.DataFrame, regs: dict[str, pd.Series], liq: np.ndarray, sym: str, tf: str) -> list[dict]:
    rows = e200_rows(d, liq, sym, tf)
    if not rows:
        return []
    al = {k: _to_bars(v.copy(), d, REGS[k][0], tf) for k, v in regs.items()}
    pos = d.index.get_indexer(pd.DatetimeIndex([r["t"] for r in rows]))
    out = []
    for r, i in zip(rows, pos):
        if i < 0:
            continue
        out.append({"symbol": r["symbol"], "tf": tf, "t": r["t"], "side": r["side"], "R": r["R"], "coin": r["ema50"],
                    **{k: float(v[i] * r["side"]) if np.isfinite(v[i]) else np.nan for k, v in al.items()}})
    return out


def collect(root: Path, syms: list[str]) -> None:
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    day = load_daily(BASE, cache, pd.Timestamp.now(tz="UTC").strftime("%Y-%m"))
    src = "спот"
    if day is None or len(day) < 400:
        btc = tf_frame(root, BASE, "4h")
        if btc is None:
            print(f"  btc200: нет данных {BASE}", flush=True)
            return
        day, src = _htf_bars(btc, "1D"), "перп"
    print(f"  btc200: BTC {src}, дневок {len(day)} с {day.index[0]:%Y-%m-%d}", flush=True)
    regs = btc_regimes(day)
    if mine([BASE]):
        pd.DataFrame({"day": day.index, "close": day["close"].to_numpy(), "d200": regs["d200"].to_numpy(),
                      "w200": regs["w200"].reindex(day.index, method="ffill").to_numpy()}).to_parquet(
            part_path("btc200_days"), index=False)
    for tf in TFS:
        rows = []
        for s in mine(syms):
            try:
                if tf != "4h" and not (root / f"{s}-{tf}.parquet").exists():
                    continue
                d = tf_frame(root, s, tf)
                if d is None or len(d) < 500:
                    continue
                rows += coin_rows(d, regs, adv30(root, s).reindex(d.index.floor("D")).to_numpy(), s, tf)
            except Exception as e:
                print(f"  btc200 {s} {tf}: пропуск ({e})", flush=True)
        print(f"  btc200 {tf}: сделок {len(rows)}", flush=True)
        if rows:
            pd.DataFrame(rows).to_parquet(part_path(f"btc200_{tf}"), index=False)


def _years_above(days: pd.DataFrame, col: str) -> str:
    z = days.dropna(subset=[col])
    return ", ".join(f"{y}: {(v > 0).mean():.0%}" for y, v in z.groupby(z.day.dt.year)[col])


def report() -> None:
    print("\n===== BTC200: дневка BTC против EMA200 как фильтр стороны для сделок бота (t по дням, прибыльных, сделок "
          "в месяц) =====")
    dp = all_parts("btc200_days")
    if dp:
        days = pd.read_parquet(dp[0])
        days["day"] = pd.to_datetime(days["day"], utc=True)
        for col, nm in (("d200", "дневка выше EMA200"), ("w200", "неделя выше EMA200")):
            print(f"  BTC: {nm}, доля дней по годам: {_years_above(days, col)}")
    passed = []
    for tf in TFS:
        parts = all_parts(f"btc200_{tf}")
        if not parts:
            print(f"  btc200 {tf}: нет данных")
            continue
        x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
        x = x.dropna(subset=["R", "coin"])
        slices = (("обе стороны", x),) if tf == "4h" else (("шорты", x[x.side < 0]), ("обе стороны", x))
        for label, z0 in slices:
            primary = tf == "4h" or label == "шорты"
            for reg, (_, rname) in REGS.items():
                z = z0.dropna(subset=[reg])
                bot = z[z.coin > 0]
                ref = reg == "d_sma"
                print(f"\n  --- {tf}, {label}, {rname}: бот {len(bot)} сделок (с прогретой средней), {_cell(bot)} ---")
                for sd, nm in ((1, "лонги"), (-1, "шорты")):
                    m = bot[bot.side == sd]
                    if len(m):
                        print(f"    {nm}: BTC по стороне {_cell(m[m[reg] > 0])} | против {_cell(m[m[reg] < 0])}")
                print("    над ботом — убираем сделки, где BTC против стороны:")
                for _, (ok, txt) in cut_verdicts(bot, reg, (0.0,)).items():
                    print(f"      {'прошло бы (справочно)' if ok and ref else 'принято' if ok and primary else 'нет'}\n{txt}")
                    if ok and primary and not ref:
                        passed.append(f"{tf} {label}: {rname}, над ботом")
                print("    вместо EMA50 монеты — сторону решает BTC:")
                ok, txt = replace_verdict(z[z[reg] > 0], bot)
                print(f"      {'прошло бы (справочно)' if ok and ref else 'принято' if ok and primary else 'нет'}\n{txt}")
                if ok and primary and not ref:
                    passed.append(f"{tf} {label}: {rname}, вместо EMA50")
    print("\n  ИТОГ: " + ("прошли — " + "; ".join(passed) if passed else "BTC против EMA200 не принят ни над ботом, "
                                                                          "ни вместо EMA50 монеты (ни дневки, ни недели)"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [s for s in a.symbols.split(",") if s])
    else:
        report()
