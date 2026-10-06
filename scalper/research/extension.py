"""Насколько рынок уже прошёл в сторону сделки к моменту сигнала (вход «вдогонку»). Бот 4h требует только, чтобы
дневка была по ту же сторону EMA50, и не смотрит, сколько цена уже прошла: шорт после падения на 40% и шорт в начале
снижения для него одинаковы. Признаки на закрытии свечи пробоя (только прошлое), со знаком сделки — больше значит
«уже далеко ушла в нашу сторону»:
    r7 / r30    изменение цены монеты за 7 / 30 дней
    ema_atr     расстояние закрытия от EMA50 дневок, в дневных ATR (последняя закрытая дневка)
    rng90       место в диапазоне закрытий за 90 дней: 1 — на самом краю в сторону сделки (шорт — у минимумов)
    rsi         RSI14 дневок: для шорта 100 − RSI (больше — сильнее перепродан)
    btc7        изменение BTC за 7 дней со знаком сделки
Квинтили — по IS; правило «пропускать 20% самых вытянутых по IS» на всех периодах, включая 2020–2021
(RESEARCH_START=2020-01). Правило бота без изменений, ретест, всё на 3R.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import tline
from .broad import ADV_MIN
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import _bot_base, _htf_bars, _years, coin_trades, market_context, tf_frame

COLS = ["symbol", "t", "side", "line", "entry", "confirm", "aggr", "with_trend", "close_loc", "adv", "risk_pct", "R3",
        "btc_ret7"]
FEATS = (("r7", "цена за 7 дней"), ("r30", "цена за 30 дней"), ("ema_atr", "от EMA50 дневок, дневных ATR"),
         ("rng90", "место в диапазоне 90 дней"), ("rsi", "RSI14 дневок (перепроданность для шорта)"),
         ("btc7", "BTC за 7 дней"))


def _rsi(c: pd.Series, n: int = 14) -> pd.Series:
    dlt = c.diff()
    up = dlt.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-dlt.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def features(d: pd.DataFrame, pos: np.ndarray, side: np.ndarray) -> pd.DataFrame:
    c = d["close"].to_numpy(dtype="float64")
    day = _htf_bars(d, "1D")
    ema = day["close"].ewm(span=50, adjust=False).mean()
    datr = _atr(day)
    rsi = _rsi(day["close"])
    closed = day.index + pd.Timedelta(days=1)                       # дневка известна после закрытия
    bar_close = d.index + pd.Timedelta(hours=4)
    k = closed.searchsorted(bar_close, side="right") - 1            # последняя закрытая дневка на закрытии свечи
    out = {n: np.full(len(pos), np.nan) for n, _ in FEATS if n != "btc7"}
    for j, (e, sd) in enumerate(zip(pos, side)):
        if e < 0:
            continue
        if e >= 42 and c[e - 42] > 0:
            out["r7"][j] = sd * (c[e] / c[e - 42] - 1)
        if e >= 180 and c[e - 180] > 0:
            out["r30"][j] = sd * (c[e] / c[e - 180] - 1)
        kk = k[e]
        if kk >= 50 and datr.iloc[kk] > 0:
            out["ema_atr"][j] = sd * (c[e] - ema.iloc[kk]) / datr.iloc[kk]
            out["rsi"][j] = rsi.iloc[kk] if sd > 0 else 100 - rsi.iloc[kk]
        if e >= 540:
            w = c[e - 540: e + 1]
            span = w.max() - w.min()
            if span > 0:
                p = (c[e] - w.min()) / span
                out["rng90"][j] = p if sd > 0 else 1 - p
    return pd.DataFrame(out)


def collect(root: Path, syms: list[str]) -> None:
    ctx = market_context(root)
    tline.LINES, tline.SCALE_LINES = ("zz",), {}
    parts = []
    for s in mine(syms):
        try:
            x = coin_trades(root, s, "4h", ctx)
            if not len(x):
                continue
            x = x[(x.line == "zz") & ~x.confirm & (x.entry == "retest")][COLS].reset_index(drop=True)
            d = tf_frame(root, s, "4h")
            pos = d.index.get_indexer(pd.to_datetime(x["t"], utc=True))
            x = pd.concat([x, features(d, pos, x["side"].to_numpy())], axis=1)
            x["btc7"] = x["side"] * x["btc_ret7"]
            parts.append(x)
        except Exception as e:
            print(f"  extension {s}: пропуск ({e})", flush=True)
    print(f"  extension: монет со сделками {len(parts)}", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path("extension"), index=False)


def _pm(z: pd.DataFrame) -> str:
    return " / ".join(f"{z[z.per == p].R3.sum() / ((pd.Timestamp(b) - pd.Timestamp(a)).days / 30.4):+.1f}"
                      for p, (a, b) in PER_ALL.items())


def _table(items: list[tuple[str, pd.DataFrame]]) -> None:
    rows = [{"вариант": nm, **{p: _cell(z[z.per == p].assign(R=z.R3)) for p in PER_ALL},
             "R/мес " + " / ".join(PER_ALL): _pm(z)} for nm, z in items]
    print(pd.DataFrame(rows).to_string(index=False))


def report() -> None:
    parts = all_parts("extension")
    print("\n===== EXTENSION: 4h, правило бота (ретест, 3R) — насколько цена уже ушла в сторону сделки к сигналу; "
          "ячейка — R на сделку (t по дням, прибыльных, сделок в месяц) =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["per"] = ""
    for p, (a, b) in PER_ALL.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    g = _bot_base(df[df.adv.isna() | (df.adv >= ADV_MIN)])
    print(f"сделок {len(g)}, шортов {(g.side < 0).mean():.0%}")
    print("медианы признаков по IS: " + ", ".join(f"{n} {g[g.per == 'is'][n].median():+.3f}" for n, _ in FEATS))
    for n, nm in FEATS:
        is_ = g[(g.per == "is") & g[n].notna()]
        if len(is_) < 50:
            continue
        qs = is_[n].quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
        q = np.digitize(g[n], qs)
        print(f"\n  --- {nm}: квинтили по IS (границы {', '.join(f'{v:+.3f}' for v in qs)}; 4 — дальше всего ушла "
              f"в сторону сделки) ---")
        _table([(f"квинтиль {i}", g[(q == i) & g[n].notna()]) for i in range(5)]
               + [("правило: без 20% самых вытянутых", g[~(g[n] >= qs[3])]), ("бот сейчас", g)])
        keep = g[~(g[n] >= qs[3])]
        print(f"  с правилом по годам: {_years(keep)}")
    print(f"\n  бот по годам: {_years(g)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 400)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x])
    else:
        report()
