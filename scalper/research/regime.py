"""Режим всего рынка для выбора стороны (4h, правило бота): когда строить восходящие линии (лонги), когда
нисходящие (шорты). Бот выбирает сторону по дневке самой монеты (выше / ниже EMA50); здесь — поверх этого режим рынка,
известный на закрытии последней дневки до сигнала:
    широта   доля монет с оборотом от $20M, чья дневка закрылась выше своей EMA50 (пороги заданы заранее: < 40%,
             40–60%, > 60%)
    BTC 200  дневка BTC выше / ниже своей 200-дневной средней
Сравнение для лонгов и шортов отдельно, правило «лонги только при широте > 60%» и «лонги только при BTC выше 200-дневной»,
и бот целиком с таким правилом. Ретест, всё на 3R, 2020–2021 (RESEARCH_START=2020-01), IS, VAL, HO.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import tline
from .broad import ADV_MIN, adv30
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _cell
from .tline import _bot_base, _years, coin_trades, market_context, tf_frame

COLS = ["symbol", "t", "side", "line", "entry", "confirm", "aggr", "with_trend", "close_loc", "adv", "risk_pct", "R3"]


def collect(root: Path, syms: list[str]) -> None:
    ctx = market_context(root)
    tline.LINES, tline.SCALE_LINES = ("zz",), {}
    days, trades = [], []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, "1d")
            if d is not None and len(d) >= 60:
                ema = d["close"].ewm(span=50, adjust=False).mean()
                days.append(pd.DataFrame({"symbol": s, "day": d.index, "close": d["close"].to_numpy(),
                                          "above": (d["close"] > ema).to_numpy(),
                                          "adv": adv30(root, s).reindex(d.index).to_numpy()}))
            x = coin_trades(root, s, "4h", ctx)
            if len(x):
                trades.append(x[(x.line == "zz") & ~x.confirm & (x.entry == "retest")][COLS])
        except Exception as e:
            print(f"  regime {s}: пропуск ({e})", flush=True)
    print(f"  regime: монет {len(days)}, со сделками {len(trades)}", flush=True)
    if days:
        pd.concat(days, ignore_index=True).to_parquet(part_path("regime_days"), index=False)
    if trades:
        pd.concat(trades, ignore_index=True).to_parquet(part_path("regime"), index=False)


def _pm(z: pd.DataFrame) -> str:
    return " / ".join(f"{z[z.per == p].R3.sum() / ((pd.Timestamp(b) - pd.Timestamp(a)).days / 30.4):+.1f}"
                      for p, (a, b) in PER_ALL.items())


def _table(items: list[tuple[str, pd.DataFrame]]) -> None:
    rows = [{"вариант": nm, **{p: _cell(z.assign(R=z.R3)[z.per == p]) for p in PER_ALL},
             "R/мес " + " / ".join(PER_ALL): _pm(z)} for nm, z in items]
    print(pd.DataFrame(rows).to_string(index=False))


def report() -> None:
    dp, tp = all_parts("regime_days"), all_parts("regime")
    print("\n===== REGIME: 4h, правило бота (ретест, 3R) — режим рынка для выбора стороны: широта (доля монет выше EMA50 "
          "дневок) и BTC относительно 200-дневной; ячейка — R на сделку (t по дням, прибыльных, сделок в месяц) =====")
    if not dp or not tp:
        print("частей нет")
        return
    days = pd.concat([pd.read_parquet(p) for p in dp], ignore_index=True)
    days["day"] = pd.to_datetime(days["day"], utc=True)
    liq = days[days.adv >= ADV_MIN]
    breadth = liq.groupby("day").above.mean()
    n_coins = liq.groupby("day").size()
    breadth = breadth[n_coins >= 10]
    btc = days[days.symbol == "BTCUSDT"].set_index("day").close.sort_index()
    btc200 = (btc > btc.rolling(200, min_periods=200).mean()).where(btc.rolling(200, min_periods=200).count() >= 200)
    reg = pd.DataFrame({"breadth": breadth, "btc200": btc200}).sort_index()
    reg.index = reg.index + pd.Timedelta(days=1)                       # известно после закрытия дневки

    df = pd.concat([pd.read_parquet(p) for p in tp], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["per"] = ""
    for p, (a, b) in PER_ALL.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    g = _bot_base(df[df.adv.isna() | (df.adv >= ADV_MIN)]).sort_values("t")
    k = reg.index.searchsorted(g.t + pd.Timedelta(hours=4), side="right") - 1    # последняя закрытая дневка
    for col in ("breadth", "btc200"):
        vals = reg[col].to_numpy(dtype="float64")
        g[col] = np.where(k >= 0, vals[np.clip(k, 0, None)], np.nan) if len(vals) else np.nan
    print("доля дней по широте: " + ", ".join(
        f"{nm} {v:.0%}" for nm, v in (("< 40%", (reg.breadth < 0.4).mean()), ("40–60%", reg.breadth.between(0.4, 0.6).mean()),
                                      ("> 60%", (reg.breadth > 0.6).mean()))))
    for sd, nm in ((1, "лонги"), (-1, "шорты")):
        z = g[g.side == sd]
        print(f"\n  --- {nm}: {len(z)} сделок ---")
        _table([("все", z), ("широта < 40%", z[z.breadth < 0.4]), ("широта 40–60%", z[z.breadth.between(0.4, 0.6)]),
                ("широта > 60%", z[z.breadth > 0.6]), ("BTC выше 200-дневной", z[z.btc200 == 1]),
                ("BTC ниже 200-дневной", z[z.btc200 == 0]),
                ("широта > 60% и BTC выше 200-дневной", z[(z.breadth > 0.6) & (z.btc200 == 1)])])
        print(f"  по годам: {_years(z)}")

    print("\n  --- бот целиком ---")
    lg, sh = g[g.side == 1], g[g.side == -1]
    _table([("как сейчас", g),
            ("лонги только при широте > 60%", pd.concat([sh, lg[~(lg.breadth <= 0.6)]])),
            ("лонги только при BTC выше 200-дневной", pd.concat([sh, lg[~(lg.btc200 == 0)]])),
            ("шорты только при широте <= 60%", pd.concat([lg, sh[~(sh.breadth > 0.6)]])),
            ("оба: лонги при > 60%, шорты при <= 60%", pd.concat([lg[~(lg.breadth <= 0.6)], sh[~(sh.breadth > 0.6)]]))])
    print(f"  режим неизвестен (мало монет / нет 200 дней BTC) — сделка остаётся как сейчас: широта у "
          f"{g.breadth.isna().mean():.0%} сделок, BTC 200 у {g.btc200.isna().mean():.0%}")


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
