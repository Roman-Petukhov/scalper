"""Правило бота 4h на новых данных (2020–2021: ни один параметр на них не подбирался) и три среза по всей истории.

Данные с 2020-01 (prepare: RESEARCH_START=2020-01). Правило бота без изменений: линии по значимым точкам, агрессоры
>= 55%, тренд старшего ТФ, закрытие в верхней половине свечи, оборот от $20M за 30 прошлых дней, всё на 3R.
Срезы (по всем годам):
    сектор        мемы, AI, L1, L2, DeFi, игры — список вручную по известным монетам; остальное — «другое».
                  Плюс скопление: в тот же день сигнал в ту же сторону ещё у 2+ монет того же сектора
    время         час UTC закрытия свечи пробоя и день недели
    уровень       пробой линии совпал с пробоем горизонтального уровня: в свече пробоя закрытие прошло цену,
                  у которой за 300 свечей до этого было 2+ разворота закрытий (фрактал 10) в пределах 0.5 ATR
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
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import LEVEL_PIV, LEVEL_TOL, PER, _bot_base, _years, coin_trades, market_context, pivots, tf_frame

PER_ALL = {"2020": ("2020-01-01", "2021-01-01"), "2021": ("2021-01-01", "2022-01-01"), **PER}
COLS = ["symbol", "t", "side", "line", "entry", "confirm", "aggr", "with_trend", "close_loc", "adv", "R3", "risk_pct"]
LEVEL_LOOKBACK = 300

SECTORS: dict[str, tuple[str, ...]] = {
    "мемы": ("DOGE", "SHIB", "PEPE", "FLOKI", "BONK", "WIF", "BOME", "MEME", "PEOPLE", "TURBO", "NEIRO", "POPCAT",
             "MOODENG", "PNUT", "BRETT", "MEW", "DOGS", "TRUMP", "FARTCOIN", "SPX", "GOAT", "ACT", "CHILLGUY", "BABYDOGE",
             "SATS", "ORDI", "MYRO", "PONKE", "BAN", "HIPPO", "WHY", "CAT", "MOG", "PENGU", "PUMP", "BANANAS31"),
    "AI": ("FET", "AGIX", "OCEAN", "RNDR", "RENDER", "TAO", "WLD", "ARKM", "AI16Z", "VIRTUAL", "NFP", "AI", "IO", "AIXBT",
           "GRIFFAIN", "ZEREBRO", "COOKIE", "PHB", "CTXC", "NMR", "ALT", "AKT", "GLM", "VANA", "KAITO", "SAHARA"),
    "L1": ("BTC", "ETH", "SOL", "ADA", "AVAX", "DOT", "NEAR", "APT", "SUI", "SEI", "TIA", "ATOM", "TRX", "TON", "ALGO",
           "ICP", "FTM", "S", "EGLD", "HBAR", "XLM", "XRP", "LTC", "BCH", "ETC", "KAS", "INJ", "BNB", "EOS", "NEO", "XTZ",
           "FIL", "KAVA", "ONE", "ZIL", "IOTA", "VET", "QTUM", "WAVES", "FLOW", "MINA", "ROSE", "CFX", "KSM", "XMR", "ZEC",
           "DASH", "THETA", "BERA", "HYPE", "MOVE", "BSV", "LUNA", "LUNA2", "CELO", "ASTR", "SXT", "IP"),
    "L2": ("ARB", "OP", "MATIC", "POL", "STRK", "ZK", "MANTA", "METIS", "IMX", "BLAST", "SCR", "ZRO", "SKL", "LRC",
           "CELR", "MNT", "STX", "TAIKO", "OMNI", "SAGA", "DYM", "ALT", "AEVO", "ZETA", "W", "ME"),
    "DeFi": ("UNI", "AAVE", "MKR", "CRV", "COMP", "SNX", "LDO", "PENDLE", "DYDX", "GMX", "SUSHI", "1INCH", "JUP", "RUNE",
             "ENA", "ETHFI", "CAKE", "BAL", "YFI", "LQTY", "RDNT", "CVX", "FXS", "SSV", "RPL", "JTO", "ONDO", "MORPHO",
             "AERO", "RAY", "ORCA", "EIGEN", "LISTA", "BAKE", "KNC", "ZRX", "BAND", "UMA", "API3", "LINK", "PYTH", "TRB"),
    "игры/NFT": ("AXS", "SAND", "MANA", "GALA", "APE", "ILV", "PIXEL", "YGG", "BEAMX", "ENJ", "CHZ", "MAGIC", "GMT",
                 "PORTAL", "XAI", "ACE", "BIGTIME", "SUPER", "ALICE", "TLM", "HIGH", "VOXEL", "NOT", "CATI", "HMSTR",
                 "BLUR", "LOOKS", "FLOW"),
}


def sector(sym: str) -> str:
    base = sym.removesuffix("USDT")
    for pfx in ("1000000", "10000", "1000", "1M"):
        if base.startswith(pfx) and len(base) > len(pfx):
            base = base[len(pfx):]
            break
    for name, coins in SECTORS.items():
        if base in coins:
            return name
    return "другое"


def level_break(c: np.ndarray, a: np.ndarray, e: int, side: int, piv_hi: np.ndarray, piv_lo: np.ndarray) -> bool:
    """В свече e закрытие прошло горизонтальный уровень: 2+ разворота закрытий (подтверждены до e, за LEVEL_LOOKBACK
    свечей) в пределах LEVEL_TOL ATR, цена уровня между прошлым и текущим закрытием (с допуском LEVEL_TOL ATR)."""
    tol = LEVEL_TOL * a[e]
    pts = np.concatenate([piv_hi[(piv_hi[:, 1] <= e) & (piv_hi[:, 0] >= e - LEVEL_LOOKBACK), 0],
                          piv_lo[(piv_lo[:, 1] <= e) & (piv_lo[:, 0] >= e - LEVEL_LOOKBACK), 0]])
    if len(pts) < 2:
        return False
    px = np.sort(c[pts])
    lo_, hi_ = (c[e - 1] - tol, c[e]) if side > 0 else (c[e], c[e - 1] + tol)
    for p in px[(px >= lo_) & (px <= hi_)]:
        if np.sum(np.abs(px - p) <= tol) >= 2:
            return True
    return False


def collect(root: Path, syms: list[str]) -> None:
    ctx = market_context(root)
    tline.LINES, tline.SCALE_LINES = ("zz",), {}
    parts = []
    for s in mine(syms):
        try:
            x = coin_trades(root, s, "4h", ctx)
            if not len(x):
                continue
            x = x[(x.line == "zz") & ~x.confirm][COLS].copy()
            d = tf_frame(root, s, "4h")
            c = d["close"].to_numpy(dtype="float64")
            a = _atr(d).to_numpy()
            ph, pl = pivots(c, LEVEL_PIV, True), pivots(c, LEVEL_PIV, False)
            pos = d.index.get_indexer(pd.to_datetime(x["t"], utc=True))
            x["level"] = [level_break(c, a, int(e), int(sd), ph, pl) if e > 0 else False
                          for e, sd in zip(pos, x["side"])]
            x["sector"] = sector(s)
            parts.append(x)
        except Exception as e:
            print(f"  oos {s}: пропуск ({e})", flush=True)
    print(f"  oos: монет со сделками {len(parts)}", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path("oos"), index=False)


def _per_month_all(g: pd.DataFrame) -> str:
    out = []
    for p, (a, b) in PER_ALL.items():
        months = (pd.Timestamp(b) - pd.Timestamp(a)).days / 30.4
        out.append(f"{g[g.per == p].R3.sum() / months:+.1f}")
    return " / ".join(out)


def _table(items: list[tuple[str, pd.DataFrame]]) -> None:
    rows = []
    for nm, z in items:
        z = z.assign(R=z["R3"])
        rows.append({"вариант": nm, **{p: _cell(z[z.per == p]) for p in PER_ALL},
                     "R/мес " + " / ".join(PER_ALL): _per_month_all(z)})
    print(pd.DataFrame(rows).to_string(index=False))


def report() -> None:
    parts = all_parts("oos")
    print("\n===== OOS: правило бота 4h на 2020–2021 (новые данные) и срезы — сектор, время, горизонтальный уровень; "
          "ячейка — R на сделку (t по дням, прибыльных, сделок в месяц), всё на 3R =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["per"] = ""
    for p, (a, b) in PER_ALL.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    base = _bot_base(df[df.adv.isna() | (df.adv >= ADV_MIN)])
    print(f"монет со сделками по годам: " + ", ".join(
        f"{y}: {g.symbol.nunique()}" for y, g in base.groupby(base.t.dt.year)))
    for entry in ("retest", "market"):
        g = base[base.entry == entry]
        en = "ретест" if entry == "retest" else "рынок"
        print(f"\n  --- {en}: 1. правило бота без изменений, 2020 и 2021 — новые данные ---")
        _table([("все", g), ("шорты", g[g.side == -1]), ("лонги", g[g.side == 1])])
        print(f"  по годам: {_years(g)}")
        print(f"  издержки x2: " + ", ".join(f"{p}: {(z.R3 - 0.0011 / z.risk_pct).mean():+.3f}"
                                         for p, z in g.groupby("per") if p))

        print(f"\n  --- {en}: 7. сектор ---")
        _table([(s, g[g.sector == s]) for s in [*SECTORS, "другое"]])
        day = g.t.dt.floor("D")
        cnt = g.groupby([day, g.side, g.sector]).symbol.transform("count")
        clustered = (cnt >= 3) & (g.sector != "другое")
        _table([("в тот же день ещё 2+ сигнала того же сектора и стороны", g[clustered]),
                ("остальные", g[~clustered])])

        print(f"\n  --- {en}: 8. время закрытия свечи пробоя (UTC) и день недели ---")
        close_h = (g.t + pd.Timedelta(hours=4)).dt.hour
        _table([(f"{h:02d}:00", g[close_h == h]) for h in (0, 4, 8, 12, 16, 20)])
        wd = (g.t + pd.Timedelta(hours=4)).dt.dayofweek
        _table([("будни", g[wd < 5]), ("суббота", g[wd == 5]), ("воскресенье", g[wd == 6])])

        print(f"\n  --- {en}: 9. пробой линии совпал с пробоем горизонтального уровня ---")
        _table([("линия + уровень", g[g.level]), ("только линия", g[~g.level])])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 360)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x])
    else:
        report()
