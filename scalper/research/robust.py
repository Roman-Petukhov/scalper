"""Насколько можно доверять 4h-стратегии бота: устойчивость (по методике CFM, Falck–Rej–Thesmar 2021) и режим паники
(Daniel–Moskowitz 2016: моментум теряет после падения рынка при высокой волатильности, на отскоке).

Устойчивость:
    параметры   каждый параметр правила сдвигается по одному: зигзаг 2.5 / 3.5 ATR (бот — 3), свинг для стопа —
                фрактал 3 / 7 (бот — 5), отступ стопа 0 / 0.25 ATR (бот — 0.1), ожидание ретеста 6 / 24 свечи
                (бот — 12), срок сделки 30 / 120 свечей (бот — 60). Подогнанное правило разваливается от соседних
                значений, настоящее — плавно меняется
    10% монет   200 раз убираем случайные 10% монет: разброс результата
    лучшие      убираем 1% и 5% самых прибыльных сделок и лучший месяц: держится ли плюс без них
Режим паники (шорты): BTC на момент сигнала — изменение за 7 и 30 дней, волатильность за 7 дней и её место среди
прошлого года. Паника — BTC упал за 30 дней больше чем на 15% и волатильность в верхней трети за год.
Правило «в панике риск x0.5» сравнивается с текущим по R в месяц и просадке.
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
from .smc import _cell
from .tline import PER, _bot_base, _per_month, _years, coin_trades, market_context, tf_frame

COLS = ["symbol", "t", "side", "line", "entry", "confirm", "aggr", "with_trend", "close_loc", "adv", "R3", "risk_pct"]
BASE = {"PIV": 5, "STOP_BUF": 0.1, "RETEST_BARS": 12, "HOLD4": 60}
VARIANTS: dict[str, dict] = {
    "бот (база)": {},
    "свинг-фрактал 3": {"PIV": 3}, "свинг-фрактал 7": {"PIV": 7},
    "отступ стопа 0 ATR": {"STOP_BUF": 0.0}, "отступ стопа 0.25 ATR": {"STOP_BUF": 0.25},
    "ретест ждём 6 свечей": {"RETEST_BARS": 6}, "ретест ждём 24 свечи": {"RETEST_BARS": 24},
    "срок сделки 30 свечей": {"HOLD4": 30}, "срок сделки 120 свечей": {"HOLD4": 120},
}
ZZ_VARIANTS = {"zz2_5": "зигзаг 2.5 ATR", "zz3_5": "зигзаг 3.5 ATR"}


def _set(p: dict) -> None:
    q = {**BASE, **p}
    tline.PIV, tline.STOP_BUF, tline.RETEST_BARS = q["PIV"], q["STOP_BUF"], q["RETEST_BARS"]
    tline.HOLD = {**tline.HOLD, "4h": q["HOLD4"]}


def btc_context(root: Path) -> pd.DataFrame | None:
    """BTC на закрытии каждого часа (известно на закрытии): изменение за 7 / 30 дней, волатильность часовых
    доходностей за 7 дней (годовая) и её доля ниже среди прошлых 365 дней."""
    d = tf_frame(root, "BTCUSDT", "1h")
    if d is None:
        return None
    c = d["close"]
    r = np.log(c).diff()
    vol = r.rolling(24 * 7, min_periods=24 * 5).std() * np.sqrt(24 * 365)
    daily = vol.resample("1D").last()
    pct = daily.rolling(365, min_periods=120).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
    out = pd.DataFrame({"btc_r7": (c / c.shift(24 * 7) - 1).to_numpy(), "btc_r30": (c / c.shift(24 * 30) - 1).to_numpy(),
                        "btc_vol": vol.to_numpy()}, index=d.index + pd.Timedelta(hours=1))
    out["btc_vol_pct"] = pct.reindex(out.index, method="ffill").to_numpy()
    return out


def collect(root: Path, syms: list[str]) -> None:
    ctx = market_context(root)
    btc = btc_context(root)
    tline.LINES = ("zz",)                                     # только линии бота: быстрее
    rows = []
    for s in mine(syms):
        for name, p in VARIANTS.items():
            _set(p)
            tline.SCALE_LINES = {"zz2_5": 2.5, "zz3_5": 3.5} if not p else {}
            try:
                x = coin_trades(root, s, "4h", ctx)
            except Exception as e:
                print(f"  robust {s} {name}: пропуск ({e})", flush=True)
                continue
            if not len(x):
                continue
            for line, vname in (("zz", name), *(((k, v) for k, v in ZZ_VARIANTS.items()) if not p else ())):
                z = x[x.line == line][COLS].copy()
                z["variant"] = vname
                rows.append(z)
    _set({})
    if not rows:
        return
    df = pd.concat(rows, ignore_index=True)
    if btc is not None:
        close_t = pd.to_datetime(df["t"], utc=True) + pd.Timedelta(hours=4)
        df = df.join(btc.reindex(close_t, method="ffill").reset_index(drop=True))
    df.to_parquet(part_path("robust"), index=False)


def _table(items: list[tuple[str, pd.DataFrame]], col: str = "R3") -> None:
    rows = []
    for nm, z in items:
        z = z.assign(R=z[col])
        rows.append({"вариант": nm, **{p: _cell(z[z.per == p]) for p in PER}, "R/мес IS / VAL / HO": _per_month(z, col)})
    print(pd.DataFrame(rows).to_string(index=False))


def _dd(z: pd.DataFrame, col: str) -> float:
    eq = z.sort_values("t")[col].cumsum().to_numpy()
    return float((np.maximum.accumulate(np.r_[0.0, eq]) - np.r_[0.0, eq]).max()) if len(eq) else 0.0


def report() -> None:
    parts = all_parts("robust")
    print("\n===== ROBUST: 4h, правило бота — устойчивость к параметрам, к составу монет и к лучшим сделкам; режим паники "
          "для шортов. Ячейка — R на сделку (t по дням, прибыльных, сделок в месяц), всё на 3R =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["per"] = ""
    for p, (a, b) in PER.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    allb = _bot_base(df[df.adv.isna() | (df.adv >= ADV_MIN)])
    rng = np.random.default_rng(7)
    for entry in ("retest", "market"):
        en = "ретест" if entry == "retest" else "рынок"
        e = allb[allb.entry == entry]
        names = list(VARIANTS) [:1] + list(ZZ_VARIANTS.values()) + list(VARIANTS)[1:]
        print(f"\n  --- {en}: 1. параметры (по одному от бота) ---")
        _table([(n, e[e.variant == n]) for n in names])
        g = e[e.variant == "бот (база)"]

        print(f"\n  --- {en}: 2. убираем случайные 10% монет (200 раз): R на сделку и R в месяц, 5% … 50% … 95% ---")
        syms = g.symbol.unique()
        res = {p: [] for p in PER}
        for _ in range(200):
            keep = set(rng.choice(syms, size=int(len(syms) * 0.9), replace=False))
            z = g[g.symbol.isin(keep)]
            for p, (a, b) in PER.items():
                zz = z[z.per == p]
                months = (pd.Timestamp(b) - pd.Timestamp(a)).days / 30.4
                res[p].append((zz.R3.mean(), zz.R3.sum() / months))
        for p in PER:
            arr = np.array(res[p])
            q = np.percentile(arr, [5, 50, 95], axis=0)
            print(f"  {p}: R на сделку {q[0, 0]:+.3f} … {q[1, 0]:+.3f} … {q[2, 0]:+.3f}; "
                  f"R в месяц {q[0, 1]:+.1f} … {q[1, 1]:+.1f} … {q[2, 1]:+.1f}")

        print(f"\n  --- {en}: 3. без лучших сделок (в каждом периоде отдельно) ---")
        items = [("все", g)]
        for share in (0.01, 0.05):
            parts_ = []
            for p in PER:
                zz = g[g.per == p]
                n = max(1, int(round(len(zz) * share)))
                parts_.append(zz.drop(zz.R3.nlargest(n).index))
            items.append((f"без {share:.0%} лучших", pd.concat(parts_)))
        parts_ = []
        for p in PER:
            zz = g[g.per == p]
            m = zz.groupby(zz.t.dt.to_period("M")).R3.sum()
            parts_.append(zz[zz.t.dt.to_period("M") != m.idxmax()] if len(m) else zz)
        items.append(("без лучшего месяца", pd.concat(parts_)))
        _table(items)
        print(f"  по годам: {_years(g)}")

        sh = g[g.side == -1]
        if "btc_r30" not in sh.columns or sh.btc_r30.isna().all():
            continue
        print(f"\n  --- {en}: 4. шорты и состояние BTC на момент сигнала ---")
        _table([("все шорты", sh),
                ("BTC за 7 дней < -10%", sh[sh.btc_r7 < -0.10]),
                ("BTC за 7 дней -10 … -3%", sh[(sh.btc_r7 >= -0.10) & (sh.btc_r7 < -0.03)]),
                ("BTC за 7 дней -3 … +3%", sh[(sh.btc_r7 >= -0.03) & (sh.btc_r7 < 0.03)]),
                ("BTC за 7 дней > +3%", sh[sh.btc_r7 >= 0.03]),
                ("волатильность BTC: нижняя треть за год", sh[sh.btc_vol_pct < 1 / 3]),
                ("средняя треть", sh[(sh.btc_vol_pct >= 1 / 3) & (sh.btc_vol_pct < 2 / 3)]),
                ("верхняя треть", sh[sh.btc_vol_pct >= 2 / 3]),
                ("ПАНИКА: BTC за 30 дней < -15% и волатильность в верхней трети",
                 sh[(sh.btc_r30 < -0.15) & (sh.btc_vol_pct >= 2 / 3)]),
                ("BTC за 30 дней < -15% (любая волатильность)", sh[sh.btc_r30 < -0.15])])
        panic = (g.side == -1) & (g.btc_r30 < -0.15) & (g.btc_vol_pct >= 2 / 3)
        g2 = g.assign(R_half=np.where(panic, g.R3 * 0.5, g.R3), R_skip=np.where(panic, 0.0, g.R3))
        rows = []
        for col, nm in (("R3", "как сейчас"), ("R_half", "в панике шорт с риском x0.5"),
                        ("R_skip", "в панике шорты пропускаем")):
            rows.append({"правило": nm, "R/мес IS / VAL / HO": _per_month(g2, col),
                         **{f"просадка {p}": f"{_dd(g2[g2.per == p], col):.1f}" for p in PER}})
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 320)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x])
    else:
        report()
