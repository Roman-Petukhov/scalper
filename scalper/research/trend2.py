"""
Неочевидные трендовые гипотезы на механике деривативов (4h, 725 монет Binance), выходы с R:R до 1:5.

Гипотезы зафиксированы до запуска (окна в 4h-барах: 3 дня = 18, неделя = 42, 90 дней = 540):
    squeeze_fuel  цена за 3 дня растёт, сумма funding за 3 дня < 0 (шорты платят против движения), OI за 3 дня
                  растёт (z > 0.5) -> лонг; зеркально: цена падает, funding > 0, OI растёт -> шорт
    coil          реализованная волатильность за 3 дня в нижних 20% своих 90 дней и OI за 3 дня в верхних 20% ->
                  вход на первом закрытии за 3-дневным диапазоном, только против перекоса толпы
                  (вверх — если funding за 3 дня < 0, вниз — если > 0)
    rs_crash      BTC за 3 дня упал на 5%+, монета за 3 дня в плюсе и её OI растёт (z > 0.5) -> лонг
    capit_v       за 12 ч цена упала, а OI упал сильнее 3σ (каскад ликвидаций); в течение 48 ч после него close
                  возвращается выше цены до каскада -> лонг (зеркально для шорт-сквиза)
    funding_flip  funding неделю был положительным (сумма за неделю в верхних 30%), за последние 3 дня < 0, цена за
                  3 дня не упала -> лонг; зеркально -> шорт
Выходы и учёт — research.trend.trade_machine: стоп 1.5/2.5 ATR(14), tp5 (тейк 5R) или trail (безубыток после
+1R, трейлинг 3 ATR), до 60 баров; издержки 2 x 6 б.п., funding. Протокол как в research.trend.

    python -m research.trend2 --root <binance 1h data with metrics> --symbols ...
"""
from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30, group_of
from .trend import COST, EXITS, MAX_HOLD, MIN_IS, PER, STOPS, summarize, trade_machine
from .wave2 import Data2

FAMILIES = ("squeeze_fuel", "coil", "rs_crash", "capit_v", "funding_flip")
DIRS = ("long", "both")
D3, WK, D90 = 18, 42, 540
KEYS = ["family", "dir", "stop", "exit"]


def _z(x: pd.Series, w: int = D90) -> pd.Series:
    return (x - x.rolling(w, min_periods=w // 3).mean()) / x.rolling(w, min_periods=w // 3).std()


def _pct(x: pd.Series, w: int = D90) -> pd.Series:
    return x.rolling(w, min_periods=w // 3).rank(pct=True)


def bars4h(h: pd.DataFrame) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "funding": "sum",
           "oi": "last"}
    d = h[list(agg)].resample("4h", label="left", closed="left").agg(agg).dropna(subset=["close"])
    pc = d["close"].shift()
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    return d


def family_signals(d: pd.DataFrame, h: pd.DataFrame, btc4: pd.Series, fam: str) -> tuple[pd.Series, pd.Series]:
    c = d["close"]
    r3 = c / c.shift(D3) - 1
    f3 = d["funding"].rolling(D3).sum()
    oi = np.log(d["oi"].replace(0, np.nan))
    oi_z = _z(oi.diff(D3))
    if fam == "squeeze_fuel":
        up = (r3 > 0) & (f3 < 0) & (oi_z > 0.5)
        dn = (r3 < 0) & (f3 > 0) & (oi_z > 0.5)
    elif fam == "coil":
        rv = np.log(c).diff().rolling(D3).std()
        coiled = (_pct(rv) < 0.2) & (_pct(oi.diff(D3)) > 0.8)
        hi, lo = d["high"].shift(1).rolling(D3).max(), d["low"].shift(1).rolling(D3).min()
        armed = coiled.shift(1).fillna(False)
        up = armed & (c > hi) & (f3 < 0)
        dn = armed & (c < lo) & (f3 > 0)
    elif fam == "rs_crash":
        b3 = btc4.reindex(d.index).ffill()
        b3 = b3 / b3.shift(D3) - 1
        up = (b3 < -0.05) & (r3 > 0) & (oi_z > 0.5)
        dn = pd.Series(False, index=d.index)
    elif fam == "capit_v":
        # каскад на часовых барах: 12 ч, цена и OI против истории (как oi_liq, но порог OI 3σ)
        hc = h["close"]
        lr = np.log(hc / hc.shift(12))
        sig = np.log(hc).diff().rolling(24 * 30, min_periods=24 * 10).std() * np.sqrt(12)
        doi = np.log(h["oi"].replace(0, np.nan)).diff(12)
        doi_z = (doi - doi.rolling(24 * 30, min_periods=240).mean()) / doi.rolling(24 * 30, min_periods=240).std()
        crash = (lr / sig < -2.5) & (doi_z < -3)
        squeeze = (lr / sig > 2.5) & (doi_z < -3)
        pre = hc.shift(12)
        lvl_dn = pre.where(crash).ffill(limit=48)            # цена до каскада, помним 48 ч
        lvl_up = pre.where(squeeze).ffill(limit=48)
        up_h = lvl_dn.notna() & (hc > lvl_dn) & (hc.shift() <= lvl_dn)
        dn_h = lvl_up.notna() & (hc < lvl_up) & (hc.shift() >= lvl_up)
        # сигнал часового бара относится к 4h-бару, в котором он закрылся
        up = up_h.resample("4h", label="left", closed="left").max().reindex(d.index).fillna(False).astype(bool)
        dn = dn_h.resample("4h", label="left", closed="left").max().reindex(d.index).fillna(False).astype(bool)
    else:
        fw = d["funding"].rolling(WK).sum()
        fpct = _pct(fw)
        up = (fpct.shift(D3) > 0.7) & (fw.shift(D3) > 0) & (f3 < 0) & (r3 >= 0)
        dn = (fpct.shift(D3) < 0.3) & (fw.shift(D3) < 0) & (f3 > 0) & (r3 <= 0)
    return up.fillna(False).astype(bool), dn.fillna(False).astype(bool)


def run(root: Path, syms: list[str]) -> pd.DataFrame:
    data = Data2(root, syms)
    btc = data.get("BTCUSDT", "1h", "full")["close"].resample("4h", label="left", closed="left").last()
    cfgs = list(itertools.product(FAMILIES, DIRS, STOPS, EXITS))
    cid = {c: i for i, c in enumerate(cfgs)}
    parts, names = [], []
    for s in syms:
        try:
            h = data.get(s, "1h", "full")
        except Exception as e:
            print(f"  {s}: пропуск ({e})", flush=True)
            continue
        if h["oi"].notna().sum() < 24 * 60:
            continue
        sid = len(names)
        names.append(s)
        d = bars4h(h)
        allowed = (adv30(root, s).reindex(d.index, method="ffill") >= ADV_MIN).to_numpy()
        arr = {c: d[c].to_numpy(dtype="float64") for c in ("open", "high", "low", "close", "atr", "funding")}
        ts = d.index.as_unit("ns").asi8
        for fam in FAMILIES:
            up, dn = family_signals(d, h, btc, fam)
            for dr in DIRS:
                L = up.to_numpy() & allowed
                S = (dn.to_numpy() & allowed) if dr == "both" else np.zeros(len(d), np.bool_)
                for sk, ex in itertools.product(STOPS, EXITS):
                    i, r, du = trade_machine(arr["open"], arr["high"], arr["low"], arr["close"], arr["atr"],
                                             arr["funding"], L, S, sk, EXITS.index(ex), MAX_HOLD, COST)
                    if len(i):
                        parts.append((cid[(fam, dr, sk, ex)], sid, ts[i], r.astype(np.float32), du))
    out = pd.DataFrame({"cfg": np.concatenate([np.full(len(p[2]), p[0], np.int16) for p in parts]),
                        "sid": np.concatenate([np.full(len(p[2]), p[1], np.int16) for p in parts]),
                        "t": pd.to_datetime(np.concatenate([p[2] for p in parts]), utc=True),
                        "R": np.concatenate([p[3] for p in parts]), "bars": np.concatenate([p[4] for p in parts])})
    out = out.join(pd.DataFrame(cfgs, columns=KEYS), on="cfg")
    nm = np.array(names)
    out["symbol"] = nm[out["sid"].to_numpy()]
    out["group"] = [group_of(x) for x in out["symbol"]]
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 260)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--symbols", required=True)
    a = ap.parse_args()
    tr = run(Path(a.root), a.symbols.split(","))
    print(f"===== TREND2: сделок {len(tr)}, монет {tr['symbol'].nunique()} =====")
    rows = []
    for _, g in tr.groupby("cfg"):
        row = {k: g[k].iloc[0] for k in KEYS}
        for p, (pa, pb) in PER.items():
            st = summarize(g[(g.t >= pa) & (g.t < pb)])
            row.update({f"{p}_{k}": v for k, v in st.items()})
        rows.append(row)
    res = pd.DataFrame(rows)
    cols = KEYS + [f"{p}_{k}" for p in PER for k in ("n", "win", "avgR", "t", "pf")]
    print("\nВсе конфигурации (средний R на сделку, после издержек и funding):")
    print(res[cols].round(3).to_string(index=False))
    top = res[res["is_n"] >= MIN_IS].sort_values("is_t", ascending=False).head(8)
    passed = top[(top["val_avgR"] > 0) & (top["val_t"] > 1.5)]
    print("\nВОРОТА: лучшие по IS (t, не меньше 300 сделок) -> VAL (R > 0, t > 1.5) -> HOLDOUT и новые монеты:")
    if not len(passed):
        print("  никто не прошёл")
    for _, f in passed.iterrows():
        g = tr[(tr["family"] == f["family"]) & (tr["dir"] == f["dir"]) & (tr["stop"] == f["stop"])
               & (tr["exit"] == f["exit"])]
        new = g[g["group"].isin(["ext54", "fresh"])]
        msg = " | ".join(f"{p}: n={s_['n']}, R={s_.get('avgR', np.nan):+.3f}, win {s_.get('win', np.nan):.0%}"
                         for p, (pa, pb) in PER.items() for s_ in [summarize(new[(new.t >= pa) & (new.t < pb)])])
        print(f"  {dict(zip(KEYS, f[KEYS].values))}: HOLDOUT все монеты R={f['ho_avgR']:+.3f} (t={f['ho_t']:.2f})\n"
              f"    новые монеты: {msg}")
