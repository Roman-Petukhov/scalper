"""Своя карта ликвидаций по OI Binance (как тепловые карты Coinglass — расчётная) и проверка гипотез о ней.

Карта (правила заданы до запуска), по часам, для каждой монеты:
    новые позиции   рост OI за час (в монетах) по типичной цене часа (H+L+C)/3; доля лонгов — доля рыночных
                    покупок часа, шортов — остальное (обе стороны с плечом)
    плечи           10× / 25× / 50× / 100× с весами 0.4 / 0.3 / 0.2 / 0.1; поддерживающая маржа 0.5%:
                    лонг ликвидируется на p·(1 − 1/L + 0.005), шорт на p·(1 + 1/L − 0.005)
    снятие          цена прошла уровень (минимум часа ≤ уровня лонгов / максимум ≥ уровня шортов) — уровень
                    ликвидирован и удаляется; падение OI — все оставшиеся уровни уменьшаются в той же доле
    сетка           логарифмическая, шаг 0.25%
Признаки на закрытии часа: ликвидации лонгов в пределах 3 ATR (4h) ниже цены и шортов выше, в долях OI;
перекос I = (шорты сверху − лонги снизу) / (сумма).
Гипотезы:
    H1 магнит        раз в сутки (00:00 UTC): I > 0.3 — цена первой коснётся +2 ATR (4h) раньше −2 ATR (за 72 ч)
                     чаще, чем при |I| ≤ 0.3, а при I < −0.3 — реже. Во всех периодах IS / VAL / HO / 2026
    H2 сбор          каскады как в oi_liq (12 ч: цена и OI за 2.5σ; против движения на 4 ч, −12 б.п.): каскад, снявший
                     по карте ≥ 1% OI ликвидаций своей стороны, лучше остальных каскадов во всех периодах
    H3 топливо       сделки бота 4h (как в боте: ретест, 3R, долив на +1.5R): у шорта — ликвидации лонгов ниже, у лонга —
                     шортов выше (доля OI); выше медианы IS — лучше, чем ниже, во всех периодах
Данные: 1h-qualified с метриками (OI с 2022 года).

    python -m research.liqmap collect --symbols <1h-qualified>
    python -m research.liqmap report
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import dconf
from .broad import ADV_MIN, adv30
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import tf_frame
from .wave2 import _price_z, _z, metrics_on_bars

TIERS = ((10, 0.4), (25, 0.3), (50, 0.2), (100, 0.1))
MM = 0.005
STEP = 0.0025
NEAR_ATR = 3.0
TOUCH_ATR, TOUCH_H = 2.0, 72
IMB = 0.3
CASCADE_K, CASCADE_THR, CASCADE_HOLD, CASCADE_COST = 12, 2.5, 4, 12e-4
SWEEP_MIN = 0.01
BOT_EXIT = "бот (3R, долив на +1.5R)"


def build_map(h: np.ndarray, lo: np.ndarray, c: np.ndarray, buy: np.ndarray, oi: np.ndarray,
              near: np.ndarray) -> dict[str, np.ndarray]:
    """Прогон карты по часам. near — полуширина окна признаков в долях цены (3 ATR 4h / цена).
    Возвращает на закрытии каждого часа: лонги снизу, шорты сверху (доли OI), снятые за час лонги / шорты (доли OI)."""
    n = len(c)
    out = {k: np.full(n, np.nan) for k in ("below", "above", "cleared_long", "cleared_short")}
    ok = np.isfinite(lo) & (lo > 0) & np.isfinite(h)
    if ok.sum() < 2:
        return out
    g0 = np.log(lo[ok].min()) - 1.0
    nb = int((np.log(h[ok].max()) + 1.0 - g0) / STEP) + 2
    L, S = np.zeros(nb), np.zeros(nb)

    def b(p: float) -> int:
        return int(np.clip((np.log(p) - g0) / STEP, 0, nb - 1))

    prev = np.nan
    for i in range(n):
        if not ok[i] or not np.isfinite(c[i]):
            continue
        il, ih = b(lo[i]), b(h[i])
        cl, cs = L[il:].sum(), S[: ih + 1].sum()                  # уровни, которые цена прошла за час
        L[il:] = 0.0
        S[: ih + 1] = 0.0
        cur = oi[i]
        if np.isfinite(cur) and cur > 0:
            if np.isfinite(prev) and prev > 0:
                d = cur - prev
                if d < 0:
                    L *= cur / prev
                    S *= cur / prev
                elif d > 0:
                    p = (h[i] + lo[i] + c[i]) / 3
                    fb = buy[i] if np.isfinite(buy[i]) else 0.5
                    for lev, w in TIERS:
                        L[b(p * (1 - 1 / lev + MM))] += d * fb * w
                        S[b(p * (1 + 1 / lev - MM))] += d * (1 - fb) * w
            ic = b(c[i])
            if np.isfinite(near[i]) and near[i] > 0:
                k = max(1, int(np.log1p(near[i]) / STEP))
                out["below"][i] = L[max(0, ic - k): ic + 1].sum() / cur
                out["above"][i] = S[ic: ic + k + 1].sum() / cur
            out["cleared_long"][i] = cl / cur
            out["cleared_short"][i] = cs / cur
            prev = cur
    return out


def first_touch(h: np.ndarray, lo: np.ndarray, i: int, up: float, dn: float, limit: int) -> int:
    for j in range(i + 1, min(len(h), i + 1 + limit)):
        hu, hd = h[j] >= up, lo[j] <= dn
        if hu and hd:
            return 0
        if hu:
            return 1
        if hd:
            return -1
    return 0


def coin_frames(root: Path, s: str) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray] | None:
    d = tf_frame(root, s, "1h")
    if d is None or len(d) < 24 * 200:
        return None
    m = metrics_on_bars(s, root, d.index, 60)
    oi = m["oi"].to_numpy(dtype="float64")
    if np.isfinite(oi).sum() < 24 * 100:
        return None
    d4 = tf_frame(root, s, "4h")
    a4 = _atr(d4) / d4["close"]                                     # ATR 4h-свечи, доля цены; известен на её закрытии
    a4.index = a4.index + pd.Timedelta(hours=4)
    near = (NEAR_ATR * a4).reindex(d.index + pd.Timedelta(hours=1), method="ffill").to_numpy()
    hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("high", "low", "close"))
    buy = (d["taker_buy_volume"] / d["volume"].replace(0, np.nan)).to_numpy()
    mp = build_map(hi, lo, c, buy, oi, near)
    d = d.assign(oi=oi)
    return d, mp, near / NEAR_ATR


def coin_rows(root: Path, s: str) -> tuple[list[dict], list[dict], list[dict]]:
    fr = coin_frames(root, s)
    if fr is None:
        return [], [], []
    d, mp, atrp = fr
    hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("high", "low", "close"))
    liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
    end = d.index + pd.Timedelta(hours=1)
    imb = (mp["above"] - mp["below"]) / (mp["above"] + mp["below"])
    h1 = []
    for i in np.flatnonzero(end.hour == 0):
        if not (liq[i] >= ADV_MIN) or not np.isfinite(imb[i]) or not np.isfinite(atrp[i]):
            continue
        w = TOUCH_ATR * atrp[i] * c[i]
        h1.append({"symbol": s, "t": end[i], "imb": imb[i], "below": mp["below"][i], "above": mp["above"][i],
                   "touch": first_touch(hi, lo, i, c[i] + w, c[i] - w, TOUCH_H),
                   "r24": c[min(len(c) - 1, i + 24)] / c[i] - 1})
    h2 = []
    dz = _z(np.log(d["oi"].replace(0, np.nan) / d["oi"].replace(0, np.nan).shift(CASCADE_K)), 24 * 30).to_numpy()
    pz = _price_z(d["close"], CASCADE_K).to_numpy()
    nxt = 0
    for i in range(CASCADE_K, len(c) - CASCADE_HOLD):
        if i < nxt or not (liq[i] >= ADV_MIN) or not (dz[i] < -CASCADE_THR) or not (abs(pz[i]) > CASCADE_THR):
            continue
        side = 1 if pz[i] < 0 else -1                               # против движения
        col = "cleared_long" if side > 0 else "cleared_short"
        swept = np.nansum(mp[col][i - CASCADE_K + 1: i + 1])
        h2.append({"symbol": s, "t": end[i], "side": side, "swept": swept,
                   "ret": side * (c[i + CASCADE_HOLD] / c[i] - 1) - CASCADE_COST})
        nxt = i + CASCADE_HOLD
    h3 = []
    d4 = tf_frame(root, s, "4h")
    d1 = tf_frame(root, s, "1d")
    liq4 = adv30(root, s).reindex(d4.index.floor("D")).to_numpy()
    pos = pd.Series(np.arange(len(d)), index=end)
    for r in dconf.coin_rows(d4, d1, liq4, s):
        k = pos.get(r["t"] + pd.Timedelta(hours=4))
        if k is None:
            continue
        fuel = mp["below"][k] if r["side"] < 0 else mp["above"][k]
        h3.append({"symbol": s, "t": r["t"], "side": r["side"], "fuel": fuel, "R": r[BOT_EXIT]})
    return h1, h2, h3


def collect(root: Path, syms: list[str]) -> None:
    out = {"liqmap_h1": [], "liqmap_h2": [], "liqmap_h3": []}
    for s in mine(syms):
        try:
            a, b, c = coin_rows(root, s)
            out["liqmap_h1"] += a
            out["liqmap_h2"] += b
            out["liqmap_h3"] += c
        except Exception as e:
            print(f"  liqmap {s}: пропуск ({e})", flush=True)
    print("  liqmap: " + ", ".join(f"{k} {len(v)}" for k, v in out.items()), flush=True)
    for k, v in out.items():
        if v:
            pd.DataFrame(v).to_parquet(part_path(k), index=False)


def _load(name: str) -> pd.DataFrame | None:
    parts = all_parts(name)
    if not parts:
        return None
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    return df


def report() -> None:
    print("\n===== LIQMAP: своя карта ликвидаций по OI Binance — магнит (H1), сбор ликвидаций в каскаде (H2), топливо "
          "для сделок бота 4h (H3) =====")
    per = {k: v for k, v in {**PER_ALL, "2026": ("2026-01-01", "2026-10-01")}.items() if k not in ("2020", "2021")}
    h1 = _load("liqmap_h1")
    if h1 is None:
        print("частей нет")
        return
    print(f"\n  H1 магнит: {len(h1)} суток-монет; доля OI в ликвидациях рядом (медиана): лонги снизу "
          f"{h1.below.median():.1%}, шорты сверху {h1.above.median():.1%}")
    ok = []
    for p, (a, b) in per.items():
        z = h1[(h1.t >= a) & (h1.t < b)]
        grp = {"шорты сверху (I > 0.3)": z[z.imb > IMB], "поровну": z[z.imb.abs() <= IMB],
               "лонги снизу (I < −0.3)": z[z.imb < -IMB]}
        ups = {}
        cells = []
        for nm, g in grp.items():
            t = g[g.touch != 0]
            ups[nm] = (t.touch > 0).mean() if len(t) else np.nan
            cells.append(f"{nm}: n={len(g)} вверх первым {ups[nm]:.0%}, 24ч {g.r24.mean():+.2%}")
        v = list(ups.values())
        if all(len(g) >= 100 for g in grp.values()):
            ok.append(v[0] > v[1] > v[2])
        print(f"    {p:>4}: " + "; ".join(cells))
    print(f"    → магнит: {'принято' if ok and all(ok) else 'не принято'} ({sum(ok)} из {len(ok)})")

    h2 = _load("liqmap_h2")
    if h2 is not None:
        print(f"\n  H2 сбор: {len(h2)} каскадов; сняли ≥ {SWEEP_MIN:.0%} OI своей стороны — {(h2.swept >= SWEEP_MIN).mean():.0%}")
        ok = []
        for p, (a, b) in per.items():
            z = h2[(h2.t >= a) & (h2.t < b)]
            y, n_ = z[z.swept >= SWEEP_MIN], z[z.swept < SWEEP_MIN]
            if len(y) >= 30 and len(n_) >= 30:
                ok.append(y.ret.mean() > n_.ret.mean())
            print(f"    {p:>4}: сняли {len(y)} → {y.ret.mean() * 1e4:+.0f} б.п.; не сняли {len(n_)} → "
                  f"{n_.ret.mean() * 1e4:+.0f} б.п.; все {z.ret.mean() * 1e4:+.0f} б.п.")
        print(f"    → сбор ликвидаций: {'принято' if ok and all(ok) else 'не принято'} ({sum(ok)} из {len(ok)})")

    h3 = _load("liqmap_h3")
    if h3 is not None:
        h3 = h3.dropna(subset=["fuel", "R"])
        a0, b0 = per["is"]
        med = h3[(h3.t >= a0) & (h3.t < b0)].fuel.median()
        print(f"\n  H3 топливо: {len(h3)} сделок бота (шортов {(h3.side < 0).mean():.0%}); медиана топлива IS {med:.2%} OI")
        ok = []
        for p, (a, b) in per.items():
            z = h3[(h3.t >= a) & (h3.t < b)]
            y, n_ = z[z.fuel > med], z[z.fuel <= med]
            if len(y) >= 10 and len(n_) >= 10:
                ok.append(y.R.mean() > n_.R.mean())
            print(f"    {p:>4}: топлива много {_cell(y)}; мало {_cell(n_)}")
        print(f"    → топливо: {'принято' if ok and all(ok) else 'не принято'} ({sum(ok)} из {len(ok)})")


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
