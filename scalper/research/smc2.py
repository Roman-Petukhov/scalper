"""
Smart Money «по учебнику» на 1h: внешняя структура (свинги 10 баров), BOS и CHoCH, снятие внешней ликвидности,
вход от ордер-блока только из discount (лонг) / premium (шорт), цель — ликвидность (вершина импульса) или 3R,
сессии Лондона и Нью-Йорка; вход сеткой из 3 лимиток или одной лимиткой.

Определения (всё известно на закрытии бара t):
    свинги        фрактал 10 баров, подтверждается через 10 баров
    слом          закрытие за последним подтверждённым внешним свингом, который ещё не был пробит; BOS — в сторону
                  текущего тренда (направления прошлого слома), CHoCH — против него (первый слом — CHoCH)
    импульс       от экстремума между пробитым свингом и баром слома (дно для лонга) до максимума после него;
                  снятие ликвидности — дно импульса ниже внешнего свинг-лоу, подтверждённого до него
    OB            последняя встречная свеча на дне импульса или до 5 баров раньше; зона [low, high]
    discount      верх зоны не выше середины диапазона импульса (лонг); для шорта — низ зоны не ниже середины
Вход: 3 лимитки (верх / середина / низ зоны), живут 72 ч; стоп — за зоной на 0.25 ATR; цели: «ликвидность» —
вершина импульса (сетап берётся, если до неё >= 1.5R от средней цены лимиток) и «3R»; до 120 ч.
Исполнение и стоп в одном баре = стоп: лимитки стоят выше стопа, поэтому любой путь к минимуму бара сначала
исполняет их, порядок внутри часа здесь не важен. Вход: 3 лимитки (сетка) или одна — у края зоны / в середине:
у сетки неблагоприятный отбор (в убыточных сделках исполняются все 3 лимитки, в прибыльных — часто одна). Издержки: maker 2 б.п. (вход, тейк), taker 5.5 б.п. (стоп, таймаут), funding.
Сессии: вход (первое исполнение) в 07–10 или 12–15 UTC.

    python -m research.smc2 collect --symbols <725 монет>   (по частям)
    python -m research.smc2 report
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from .broad import ADV_MIN, adv30, group_of
from .shard import all_parts, mine, part_path
from .smc import MAKER, TAKER, _atr, _cell, swings
from .wave2 import Data2

PER = {"is": ("2022-01-01", "2024-07-01"), "val": ("2024-07-01", "2025-07-01"), "ho": ("2025-07-01", "2026-10-01")}
SWING_N = 10
VALID = 72
MAX_HOLD = 120
MIN_RR_LIQ = 1.5
KILL = {7, 8, 9, 12, 13, 14}


@njit(cache=True)
def ladder_tp(o, h, lo, c, fund, t0, side, l1, l2, l3, stop, tp, valid, max_hold):
    """R сделки (NaN — не исполнилась), число лимиток, бар выхода, бар первого исполнения."""
    n = len(c)
    legs = np.array([l1, l2, l3])
    filled = np.zeros(3, np.bool_)
    avg = (l1 + l2 + l3) / 3.0
    risk = side * (avg - stop)
    if not (risk > 0) or not (side * (tp - avg) > 0):
        return np.nan, 0, t0, -1
    first = -1
    paid = 0.0
    j = t0 + 1
    while j < n:
        new = False
        if j <= t0 + valid:
            for k in range(3):
                if not filled[k] and ((side > 0 and lo[j] < legs[k]) or (side < 0 and h[j] > legs[k])):
                    filled[k] = True
                    new = True
                    if first < 0:
                        first = j
        nf = filled.sum()
        if nf == 0:
            if (side > 0 and h[j] >= tp) or (side < 0 and lo[j] <= tp) or j >= t0 + valid:
                return np.nan, 0, j, -1
            j += 1
            continue
        paid += fund[j]
        ex, fee = np.nan, TAKER
        hit_stop = (side > 0 and lo[j] <= stop) or (side < 0 and h[j] >= stop)
        if hit_stop:
            ex = min(stop, o[j]) if side > 0 else max(stop, o[j])
            if new:
                ex = stop
        elif not new and ((side > 0 and h[j] >= tp) or (side < 0 and lo[j] <= tp)):
            ex, fee = tp, MAKER
        elif j - first >= max_hold or j == n - 1:
            ex = c[j]
        if not np.isnan(ex):
            r = 0.0
            for k in range(3):
                if filled[k]:
                    r += (side * (ex - legs[k]) - (MAKER + fee) * legs[k]) / risk / 3.0
            r -= side * paid * avg * nf / 3.0 / risk
            return r, nf, j, first
        j += 1
    return np.nan, 0, n - 1, first


def structure_setups(h: pd.DataFrame, n: int = SWING_N) -> list[dict]:
    """Сломы внешней структуры с ордер-блоками на дне (вершине) импульса."""
    o, hi, lo, c = (h[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    m = len(c)
    sh, sl = swings(hi, lo, n)
    # индекс подтверждённого свинга: значение меняется — запоминаем, где был экстремум
    sh_i, sl_i = np.full(m, -1), np.full(m, -1)
    cur_h = cur_l = -1
    for i in range(m):
        p = i - n
        if p >= n:
            if hi[p] == hi[p - n: p + n + 1].max():
                cur_h = p
            if lo[p] == lo[p - n: p + n + 1].min():
                cur_l = p
        sh_i[i], sl_i[i] = cur_h, cur_l
    out, trend = [], 0
    broken_h = broken_l = -1
    for t in range(1, m):
        for side in (1, -1):
            si = sh_i[t - 1] if side > 0 else sl_i[t - 1]
            if si < 0 or si == (broken_h if side > 0 else broken_l):
                continue
            lvl = hi[si] if side > 0 else lo[si]
            if not (side * (c[t] - lvl) > 0):
                continue
            if side > 0:
                broken_h = si
            else:
                broken_l = si
            kind = "BOS" if trend == side else "CHoCH"
            trend = side
            seg = slice(si, t + 1)
            j = si + (int(np.argmin(lo[seg])) if side > 0 else int(np.argmax(hi[seg])))
            leg_ext = lo[j] if side > 0 else hi[j]
            top = hi[j:t + 1].max() if side > 0 else lo[j:t + 1].min()
            ob = -1
            for q in range(j, max(j - 6, -1), -1):
                if (side > 0 and c[q] < o[q]) or (side < 0 and c[q] > o[q]):
                    ob = q
                    break
            if ob < 0:
                continue
            prior = sl[j - 1] if side > 0 else sh[j - 1]
            swept = bool(j > 0 and not np.isnan(prior) and side * (prior - leg_ext) > 0)
            eq = (leg_ext + top) / 2
            zl, zh = lo[ob], hi[ob]
            discount = (zh <= eq) if side > 0 else (zl >= eq)
            out.append({"t": t, "side": side, "kind": kind, "swept": swept, "zl": zl, "zh": zh, "target": top,
                        "discount": discount})
    return out


def coin_setups(h: pd.DataFrame, adv: pd.Series, sym: str) -> pd.DataFrame:
    idx = h.index
    a = _atr(h).to_numpy()
    o, hi, lo, c = (h[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = h["funding"].fillna(0.0).to_numpy(dtype="float64")
    ok_adv = adv.reindex(idx.floor("D")).to_numpy() >= ADV_MIN
    rows = []
    for s in structure_setups(h):
        t, side = s["t"], s["side"]
        if not ok_adv[t] or not (a[t] > 0) or not s["discount"]:
            continue
        zl, zh = s["zl"], s["zh"]
        l1, l2, l3 = (zh, (zl + zh) / 2, zl) if side > 0 else (zl, (zl + zh) / 2, zh)
        stop = zl - 0.25 * a[t] if side > 0 else zh + 0.25 * a[t]
        avg = (l1 + l2 + l3) / 3
        risk = side * (avg - stop)
        rr_liq = side * (s["target"] - avg) / risk if risk > 0 else np.nan
        row = {"symbol": sym, "t": idx[t], "kind": s["kind"], "swept": s["swept"], "side": side, "rr_liq": rr_liq}
        for mode, legs in (("ladder", (l1, l2, l3)), ("top", (l1, l1, l1)), ("mid", (l2, l2, l2))):
            avg_m = sum(legs) / 3
            risk_m = side * (avg_m - stop)
            for tgt, tp in (("liq", s["target"]), ("3r", avg_m + side * 3 * risk_m)):
                r, nf, ex, first = ladder_tp(o, hi, lo, c, f, t, side, *legs, stop, tp, VALID, MAX_HOLD)
                row[f"R_{tgt}_{mode}"] = r
                if tgt == "3r" and mode == "ladder":
                    row["fill_hour"] = idx[first].hour if first >= 0 else -1
        rows.append(row)
    return pd.DataFrame(rows)


def collect(root: Path, syms: list[str]) -> None:
    data = Data2(root, syms)
    parts = []
    for s in mine(syms):
        try:
            h = data.get(s, "1h", "full")
            if len(h) < 24 * 60:
                continue
            x = coin_setups(h, adv30(root, s), s)
            if len(x):
                parts.append(x)
        except Exception as e:
            print(f"  {s}: пропуск ({e})", flush=True)
        finally:
            data.cache.pop((s, "1h", "m"), None)
            data.cache.pop((s, "1h"), None)
    print(f"  монет с сетапами: {len(parts)}", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path("smc2"), index=False)


def report() -> None:
    parts = all_parts("smc2")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["group"] = [group_of(s) for s in df["symbol"]]
    df["per"] = ""
    for p, (a, b) in PER.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    df["kill"] = df["fill_hour"].isin(list(KILL))
    print(f"===== SMC2: сетапов в discount / premium {len(df):,}, монет {df.symbol.nunique()}, частей {len(parts)} =====")
    print("ячейка: средний R на сделку (t по дням, прибыльных, сделок в месяц на весь рынок)")
    rows = []
    for (k, sw, sd), g in df.groupby(["kind", "swept", "side"]):
        for tgt in ("liq", "3r"):
            gg = g[g.rr_liq >= MIN_RR_LIQ] if tgt == "liq" else g
            for mode in ("ladder", "top", "mid"):
                x = gg.assign(R=gg[f"R_{tgt}_{mode}"])
                rows.append({"слом": k, "снятие ликв.": "да" if sw else "нет", "сторона": "лонг" if sd > 0 else "шорт",
                             "цель": "ликвидность" if tgt == "liq" else "3R",
                             "вход": {"ladder": "3 лимитки", "top": "1 у края зоны", "mid": "1 в середине"}[mode],
                             **{p: _cell(x[x.per == p]) for p in PER},
                             "сессии, VAL+HO": _cell(x[(x.per != "is") & x.kill]),
                             "вне подбора, VAL+HO": _cell(x[(x.per != "is") & x.group.isin(["ext54", "fresh"])])})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 360)
    pd.set_option("display.max_columns", 30)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x])
    else:
        report()
