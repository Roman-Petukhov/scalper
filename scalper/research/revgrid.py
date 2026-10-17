"""Обратная сетка на 4h: цена прошла шаг вверх — докупаем, вниз — продаём (покупка волатильности, «черепахи» без канала).

Правило (задано до запуска), 4h-свечи, монеты с оборотом от $20M на день открытия:
    вне позиции   точка отсчёта A и шаг S = m × ATR14 (замораживаются); стоп-ордера на A + S (лонг) и A − S (шорт)
    в позиции     долив 1 частью стоп-ордером на каждом следующем шаге от последнего долива, до N частей; выход всем
                  объёмом, когда цена откатила на X шагов от лучшей цены кампании (скользящий стоп); после выхода
                  A = цена выхода, S — по новому ATR
    исполнение    стоп-ордер — по уровню, гэп — по открытию; путь внутри свечи: растущая — открытие → минимум →
                  максимум → закрытие, падающая — открытие → максимум → минимум → закрытие
    фильтр тренда входы и доливы только по дневному тренду (закрытие выше / ниже EMA50, как у бота; известен до свечи)
    издержки      taker 5.5 б.п. на каждое исполнение, funding; отдельно — издержки x2
Результат кампании (от входа до выхода) в R, где 1R = X × S на одну часть (потеря одной части, сразу выбитой стопом).
Варианты: m = 1 / 2 / 3 ATR × X = 1 / 2 шага, N = 4, голая и с фильтром; справка — N = 1 (без доливов) при m 2, X 2.
Перепроверка (после первого прогона): прошедшие на 4h варианты (шаг 1 / 1 / 2 ATR, выход 1 / 2 / 1) — те же правила
на часовых свечах с ATR и трендом последней закрытой 4h-свечи: на 4h кампании по 2–3 свечи, исход решает допущение о
пути цены внутри свечи; на 1h неопределённость вчетверо меньше.
Решение: стратегия работает, если R на кампанию > 0 во всех периодах 2020 / 2021 / IS / VAL / HO / 2026 и знак тот же
при издержках x2; фильтр помогает, если с ним R в месяц выше голой во всех периодах.

    python -m research.revgrid collect --symbols <1h-qualified>
    python -m research.revgrid report
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import TAKER, htf_trend, tf_frame

STEPS, EXITS, UNITS = (1.0, 2.0, 3.0), (1, 2), 4
MIN_ATR = 1e-4                 # ATR меньше 0.01% цены — битые данные (иначе R делится почти на ноль)
PER = {**PER_ALL, "2026": ("2026-01-01", "2026-10-01")}


def variants() -> list[tuple[float, int, int, bool]]:
    out = [(m, x, UNITS, f) for m in STEPS for x in EXITS for f in (False, True)]
    return out + [(2.0, 2, 1, False), (2.0, 2, 1, True)]


def simulate(o: np.ndarray, hi: np.ndarray, lo: np.ndarray, c: np.ndarray, atr: np.ndarray, fund: np.ndarray,
             trend: np.ndarray, liq: np.ndarray, m: float, x: int, n: int, use_trend: bool) -> list[dict]:
    """Кампании обратной сетки: {'i0', 'i1', 'side', 'units', 'r', 'r2'} (r2 — при издержках x2).

    Путь цены внутри свечи: растущая — открытие → минимум → максимум → закрытие, падающая — открытие → максимум →
    минимум → закрытие; переход от прошлого закрытия к открытию — гэп (исполнение по открытию)."""
    o, hi, lo, c = o.tolist(), hi.tolist(), lo.tolist(), c.tolist()
    atr, fund, trend, liq = atr.tolist(), fund.tolist(), trend.tolist(), liq.tolist()
    out: list[dict] = []
    st = {"anchor": None, "step": None, "side": 0, "units": 0, "last": 0.0, "ext": 0.0, "gross": 0.0, "fees": 0.0, "i0": 0,
          "entries": []}

    def valid(i: int, price: float) -> bool:
        return atr[i] == atr[i] and atr[i] > MIN_ATR * price          # битые свечи с ATR ≈ 0 — шаг не задаём

    def reset(i: int, price: float) -> None:
        st["anchor"], st["step"] = price, (m * atr[i] if valid(i, price) else None)
        st["side"], st["units"], st["entries"] = 0, 0, []

    def close(i: int, price: float) -> None:
        side, step = st["side"], st["step"]
        gross = st["gross"] + sum(side * (price - e) for e in st["entries"])
        fees = st["fees"] + TAKER * price * st["units"]
        out.append({"i0": st["i0"], "i1": i, "side": side, "units": st["units"], "r": (gross - fees) / (x * step),
                    "r2": (gross - 2 * fees) / (x * step)})
        reset(i, price)

    def fill(i: int, px: float) -> None:
        st["entries"].append(px)
        st["fees"] += TAKER * px
        st["units"] += 1

    for i in range(1, len(c)):
        if st["anchor"] is None or st["step"] is None:
            if valid(i, c[i]):
                st["anchor"], st["step"] = c[i], m * atr[i]
            continue
        ok = (lambda s_: trend[i - 1] == s_) if use_trend else (lambda s_: True)
        path = [c[i - 1], o[i], lo[i], hi[i], c[i]] if c[i] >= o[i] else [c[i - 1], o[i], hi[i], lo[i], c[i]]
        skipped = done = False
        for k in range(1, len(path)):
            a_, b_ = path[k - 1], path[k]
            gap = k == 1
            if b_ == a_:
                continue
            mv = 1 if b_ > a_ else -1
            if st["side"] == 0:
                if skipped:
                    continue
                lvl = st["anchor"] + mv * st["step"]
                if mv * (b_ - lvl) < 0:
                    continue
                if not (liq[i - 1] >= ADV_MIN) or not ok(mv):
                    skipped = True
                    continue
                st.update(side=mv, last=b_ if gap else lvl, ext=b_, gross=0.0, fees=0.0, i0=i, entries=[], units=0)
                fill(i, b_ if gap else lvl)
            side, step = st["side"], st["step"]
            if mv == side:
                st["ext"] = max(st["ext"], b_) if side > 0 else min(st["ext"], b_)
                while st["units"] < n and ok(side):
                    nxt = st["last"] + side * step
                    if side * (b_ - nxt) < 0:
                        break
                    fill(i, b_ if gap else nxt)
                    st["last"] = b_ if gap else nxt         # шаги дальше — от фактической цены гэпа
            else:
                stop = st["ext"] - side * x * step
                if side * (b_ - stop) <= 0:
                    close(i, b_ if gap else stop)
                    done = True
                    break
        if skipped and st["side"] == 0:
            st["anchor"] = c[i]                     # вход не взят — отсчёт заново от закрытия
        if not done and st["side"] != 0:
            st["gross"] -= st["side"] * st["units"] * c[i] * fund[i]    # лонг платит положительный funding
    return out


FINE = ((1.0, 1), (1.0, 2), (2.0, 1))          # прошедшие на 4h — перепроверка на 1h-свечах (меньше допущений внутри свечи)


def _fine(d4: pd.DataFrame, d1: pd.DataFrame, atr4: np.ndarray, trend4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ATR и тренд 4h последней закрытой 4h-свечи — на часовых свечах (без заглядывания вперёд)."""
    a = pd.Series(atr4, index=d4.index).shift(1).reindex(d1.index.floor("4h")).to_numpy()
    t = pd.Series(trend4, index=d4.index).shift(1).reindex(d1.index.floor("4h")).to_numpy()
    return a, t


def collect(root: Path, syms: list[str]) -> None:
    rows = []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, "4h")
            if d is None or len(d) < 500:
                continue
            atr = _atr(d).to_numpy()
            trend = htf_trend(d, "4h")
            adv = adv30(root, s)
            frames = [("4h", d, atr, trend, variants())]
            d1 = tf_frame(root, s, "1h")
            if d1 is not None and len(d1) > 2000:
                a1, t1 = _fine(d, d1, atr, trend)
                frames.append(("1h", d1, a1, t1, [v for v in variants() if (v[0], v[1]) in FINE and v[2] == UNITS]))
            for tf, fr, a_, t_, vs in frames:
                arr = [fr[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close")]
                fund = fr["funding"].fillna(0.0).to_numpy(dtype="float64")
                liq = adv.reindex(fr.index.floor("D")).to_numpy()
                for m, x, n, f in vs:
                    for r in simulate(*arr, a_, fund, t_, liq, m, x, n, f):
                        rows.append({"symbol": s, "t": fr.index[r["i0"]], "bars": r["i1"] - r["i0"], "side": r["side"],
                                     "units": r["units"], "R": r["r"], "R2": r["r2"], "m": m, "x": x, "n": n,
                                     "trend": f, "tf": tf})
        except Exception as e:
            print(f"  revgrid {s}: пропуск ({e})", flush=True)
    print(f"  revgrid: кампаний {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("revgrid"), index=False)


def _months(a: str, b: str) -> float:
    return (pd.Timestamp(b) - pd.Timestamp(a)).days / 30.44


def report() -> None:
    parts = all_parts("revgrid")
    print("\n===== REVGRID: обратная сетка 4h (вверх — докупаем, вниз — продаём), голая и по дневному тренду =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    print(f"кампаний {len(df)}, монет {df.symbol.nunique()}")
    per_month: dict[tuple, dict[str, float]] = {}
    if "tf" not in df:
        df["tf"] = "4h"
    for key, g in df.groupby(["tf", "m", "x", "n", "trend"], sort=True):
        tf, m, x, n, f = key
        cells, ok, ok2 = [], [], []
        per_month[key] = {}
        for p, (a, b) in PER.items():
            z = g[(g.t >= a) & (g.t < b)]
            per_month[key][p] = z.R.sum() / _months(a, b)
            ok.append(len(z) >= 10 and z.R.mean() > 0)
            ok2.append(len(z) >= 10 and z.R2.mean() > 0)
            cells.append(f"{p}: {_cell(z)}, {per_month[key][p]:+.1f}R/мес")
        name = f"{tf}: шаг {m:g} ATR(4h), выход {x} шаг., частей {n}, {'по тренду' if f else 'голая'}"
        verdict = "РАБОТАЕТ" if all(ok) and all(ok2) else f"нет ({sum(ok)} из {len(ok)} периодов в плюсе, x2: {sum(ok2)})"
        print(f"\n  {name}: доля прибыльных {np.mean(g.R > 0):.0%}, средняя длина {g.bars.mean() * (4 if tf == '4h' else 1) / 24:.1f} дн., "
              f"частей в среднем {g.units.mean():.2f} → {verdict}")
        print("    " + "  ".join(cells))
    print("\n  фильтр тренда лучше голой (R в месяц во всех периодах)?")
    for (tf, m, x, n, f), pm in per_month.items():
        if f and (tf, m, x, n, False) in per_month:
            base = per_month[(tf, m, x, n, False)]
            better = [pm[p] > base[p] for p in PER]
            print(f"    {tf} шаг {m:g}, выход {x}, частей {n}: "
                  f"{'да' if all(better) else f'нет ({sum(better)} из {len(better)})'}")


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
