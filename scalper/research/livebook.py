"""Три проверки по журналу владельца 2026-10-26 (в момент отказа Bybit 110007 «не хватает маржи» было открыто 9 позиций —
около 37% счёта под риском при 13 капиталах номинала, а бэктест риска считал 15m с лимитом 3 позиции).

1. Портфель в живых настройках. Сделки бота 15m (по рынку, обе стороны, наклон <= 0.04) и 4h (ретест) вместе, одним
   счётом (research/crisk.py: время входа и выхода, R с комиссиями и funding, стоп в % цены). Как в боте: риск от счёта,
   лестница на 15m (×1.5 после сделки >= +2R, сброс после убытка, 3 ступени), общий лимит позиций, дневной стоп −15%
   (до 00:00 UTC новых входов нет), маржа — номинал / плечо (15m ×15, 4h ×12) не больше счёта, иначе вход
   отклоняется, как 110007. Варианты, заданы до запуска:
     live            — 15m 2% + лестница, 4h 4.5%, лимит 15, без лимита 15m, без потолка риска (как сейчас);
     live, 15m <= 3 / 5 / 8 позиций;
     live, потолок открытого риска 10 / 15 / 20% счёта (сделка, которая его превысит, пропускается);
     умеренный       — 15m 0.5% + лестница, 4h 1%, лимит 15, 15m <= 3 (масштаб принятых проверок).
   Просадка — по закрытым сделкам и по блочному бутстрепу недель (недели берутся целиком — одновременные сделки
   остаются вместе, связь между позициями не рвётся), 200 путей. «Безопасный» — P(просадка > 50%) <= 5% и
   историческая просадка <= 35%; рекомендация — безопасный вариант с наибольшим медианным ростом в год IS / VAL / HO.
2. Совпадение пробоя линии с пробоем горизонтального уровня (метка бота «+ уровень», research/tline.level_hit) на 15m —
   в журнале +0.85R на 29 сделках против −1.06R на 7. Фильтр «только с уровнем»: отрезанные убыточны в каждом
   периоде, где их >= 20, t по дням <= −3, R в месяц не ниже (research/spec.cut_verdicts). Решают 15m шорты и 4h.
3. Размер по волатильности рынка (Moreira, Muir 2017; Grobys и др. 2025 для крипто-моментума): риск × медиана IS /
   реализованная волатильность BTC за 20 дней (известна на начало дня), в пределах ×0.5…×2. Против фиксированного
   риска того же среднего — research/squeezesize.compare (рост выше в IS, VAL, HO, перестановки p <= 0.05,
   P(просадка > 50%) <= 5%). 15m шорты (лимит 3) и 4h (лимит 12), база 1%.

    python -m research.crisk collect --symbols <монеты>      # CRISK_TFS=4h и 15m
    python -m research.livebook
"""
from __future__ import annotations

import math
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .crisk import LIMIT, MAIN, RUNS, period_growth, worst_month
from .oos import PER, PER_ALL
from .shard import all_parts
from .smc import _cell
from .spec import cut_verdicts
from .squeezesize import compare

DAY_NS = 86_400 * 10**9
PATHS = 200
LADDER_TAKE = 2.0


@dataclass(frozen=True)
class Book:
    name: str
    r15: float = 0.02
    ladder_x: float = 1.5
    steps: int = 3
    r4h: float = 0.045
    limit: int = 15
    limit15: int = 0              # 0 — без своего лимита
    heat: float = 0.0             # 0 — без потолка открытого риска
    day_stop: float = 0.15
    lev15: float = 15.0
    lev4h: float = 12.0


BOOKS = (Book("live"), Book("live, 15m <= 3", limit15=3), Book("live, 15m <= 5", limit15=5),
         Book("live, 15m <= 8", limit15=8), Book("live, риск <= 10%", heat=0.10), Book("live, риск <= 15%", heat=0.15),
         Book("live, риск <= 20%", heat=0.20), Book("умеренный", r15=0.005, r4h=0.01, limit15=3))


def load() -> pd.DataFrame | None:
    parts = all_parts("crisk_15m") + all_parts("crisk_4h")
    if not parts:
        return None
    z = pd.concat([pd.read_parquet(q) for q in parts], ignore_index=True)
    for c in ("fill", "exit", "t"):
        z[c] = pd.to_datetime(z[c], utc=True).dt.tz_localize(None)
    z = z.dropna(subset=["R", "stop_pct"])
    z = z[z.stop_pct > 0]
    return z.sort_values(["fill", "tf"]).reset_index(drop=True)


def simulate_book(z: pd.DataFrame, b: Book, R: np.ndarray | None = None) -> dict:
    """Один счёт на 15m и 4h; по закрытым сделкам, сложный процент."""
    R = z.R.to_numpy() if R is None else R
    f = z.fill.to_numpy("datetime64[ns]").astype("int64")
    x = z.exit.to_numpy("datetime64[ns]").astype("int64")
    is15 = (z.tf == "15m").to_numpy()
    stop = z.stop_pct.to_numpy()
    ev = sorted([(int(f[i]), 2, i) for i in range(len(z))] +
                [(int(x[i]), 3 if x[i] <= f[i] else 0, i) for i in range(len(z))], key=lambda e: (e[0], e[1]))
    eq, peak, mdd = 1.0, 1.0, 0.0
    open_: dict[int, tuple[float, float, float]] = {}       # номер → (риск в деньгах, доля риска, маржа в деньгах)
    step, day, day_eq, halted = 0, -1, 1.0, False
    skip = {"лимит": 0, "лимит 15m": 0, "потолок риска": 0, "маржа": 0, "дневной стоп": 0}
    times, curve, heat_max, n15 = [], [], 0.0, 0
    for t, kind, i in ev:
        d = t // DAY_NS
        if d != day:
            day, day_eq, halted = d, eq, False
        if kind in (0, 3):
            if i not in open_:
                continue
            amt, _, _ = open_.pop(i)
            eq += amt * R[i]
            if is15[i]:
                n15 -= 1
                step = min(step + 1, b.steps - 1) if R[i] >= LADDER_TAKE else 0 if R[i] < 0 else step
            times.append(t)
            curve.append(eq)
            peak = max(peak, eq)
            mdd = max(mdd, 1 - eq / peak)
            if not halted and eq <= day_eq * (1 - b.day_stop):
                halted = True
            continue
        if eq <= 0:
            continue
        if halted:
            skip["дневной стоп"] += 1
            continue
        if len(open_) >= b.limit:
            skip["лимит"] += 1
            continue
        if is15[i] and b.limit15 and n15 >= b.limit15:
            skip["лимит 15m"] += 1
            continue
        r = min(b.r15 * b.ladder_x ** step, 0.06) if is15[i] else b.r4h
        used = sum(v[1] for v in open_.values())
        if b.heat and used + r > b.heat + 1e-12:
            skip["потолок риска"] += 1
            continue
        margin = r * eq / stop[i] / (b.lev15 if is15[i] else b.lev4h)
        if sum(v[2] for v in open_.values()) + margin > eq:
            skip["маржа"] += 1
            continue
        open_[i] = (r * eq, r, margin)
        n15 += int(is15[i])
        heat_max = max(heat_max, used + r)
    c = pd.Series(curve, index=pd.to_datetime(np.array(times, dtype="int64")), dtype="float64")
    return {"eq": eq, "mdd": mdd, "curve": c, "skip": skip, "heat_max": heat_max}


def week_blocks(z: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Блочный бутстреп недель: недели (по входу, с понедельника) берутся целиком и с возвращением, время сдвигается
    на место недели в новом ряду — одновременные сделки остаются одновременными."""
    wk = (z.fill - pd.to_timedelta(z.fill.dt.dayofweek, unit="D")).dt.normalize()
    weeks = np.sort(wk.unique())
    groups = {w: np.flatnonzero(wk.to_numpy() == w) for w in weeks}
    start = pd.Timestamp(weeks[0])
    idx, shift = [], []
    for slot, w in enumerate(rng.choice(weeks, size=len(weeks), replace=True)):
        g = groups[w]
        idx.append(g)
        shift.append(np.full(len(g), (start + pd.Timedelta(weeks=slot) - pd.Timestamp(w)).value, dtype="int64"))
    idx_a, sh = np.concatenate(idx), np.concatenate(shift)
    out = z.iloc[idx_a].copy()
    out["fill"] = pd.to_datetime(out.fill.to_numpy("datetime64[ns]").astype("int64") + sh)
    out["exit"] = pd.to_datetime(out.exit.to_numpy("datetime64[ns]").astype("int64") + sh)
    return out.sort_values("fill").reset_index(drop=True)


def _growth(curve: pd.Series) -> dict[str, float]:
    return {k: period_growth(curve, a, b) for k, (a, b) in PER_ALL.items()}


def portfolio_report(z: pd.DataFrame) -> None:
    print("\n  === 1. Портфель в живых настройках: 15m (обе стороны, наклон <= 0.04) + 4h одним счётом, маржа, дневной "
          "стоп; P(просадка > 50%) — блочный бутстреп недель ===")
    print(f"  сделок: 15m {int((z.tf == '15m').sum())}, 4h {int((z.tf == '4h').sum())}")
    rng = np.random.default_rng(23)
    boots = [week_blocks(z, rng) for _ in range(PATHS)]
    res = {}
    for b in BOOKS:
        s = simulate_book(z, b)
        g = _growth(s["curve"])
        dds = np.array([simulate_book(q, b)["mdd"] for q in boots])
        p50 = float(np.mean(dds > 0.5))
        main = [g[k] for k in MAIN if np.isfinite(g[k])]
        med = float(np.median(main)) if main else float("nan")
        safe = p50 <= 0.05 and s["mdd"] <= 0.35
        res[b.name] = (safe, med)
        cells = "  ".join(f"{k} {np.expm1(v):+.0%}" if np.isfinite(v) else f"{k} —" for k, v in g.items())
        skips = ", ".join(f"{k} {v}" for k, v in s["skip"].items() if v)
        print(f"    {b.name:<20} в год: {cells}\n{'':26}просадка {s['mdd']:.0%}, худший месяц "
              f"{worst_month(s['curve']):+.0%}, бутстреп: медиана просадки {np.median(dds):.0%}, P(> 50%) {p50:.0%}; "
              f"макс. открытый риск {s['heat_max']:.0%}; пропущено: {skips or '—'} → "
              f"{'безопасный' if safe else 'опасный'}")
    ok = {k: v for k, v in res.items() if v[0] and np.isfinite(v[1])}
    if ok:
        best = max(ok, key=lambda k: ok[k][1])
        print(f"  → рекомендация: «{best}» (медианный рост в год IS / VAL / HO {np.expm1(ok[best][1]):+.0%})")
    else:
        print("  → ни один вариант не безопасен по заданному критерию")


def level_report(z: pd.DataFrame) -> list[str]:
    print("\n  === 2. Пробой линии вместе с горизонтальным уровнем (метка «+ уровень») ===")
    passed = []
    slices = (("4h, обе стороны", z[z.tf == "4h"], True), ("15m, шорты", z[(z.tf == "15m") & (z.side < 0)], True),
              ("15m, обе стороны", z[z.tf == "15m"], False))
    for label, m, primary in slices:
        m = m.assign(lvl=m.level.astype(float))
        print(f"\n    --- {label}: сделок {len(m)}, с уровнем {m.lvl.mean():.0%} ---")
        for v, nm in ((1.0, "с уровнем"), (0.0, "без уровня")):
            q = m[m.lvl == v]
            cells = "  ".join(f"{p}: {_cell(q[(q.t >= a) & (q.t < b)])}" for p, (a, b) in PER_ALL.items())
            print(f"      {nm}: {cells}  всё время: {_cell(q)}")
        for _, (ok, txt) in cut_verdicts(m, "lvl", (0.5,)).items():
            print(f"      только с уровнем: {'принято' if ok and primary else 'нет'}\n{txt}")
            if ok and primary:
                passed.append(f"{label}: только с уровнем")
    return passed


def vol_report(z: pd.DataFrame) -> list[str]:
    print("\n  === 3. Риск по волатильности рынка: × медиана IS / волатильность BTC за 20 дней (×0.5…×2) ===")
    if "btc_vol" not in z:
        print("    нет btc_vol в частях")
        return []
    passed = []
    rng = np.random.default_rng(29)
    for tf, label, m in (("4h", "4h, обе стороны", z[z.tf == "4h"]),
                         ("15m", "15m, шорты", z[(z.tf == "15m") & (z.side < 0)])):
        m = m.dropna(subset=["btc_vol", "R_be"]).sort_values("fill").reset_index(drop=True)
        a, b = PER["is"]
        med = float(m[(m.t >= a) & (m.t < b)].btc_vol.median())
        w = np.clip(med / m.btc_vol.to_numpy(), 0.5, 2.0)
        print(f"\n    --- {label}: сделок {len(m)}, медиана волатильности BTC в IS {med:.0%} годовых, лимит {LIMIT[tf]} ---")
        if compare(m, w, 0.01, "1% × по волатильности", LIMIT[tf], RUNS[tf], rng):
            passed.append(label)
    return passed


def report() -> None:
    print("\n===== LIVEBOOK: живые настройки одним счётом, «+ уровень» на 15m, риск по волатильности рынка =====")
    z = load()
    if z is None:
        print("  crisk: нет данных")
        return
    portfolio_report(z)
    lv = level_report(z)
    vo = vol_report(z)
    print("\n  ИТОГ: уровень — " + ("; ".join(lv) if lv else "не принят") + "; риск по волатильности — "
          + ("принят: " + "; ".join(vo) if vo else "не принят"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    report()
