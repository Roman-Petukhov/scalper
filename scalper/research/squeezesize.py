"""Повышенный риск на сжатых 4h-пробоях вместо фильтра (research/sharp.py: ATR на пробое <= нижнего квинтиля IS от ATR
на опоре A — +1.0R, сильнее остальных в каждом периоде, но остальные тоже в плюсе, резать их нельзя).

Сделки — правило бота на 4h (ретест, research/crisk.py: те же входы, выходы, лимит 12 позиций). Сжатая сделка —
squeeze <= нижней квинтильной границы IS (без заглядывания в VAL / HO). Варианты, заданы до запуска:
  ×1.5 и ×2 на сжатых, ×1 на остальных — против фиксированного риска того же среднего (чтобы выигрыш не был просто
  «больше риска»); база 1% и 2% (на 2% — справочно, решает 1%).
Принятие (как у лестницы, research/crisk.ladder_report): рост в год выше, чем у фиксированного риска того же
среднего, в IS, VAL и HO; перестановки исходов при тех же множителях — p <= 0.05; P(просадка > 50%) по бутстрепу
<= 5%.

    CRISK_TFS=4h python -m research.crisk collect --symbols <монеты>
    python -m research.squeezesize
"""
from __future__ import annotations

import math
import sys
import warnings

import numpy as np
import pandas as pd

from .crisk import LIMIT, MAIN, RUNS, Policy, period_growth, simulate, worst_month
from .formation import is_edges
from .oos import PER_ALL
from .shard import all_parts

MULTS = (1.5, 2.0)
BASES = (0.01, 0.02)


def load() -> pd.DataFrame | None:
    parts = all_parts("crisk_4h")
    if not parts:
        return None
    z = pd.concat([pd.read_parquet(q) for q in parts], ignore_index=True)
    for c in ("fill", "exit", "exit_be", "be_at", "t"):
        z[c] = pd.to_datetime(z[c], utc=True).dt.tz_localize(None)
    return z.dropna(subset=["R", "R_be", "squeeze"]).sort_values("fill").reset_index(drop=True)


def weights(z: pd.DataFrame, mult: float, edge: float) -> np.ndarray:
    return np.where(z["squeeze"].to_numpy() <= edge, mult, 1.0)


def dd50w(z: pd.DataFrame, p: Policy, w: np.ndarray, runs: int, rng: np.random.Generator) -> float:
    """Доля бутстреп-путей (исход и множитель сделки берутся парой) с просадкой > 50%."""
    hits = 0
    for _ in range(runs):
        k = rng.integers(0, len(z), len(z))
        hits += simulate(z, p, LIMIT["4h"], z.R.to_numpy()[k], z.R_be.to_numpy()[k], mult=w[k])["mdd"] > 0.5
    return hits / runs


def _row(name: str, s: dict, dd: float) -> str:
    g = {k: period_growth(s["curve"], a, b) for k, (a, b) in PER_ALL.items()}
    cells = "  ".join(f"{k} {np.expm1(v):+.0%}" if np.isfinite(v) else f"{k} —" for k, v in g.items())
    return (f"    {name:<26} в год: {cells} | просадка {s['mdd']:.0%}, худший месяц {worst_month(s['curve']):+.0%}, "
            f"P(просадка > 50%) {dd:.0%}, средний риск {s['avg_risk']:.2%}")


def report() -> None:
    print("\n===== SQUEEZESIZE: повышенный риск на сжатых 4h-пробоях (ATR пробоя / ATR опоры <= квинтиль 1 IS) против "
          "фиксированного риска того же среднего =====")
    z = load()
    if z is None:
        print("  crisk 4h: нет данных")
        return
    e = is_edges(z, "squeeze")
    if not len(e):
        print("  мало сделок IS для границ")
        return
    edge, lim, runs = float(e[0]), LIMIT["4h"], RUNS["4h"]
    sq = z["squeeze"] <= edge
    print(f"  сделок {len(z)}, сжатых {int(sq.sum())} ({sq.mean():.0%}), граница IS {edge:.3f}; R сжатых "
          f"{z.R[sq].mean():+.3f}, остальных {z.R[~sq].mean():+.3f}; лимит {lim}")
    rng = np.random.default_rng(17)
    passed = []
    for base in BASES:
        for mult in MULTS:
            w = weights(z, mult, edge)
            pw = Policy(f"{base:.0%} ×{mult:g} на сжатых", flat=base)
            s_w = simulate(z, pw, lim, mult=w)
            pa = Policy(f"flat {s_w['avg_risk']:.2%}", flat=s_w["avg_risk"])
            s_a = simulate(z, pa, lim)
            dd_w = dd50w(z, pw, w, runs, rng)
            dd_a = dd50w(z, pa, np.ones(len(z)), runs, rng)
            print(f"\n  база {base:.0%}, множитель ×{mult:g}:")
            print(_row(f"flat {base:.0%}", simulate(z, Policy("", flat=base), lim), float("nan")))
            print(_row(pa.name, s_a, dd_a))
            print(_row(pw.name, s_w, dd_w))
            gw = {k: period_growth(s_w["curve"], *PER_ALL[k]) for k in MAIN}
            ga = {k: period_growth(s_a["curve"], *PER_ALL[k]) for k in MAIN}
            better = [gw[k] > ga[k] for k in MAIN if np.isfinite(gw[k]) and np.isfinite(ga[k])]
            real = math.log(max(s_w["eq"], 1e-9)) - math.log(max(s_a["eq"], 1e-9))
            perm = []
            for _ in range(runs):
                k = rng.permutation(len(z))
                a = simulate(z, pw, lim, z.R.to_numpy()[k], z.R_be.to_numpy()[k], mult=w)["eq"]
                b = simulate(z, pa, lim, z.R.to_numpy()[k], z.R_be.to_numpy()[k])["eq"]
                perm.append(math.log(max(a, 1e-9)) - math.log(max(b, 1e-9)))
            pperm = float(np.mean(np.array(perm) >= real))
            ok = bool(better) and all(better) and pperm <= 0.05 and dd_w <= 0.05
            decide = base == BASES[0]
            print(f"    лог-рост против того же среднего {real:+.2f} (перестановки: медиана {np.median(perm):+.2f}, "
                  f"p {pperm:.2f}), лучше в IS / VAL / HO: {better} → "
                  f"{('принято' if ok else 'нет') if decide else ('прошло бы (справочно)' if ok else 'нет (справочно)')}")
            if ok and decide:
                passed.append(pw.name)
    print("\n  ИТОГ: " + ("прошли — " + ", ".join(passed) if passed else "повышенный риск на сжатых 4h-пробоях не принят"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    report()
