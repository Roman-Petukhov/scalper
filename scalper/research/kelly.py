"""Размер риска по Келли для бота 4h (MacLean, Thorp, Ziemba 2010: максимум роста — у доли Келли, на практике берут
½ и меньше из-за ошибки в оценке среднего).

Сделки — те же, что в research/margsim.py (правило бота 4h: линия, ретест, 3R; бета и ход BTC для хеджа), счёт —
margsim.simulate в режиме «по риску»: размер от капитала на момент входа, хедж BTC, плечо 10×, лимит 12 позиций,
дневной стоп 4%, свободная маржа — как в боте. Одновременные сделки по связанным монетам учитываются сами собой.
Части отчёта:
  1. Келли по сделке без учёта одновременности: f* = argmax среднего log(1 + f·R) по периодам — насколько он
     устойчив от периода к периоду.
  2. Счёт при риске 0.5…15% на сделку: рост капитала в год (по логарифму), просадка, худший месяц, урезанные сделки.
  3. «Вперёд»: риск с наибольшим ростом на 2020 … IS (до VAL) и как он, его ½ и ¼ и нынешний 1% прожили VAL и HO.
  4. Блочный бутстреп по месяцам (месяц сделок целиком, с их одновременностью): медиана роста и вероятность
     просадки больше 30% и 50% при каждом риске.
Решение (задано до запуска): рекомендуемый риск — не больше ½ «вперёд»-оптимума и такой, что в бутстрепе вероятность
просадки > 50% не выше 5%.

    python -m research.margsim collect --symbols <1h-qualified>     (данные — части margsim)
    python -m research.kelly report
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from .margsim import E0, Cfg, max_dd, monthly, simulate
from .oos import PER_ALL
from .shard import all_parts

RISKS = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0)
LEV, BOOT, SEED = 10, 300, 11
F_GRID = np.round(np.arange(0.0, 0.501, 0.005), 3)


def kelly_f(r: np.ndarray) -> float:
    """Доля капитала под риск 1R, максимизирующая средний log(1 + f·R); 0 — если край не положителен."""
    r = r[np.isfinite(r)]
    if len(r) < 20:
        return np.nan
    ok = F_GRID[1 + F_GRID * r.min() > 0] if r.min() < 0 else F_GRID
    g = np.array([np.log1p(f * r).mean() for f in ok])
    return float(ok[int(np.argmax(g))])


def cfg(risk: float) -> Cfg:
    return Cfg(f"риск {risk:g}%", False, risk, LEV, True)


def growth(eq: pd.Series, e0: float, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Рост капитала в год (лог, %) на отрезке [start, end) по кривой закрытых сделок."""
    if eq.empty:
        return np.nan
    before = eq[eq.index < start]
    a = before.iloc[-1] if len(before) else e0
    inside = eq[(eq.index >= start) & (eq.index < end)]
    b = inside.iloc[-1] if len(inside) else a
    years = (end - start).days / 365.25
    return float((np.exp(np.log(b / a) / years) - 1) * 100) if a > 0 and b > 0 and years > 0 else np.nan


def bootstrap(df: pd.DataFrame, risk: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Блоки — календарные месяцы входа; месяц переносится целиком со сдвигом времени. Рост в год и просадка, %."""
    m = df.t_in.dt.to_period("M")
    months = m.unique()
    groups = {k: df[m == k] for k in months}
    base = df.t_in.min().floor("D")
    g_out, dd_out = [], []
    for _ in range(BOOT):
        pick = rng.choice(len(months), size=len(months), replace=True)
        chunks = []
        for slot, i in enumerate(pick):
            g = groups[months[i]]
            shift = (base + pd.DateOffset(months=slot)) - months[i].start_time.tz_localize(df.t_in.dt.tz)
            chunks.append(g.assign(t_in=g.t_in + shift, t_out=g.t_out + shift))
        res = simulate(pd.concat(chunks, ignore_index=True), cfg(risk))
        eq = res.eq
        years = len(months) / 12
        g_out.append((np.exp(np.log(eq.iloc[-1] / E0) / years) - 1) * 100 if len(eq) and eq.iloc[-1] > 0 else -100.0)
        dd_out.append(max_dd(eq))
    return np.asarray(g_out), np.asarray(dd_out)


def report() -> None:
    parts = all_parts("margsim")
    print("\n===== KELLY: размер риска на сделку для бота 4h (линия, ретест, 3R; хедж BTC, плечо 10×, лимит 12 позиций, "
          "дневной стоп 4%) =====")
    if not parts:
        print("  kelly: нет данных (нужны части margsim)")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    for c in ("t", "t_in", "t_out"):
        df[c] = pd.to_datetime(df[c], utc=True)
    df = df.dropna(subset=["R3"]).sort_values("t_in", kind="stable").reset_index(drop=True)
    tz = df.t_in.dt.tz
    print(f"  сделок {len(df)} с {df.t.min():%Y-%m} по {df.t.max():%Y-%m}, R на сделку {df.R3.mean():+.3f}")

    print("\n  1. Келли по сделке (без одновременности), доля капитала под риск 1R:")
    cells = []
    for p, (a, b) in PER_ALL.items():
        z = df[(df.t >= pd.Timestamp(a, tz=tz)) & (df.t < pd.Timestamp(b, tz=tz))]
        cells.append(f"{p}: {kelly_f(z.R3.to_numpy()):.1%} ({len(z)}, {z.R3.mean():+.2f}R)")
    print("     " + "  ".join(cells) + f"  всё время: {kelly_f(df.R3.to_numpy()):.1%}")

    print("\n  2. Счёт при разном риске на сделку (рост в год по периодам, %):")
    rows, curves = [], {}
    for r in RISKS:
        res = simulate(df, cfg(r))
        curves[r] = res.eq
        mo = monthly(res.eq)
        row = {"риск %": r, "итог ×": f"{res.eq.iloc[-1] / E0:.3g}" if len(res.eq) else "-",
               "просадка %": f"{max_dd(res.eq):.0f}", "худший мес. %": f"{mo.min():.0f}" if len(mo) else "-"}
        for p, (a, b) in PER_ALL.items():
            row[p] = f"{growth(res.eq, E0, pd.Timestamp(a, tz=tz), pd.Timestamp(b, tz=tz)):+.0f}"
        row |= {"урезано": res.cut, "нет маржи": res.rej_margin, "дн. стоп": res.rej_day}
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n  3. Вперёд: риск с наибольшим ростом на 2020 … IS, затем VAL и HO (рост в год, %):")
    start = pd.Timestamp(PER_ALL["2020"][0], tz=tz)
    val_a, val_b = (pd.Timestamp(x, tz=tz) for x in PER_ALL["val"])
    ho_a, ho_b = (pd.Timestamp(x, tz=tz) for x in PER_ALL["ho"])
    train = {r: growth(curves[r], E0, start, val_a) for r in RISKS}
    best = max(train, key=lambda r: train[r] if np.isfinite(train[r]) else -np.inf)
    picks = {"оптимум": best, "½": best / 2, "¼": best / 4, "сейчас": 1.0}
    for nm, r in picks.items():
        eq = curves[r] if r in curves else simulate(df, cfg(r)).eq
        print(f"     {nm} — риск {r:g}%: обучение {growth(eq, E0, start, val_a):+.0f}%  VAL {growth(eq, E0, val_a, val_b):+.0f}%  "
              f"HO {growth(eq, E0, ho_a, ho_b):+.0f}%  просадка за всё время {max_dd(eq):.0f}%")

    print(f"\n  4. Бутстреп по месяцам ({BOOT} прогонов): медиана роста в год, p10, вероятность просадки > 30% / > 50%:")
    rng = np.random.default_rng(SEED)
    ok_risks = []
    for r in RISKS:
        g, dd = bootstrap(df, r, rng)
        p50 = float((dd > 50).mean())
        print(f"     риск {r:>4g}%: рост {np.median(g):+.0f}% (p10 {np.percentile(g, 10):+.0f}%), просадка медиана "
              f"{np.median(dd):.0f}%, > 30% — {(dd > 30).mean():.0%}, > 50% — {p50:.0%}")
        if p50 <= 0.05 and r <= best / 2:
            ok_risks.append(r)
    print("\n  ИТОГ: " + (f"рекомендуемый риск на сделку — до {max(ok_risks):g}% (½ «вперёд»-оптимума {best:g}% и "
                         f"просадка > 50% в бутстрепе не чаще 5%)" if ok_risks else
                         f"ни один риск не проходит оба условия (оптимум «вперёд» {best:g}%)"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 400)
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
