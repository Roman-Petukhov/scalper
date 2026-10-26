"""Риск-менеджмент из документа «стратегии чемпионов» (раздел 10) на сделках бота: лестница риска, бюджет открытого
риска, безубыток с перевыпуском риска, дневной стоп и «три страйка» — против фиксированного риска.

Сделки — правило бота: 4h ретест (обе стороны, лимит 12 позиций), 15m по рынку (шорты, наклон <= 0.025, лимит 3);
время входа и выхода — по свечам, R с комиссиями и funding, доливка 4h не моделируется. Для безубытка у каждой сделки
считается второй исход: после хода +1.2R стоп в цену входа (риск позиции освобождается), возврат к входу — 0R минус
комиссии. Счёт по закрытым сделкам (просадка внутри сделок не видна — занижена), сложный процент.

Политики:
  flat r          — r% от счёта на каждую сделку (1 / 2 / 3 / 4.57 — «четверть Келли» документа), лимит позиций бота;
  ladder          — 2 → 3 → 4.5% после каждого тейка (сделка >= +2R), сброс на 2% после убыточной, лимит бота;
  flat avg        — фиксированный риск, равный среднему риску лестницы (честное сравнение с ней);
  doc             — весь раздел 10: лестница, бюджет риска 11.5% (новая сделка — min(риск ступени, max(1%, 11.5% −
                    открытый риск))), безубыток на +1.2R с обнулением риска, дневной стоп −15% до 00:00 UTC, две такие
                    остановки за календарную неделю — риск 1% до восстановления счёта на +10%; без лимита позиций;
  doc + лимит     — то же с лимитом позиций бота.
Принятие, задано до запуска:
  предпосылка — победы кучкуются: корреляция результата сделки с предыдущей (по времени закрытия) > 0 с p < 0.05;
  лестница    — рост (лог) выше «flat avg» в IS, VAL и HO, перестановка результатов между сделками (то же время,
                200 / 100 раз) даёт такой перевес не чаще 5%, и P(просадка > 50%) <= 5%;
  doc         — P(просадка > 50%) <= 5% (бутстреп результатов при том же времени) и рост не ниже «flat 2%» в IS, VAL, HO.

    python -m research.crisk collect --symbols <монеты>      # CRISK_TFS=4h или 15m
    python -m research.crisk report
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
import weakref
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import noline
from .broad import adv30
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _atr
from .tline import MAKER, TAKER, tf_frame, zz_lines

TFS = tuple(os.environ.get("CRISK_TFS", "4h,15m").split(","))
ENTRY = {"4h": "retest", "15m": "market"}
LIMIT = {"4h": 12, "15m": 3}
MAX_SLOPE = 0.025
BE_AT = 1.2
TAKE_R = 2.0
LADDER = (0.02, 0.03, 0.045)
HEAT, HEAT_FLOOR = 0.115, 0.01
DAY_STOP, WEEK_RISK, RECOVER = 0.15, 0.01, 0.10
RUNS = {"4h": 200, "15m": 100}
MAIN = ("is", "val", "ho")


def be_path(o, hi, lo, side: int, px: float, stop: float, fill: int, ex: int, r: float,
            fee_in: float) -> tuple[int, float, int]:
    """(бар, где ход достиг +BE_AT·R, или −1; R с безубытком; бар выхода с безубытком)."""
    risk = side * (px - stop)
    lvl = px + side * BE_AT * risk
    for j in range(fill + 1, ex + 1):
        if (side > 0 and hi[j] >= lvl) or (side < 0 and lo[j] <= lvl):
            if j == ex and r < 0:
                return -1, r, ex                       # в баре выхода стоп раньше: порядок неизвестен — без безубытка
            for k in range(j + 1, ex + 1):
                back = (lo[k] <= px) if side > 0 else (hi[k] >= px)
                if back and (k < ex or r < 2.0):
                    return j, -(fee_in + TAKER) * px / risk, k
            return j, r, ex
    return -1, r, ex


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str) -> list[dict]:
    noline.TF = tf
    x = noline.coin_trades(d, liq, sym, lines_only=True)
    if not len(x):
        return []
    x = x[x.entry == ENTRY[tf]]
    o, hi, lo = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low"))
    atr = _atr(d).to_numpy()
    recs = {(int(r["t"]), int(r["side"])): r for r in zz_lines(d) if r["t"] > 0}
    fee_in = MAKER if ENTRY[tf] == "retest" else TAKER
    rows = []
    for r in x.itertuples(index=False):
        t, side = d.index.get_loc(r.t), int(r.side)
        rec = recs.get((t, side))
        if rec is None or not (atr[t] > 0) or not np.isfinite(r.R3):
            continue
        if tf != "4h" and abs(float(rec["slope"])) / atr[t] > MAX_SLOPE:
            continue
        fill, ex = int(r.fill_i), int(r.exit_i)
        j, rb, exb = be_path(o, hi, lo, side, float(r.px), float(r.stop), fill, ex, float(r.R3), fee_in)
        rows.append({"symbol": sym, "tf": tf, "side": side, "R": float(r.R3), "R_be": float(rb),
                     "fill": d.index[fill], "exit": d.index[ex], "exit_be": d.index[exb],
                     "be_at": d.index[j] if j >= 0 else pd.NaT})
    return rows


def collect(root: Path, syms: list[str]) -> None:
    for tf in TFS:
        rows = []
        for s in mine(syms):
            try:
                if tf != "4h" and not (root / f"{s}-{tf}.parquet").exists():
                    continue
                d = tf_frame(root, s, tf)
                if d is None or len(d) < 500:
                    continue
                rows += coin_rows(d, adv30(root, s).reindex(d.index.floor("D")).to_numpy(), s, tf)
            except Exception as e:
                print(f"  crisk {s} {tf}: пропуск ({e})", flush=True)
        print(f"  crisk {tf}: сделок {len(rows)}", flush=True)
        if rows:
            pd.DataFrame(rows).to_parquet(part_path(f"crisk_{tf}"), index=False)


@dataclass(frozen=True)
class Policy:
    name: str
    flat: float = 0.0            # > 0 — фиксированный риск
    ladder: bool = False
    heat: bool = False
    be: bool = False
    breakers: bool = False
    limit: bool = True
    steps: tuple[float, ...] = LADDER     # ступени лестницы


DAY_NS = 86_400 * 10**9
_EVENTS: dict[tuple[int, bool], tuple[weakref.ref, list[tuple[int, int, int]]]] = {}


def _events(z: pd.DataFrame, be: bool) -> list[tuple[int, int, int]]:
    """События (время в нс, порядок, номер сделки): выходы (0) и освобождение риска (1) раньше входов (2)."""
    key = (id(z), be)
    hit = _EVENTS.get(key)
    if hit is None or hit[0]() is not z:
        f = z.fill.to_numpy("datetime64[ns]").astype("int64")
        x = (z.exit_be if be else z.exit).to_numpy("datetime64[ns]").astype("int64")
        # выход в ту же свечу, что и вход (ретест исполнен и выбит) — после своего входа
        ev = [(int(f[i]), 2, i) for i in range(len(z))] + [(int(x[i]), 3 if x[i] <= f[i] else 0, i)
                                                           for i in range(len(z))]
        if be:
            b = z.be_at.to_numpy("datetime64[ns]")
            ev += [(int(b[i].astype("int64")), 1, i) for i in range(len(z))
                   if not np.isnat(b[i]) and int(b[i].astype("int64")) < x[i]]
        ev.sort(key=lambda e: (e[0], e[1]))
        _EVENTS[key] = (weakref.ref(z), ev)
    return _EVENTS[key][1]


def simulate(z: pd.DataFrame, p: Policy, limit: int, R: np.ndarray | None = None,
             R_be: np.ndarray | None = None) -> dict:
    """Счёт по сделкам z (fill, exit, exit_be, be_at); R / R_be — подмена исходов (перестановка, бутстреп)."""
    R = z.R.to_numpy() if R is None else R
    R_be = z.R_be.to_numpy() if R_be is None else R_be
    out_r = R_be if p.be else R
    eq, peak, mdd = 1.0, 1.0, 0.0
    open_: dict[int, tuple[float, float]] = {}            # номер → (риск в деньгах, доля риска для бюджета)
    step, risks, times, curve = 0, [], [], []
    day, day_eq, halted = -1, 1.0, False
    week, strikes, guard_to = -1, 0, 0.0
    for t, kind, i in _events(z, p.be):
        if p.breakers:
            d = t // DAY_NS
            w = (d + 3) // 7                                   # календарная неделя с понедельника (1970-01-01 — чт)
            if d != day:
                day, day_eq, halted = d, eq, False
            if w != week:
                week, strikes = w, 0
        if kind in (0, 3):
            if i not in open_:
                continue
            amt, _ = open_.pop(i)
            eq += amt * out_r[i]
            times.append(t)
            curve.append(eq)
            peak = max(peak, eq)
            mdd = max(mdd, 1 - eq / peak)
            if p.ladder:
                step = min(step + 1, len(p.steps) - 1) if out_r[i] >= TAKE_R else 0 if out_r[i] < 0 else step
            if p.breakers and not halted and eq <= day_eq * (1 - DAY_STOP):
                halted, strikes = True, strikes + 1
                if strikes >= 2:
                    guard_to = max(guard_to, eq * (1 + RECOVER))
        elif kind == 1:
            if i in open_:
                open_[i] = (open_[i][0], 0.0)
        else:
            if eq <= 0 or (p.limit and len(open_) >= limit) or (p.breakers and halted):
                continue
            r = p.flat if p.flat > 0 else p.steps[step] if p.ladder else p.steps[0]
            if p.breakers and guard_to > 0:
                if eq >= guard_to:
                    guard_to = 0.0
                else:
                    r = WEEK_RISK
            if p.heat:
                used = sum(v[1] for v in open_.values())
                r = min(r, max(HEAT_FLOOR, HEAT - used))
            open_[i] = (r * eq, r)
            risks.append(r)
    c = pd.Series(curve, index=pd.to_datetime(np.array(times, dtype="int64")), dtype="float64")
    return {"eq": eq, "mdd": mdd, "curve": c, "avg_risk": float(np.mean(risks)) if risks else 0.0, "n": len(risks)}


def period_growth(curve: pd.Series, a: str, b: str) -> float:
    """Рост в год за период по кривой закрытых сделок (лог-рост, приведённый к году)."""
    if curve.empty:
        return float("nan")
    s = curve.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    start = s[s.index < pd.Timestamp(a)]
    e0 = float(start.iloc[-1]) if len(start) else 1.0
    inside = s[(s.index >= pd.Timestamp(a)) & (s.index < pd.Timestamp(b))]
    if inside.empty or e0 <= 0:
        return float("nan")
    e1 = float(inside.iloc[-1])
    years = (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25
    return math.log(max(e1, 1e-9) / e0) / years


def worst_month(curve: pd.Series) -> float:
    if curve.empty:
        return float("nan")
    m = curve.sort_index().resample("ME").last().ffill()
    return float((m / m.shift(1).fillna(1.0) - 1).min())


def autocorr(z: pd.DataFrame) -> tuple[float, float, float, float]:
    """(P(+ | прошлая +), P(+ | прошлая −), корреляция R с предыдущей по закрытию, p двусторонний)."""
    r = z.sort_values("exit").R.to_numpy()
    w = r > 0
    prev, cur = w[:-1], w[1:]
    pw = cur[prev].mean() if prev.any() else float("nan")
    pl = cur[~prev].mean() if (~prev).any() else float("nan")
    rho = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if len(r) > 3 else float("nan")
    zstat = rho * math.sqrt(len(r) - 1)
    p = math.erfc(abs(zstat) / math.sqrt(2))
    return float(pw), float(pl), rho, p


def policies(avg: float) -> list[Policy]:
    return [Policy("flat 1%", flat=0.01), Policy("flat 2%", flat=0.02), Policy("flat 3%", flat=0.03),
            Policy("flat 4.57%", flat=0.0457), Policy("ladder 2/3/4.5", ladder=True),
            Policy(f"flat avg {avg:.2%}", flat=avg),
            Policy("doc (без лимита)", ladder=True, heat=True, be=True, breakers=True, limit=False),
            Policy("doc + лимит бота", ladder=True, heat=True, be=True, breakers=True, limit=True)]


def dd50(z: pd.DataFrame, p: Policy, limit: int, runs: int, rng: np.random.Generator) -> float:
    """Доля бутстреп-путей (исходы сделок с возвращением, время то же) с просадкой > 50%."""
    hits = 0
    for _ in range(runs):
        k = rng.integers(0, len(z), len(z))
        hits += simulate(z, p, limit, z.R.to_numpy()[k], z.R_be.to_numpy()[k])["mdd"] > 0.5
    return hits / runs


def report() -> None:
    print("\n===== CRISK: риск-менеджмент «чемпионов» (раздел 10) на сделках бота — лестница, бюджет риска, безубыток, "
          "дневной стоп — против фиксированного риска (счёт по закрытым сделкам) =====")
    passed = []
    for tf in TFS:
        parts = all_parts(f"crisk_{tf}")
        if not parts:
            print(f"  crisk {tf}: нет данных")
            continue
        z = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        for c in ("fill", "exit", "exit_be", "be_at"):
            z[c] = pd.to_datetime(z[c], utc=True).dt.tz_localize(None)
        z = z.dropna(subset=["R", "R_be"])
        if tf != "4h":
            z = z[z.side < 0]
        z = z.sort_values("fill").reset_index(drop=True)
        lim, runs = LIMIT[tf], RUNS[tf]
        rng = np.random.default_rng(7)
        pw, pl, rho, pv = autocorr(z)
        print(f"\n  --- {tf}: сделок {len(z)}, лимит позиций {lim}, R {z.R.mean():+.3f}, с безубытком {z.R_be.mean():+.3f}, "
              f"дошли до +{BE_AT}R {z.be_at.notna().mean():.0%} ---")
        print(f"    предпосылка «победы кучкуются»: P(+ | прошлая +) {pw:.1%}, P(+ | прошлая −) {pl:.1%}, "
              f"корреляция R с прошлой {rho:+.3f} (p {pv:.3f}) → {'да' if rho > 0 and pv < 0.05 else 'нет'}")
        lad = simulate(z, Policy("ladder", ladder=True), lim)
        res = {}
        for p in policies(lad["avg_risk"]):
            s = simulate(z, p, lim)
            g = {k: period_growth(s["curve"], a, b) for k, (a, b) in PER_ALL.items()}
            dd = dd50(z, p, lim, runs, rng)
            cells = "  ".join(f"{k} {np.expm1(v):+.0%}" if np.isfinite(v) else f"{k} —" for k, v in g.items())
            print(f"    {p.name:<18} в год: {cells} | просадка {s['mdd']:.0%}, худший месяц {worst_month(s['curve']):+.0%}, "
                  f"P(просадка > 50%) {dd:.0%}, сделок {s['n']}, средний риск {s['avg_risk']:.2%}")
            res[p.name] = (p, s, g, dd)
        p_lad, s_lad, g_lad, dd_lad = res["ladder 2/3/4.5"]
        avg_name = next(k for k in res if k.startswith("flat avg"))
        p_avg, s_avg, g_avg, _ = res[avg_name]
        better = [g_lad[k] > g_avg[k] for k in MAIN if np.isfinite(g_lad[k]) and np.isfinite(g_avg[k])]
        real = math.log(max(s_lad["eq"], 1e-9)) - math.log(max(s_avg["eq"], 1e-9))
        perm = []
        for _ in range(runs):
            k = rng.permutation(len(z))
            a = simulate(z, p_lad, lim, z.R.to_numpy()[k], z.R_be.to_numpy()[k])["eq"]
            b = simulate(z, p_avg, lim, z.R.to_numpy()[k], z.R_be.to_numpy()[k])["eq"]
            perm.append(math.log(max(a, 1e-9)) - math.log(max(b, 1e-9)))
        pperm = float(np.mean(np.array(perm) >= real))
        ok = bool(better) and all(better) and pperm <= 0.05 and dd_lad <= 0.05
        print(f"    лестница против «{avg_name}»: лог-рост за всё время {real:+.2f} (перестановки: медиана "
              f"{np.median(perm):+.2f}, p {pperm:.2f}), лучше в IS / VAL / HO: {better} → {'принято' if ok else 'нет'}")
        if ok:
            passed.append(f"{tf} лестница")
        for nm in ("doc (без лимита)", "doc + лимит бота"):
            p, s, g, dd = res[nm]
            g2 = res["flat 2%"][2]
            ok = dd <= 0.05 and all(g[k] >= g2[k] for k in MAIN if np.isfinite(g[k]) and np.isfinite(g2[k]))
            print(f"    {nm}: P(просадка > 50%) {dd:.0%}, рост не ниже flat 2% в IS / VAL / HO — "
                  f"{'принято' if ok else 'нет'}")
            if ok:
                passed.append(f"{tf} {nm}")
    print("\n  ИТОГ: " + ("прошли — " + ", ".join(passed) if passed else "риск-модель раздела 10 не принята"))


LADDERS15 = ((0.0025, 0.00375, 0.005625), (0.005, 0.0075, 0.01125))


def ladder_report() -> None:
    """Лестница риска на 15m в рабочем масштабе (задано до запуска): ступени ×1.5 от 0.25% и от 0.5%, лимит 3, против
    фиксированного риска того же среднего; принятие — как у лестницы в report()."""
    print("\n===== CRISK15: лестница риска на 15m в масштабе бота (0.25 / 0.375 / 0.56% и 0.5 / 0.75 / 1.125%) против "
          "фиксированного риска того же среднего =====")
    parts = all_parts("crisk_15m")
    if not parts:
        print("  crisk 15m: нет данных")
        return
    z = pd.concat([pd.read_parquet(q) for q in parts], ignore_index=True)
    for c in ("fill", "exit", "exit_be", "be_at"):
        z[c] = pd.to_datetime(z[c], utc=True).dt.tz_localize(None)
    z = z.dropna(subset=["R", "R_be"])
    z = z[z.side < 0].sort_values("fill").reset_index(drop=True)
    lim, runs, passed = LIMIT["15m"], RUNS["15m"], []
    rng = np.random.default_rng(11)
    print(f"  15m шорты: сигналов {len(z)}, лимит {lim}")
    for steps in LADDERS15:
        lad = Policy(f"ladder {'/'.join(f'{x:.3%}' for x in steps)}", ladder=True, steps=steps)
        s_l = simulate(z, lad, lim)
        avg = Policy(f"flat {s_l['avg_risk']:.3%}", flat=s_l["avg_risk"])
        rows = {}
        for p in (Policy(f"flat {steps[0]:.3%}", flat=steps[0]), avg, lad):
            s = simulate(z, p, lim)
            g = {k: period_growth(s["curve"], a, b) for k, (a, b) in PER_ALL.items()}
            dd = dd50(z, p, lim, runs, rng)
            rows[p.name] = (s, g, dd)
            cells = "  ".join(f"{k} {np.expm1(v):+.0%}" if np.isfinite(v) else f"{k} —" for k, v in g.items())
            print(f"    {p.name:<28} в год: {cells} | просадка {s['mdd']:.0%}, худший месяц "
                  f"{worst_month(s['curve']):+.0%}, P(просадка > 50%) {dd:.0%}, средний риск {s['avg_risk']:.3%}")
        (s_l, g_l, dd_l), (s_a, g_a, _) = rows[lad.name], rows[avg.name]
        better = [g_l[k] > g_a[k] for k in MAIN if np.isfinite(g_l[k]) and np.isfinite(g_a[k])]
        real = math.log(max(s_l["eq"], 1e-9)) - math.log(max(s_a["eq"], 1e-9))
        perm = []
        for _ in range(runs):
            k = rng.permutation(len(z))
            a = simulate(z, lad, lim, z.R.to_numpy()[k], z.R_be.to_numpy()[k])["eq"]
            b = simulate(z, avg, lim, z.R.to_numpy()[k], z.R_be.to_numpy()[k])["eq"]
            perm.append(math.log(max(a, 1e-9)) - math.log(max(b, 1e-9)))
        pperm = float(np.mean(np.array(perm) >= real))
        ok = bool(better) and all(better) and pperm <= 0.05 and dd_l <= 0.05
        print(f"    {lad.name} против {avg.name}: лог-рост {real:+.2f} (перестановки: медиана {np.median(perm):+.2f}, "
              f"p {pperm:.2f}), лучше в IS / VAL / HO: {better}, просадка {s_l['mdd']:.0%} против {s_a['mdd']:.0%} → "
              f"{'принято' if ok else 'нет'}")
        if ok:
            passed.append(lad.name)
    print("\n  ИТОГ: " + ("прошли — " + ", ".join(passed) if passed else "лестница риска на 15m не принята"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report", "ladder"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [s for s in a.symbols.split(",") if s])
    elif a.mode == "ladder":
        ladder_report()
    else:
        report()
