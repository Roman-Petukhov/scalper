"""Сетка вокруг EMA200 — стратегия владельца (график AVAX 15m, 2026-10-26): горизонтальные линии от EMA200; выше EMA200
только покупки, ниже — только продажи, «мягкий мартингейл»; касание EMA200 закрывает всё и перестраивает сетку так, чтобы
ближние («жёлтые») линии обвивали EMA200. Проблема владельца — пила у EMA200 и внутри сетки; вопрос — шаг сетки по
волатильности ТФ и фильтр пилы. По просьбе «протестируй разные варианты» — сетка вариантов, заданная до запуска.

Литература. Taranto, Khan (2020), «Gambler's ruin problem and bi-directional grid constrained trading and investment
strategies», Investment Management and Financial Innovations 17(3): сетка без преимущества в цене — задача о разорении
игрока; на коротком отрезке может обгонять, на длинном почти наверняка разоряет (теорема Дуба: правило входа / выхода
без дрейфа меняет форму распределения, не матожидание — см. docs/knowledge.md, «Литература по выходам»). Против пилы —
фильтры режима: Efficiency Ratio Кауфмана (Smarter Trading, 1995), наклон медленной средней.

Правила. Сигналы — на свечах ТФ (15m или 1h; EMA200, ATR14, ER, наклон — по последней закрытой свече ТФ), исполнение —
на 5m-свечах (путь внутри 5m: растущая — открытие → минимум → максимум → закрытие, падающая — открытие → максимум →
минимум → закрытие). Центр сетки c0 — значение EMA200 в момент касания; шаг s = k·ATR ТФ на тот же момент; линии стороны
S (S = +1 над EMA200, −1 под ней): c0 + S·(0.5 + i)·s, i = 0..3 (i = 0 — «жёлтая»). Линия годна, пока она по свою
сторону текущей EMA200.
  вход        «лимит» — цена пришла к линии со стороны тренда (откат к EMA200), maker; «стоп» — пробила линию от EMA200
              (вдогонку), taker; «оба» — любое касание. Одна позиция на линию, пока она открыта.
  объём       n-я одновременно открытая часть: ×1 (без мартингейла), ×1.5^n, ×2^n, n + 1 (линейно); доли нормированы —
              все 4 части вместе = 1 капитал монеты (плечо 1 на полную сетку).
  выход       всегда — касание EMA200: закрыть всё (taker), перестроить сетку. Плюс: «тейк на соседней линии» (каждая часть
              на линии + S·s, maker, линия снова свободна), «общий тейк» (всё на средней + S·s, maker), «только EMA200»,
              «трейлинг» (всё, когда цена откатила на 2 шага от лучшей с первой части, taker).
  шаг         k = 0.5 / 1 / 2 ATR ТФ.
  фильтр      новые части только если: нет фильтра / ER(48 свечей ТФ) >= 0.3 / EMA200 наклонена в сторону S за 24 свечи.
Издержки maker 0.02%, taker 0.055%, funding перпетуала. Монеты core70 (ликвидные), с 2020. Счёт монеты — капитал полной
сетки, переоценка по закрытию дня; портфель — среднее по монетам за день.
Принятие (на вариант): средний дневной результат > 0 в каждом периоде 2020 / 2021 / IS / VAL / HO, t по дням >= 3,
плюс при издержках ×2. Вариантов 864 — порог t >= 3 мягкий, смотрим ещё на соседей (устойчивость по шагу и фильтру).

    python -m research.emagrid collect --symbols <монеты>
    python -m research.emagrid report
"""
from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _atr
from .tline import MAKER, TAKER, tf_frame

TFS = {"15m": "15min", "1h": "1h"}
ENTRIES = ("лимит", "стоп", "оба")
EXITS = ("тейк на соседней линии", "общий тейк", "только EMA200", "трейлинг")
SIZES = ("×1", "×1.5", "×2", "n+1")
STEPS = (0.5, 1.0, 2.0)
FILTERS = ("нет", "ER >= 0.3", "наклон EMA200")
LEVELS, ER_N, ER_MIN, SLOPE_N, TRAIL = 4, 48, 0.3, 24, 2.0
DAY0 = pd.Timestamp("2020-01-01", tz="UTC")
VARIANTS = tuple(itertools.product(TFS, ENTRIES, EXITS, SIZES, STEPS, FILTERS))


def vname(v: tuple) -> str:
    tf, en, ex, sz, k, fl = v
    return f"{tf} | вход {en} | {ex} | объём {sz} | шаг {k:g} ATR | фильтр {fl}"


def size_table(kind: str) -> np.ndarray:
    n = np.arange(LEVELS, dtype=np.float64)
    w = {"×1": np.ones(LEVELS), "×1.5": 1.5 ** n, "×2": 2.0 ** n, "n+1": n + 1}[kind]
    return w / w.sum()


@njit(cache=True)
def _close_all(occ, q, px, side, price, fee, eq_real, fees):
    for i in range(occ.shape[0]):
        if occ[i]:
            eq_real += q[i] * side * (price - px[i]) / px[i] - fee * q[i] * price / px[i]
            fees += fee * q[i] * price / px[i]
            occ[i] = False
    return eq_real, fees


@njit(cache=True)
def simulate(o, h, lo, c, f, day, ema, atr, gate_l, gate_s, n_days, entry_mode, exit_mode, sizes, k_step):
    """Дневной результат (переоценка) и издержки одной монеты. entry_mode: 0 лимит, 1 стоп, 2 оба; exit_mode: 0 тейк на
    соседней линии, 1 общий тейк, 2 только EMA200, 3 трейлинг. gate_l / gate_s — фильтр новых частей по сторонам."""
    L = sizes.shape[0]
    pnl = np.zeros(n_days)
    fee_d = np.zeros(n_days)
    occ = np.zeros(L, dtype=np.bool_)
    q = np.zeros(L)
    px = np.zeros(L)
    tp = np.zeros(L)
    eq_real, fees = 0.0, 0.0
    prev_eq, prev_fees = 0.0, 0.0
    c0, s, side = np.nan, np.nan, 0
    best = np.nan
    pts = np.zeros(4)
    n = c.shape[0]
    for j in range(1, n):
        e = ema[j]
        if not (np.isfinite(e) and atr[j] > 0):
            continue
        if side == 0:                                              # первая сетка
            side = 1 if c[j - 1] > e else -1
            c0, s = e, k_step * atr[j]
        nopen = 0
        for i in range(L):
            if occ[i]:
                nopen += 1
                eq_real -= side * q[i] * f[j]                      # funding: лонг платит при f > 0
        if c[j] >= o[j]:
            pts[0], pts[1], pts[2], pts[3] = o[j], lo[j], h[j], c[j]
        else:
            pts[0], pts[1], pts[2], pts[3] = o[j], h[j], lo[j], c[j]
        p = c[j - 1]
        for leg in range(4):
            b = pts[leg]
            if b == p:
                continue
            d = 1 if b > p else -1
            for _ in range(64):                                    # события по ходу цены от p к b
                if side * (p - e) < 0:                             # EMA200 сама прошла через цену — это касание
                    if nopen > 0:
                        eq_real, fees = _close_all(occ, q, px, side, p, TAKER, eq_real, fees)
                        nopen = 0
                        best = np.nan
                    side = -side
                    c0, s = e, k_step * atr[j]
                best_px, kind, idx = np.inf, -1, -1
                # касание EMA200 (движение к ней)
                if d == -side and (e - p) * d > 0 and (b - e) * d >= 0:
                    best_px, kind = abs(e - p), 0
                # части сетки
                for i in range(L):
                    lvl = c0 + side * (0.5 + i) * s
                    if not occ[i] and side * (lvl - e) > 0:
                        ok_dir = (entry_mode == 2) or (entry_mode == 0 and d == -side) or (entry_mode == 1 and d == side)
                        gate = gate_l[j] if side > 0 else gate_s[j]
                        if ok_dir and gate and (lvl - p) * d >= 0 and (b - lvl) * d >= 0 and abs(lvl - p) < best_px:
                            best_px, kind, idx = abs(lvl - p), 1, i
                    if occ[i] and exit_mode == 0 and d == side and (tp[i] - p) * d > 0 and (b - tp[i]) * d >= 0 \
                            and abs(tp[i] - p) < best_px:
                        best_px, kind, idx = abs(tp[i] - p), 2, i
                if nopen > 0 and exit_mode == 1 and d == side:
                    qs, wsum = 0.0, 0.0
                    for i in range(L):
                        if occ[i]:
                            qs += q[i]
                            wsum += q[i] / px[i]
                    avg = qs / wsum
                    t_ = avg + side * s
                    if (t_ - p) * d > 0 and (b - t_) * d >= 0 and abs(t_ - p) < best_px:
                        best_px, kind = abs(t_ - p), 3
                if nopen > 0 and exit_mode == 3 and d == -side:
                    t_ = best - side * TRAIL * s
                    if (t_ - p) * d > 0 and (b - t_) * d >= 0 and abs(t_ - p) < best_px:
                        best_px, kind = abs(t_ - p), 4
                if kind < 0:
                    break
                price = p + d * best_px
                if kind == 0:                                      # касание EMA200: всё закрыть, перестроить
                    eq_real, fees = _close_all(occ, q, px, side, price, TAKER, eq_real, fees)
                    nopen = 0
                    side = -side if d == -side else side
                    c0, s = e, k_step * atr[j]
                    best = np.nan
                elif kind == 1:
                    fee = MAKER if d == -side else TAKER
                    occ[idx] = True
                    q[idx] = sizes[nopen]
                    px[idx] = price
                    tp[idx] = price + side * s
                    eq_real -= fee * q[idx]
                    fees += fee * q[idx]
                    if nopen == 0:
                        best = price
                    nopen += 1
                elif kind == 2:
                    eq_real += q[idx] * side * (price - px[idx]) / px[idx] - MAKER * q[idx] * price / px[idx]
                    fees += MAKER * q[idx] * price / px[idx]
                    occ[idx] = False
                    nopen -= 1
                    if nopen == 0:
                        best = np.nan
                else:
                    fee = MAKER if kind == 3 else TAKER
                    eq_real, fees = _close_all(occ, q, px, side, price, fee, eq_real, fees)
                    nopen = 0
                    best = np.nan
                p = price
            if nopen > 0 and d == side:
                best = b if not np.isfinite(best) else (max(best, b) if side > 0 else min(best, b))
            p = b
        if j == n - 1 or day[j + 1] != day[j]:                    # переоценка на закрытии дня
            unreal = 0.0
            for i in range(L):
                if occ[i]:
                    unreal += q[i] * side * (c[j] - px[i]) / px[i]
            eq = eq_real + unreal
            dd = day[j]
            if 0 <= dd < n_days:
                pnl[dd] += eq - prev_eq
                fee_d[dd] += fees - prev_fees
            prev_eq, prev_fees = eq, fees
    return pnl, fee_d


def tf_view(b5: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Признаки ТФ на 5m-свечах: значение последней закрытой свечи ТФ (известно после её закрытия)."""
    g = b5.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    cl = g["close"]
    ema = cl.ewm(span=200, adjust=False, min_periods=200).mean()
    move = (cl - cl.shift(ER_N)).abs()
    path = cl.diff().abs().rolling(ER_N, min_periods=ER_N).sum()
    x = pd.DataFrame({"ema": ema, "atr": _atr(g), "er": move / path, "slope": ema - ema.shift(SLOPE_N)})
    x.index = x.index + pd.Timedelta(rule)                         # известно на закрытии свечи ТФ
    return x.reindex(b5.index, method="ffill")


def coin_days(b5: pd.DataFrame, n_days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(pnl [вариант, день], издержки [вариант, день], активные дни монеты)."""
    o, h, lo, c = (b5[k].to_numpy(dtype=np.float64) for k in ("open", "high", "low", "close"))
    f = b5["funding"].fillna(0.0).to_numpy(dtype=np.float64) if "funding" in b5 else np.zeros(len(c))
    day = ((b5.index - DAY0) // pd.Timedelta("1D")).to_numpy().astype(np.int64)
    views = {tf: tf_view(b5, rule) for tf, rule in TFS.items()}
    out_p = np.zeros((len(VARIANTS), n_days), dtype=np.float32)
    out_f = np.zeros((len(VARIANTS), n_days), dtype=np.float32)
    sizes = {k: size_table(k) for k in SIZES}
    for vi, (tf, en, ex, sz, k, fl) in enumerate(VARIANTS):
        v = views[tf]
        ema, atr = v["ema"].to_numpy(dtype=np.float64), v["atr"].to_numpy(dtype=np.float64)
        if fl == "нет":
            gl = gs = np.ones(len(c), dtype=np.bool_)
        elif fl == "ER >= 0.3":
            gl = gs = (v["er"].to_numpy() >= ER_MIN)
        else:
            sl = v["slope"].to_numpy()
            gl, gs = sl > 0, sl < 0
        p, fe = simulate(o, h, lo, c, f, day, ema, atr, np.asarray(gl, np.bool_), np.asarray(gs, np.bool_), n_days,
                         ENTRIES.index(en), EXITS.index(ex), sizes[sz], float(k))
        out_p[vi], out_f[vi] = p, fe
    active = np.zeros(n_days, dtype=np.float32)
    dd = np.unique(day[(day >= 0) & (day < n_days)])
    active[dd] = 1.0
    return out_p, out_f, active


def collect(root: Path, syms: list[str]) -> None:
    n_days = int((pd.Timestamp.now(tz="UTC").normalize() - DAY0).days) + 2
    sum_p = np.zeros((len(VARIANTS), n_days), dtype=np.float64)
    sum_f = np.zeros_like(sum_p)
    cnt = np.zeros(n_days)
    for s in mine(syms):
        try:
            b5 = tf_frame(root, s, "5m")
            if b5 is None or len(b5) < 288 * 120:
                continue
            p, fe, act = coin_days(b5, n_days)
            sum_p += p
            sum_f += fe
            cnt += act
            print(f"  emagrid {s}: дней {int(act.sum())}", flush=True)
        except Exception as e:
            print(f"  emagrid {s}: пропуск ({e})", flush=True)
    cols = {f"p{i}": sum_p[i] for i in range(len(VARIANTS))} | {f"f{i}": sum_f[i] for i in range(len(VARIANTS))}
    pd.DataFrame({"day": np.arange(n_days), "n": cnt, **cols}).to_parquet(part_path("emagrid"), index=False)


def stats(r: pd.Series) -> dict:
    """Дневной ряд портфеля: в год, Sharpe, просадка, t."""
    if len(r) < 30 or r.std() == 0:
        return {"ann": np.nan, "sh": np.nan, "dd": np.nan, "t": np.nan}
    eq = r.cumsum()
    return {"ann": r.mean() * 365, "sh": r.mean() / r.std() * np.sqrt(365), "dd": float((eq.cummax() - eq).max()),
            "t": r.mean() / r.std() * np.sqrt(len(r))}


def report() -> None:
    print("\n===== EMAGRID: сетка вокруг EMA200 (стратегия владельца) — вход, выход, мартингейл, шаг, фильтр пилы; "
          "портфель core70 с 2020, капитал монеты = полная сетка, плечо 1 =====")
    parts = all_parts("emagrid")
    if not parts:
        print("  emagrid: нет данных")
        return
    x = None
    for q in parts:
        y = pd.read_parquet(q).set_index("day")
        x = y if x is None else x.add(y, fill_value=0.0)
    x = x[x.n > 0]
    days = DAY0.tz_localize(None) + pd.to_timedelta(x.index, unit="D")
    rows = []
    for i, v in enumerate(VARIANTS):
        r = pd.Series((x[f"p{i}"] / x.n).to_numpy(), index=days)
        r2 = pd.Series(((x[f"p{i}"] - x[f"f{i}"]) / x.n).to_numpy(), index=days)
        st = stats(r)
        per = {k: r[(r.index >= a) & (r.index < b)] for k, (a, b) in PER_ALL.items()}
        pos = [m.mean() > 0 for m in per.values() if len(m) >= 30]
        ok = bool(pos) and all(pos) and st["t"] >= 3 and r2.mean() > 0
        rows.append({"v": i, "name": vname(v), "tf": v[0], "entry": v[1], "exit": v[2], "size": v[3], "step": v[4],
                     "filter": v[5], **st, "x2": r2.mean() * 365, "ok": ok,
                     **{f"y_{k}": m.mean() * 365 if len(m) else np.nan for k, m in per.items()}})
    z = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(f"  дней {len(x)}, монет в среднем {x.n.mean():.0f}; вариантов {len(z)}, в плюсе за всё время "
          f"{int((z.ann > 0).sum())}, принято {int(z.ok.sum())}")
    print("\n  Средний годовой результат вариантов по каждому фактору (медиана по вариантам, % капитала сетки в год):")
    for col in ("tf", "entry", "exit", "size", "step", "filter"):
        g = z.groupby(col)[["ann", "sh", "dd"]].median()
        print(f"    {col}: " + "; ".join(f"{k}: {r.ann:+.1%} (Sharpe {r.sh:+.2f}, просадка {r.dd:.0%})"
                                          for k, r in g.iterrows()))
    print("\n  Лучшие 20 по Sharpe за всё время (в год по периодам 2020 / 2021 / IS / VAL / HO):")
    for _, r in z.sort_values("sh", ascending=False).head(20).iterrows():
        per = " / ".join(f"{r[f'y_{k}']:+.0%}" for k in PER_ALL)
        print(f"    {r['name']}: в год {r.ann:+.1%}, Sharpe {r.sh:+.2f}, t {r.t:.1f}, просадка {r.dd:.0%}, x2 "
              f"{r.x2:+.1%}; {per} → {'принято' if r.ok else 'нет'}")
    print("\n  Худшие 5 (хвост мартингейла):")
    for _, r in z.sort_values("ann").head(5).iterrows():
        print(f"    {r['name']}: в год {r.ann:+.1%}, просадка {r.dd:.0%}")
    acc = z[z.ok]
    print("\n  ИТОГ: " + ("приняты — " + "; ".join(acc.sort_values("sh", ascending=False).name.head(10))
                         if len(acc) else "ни один вариант сетки вокруг EMA200 не прошёл критерий"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [s for s in a.symbols.split(",") if s])
    else:
        report()
