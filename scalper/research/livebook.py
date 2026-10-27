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
from pathlib import Path

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
SQUEEZE_EDGE = 0.5        # как SQUEEZE_MAX бота (trader/domain/strategy.py)


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
    sides15: str = "both"         # "short" — только шорты 15m
    slope15: float = 0.04         # наклон линии 15m не круче
    squeeze4h: float = 1.0        # риск 4h × на сжатых пробоях (research/squeezesize.py)
    long_x: float = 1.0           # риск лонгов 15m × (как long_risk_x бота, после лестницы); при sides15="both"
    breaker: int = 0              # прострелы: столько исполнений за час — до конца дня (UTC) новых не берём; 0 — нет
    extra: tuple = ()             # другие стратегии: (tf, доля номинала от счёта, плечо, лимит позиций или 0); их R —
                                  # доходность на номинал (stop_pct = 1), в общий лимит бота не входят (research/livebook 10)


# второй заход (после первого: живые и «умеренный» опасны) — где безопасно; задан до запуска, критерий тот же
BOOKS2 = (Book("4h 1%", r15=0.0, r4h=0.01), Book("4h 2%", r15=0.0, r4h=0.02), Book("4h 3%", r15=0.0, r4h=0.03),
          Book("4h 2% + сжатие", r15=0.0, r4h=0.02, squeeze4h=1.5),
          Book("15m ш 0.25% ≤.025 ≤3 + 4h 1%", r15=0.0025, r4h=0.01, limit15=3, sides15="short", slope15=0.025),
          Book("15m ш 0.25% ≤.025 ≤3 + 4h 2% сж", r15=0.0025, r4h=0.02, limit15=3, sides15="short", slope15=0.025,
               squeeze4h=1.5),
          Book("15m ш 0.5% ≤.025 ≤3 + 4h 2% сж", r15=0.005, r4h=0.02, limit15=3, sides15="short", slope15=0.025,
               squeeze4h=1.5),
          Book("15m ш 0.5% ≤.04 ≤3 + 4h 2% сж", r15=0.005, r4h=0.02, limit15=3, sides15="short", slope15=0.04,
               squeeze4h=1.5),
          Book("15m ш 0.25% ≤.025 риск≤10% + 4h 2% сж", r15=0.0025, r4h=0.02, heat=0.10, sides15="short",
               slope15=0.025, squeeze4h=1.5),
          Book("15m ш 1% ≤.025 ≤3 + 4h 2% сж", r15=0.01, r4h=0.02, limit15=3, sides15="short", slope15=0.025,
               squeeze4h=1.5))
# третий заход — граница риска для старта с $1000 (между «15m 0.25% + 4h 2% сж», P 1%, и «4h 3%», P 8%);
# задан до запуска, критерий тот же, максимум — безопасный вариант с наибольшим медианным ростом
_S15 = dict(limit15=3, sides15="short", slope15=0.025, squeeze4h=1.5)
BOOKS3 = (Book("4h 2.5% сж", r15=0.0, r4h=0.025, squeeze4h=1.5), Book("4h 3% сж", r15=0.0, r4h=0.03, squeeze4h=1.5),
          Book("15m ш 0.25% + 4h 2% сж", r15=0.0025, r4h=0.02, **_S15),
          Book("15m ш 0.25% + 4h 2.25% сж", r15=0.0025, r4h=0.0225, **_S15),
          Book("15m ш 0.25% + 4h 2.5% сж", r15=0.0025, r4h=0.025, **_S15),
          Book("15m ш 0.25% + 4h 2.75% сж", r15=0.0025, r4h=0.0275, **_S15),
          Book("15m ш 0.35% + 4h 2% сж", r15=0.0035, r4h=0.02, **_S15),
          Book("15m ш 0.35% + 4h 2.5% сж", r15=0.0035, r4h=0.025, **_S15))
# четвёртый заход — наклон линии 15m и лонги 15m при безопасном масштабе (15m 0.25% + 4h 2.25% сж); задан до запуска.
# Вариант принимается, если безопасен (P(просадка > 50%) <= 5%, просадка <= 35%) и его медианный рост в год IS / VAL / HO
# выше, чем у базы (наклон 0.025, шорты)
_S4 = dict(r15=0.0025, r4h=0.0225, limit15=3, squeeze4h=1.5)
BOOKS4 = (Book("база: шорты ≤.025", sides15="short", slope15=0.025, **_S4),
          Book("шорты ≤.03", sides15="short", slope15=0.03, **_S4),
          Book("шорты ≤.04", sides15="short", slope15=0.04, **_S4),
          Book("обе ≤.025, лонги ×0.25", slope15=0.025, long_x=0.25, **_S4),
          Book("обе ≤.025, лонги ×0.5", slope15=0.025, long_x=0.5, **_S4),
          Book("обе ≤.025, лонги ×1", slope15=0.025, **_S4),
          Book("обе ≤.04, лонги ×0.5", slope15=0.04, long_x=0.5, **_S4))
# пятый заход — связка, которую владелец поставил в панели (15m 0.35%, 4h 2.25%): между проверенными 0.35 / 2% (36%, P 3%) и
# 0.35 / 2.5% (40%, P 11%); 500 путей вместо 200 (первые 200 те же, seed), критерий тот же; плюс 0.3% и дневной стоп 10%
BOOKS5 = (Book("база 0.25 / 2.25", sides15="short", slope15=0.025, **_S4 | {"r4h": 0.0225}),
          Book("владелец 0.35 / 2.25", sides15="short", slope15=0.025, **_S4 | {"r15": 0.0035, "r4h": 0.0225}),
          Book("0.30 / 2.25", sides15="short", slope15=0.025, **_S4 | {"r15": 0.003, "r4h": 0.0225}),
          Book("0.35 / 2.25, дневной стоп 10%", sides15="short", slope15=0.025, day_stop=0.10,
               **_S4 | {"r15": 0.0035, "r4h": 0.0225}),
          Book("0.35 / 2.25, потолок риска 15%", sides15="short", slope15=0.025, heat=0.15,
               **_S4 | {"r15": 0.0035, "r4h": 0.0225}))
# шестой заход — 15m один, без 4h (вклад в просадку в чистом виде): шорты, наклон <= 0.025, <= 3 позиции; 500 путей
_S6 = dict(r4h=0.0, sides15="short", slope15=0.025, limit15=3)
BOOKS6 = (Book("15m один 0.25%", r15=0.0025, **_S6), Book("15m один 0.35%", r15=0.0035, **_S6),
          Book("15m один 0.35%, без лестницы", r15=0.0035, ladder_x=1.0, **_S6),
          Book("15m один 0.5%", r15=0.005, **_S6), Book("15m один 1%", r15=0.01, **_S6))
# седьмой заход — 4h отдельно против 4h + 15m при одинаковой просадке (вопрос владельца: нужны ли обе стратегии вместе);
# 15m шорты, наклон <= 0.025, <= 3 позиции, лестница; 4h со сжатием ×1.5; 500 путей
_S7 = dict(sides15="short", slope15=0.025, limit15=3, squeeze4h=1.5)
BOOKS7 = tuple(Book(f"4h один {r:g}%", r15=0.0, r4h=r / 100, **_S7) for r in (2.0, 2.2, 2.25, 2.4, 2.5)) + (
    Book("15m 0.15% + 4h 2.25%", r15=0.0015, r4h=0.0225, **_S7), Book("15m 0.15% + 4h 2.4%", r15=0.0015, r4h=0.024, **_S7),
    Book("15m 0.25% + 4h 2%", r15=0.0025, r4h=0.02, **_S7), Book("15m 0.25% + 4h 2.25%", r15=0.0025, r4h=0.0225, **_S7))
# восьмой заход — граница для владельца, готового терпеть глубокую просадку ради роста: где рост типичного пути перестаёт расти
# (потеря от разброса), как растут хвосты (P просадки > 50 / 60 / 70%, 10-й процентиль роста). 15m шорты ≤ 0.025, ≤ 3 позиции
_S8 = dict(sides15="short", slope15=0.025, limit15=3, squeeze4h=1.5)
BOOKS8 = (Book("15m 0.15% + 4h 2.4% (эталон)", r15=0.0015, r4h=0.024, **_S8),
          Book("4h один 3%", r15=0.0, r4h=0.03, **_S8), Book("4h один 4%", r15=0.0, r4h=0.04, **_S8),
          Book("15m 0.25% + 4h 3%", r15=0.0025, r4h=0.03, **_S8), Book("15m 0.35% + 4h 3%", r15=0.0035, r4h=0.03, **_S8),
          Book("15m 0.5% + 4h 3%", r15=0.005, r4h=0.03, **_S8), Book("15m 0.35% + 4h 4%", r15=0.0035, r4h=0.04, **_S8),
          Book("15m 0.5% + 4h 4%", r15=0.005, r4h=0.04, **_S8), Book("15m 1% + 4h 4%", r15=0.01, r4h=0.04, **_S8))
# девятый заход — вопрос владельца «за сколько $3000 → $100 000 (×33)»: шанс дойти за 2 года и срок по путям бутстрепа,
# по бэктесту и при вдвое меньшем ожидании на сделку (каждый R минус половина среднего своего ТФ — разброс тот же)
BOOKS9 = (Book("B2: 15m 0.35% + 4h 3% (текущие)", r15=0.0035, r4h=0.03, **_S8),
          Book("C1: 15m 0.35% + 4h 4%", r15=0.0035, r4h=0.04, **_S8),
          Book("15m 0.5% + 4h 4%", r15=0.005, r4h=0.04, **_S8), Book("15m 1% + 4h 4%", r15=0.01, r4h=0.04, **_S8))
# десятый заход — бот B2 вместе с oi_liq (откуп ликвидационных каскадов, лесенка 5 лимитов, 4 ч) и ловлей прострелов 6σ
# (лимитка на 6σ ниже, тейк на половине отката, до 60 мин; только монеты Bybit, исполнение при проколе на 0.2%) одним
# счётом; размеры как research/portfolio_sim2 («умеренный» / «агрессивный»). Сделки — research/combo_data
_B2 = dict(r15=0.0035, r4h=0.03, **_S8)
OI, SPIKE = ("oi", 0.3, 5.0, 3), ("spike", 0.05, 10.0, 0)
BOOKS10 = (Book("B2", **_B2), Book("B2 + oi 0.3 (≤ 3)", extra=(OI,), **_B2),
           Book("B2 + oi 0.6 (≤ 5)", extra=(("oi", 0.6, 5.0, 5),), **_B2),
           Book("B2 + прострелы 0.05", extra=(SPIKE,), **_B2), Book("B2 + прострелы 0.10", extra=(("spike", 0.10, 10.0, 0),), **_B2),
           Book("B2 + oi 0.3 + прострелы 0.05", extra=(OI, SPIKE), **_B2),
           Book("без бота: oi 0.3 + прострелы 0.05", extra=(OI, SPIKE), **(_B2 | {"r15": 0.0, "r4h": 0.0})))
COMBO_DIR = Path(__file__).resolve().parent / "combo_data"
CRASH_DAY = "2025-10-10"
# одиннадцатый заход — прострелы с ограничениями риска (просьба владельца «без стопа можно слить»): потолок позиций,
# предохранитель на обвал (3 исполнения за час — до конца дня новых нет), аварийный стоп 5D (отдельный ряд сделок)
SP5 = ("spike", 0.05, 10.0, 5)
BOOKS11 = (Book("B2", **_B2), Book("B2 + прострелы 5%, без ограничений", extra=(SPIKE,), **_B2),
           Book("B2 + прострелы 5%, ≤ 5 позиций", extra=(SP5,), **_B2),
           Book("B2 + прострелы 5%, ≤ 5, предохранитель 3/ч", extra=(SP5,), breaker=3, **_B2),
           Book("B2 + прострелы 5% стоп 5D, ≤ 5, предохр.", extra=(("spike5d", 0.05, 10.0, 5),), breaker=3, **_B2),
           Book("B2 + прострелы 10%, ≤ 5, предохранитель 3/ч", extra=(("spike", 0.10, 10.0, 5),), breaker=3, **_B2),
           Book("B2 + прострелы 4σ 5%, ≤ 5, предохранитель 3/ч", extra=(("spike4", 0.05, 10.0, 5),), breaker=3, **_B2),
           Book("B2 + прострелы 4σ 5%, без ограничений", extra=(("spike4", 0.05, 10.0, 0),), **_B2))
GOAL, GOAL_YEARS = 100_000 / 3_000, 2.0
EDGES = (1.0, 0.5)
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
    keep = keep_mask(z, b)
    if not keep.all():
        z = z[keep]
        R = None if R is None else R[keep]
    R = z.R.to_numpy() if R is None else R
    sq = (z["squeeze"].to_numpy() <= SQUEEZE_EDGE) if "squeeze" in z else np.zeros(len(z), bool)
    f = z.fill.to_numpy("datetime64[ns]").astype("int64")
    x = z.exit.to_numpy("datetime64[ns]").astype("int64")
    is15 = (z.tf == "15m").to_numpy()
    long = (z.side > 0).to_numpy()
    tfs = z.tf.to_numpy()
    ext = {e[0]: e[1:] for e in b.extra if e[1] > 0}
    n_ext = dict.fromkeys(ext, 0)
    spike_times: list[int] = []
    spike_off = -1                                           # день, до конца которого прострелы выключены
    taken: list[tuple[int, float]] = []                      # (номер сделки доп. стратегии, доля счёта на входе)
    stop = z.stop_pct.to_numpy()
    ev = sorted([(int(f[i]), 2, i) for i in range(len(z))] +
                [(int(x[i]), 3 if x[i] <= f[i] else 0, i) for i in range(len(z))], key=lambda e: (e[0], e[1]))
    eq, peak, mdd = 1.0, 1.0, 0.0
    open_: dict[int, tuple[float, float, float]] = {}       # номер → (риск в деньгах, доля риска, маржа в деньгах)
    step, day, day_eq, halted = 0, -1, 1.0, False
    skip = {"лимит": 0, "лимит 15m": 0, "лимит доп.": 0, "предохранитель": 0, "потолок риска": 0, "маржа": 0, "дневной стоп": 0}
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
            if tfs[i] in ext:
                n_ext[tfs[i]] -= 1
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
        if tfs[i] in ext:
            frac, lev, lim = ext[tfs[i]]
            if tfs[i].startswith("spike") and b.breaker:
                if spike_off == d:
                    skip["предохранитель"] += 1
                    continue
                spike_times = [x for x in spike_times if x > t - 3600 * 10**9]
                if len(spike_times) >= b.breaker:
                    spike_off = d
                    skip["предохранитель"] += 1
                    continue
            if lim and n_ext[tfs[i]] >= lim:
                skip["лимит доп."] += 1
                continue
            margin = frac * eq / lev
            if sum(v[2] for v in open_.values()) + margin > eq:
                skip["маржа"] += 1
                continue
            open_[i] = (frac * eq, 0.0, margin)
            n_ext[tfs[i]] += 1
            taken.append((i, frac))
            if tfs[i].startswith("spike"):
                spike_times.append(t)
            continue
        if len(open_) - sum(n_ext.values()) >= b.limit:
            skip["лимит"] += 1
            continue
        if is15[i] and b.limit15 and n15 >= b.limit15:
            skip["лимит 15m"] += 1
            continue
        r = min(b.r15 * b.ladder_x ** step * (b.long_x if long[i] else 1.0), 0.06) if is15[i] else min(b.r4h * (b.squeeze4h if sq[i] else 1.0), 0.06)
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
    tk = pd.DataFrame(taken, columns=["i", "frac"])
    if len(tk):
        tk = tk.assign(tf=tfs[tk.i], R=R[tk.i], fill=z.fill.to_numpy()[tk.i])
    return {"eq": eq, "mdd": mdd, "curve": c, "skip": skip, "heat_max": heat_max, "taken": tk}


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


def portfolio_report(z: pd.DataFrame, books: tuple[Book, ...] = BOOKS,
                     title: str = "1. Портфель в живых настройках: 15m (обе стороны, наклон <= 0.04) + 4h",
                     paths: int = PATHS) -> None:
    print(f"\n  === {title} одним счётом, маржа, дневной стоп; P(просадка > 50%) — блочный бутстреп недель ===")
    print(f"  сделок: 15m {int((z.tf == '15m').sum())}, 4h {int((z.tf == '4h').sum())}")
    rng = np.random.default_rng(23)
    boots = [week_blocks(z, rng) for _ in range(paths)]
    years = (z.fill.max() - z.fill.min()).days / 365.25
    res = {}
    for b in books:
        s = simulate_book(z, b)
        g = _growth(s["curve"])
        sims = [simulate_book(q, b) for q in boots]
        dds = np.array([x["mdd"] for x in sims])
        p50 = float(np.mean(dds > 0.5))
        cagr = np.array([x["eq"] ** (1 / years) - 1 for x in sims])
        main = [g[k] for k in MAIN if np.isfinite(g[k])]
        med = float(np.median(main)) if main else float("nan")
        safe = p50 <= 0.05 and s["mdd"] <= 0.35
        res[b.name] = (safe, med)
        cells = "  ".join(f"{k} {np.expm1(v):+.0%}" if np.isfinite(v) else f"{k} —" for k, v in g.items())
        skips = ", ".join(f"{k} {v}" for k, v in s["skip"].items() if v)
        print(f"    {b.name:<38} в год: {cells}\n{'':26}просадка {s['mdd']:.0%}, худший месяц "
              f"{worst_month(s['curve']):+.0%}, бутстреп: медиана просадки {np.median(dds):.0%}, P(> 50%) {p50:.0%}; "
              f"макс. открытый риск {s['heat_max']:.0%}; пропущено: {skips or '—'}\n{'':26}пути бутстрепа, рост в год: 10-й процентиль "
              f"{np.percentile(cagr, 10):+.0%}, медиана {np.median(cagr):+.0%}; P(просадка > 60%) {np.mean(dds > 0.6):.0%}, "
              f"P(> 70%) {np.mean(dds > 0.7):.0%} → "
              f"{'безопасный' if safe else 'опасный'}")
    ok = {k: v for k, v in res.items() if v[0] and np.isfinite(v[1])}
    if ok:
        best = max(ok, key=lambda k: ok[k][1])
        print(f"  → рекомендация: «{best}» (медианный рост в год IS / VAL / HO {np.expm1(ok[best][1]):+.0%})")
    else:
        print("  → ни один вариант не безопасен по заданному критерию")


def keep_mask(z: pd.DataFrame, b: Book) -> np.ndarray:
    extra = [e[0] for e in b.extra if e[1] > 0]
    return (((z.tf == "4h") & (b.r4h > 0)) | ((z.tf == "15m") & (b.r15 > 0) & (z.slope_atr <= b.slope15)
                                               & ((b.sides15 == "both") | (z.side < 0))) | z.tf.isin(extra)).to_numpy()


def haircut(z: pd.DataFrame, b: Book, edge: float) -> np.ndarray:
    """R сделок с ожиданием × edge: из каждого R вычитается (1 − edge) × среднее R своего ТФ по сделкам книги."""
    R = z.R.to_numpy().copy()
    k = keep_mask(z, b)
    for tf in ("15m", "4h", *(e[0] for e in b.extra)):
        m = k & (z.tf == tf).to_numpy()
        if m.any():
            R[(z.tf == tf).to_numpy()] -= (1 - edge) * R[m].mean()
    return R


def goal_stats(curve: pd.Series, start: pd.Timestamp, goal: float, years: float) -> tuple[float, bool, float, float]:
    """(счёт × через years лет, дошёл ли до goal за years лет, лет до goal или nan, просадка за years лет)."""
    if curve.empty:
        return 1.0, False, float("nan"), 0.0
    end = start + pd.Timedelta(days=365.25 * years)
    head = curve[curve.index <= end]
    x = float(head.iloc[-1]) if len(head) else 1.0
    hit = curve[curve >= goal]
    t_goal = (hit.index[0] - start).days / 365.25 if len(hit) else float("nan")
    eq = pd.concat([pd.Series([1.0], index=[start]), head])
    dd = float((1 - eq / eq.cummax()).max())
    return x, bool(len(hit) and hit.index[0] <= end), t_goal, dd


def rolling_windows(curve: pd.Series, years: float) -> np.ndarray:
    """Рост счёта за years лет из каждого месяца истории (старт — конец месяца)."""
    m = curve.resample("ME").last().ffill()
    k = int(round(12 * years))
    return (m.shift(-k) / m).dropna().to_numpy() if len(m) > k else np.array([])


def goal_report(z: pd.DataFrame, books: tuple[Book, ...] = BOOKS9, paths: int = 500) -> None:
    print(f"\n  === ×{GOAL:.1f} ($3000 → $100 000): за {GOAL_YEARS:g} года и срок; {paths} путей бутстрепа недель ===")
    rng = np.random.default_rng(29)
    boots = [week_blocks(z, rng) for _ in range(paths)]
    for b in books:
        for edge in EDGES:
            tag = "бэктест" if edge == 1.0 else f"ожидание ×{edge:g}"
            s = simulate_book(z, b, haircut(z, b, edge))
            win = rolling_windows(s["curve"], GOAL_YEARS)
            hist = (f"история, окна {GOAL_YEARS:g} года из каждого месяца ({len(win)}): медиана ×{np.median(win):.1f}, "
                    f"худшее ×{win.min():.2f}, лучшее ×{win.max():.0f}, дошли до ×{GOAL:.0f} {np.mean(win >= GOAL):.0%}"
                    if len(win) else "история: окон нет")
            res = [goal_stats(simulate_book(q, b, haircut(q, b, edge))["curve"], q.fill.min(), GOAL, GOAL_YEARS)
                   for q in boots]
            x2 = np.array([r[0] for r in res])
            hit = np.array([r[1] for r in res])
            tg = np.array([r[2] for r in res])
            dd2 = np.array([r[3] for r in res])
            reached = np.isfinite(tg)
            t_med = f"{np.median(tg[reached]):.1f} года" if reached.mean() >= 0.5 else "больше половины путей не дошли"
            print(f"    {b.name} — {tag}\n{'':6}за {GOAL_YEARS:g} года: счёт × 10-й процентиль {np.percentile(x2, 10):.2f}, "
                  f"медиана {np.median(x2):.1f}, 90-й {np.percentile(x2, 90):.0f}; дошли до ×{GOAL:.0f} {hit.mean():.0%}; "
                  f"меньше старта {np.mean(x2 < 1):.0%}; просадка > 50% за это время {np.mean(dd2 > 0.5):.0%}\n{'':6}"
                  f"до ×{GOAL:.0f}: медиана {t_med}, дошли вообще (за всю длину пути) {reached.mean():.0%}\n{'':6}{hist}")


def extra_trades(root: Path = COMBO_DIR) -> pd.DataFrame:
    """Сделки oi_liq и прострелов в формате load(): R — доходность на номинал (комиссии внутри), stop_pct = 1."""
    oi = pd.read_csv(root / "oi_liq_trades.csv")
    t = pd.to_datetime(oi["t"], utc=True, format="ISO8601").dt.tz_localize(None)
    a = pd.DataFrame({"tf": "oi", "fill": t + pd.Timedelta(hours=1), "exit": t + pd.Timedelta(hours=5),
                      "R": oi["L5x0.5"].to_numpy(), "symbol": oi["symbol"]})
    sp = pd.read_csv(root / "spikes_6s_bybit.csv")
    t = pd.to_datetime(sp["t"], utc=True, format="ISO8601").dt.tz_localize(None)
    b = pd.DataFrame({"tf": "spike", "fill": t, "exit": t + pd.Timedelta(minutes=60), "R": sp["r"].to_numpy(),
                      "symbol": sp["symbol"]})
    sp5 = pd.read_csv(root / "spikes_6s_stop5d_bybit.csv")
    t = pd.to_datetime(sp5["t"], utc=True, format="ISO8601").dt.tz_localize(None)
    b5 = pd.DataFrame({"tf": "spike5d", "fill": t, "exit": t + pd.Timedelta(minutes=60), "R": sp5["r"].to_numpy(),
                       "symbol": sp5["symbol"]})
    sp4 = pd.read_csv(root / "spikes_4s_bybit.csv")
    t = pd.to_datetime(sp4["t"], utc=True, format="ISO8601").dt.tz_localize(None)
    b4 = pd.DataFrame({"tf": "spike4", "fill": t, "exit": t + pd.Timedelta(minutes=60), "R": sp4["r"].to_numpy(),
                       "symbol": sp4["symbol"]})
    x = pd.concat([a, b, b5, b4], ignore_index=True)
    return x.assign(stop_pct=1.0, side=1, t=x.fill, slope_atr=0.0, squeeze=1.0).dropna(subset=["R"])


def combo_report(z: pd.DataFrame, ex: pd.DataFrame, paths: int = 300) -> None:
    windows = (("с 2022-01 (бот + oi_liq)", "2022-01-21", "2026-10-01", False, ("прострел",)),
               ("2025-07 … 2026-09 (все три)", "2025-07-03", "2026-10-01", False, ()),
               ("2025-07 … 2026-09 без обвала 10.10.2025", "2025-07-03", "2026-10-01", True, ()))
    cols = [c for c in ("tf", "fill", "exit", "R", "stop_pct", "side", "t", "slope_atr", "squeeze") if c in z]
    for title, a, e, drop, skip in windows:
        q = pd.concat([z[cols], ex[cols]], ignore_index=True)
        q = q[(q.fill >= a) & (q.fill < e)]
        if drop:
            q = q[q.fill.dt.strftime("%Y-%m-%d") != CRASH_DAY]
        q = q.sort_values(["fill", "tf"]).reset_index(drop=True)
        years = (q.fill.max() - q.fill.min()).days / 365.25
        n = q.tf.value_counts().to_dict()
        print(f"\n  === {title}: {years:.1f} года, сделок {n} ===")
        rng = np.random.default_rng(31)
        boots = [week_blocks(q, rng) for _ in range(paths)]
        for edge in EDGES:
            print(f"    -- {'бэктест' if edge == 1.0 else f'ожидание ×{edge:g} (у каждой стратегии)'} --")
            for b in BOOKS10:
                if any(k in b.name for k in skip):
                    continue
                s = simulate_book(q, b, haircut(q, b, edge))
                cagr = s["eq"] ** (1 / years) - 1
                sims = [simulate_book(w, b, haircut(w, b, edge)) for w in boots]
                cg = np.array([x["eq"] ** (1 / years) - 1 for x in sims])
                dd = np.array([x["mdd"] for x in sims])
                skips = ", ".join(f"{k} {v}" for k, v in s["skip"].items() if v)
                print(f"      {b.name:<36} история {cagr:+.0%} в год, просадка {s['mdd']:.0%}, худший месяц "
                      f"{worst_month(s['curve']):+.0%}; бутстреп: медиана {np.median(cg):+.0%}, 10-й процентиль "
                      f"{np.percentile(cg, 10):+.0%}, P(просадка > 50%) {np.mean(dd > 0.5):.0%}; пропущено: {skips or '—'}")


def report10() -> None:
    print("\n===== LIVEBOOK10: бот B2 вместе с oi_liq и ловлей прострелов одним счётом (маржа Bybit; 300 путей) =====")
    z = load()
    if z is None:
        print("  crisk: нет данных")
        return
    combo_report(z, extra_trades())


def spike_stats(tk: pd.DataFrame) -> str:
    sp = tk[tk.tf.str.startswith("spike")] if len(tk) else tk
    if not len(sp):
        return "прострелов не взято"
    day = (sp.frac * sp.R).groupby(pd.to_datetime(sp.fill).dt.normalize()).sum()
    return (f"прострелов {len(sp)}, прибыльных {(sp.R > 0).mean():.0%}, средняя {sp.R.mean():+.2%}, худшая {sp.R.min():+.0%}"
            f" (на счёт {(sp.frac * sp.R).min():+.1%}), худший день прострелов {day.min():+.1%} счёта")


def guarded_report(z: pd.DataFrame, ex: pd.DataFrame, paths: int = 300) -> None:
    cols = [c for c in ("tf", "fill", "exit", "R", "stop_pct", "side", "t", "slope_atr", "squeeze") if c in z]
    for title, drop in (("2025-07 … 2026-09", False), ("2025-07 … 2026-09 без обвала 10.10.2025", True)):
        q = pd.concat([z[cols], ex[cols]], ignore_index=True)
        q = q[(q.fill >= "2025-07-03") & (q.fill < "2026-10-01")]
        if drop:
            q = q[q.fill.dt.strftime("%Y-%m-%d") != CRASH_DAY]
        q = q.sort_values(["fill", "tf"]).reset_index(drop=True)
        years = (q.fill.max() - q.fill.min()).days / 365.25
        print(f"\n  === {title}: {years:.1f} года ===")
        rng = np.random.default_rng(37)
        boots = [week_blocks(q, rng) for _ in range(paths)]
        for edge in EDGES:
            print(f"    -- {'бэктест' if edge == 1.0 else f'ожидание ×{edge:g} (у каждой стратегии)'} --")
            for b in BOOKS11:
                s = simulate_book(q, b, haircut(q, b, edge))
                sims = [simulate_book(w, b, haircut(w, b, edge)) for w in boots]
                cg = np.array([x["eq"] ** (1 / years) - 1 for x in sims])
                dd = np.array([x["mdd"] for x in sims])
                skips = ", ".join(f"{k} {v}" for k, v in s["skip"].items() if v)
                print(f"      {b.name:<44} история {s['eq'] ** (1 / years) - 1:+.0%} в год, просадка {s['mdd']:.0%}, худший "
                      f"месяц {worst_month(s['curve']):+.0%}; бутстреп: медиана {np.median(cg):+.0%}, 10-й процентиль "
                      f"{np.percentile(cg, 10):+.0%}, P(просадка > 50%) {np.mean(dd > 0.5):.0%}\n{'':8}{spike_stats(s['taken'])}; "
                      f"пропущено: {skips or '—'}")


def report11() -> None:
    print("\n===== LIVEBOOK11: прострелы с ограничениями риска рядом с ботом B2 (потолок, предохранитель, стоп 5D) =====")
    z = load()
    if z is None:
        print("  crisk: нет данных")
        return
    guarded_report(z, extra_trades())


def report9() -> None:
    print("\n===== LIVEBOOK9: $3000 → $100 000 — шанс за 2 года и срок; бэктест и вдвое меньшее ожидание (500 путей) =====")
    z = load()
    if z is None:
        print("  crisk: нет данных")
        return
    goal_report(z)


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


def report2() -> None:
    print("\n===== LIVEBOOK2: где безопасно — 4h отдельно и с 15m-шортами в масштабе проверок =====")
    z = load()
    if z is None:
        print("  crisk: нет данных")
        return
    portfolio_report(z, BOOKS2, "Варианты")


def report8() -> None:
    print("\n===== LIVEBOOK8: высокий риск — где рост типичного пути перестаёт расти и как растут хвосты (500 путей) =====")
    z = load()
    if z is None:
        print("  crisk: нет данных")
        return
    portfolio_report(z, BOOKS8, "Высокий риск", paths=500)


def report7() -> None:
    print("\n===== LIVEBOOK7: 4h отдельно против 4h + 15m при одинаковой просадке (500 путей) =====")
    z = load()
    if z is None:
        print("  crisk: нет данных")
        return
    portfolio_report(z, BOOKS7, "4h отдельно и вместе с 15m", paths=500)


def report6() -> None:
    print("\n===== LIVEBOOK6: 15m шорты отдельно, без 4h (500 путей) =====")
    z = load()
    if z is None:
        print("  crisk: нет данных")
        return
    portfolio_report(z, BOOKS6, "15m отдельно", paths=500)


def report5() -> None:
    print("\n===== LIVEBOOK5: связка владельца 15m 0.35% + 4h 2.25% (500 путей) =====")
    z = load()
    if z is None:
        print("  crisk: нет данных")
        return
    portfolio_report(z, BOOKS5, "Связка владельца", paths=500)


def report4() -> None:
    print("\n===== LIVEBOOK4: наклон линии 15m и лонги 15m при безопасном масштабе =====")
    z = load()
    if z is None:
        print("  crisk: нет данных")
        return
    portfolio_report(z, BOOKS4, "Наклон и лонги")


def report3() -> None:
    print("\n===== LIVEBOOK3: максимальный безопасный риск — сетка между P 1% и P 8% =====")
    z = load()
    if z is None:
        print("  crisk: нет данных")
        return
    portfolio_report(z, BOOKS3, "Граница риска")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    {"2": report2, "3": report3, "4": report4, "5": report5, "6": report6, "7": report7, "8": report8,
     "9": report9, "10": report10, "11": report11}.get((sys.argv[1:] or [""])[0], report)()
