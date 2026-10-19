"""Счёт бумажной книги «пробой линии на дневках» в деньгах по истории (research/htftrend.py, колонки acct_*).

Тот же учёт, что trader/domain/daily_break.py: риск — доля текущего капитала (старт 1.0 плюс итог сделок, закрытых
раньше дня входа); номинал = риск × уровень / расстояние до стопа, не больше плеча × капитал и того, что помещается в
95% свободной маржи вместе с хеджем BTC (бета × номинал); нет места — сделка пропущена. Доливка (+0.5 на +1.5R) —
в её день, только если её маржа (без хеджа) помещается в свободную, иначе сделка идёт без неё (R без доливки).
Хедж — только основной части (нога BTC в R сделки из htftrend). Итог сделки = (R + нога BTC) × риск.
Капитал — по закрытым сделкам (без переоценки открытых), поэтому просадка — нижняя оценка; «тепло» — сумма рисков
открытых позиций к капиталу, сколько можно потерять, если все стопы сработают разом.

Варианты (заданы до запуска, по вопросу владельца «как уменьшить просадки при заданном риске»):
    база                   все сделки, риск 6%, плечо 12
    не больше N позиций     одновременно открытых — 3 / 4 / 6
    только шорты
    агрессоры выше медианы IS
    риск вдвое меньше в просадке  3% при просадке больше 15% от максимума капитала, 6% — после нового максимума
    шорты, не больше 4 позиций
    справка                риск 2% и 3%, все сделки
Прогон 7 (по вопросу владельца «просадка — от многих убыточных позиций в одну сторону?», заданы до запуска):
    не больше N в сторону   одновременно открытых лонгов и шортов отдельно — 2 / 3; встречная сделка не мешает
    чистый риск в сторону   |риск лонгов − риск шортов| открытых к капиталу не больше 12% / 18%; новая сделка урезается
                           до остатка, остаток меньше четверти риска — пропуск (бета не учитывается: основа под хеджем)
    риск по очереди         k-я одновременная сделка в ту же сторону — риск 6% / k (6, 3, 2, 1.5…)
Решение: вариант лучше базы, если максимальная просадка и худший месяц меньше, а годовая доходность к просадке выше,
и на перпетуалах, и на споте.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LEVERAGE, MARGIN_USE, ADD_SIZE = 12, 0.95, 0.5


@dataclass(frozen=True)
class Policy:
    name: str
    risk: float = 0.06
    max_open: int | None = None
    dd_cut: float | None = None         # просадка, после которой риск — dd_risk
    dd_risk: float | None = None
    side_max: int | None = None         # одновременно открытых в одну сторону
    net_cap: float | None = None        # |риск лонгов − риск шортов| к капиталу
    decay: bool = False                 # k-я сделка в ту же сторону — риск / k


@dataclass(frozen=True)
class Result:
    taken: int
    skip_cap: int
    skip_margin: int
    add_refused: int
    final: float                        # капитал в конце, старт 1.0
    cagr: float
    max_dd: float
    worst_month: float
    max_heat: float
    years: dict[int, float]


def _day(x) -> pd.Timestamp:
    return pd.Timestamp(x).tz_localize(None).normalize() if pd.Timestamp(x).tzinfo else pd.Timestamp(x).normalize()


def simulate(trades: pd.DataFrame, pol: Policy, leverage: int = LEVERAGE) -> Result:
    """Счёт по сделкам с колонками acct_fill, acct_exit, acct_r, acct_r0, acct_h, acct_add, acct_addpx, acct_beta,
    acct_lr (порядок сделок одного дня — по времени пробоя и монете)."""
    tr = trades.dropna(subset=["acct_fill", "acct_exit", "acct_r"]).copy()
    tr["fd"] = tr.acct_fill.map(_day)
    tr["xd"] = tr.acct_exit.map(_day)
    tr["ad"] = tr.acct_add.map(lambda v: _day(v) if pd.notna(v) else pd.NaT)
    tr = tr.sort_values(["fd", "t", "symbol"]).reset_index(drop=True)
    days = sorted(set(tr.fd) | set(tr.xd) | set(tr.ad.dropna()))
    pos: list[dict] = []                # открытые и закрытые позиции, взятые в счёт
    closed_eq: list[tuple[pd.Timestamp, float]] = []
    realized, peak = 0.0, 1.0
    skip_cap = skip_margin = refused = 0
    heat_max = 0.0
    by_fill = {d: g for d, g in tr.groupby("fd")}

    def margin(p: dict, day: pd.Timestamp) -> float:
        b = p["beta"] if p["beta"] > 0 else 0.0
        m = p["notional"] * (1 + b)
        if p["add_ok"] and p["ad"] <= day:
            m += ADD_SIZE * p["notional"] * p["addpx"]
        return m / leverage

    for day in days:
        # 1. капитал на начало дня — сделки, закрытые раньше
        for p in sorted((p for p in pos if p["xd"] < day and not p["booked"]), key=lambda p: p["xd"]):
            p["booked"] = True
            realized += p["pnl"]
            closed_eq.append((p["xd"], 1.0 + realized))
            peak = max(peak, 1.0 + realized)
        equity = 1.0 + realized
        live = [p for p in pos if p["fd"] <= day <= p["xd"]]
        # 2. входы дня
        for _, x in by_fill.get(day, pd.DataFrame()).iterrows():
            if pol.max_open is not None and len(live) >= pol.max_open:
                skip_cap += 1
                continue
            side = 1 if x.side > 0 else -1
            same = sum(p["side"] == side for p in live)
            if pol.side_max is not None and same >= pol.side_max:
                skip_cap += 1
                continue
            risk = pol.dd_risk if pol.dd_cut is not None and 1 - equity / peak > pol.dd_cut else pol.risk
            if pol.decay:
                risk /= 1 + same
            if pol.net_cap is not None:
                room = pol.net_cap - side * sum(p["side"] * p["risk"] for p in live) / equity
                if room < risk / 4:
                    skip_cap += 1
                    continue
                risk = min(risk, room)
            b = float(x.acct_beta) if np.isfinite(x.acct_beta) and x.acct_beta > 0 else 0.0
            used = sum(margin(p, day) for p in live)
            notional = min(risk * equity * x.acct_lr, leverage * equity,
                           max(MARGIN_USE * (equity - used), 0.0) * leverage / (1 + b))
            if not notional > 0:
                skip_margin += 1
                continue
            p = {"side": side, "fd": day, "xd": x.xd, "ad": x.ad,
                 "addpx": float(x.acct_addpx) if pd.notna(x.acct_addpx) else 1.0, "beta": b, "notional": notional,
                 "risk": notional / x.acct_lr, "r": float(x.acct_r),
                 "r0": float(x.acct_r0), "h": float(x.acct_h) if b > 0 else 0.0, "add_ok": False, "booked": False,
                 "pending_add": pd.notna(x.ad)}
            pos.append(p)
            live.append(p)
        # 3. доливки дня — по свободной марже
        for p in live:
            if p["pending_add"] and p["ad"] == day:
                p["pending_add"] = False
                need = ADD_SIZE * p["notional"] * p["addpx"] / leverage
                used = sum(margin(q, day) for q in live)
                p["add_ok"] = need <= MARGIN_USE * (equity - used)
                refused += not p["add_ok"]
        for p in live:
            if not p["pending_add"]:
                p["pnl"] = ((p["r"] if p["add_ok"] or pd.isna(p["ad"]) else p["r0"]) + p["h"]) * p["risk"]
        heat_max = max(heat_max, sum(p["risk"] for p in live) / equity if equity > 0 else 0.0)
    for p in sorted((p for p in pos if not p["booked"]), key=lambda p: p["xd"]):
        realized += p["pnl"]
        closed_eq.append((p["xd"], 1.0 + realized))
    if not closed_eq:
        return Result(0, skip_cap, skip_margin, refused, 1.0, 0.0, 0.0, 0.0, heat_max, {})
    eq = pd.Series([v for _, v in closed_eq], index=pd.DatetimeIndex([d for d, _ in closed_eq])).groupby(level=0).last()
    start = tr.fd.min()
    full = pd.concat([pd.Series([1.0], index=[start - pd.Timedelta(days=1)]), eq]).sort_index()
    full = full.groupby(level=0).last()
    dd = float((1 - full / full.cummax()).max())
    month = full.resample("ME").last().ffill()
    month = pd.concat([pd.Series([1.0], index=[month.index[0] - pd.offsets.MonthEnd()]), month])
    mret = month.pct_change().dropna()
    years = (full.index[-1] - full.index[0]).days / 365.25
    final = float(full.iloc[-1])
    cagr = final ** (1 / years) - 1 if years > 0 and final > 0 else -1.0
    yearly = full.resample("YE").last().ffill()
    yearly = pd.concat([pd.Series([1.0], index=[yearly.index[0] - pd.offsets.YearEnd()]), yearly]).pct_change().dropna()
    return Result(len(pos), skip_cap, skip_margin, refused, final, float(cagr), dd, float(mret.min()), heat_max,
                  {int(k.year): float(v) for k, v in yearly.items()})


def policies(aggr_median: float | None = None) -> list[tuple[Policy, str]]:
    """(политика, фильтр сделок: all / short / aggr)."""
    out = [(Policy("база: риск 6%, все сделки"), "all")]
    out += [(Policy(f"не больше {n} позиций", max_open=n), "all") for n in (3, 4, 6)]
    out += [(Policy("только шорты"), "short")]
    if aggr_median is not None:
        out += [(Policy(f"агрессоры выше медианы IS ({aggr_median:.3f})"), "aggr")]
    out += [(Policy("риск 3% в просадке > 15%", dd_cut=0.15, dd_risk=0.03), "all"),
            (Policy("шорты, не больше 4 позиций", max_open=4), "short"),
            (Policy("справка: риск 3%", risk=0.03), "all"), (Policy("справка: риск 2%", risk=0.02), "all")]
    out += [(Policy(f"не больше {n} в сторону", side_max=n), "all") for n in (2, 3)]
    out += [(Policy(f"чистый риск в сторону ≤ {c:.0%}", net_cap=c), "all") for c in (0.12, 0.18)]
    out += [(Policy("риск по очереди: 6% / k", decay=True), "all")]
    return out


def report(df: pd.DataFrame, title: str) -> None:
    x = df.dropna(subset=["acct_fill", "acct_r"])
    if not len(x):
        print(f"     {title}: сделок для счёта нет")
        return
    is_ = x[(x.t >= "2022-01-01") & (x.t < "2024-07-01")]
    am = float(is_.aggr.median()) if len(is_) >= 30 else float(x.aggr.median())
    print(f"     {title}: сделок {len(x)}, с {x.acct_fill.min():%Y-%m} по {x.acct_exit.max():%Y-%m}")
    for pol, flt in policies(am):
        z = x if flt == "all" else x[x.side < 0] if flt == "short" else x[x.aggr > am]
        r = simulate(z, pol)
        yrs = " ".join(f"{y}: {v:+.0%}" for y, v in sorted(r.years.items()))
        print(f"       {pol.name}: итог ×{r.final:.2f}, в год {r.cagr:+.0%}, просадка {r.max_dd:.0%}, худший месяц "
              f"{r.worst_month:+.0%}, доходность/просадка {r.cagr / r.max_dd if r.max_dd > 0 else float('nan'):.2f}, "
              f"тепло до {r.max_heat:.0%}; взято {r.taken}, мимо по лимиту {r.skip_cap}, по марже {r.skip_margin}, "
              f"без доливки {r.add_refused}")
        print(f"         по годам: {yrs}")
