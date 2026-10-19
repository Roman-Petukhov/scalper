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
Прогон 8 (владелец: «при 3–4% какая прибыль в месяц; 6% ставил с учётом хеджа, без хеджа поставил бы 4%»):
    риск 3 / 4 / 5 / 6% с хеджем и 4 / 6% без хеджа (без ноги BTC и её маржи); по каждому — медианный месяц, доля
    месяцев в минусе, самый долгий срок без нового максимума капитала.
Прогон 9 (владелец: «со $100 дойти до $1000–2000, не снимая»): счёт в долларах со старта $100, минимальные ордера
Bybit — номинал монеты от $5, хедж BTCUSDT шагом 0.001 BTC (округление до шага; меньше шага — без хеджа). Сценарии:
    план             3% без хеджа, пока счёт < $500, дальше 4% с хеджем
    4% с хеджем      с первого дня (хедж — только когда набирается шаг)
    3% / 4% без хеджа всё время
Старт — с каждого месяца истории (пути перекрываются, это не независимые испытания); по каждому сценарию — доля путей,
дошедших до $1000 и $2000 за 1 / 2 / 3 года, медиана срока среди дошедших, доля путей, где счёт падал ниже $30.
Решение: вариант лучше базы, если максимальная просадка и худший месяц меньше, а годовая доходность к просадке выше,
и на перпетуалах, и на споте.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LEVERAGE, MARGIN_USE, ADD_SIZE = 12, 0.95, 0.5
MIN_NOTIONAL, BTC_STEP = 5.0, 0.001     # Bybit: минимальный номинал ордера USDT-перпетуала, шаг количества BTCUSDT


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
    hedge: bool = True                  # нога BTC на основу (её итог и маржа)
    usd: float | None = None            # старт в долларах — включает минимальные ордера Bybit
    switch_usd: float | None = None     # пока счёт ниже — риск low_risk и без хеджа
    low_risk: float | None = None


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
    median_month: float = 0.0
    neg_months: float = 0.0             # доля месяцев в минусе
    underwater_days: int = 0            # самый долгий срок без нового максимума капитала
    skip_min: int = 0                   # номинал меньше минимального ордера
    curve: pd.Series | None = None      # капитал по закрытым сделкам (старт 1.0)


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
    skip_cap = skip_margin = refused = skip_min = 0
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
            low = pol.usd is not None and pol.switch_usd is not None and equity * pol.usd < pol.switch_usd
            if low:
                risk = pol.low_risk if pol.low_risk is not None else risk
            if pol.decay:
                risk /= 1 + same
            if pol.net_cap is not None:
                room = pol.net_cap - side * sum(p["side"] * p["risk"] for p in live) / equity
                if room < risk / 4:
                    skip_cap += 1
                    continue
                risk = min(risk, room)
            b = float(x.acct_beta) if pol.hedge and not low and np.isfinite(x.acct_beta) and x.acct_beta > 0 else 0.0
            used = sum(margin(p, day) for p in live)
            notional = min(risk * equity * x.acct_lr, leverage * equity,
                           max(MARGIN_USE * (equity - used), 0.0) * leverage / (1 + b))
            if not notional > 0:
                skip_margin += 1
                continue
            hr = 1.0                            # доля ноги BTC после округления до шага
            if pol.usd is not None:
                if notional * pol.usd < MIN_NOTIONAL:
                    skip_min += 1
                    continue
                px = float(x.get("acct_btc", np.nan))
                if b > 0:
                    q = np.floor(b * notional * pol.usd / px / BTC_STEP + 0.5) * BTC_STEP if px > 0 else 0.0
                    hr = q * px / (b * notional * pol.usd) if q > 0 else 0.0
                    b *= hr
            p = {"side": side, "fd": day, "xd": x.xd, "ad": x.ad,
                 "addpx": float(x.acct_addpx) if pd.notna(x.acct_addpx) else 1.0, "beta": b, "notional": notional,
                 "risk": notional / x.acct_lr, "r": float(x.acct_r),
                 "r0": float(x.acct_r0), "h": float(x.acct_h) * hr if b > 0 else 0.0, "add_ok": False, "booked": False,
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
        return Result(0, skip_cap, skip_margin, refused, 1.0, 0.0, 0.0, 0.0, heat_max, {}, skip_min=skip_min)
    eq = pd.Series([v for _, v in closed_eq], index=pd.DatetimeIndex([d for d, _ in closed_eq])).groupby(level=0).last()
    start = tr.fd.min()
    full = pd.concat([pd.Series([1.0], index=[start - pd.Timedelta(days=1)]), eq]).sort_index()
    full = full.groupby(level=0).last()
    dd = float((1 - full / full.cummax()).max())
    month = full.resample("ME").last().ffill()
    month = pd.concat([pd.Series([1.0], index=[month.index[0] - pd.offsets.MonthEnd()]), month])
    mret = month.pct_change().dropna()
    mret = mret[mret.index >= start]
    years = (full.index[-1] - full.index[0]).days / 365.25
    final = float(full.iloc[-1])
    cagr = final ** (1 / years) - 1 if years > 0 and final > 0 else -1.0
    yearly = full.resample("YE").last().ffill()
    yearly = pd.concat([pd.Series([1.0], index=[yearly.index[0] - pd.offsets.YearEnd()]), yearly]).pct_change().dropna()
    top = full.cummax()
    highs = full.index[full >= top]
    gaps = np.diff(np.append(highs.values, full.index[-1].to_datetime64())).astype("timedelta64[D]").astype(int)
    return Result(len(pos), skip_cap, skip_margin, refused, final, float(cagr), dd, float(mret.min()), heat_max,
                  {int(k.year): float(v) for k, v in yearly.items()}, float(mret.median()), float((mret < 0).mean()),
                  int(gaps.max()) if len(gaps) else 0, skip_min, full)


@dataclass(frozen=True)
class Journey:
    starts: int
    reach: dict[float, tuple[float, float, float, float]]   # цель → доля за 1 / 2 / 3 года, медиана дней дошедших
    below_floor: float                  # доля путей, где счёт был ниже floor
    median_final_1y: float              # медиана счёта через год, $


def journey(trades: pd.DataFrame, pol: Policy, targets: tuple[float, ...] = (1000.0, 2000.0),
            floor: float = 30.0) -> Journey:
    """Путь счёта pol.usd долларов со старта в начале каждого месяца истории (последний старт — за год до конца)."""
    tr = trades.dropna(subset=["acct_fill", "acct_exit", "acct_r"])
    fills = tr.acct_fill.map(_day)
    first, last = fills.min(), tr.acct_exit.map(_day).max()
    hits: dict[float, list[float]] = {t: [] for t in targets}
    low, final = [], []
    for m in pd.date_range(first.to_period("M").to_timestamp(), last - pd.DateOffset(years=1), freq="MS"):
        r = simulate(tr[fills >= m], pol)
        if r.curve is None:
            continue
        usd = r.curve * pol.usd
        low.append(bool(usd.min() < floor))
        at = usd[usd.index <= m + pd.DateOffset(years=1)]
        final.append(float(at.iloc[-1]))
        for t in targets:
            up = usd.index[usd >= t]
            hits[t].append(float((up[0] - m).days) if len(up) else np.inf)
    n = len(low)
    if not n:
        return Journey(0, {}, 0.0, float("nan"))
    reach = {}
    for t, d in hits.items():
        a = np.array(d)
        done = a[np.isfinite(a)]
        reach[t] = (float((a <= 365).mean()), float((a <= 730).mean()), float((a <= 1096).mean()),
                    float(np.median(done)) if len(done) else float("nan"))
    return Journey(n, reach, float(np.mean(low)), float(np.median(final)))


def scenarios(usd: float = 100.0) -> list[Policy]:
    return [Policy("план: 3% без хеджа до $500, дальше 4% с хеджем", risk=0.04, usd=usd, switch_usd=500.0,
                   low_risk=0.03),
            Policy("4% с хеджем с первого дня", risk=0.04, usd=usd),
            Policy("3% без хеджа всё время", risk=0.03, hedge=False, usd=usd),
            Policy("4% без хеджа всё время", risk=0.04, hedge=False, usd=usd)]


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
    out += [(Policy(f"риск {r:.0%}, хедж", risk=r), "all") for r in (0.03, 0.04, 0.05, 0.06)]
    out += [(Policy(f"риск {r:.0%}, без хеджа", risk=r, hedge=False), "all") for r in (0.04, 0.06)]
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
        print(f"         месяц: в среднем {(1 + r.cagr) ** (1 / 12) - 1:+.1%}, медиана {r.median_month:+.1%}, в минусе "
              f"{r.neg_months:.0%} месяцев; дольше всего без нового максимума {r.underwater_days} дн.")
        print(f"         по годам: {yrs}")


def report_journey(df: pd.DataFrame, title: str, usd: float = 100.0) -> None:
    x = df.dropna(subset=["acct_fill", "acct_r"])
    if not len(x):
        return
    print(f"     {title}: счёт ${usd:.0f}, старт с каждого месяца (пути перекрываются)")
    for pol in scenarios(usd):
        j = journey(x, pol)
        r = simulate(x, pol)
        parts = "; ".join(f"${t:.0f}: за год {a:.0%}, 2 года {b:.0%}, 3 года {c:.0%}, медиана {m:.0f} дн."
                          for t, (a, b, c, m) in j.reach.items())
        print(f"       {pol.name}: путей {j.starts}; {parts}; ниже $30 было в {j.below_floor:.0%} путей; через год "
              f"медиана ${j.median_final_1y:.0f}; со старта истории — в год {r.cagr:+.0%}, просадка {r.max_dd:.0%}, "
              f"мимо по мин. ордеру {r.skip_min}")
