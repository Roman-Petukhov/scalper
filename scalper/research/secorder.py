"""Очередь секторов в бычьем рынке: можно ли заранее угадать, какой сектор альтов выстрелит следующим (вопрос владельца
«сектора стреляют не разом, а по очереди — как угадать очередь»).

Прошлый прогон (research/sectors.py): «лучший сектор за 7 дней — на неделю» плюс во всех периодах, t 2.3; фиксированного
опережения сектор → сектор нет. Здесь — признаки из литературы и практики, которые могут сказать о следующем секторе
раньше цены:
    O1 лестница размера   (Hou 2007: крупные ведут мелких внутри отрасли; «деньги текут BTC → ETH → крупные → мелкие»)
       a: крупные альты (топ-20 по обороту без BTC / ETH) обогнали BTC за 28 дней → следующую неделю мелкие
          (места 51–200) против крупных;
       b: ETH / BTC вырос за 28 дней → следующую неделю корзина альтов против BTC.
    O2 внимание           (Maître, Pugachyov, Weigert 2025: аномальное внимание → доходность следующего дня; оборот —
                           доступная замена твитам) — сектор с наибольшим ростом оборота за 7 дней к своему среднему за 90
                           (сверх такого же отношения рынка) → держим неделю; b — то же, но только если сектор сам за 7 дней
                           в плюсе сверх рынка (внимание + подтверждение ценой).
    O3 ширина пробоя      сектор с наибольшей долей монет на 30-дневном максимуме закрытия (>= 5 монет) → неделя.
    O4 очередь по циклам  (справка) бычьи отрезки 2020-07 … 2021-11 и 2023-01 … 2025-01: день, когда сектор достиг
                           максимума накопленной доходности сверх рынка, и место в очереди; совпадает ли очередь двух
                           циклов (ранговая корреляция Спирмена). Цикла два — это описание, не доказательство.
Издержки taker 11 б.п. за круг (у O1 — две ноги, 22 б.п.). Мерило — сверх рынка / BTC того же срока.
Принятие (O1–O3, задано до запуска): > 0 в каждом периоде IS / VAL / HO (недель >= 20), t >= 3, > 0 при издержках ×2.

    python -m research.sectors collect --symbols <1h-all> && python -m research.secorder
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from .sectors import ADV_MIN, COST, load_daily, panel, sector_map, verdict

LOOK = 28
LARGE, SMALL = (0, 20), (50, 200)
ATT_SHORT, ATT_LONG = 7, 90
HIGH_N, BREADTH_MIN = 30, 5
CYCLES = {"цикл 2020–2021": ("2020-07-01", "2021-11-30"), "цикл 2023–2025": ("2023-01-01", "2025-01-31")}


def mondays(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx[idx.dayofweek == 0]


def week_ret(r: pd.Series, t: pd.Timestamp) -> float | None:
    """Доходность дневного ряда с понедельника t на 7 дней (сложная); мало дней — None."""
    w = r.loc[t:t + pd.Timedelta(days=6)].dropna()
    return float(np.expm1(np.log1p(w).sum())) if len(w) >= 5 else None


def past(r: pd.Series, t: pd.Timestamp, n: int) -> float | None:
    w = r.loc[t - pd.Timedelta(days=n):t - pd.Timedelta(days=1)].dropna()
    return float(np.expm1(np.log1p(w).sum())) if len(w) >= n - 2 else None


def size_baskets(r: pd.DataFrame, qv: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Дневные доходности крупных и мелких альтов: места по обороту за 30 дней на вчера, без BTC / ETH."""
    adv = qv.rolling(30, min_periods=20).mean().shift(1).drop(columns=["BTCUSDT", "ETHUSDT"], errors="ignore")
    rank = adv.where(adv >= ADV_MIN).rank(axis=1, ascending=False)
    ra = r.reindex(columns=rank.columns)
    big = ra.where((rank > LARGE[0]) & (rank <= LARGE[1])).mean(axis=1)
    small = ra.where((rank > SMALL[0]) & (rank <= SMALL[1])).mean(axis=1)
    return big, small


def ladder(r: pd.DataFrame, qv: pd.DataFrame, mkt: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    btc = r["BTCUSDT"] if "BTCUSDT" in r else pd.Series(dtype="float64")
    eth = r["ETHUSDT"] if "ETHUSDT" in r else pd.Series(dtype="float64")
    big, small = size_baskets(r, qv)
    a, b = [], []
    for t in mondays(r.index):
        pb, pbtc, peth = past(big, t, LOOK), past(btc, t, LOOK), past(eth, t, LOOK)
        if pb is not None and pbtc is not None and pb > pbtc:
            s, g = week_ret(small, t), week_ret(big, t)
            if s is not None and g is not None:
                a.append({"t": t, "X": s - g - 2 * COST})
        if peth is not None and pbtc is not None and peth > pbtc:
            m, x = week_ret(mkt, t), week_ret(btc, t)
            if m is not None and x is not None:
                b.append({"t": t, "X": m - x - 2 * COST})
    return pd.DataFrame(a), pd.DataFrame(b)


def sector_turnover(qv: pd.DataFrame) -> dict[str, pd.Series]:
    smap = sector_map()
    out = {}
    for name in dict.fromkeys(smap.values()):
        cols = [c for c in qv.columns if smap.get(c) == name]
        if cols:
            out[name] = qv[cols].sum(axis=1, min_count=1)
    return out


def pick_weekly(score: pd.DataFrame, ex: pd.DataFrame, gate: pd.DataFrame | None = None) -> pd.DataFrame:
    """Каждый понедельник — сектор с наибольшим score на вчера (при gate — только где gate > 0); держим неделю."""
    rows = []
    for t in mondays(score.index):
        prev = t - pd.Timedelta(days=1)
        if prev not in score.index:
            continue
        sc = score.loc[prev].dropna()
        sc = sc[ex.loc[prev].reindex(sc.index).notna()]          # у сектора есть индекс (>= 3 ликвидных монет)
        if gate is not None:
            g = gate.loc[prev].reindex(sc.index)
            sc = sc[g > 0]
        if len(sc) < 1 or score.loc[prev].notna().sum() < 4:
            continue
        pick = sc.idxmax()
        x = week_ret(ex[pick], t)
        if x is not None:
            rows.append({"t": t, "X": x - COST, "sector": pick})
    return pd.DataFrame(rows)


def attention(qv: pd.DataFrame, idx: dict[str, pd.Series], mkt: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    turn = pd.DataFrame(sector_turnover(qv)).reindex(columns=list(idx))
    tot = qv.drop(columns=["BTCUSDT"], errors="ignore").sum(axis=1, min_count=1)
    rel = turn.rolling(ATT_SHORT, min_periods=6).mean() / turn.rolling(ATT_LONG, min_periods=60).mean()
    mrel = tot.rolling(ATT_SHORT, min_periods=6).mean() / tot.rolling(ATT_LONG, min_periods=60).mean()
    score = np.log(rel.div(mrel, axis=0))
    ex = pd.DataFrame(idx).sub(mkt, axis=0)
    mom = np.log1p(ex.fillna(0.0)).rolling(ATT_SHORT, min_periods=6).sum()
    return pick_weekly(score, ex), pick_weekly(score, ex, gate=mom)


def breadth(close: pd.DataFrame, r: pd.DataFrame, idx: dict[str, pd.Series], mkt: pd.Series) -> pd.DataFrame:
    smap = sector_map()
    hi = close >= close.rolling(HIGH_N, min_periods=HIGH_N).max()
    live = r.notna()
    sc = {}
    for name in idx:
        cols = [c for c in close.columns if smap.get(c) == name]
        n = live[cols].sum(axis=1)
        sc[name] = (hi[cols] & live[cols]).sum(axis=1).div(n).where(n >= BREADTH_MIN)
    ex = pd.DataFrame(idx).sub(mkt, axis=0)
    return pick_weekly(pd.DataFrame(sc), ex)


def cycle_order(idx: dict[str, pd.Series], mkt: pd.Series) -> None:
    ex = pd.DataFrame(idx).sub(mkt, axis=0)
    orders = {}
    print("\n  --- O4: очередь секторов по циклам (справка; день максимума накопленной доходности сверх рынка) ---")
    for name, (a, b) in CYCLES.items():
        seg = ex.loc[a:b]
        cum = np.expm1(np.log1p(seg.fillna(0.0)).cumsum())
        have = seg.notna().sum() >= max(1.0, 0.6 * len(seg))
        rows = []
        for s in seg.columns[have.to_numpy()]:
            rows.append({"сектор": s, "пик": cum[s].idxmax().date(), "сверх рынка в пике": cum[s].max(),
                         "к концу": cum[s].iloc[-1]})
        if not rows:
            print(f"  {name}: секторов с данными нет")
            continue
        tab = pd.DataFrame(rows).sort_values("пик").reset_index(drop=True)
        tab.index += 1
        orders[name] = {r["сектор"]: i for i, r in tab.iterrows()}
        print(f"  {name} ({a} … {b}):")
        print(tab.to_string(formatters={"сверх рынка в пике": "{:+.0%}".format, "к концу": "{:+.0%}".format}))
    names = list(orders)
    if len(names) == 2:
        common = sorted(set(orders[names[0]]) & set(orders[names[1]]))
        if len(common) >= 4:
            x = pd.Series([orders[names[0]][s] for s in common]).rank()
            y = pd.Series([orders[names[1]][s] for s in common]).rank()
            rho = float(np.corrcoef(x, y)[0, 1])
            print(f"  Спирмен очередей двух циклов по {len(common)} общим секторам: {rho:+.2f} "
                  f"({'очередь похожа' if rho >= 0.6 else 'очередь не повторилась'})")


def show(name: str, z: pd.DataFrame, passed: list[str], legs: int = 1) -> None:
    if z.empty:
        print(f"    {name}: сделок нет")
        return
    good, txt = verdict(z)                            # «×2» в verdict вычитает ещё одну ногу; у двух ног — проверяем две
    good = good and (z.X - COST * legs).mean() > 0
    print(f"    {name}: {'ПРИНЯТО' if good else 'нет'} (недель {len(z)})\n        {txt}")
    if "sector" in z:
        top = z.sector.value_counts().head(3)
        print("        чаще всего: " + ", ".join(f"{k} {v}" for k, v in top.items()))
    if good:
        passed.append(name)


def report() -> None:
    print("\n===== SECORDER: можно ли угадать следующий сектор (перпетуалы Binance с делистнутыми, дневки) =====")
    got = load_daily()
    if got is None:
        print("  secorder: нет данных")
        return
    close, qv = got
    r, mkt, idx = panel(close, qv)
    print(f"  монет {close.shape[1]}, секторов {len(idx)}, дни {close.index[0].date()} … {close.index[-1].date()}")
    passed: list[str] = []
    print("\n  --- O1: лестница размера ---")
    a, b = ladder(r, qv, mkt)
    show("a: крупные обогнали BTC за 28 дн. → мелкие против крупных, неделя", a, passed, legs=2)
    show("b: ETH/BTC вырос за 28 дн. → альты против BTC, неделя", b, passed, legs=2)
    print("\n  --- O2: внимание (рост оборота сектора) ---")
    a2, b2 = attention(qv, idx, mkt)
    show("a: наибольший рост оборота → неделя", a2, passed)
    show("b: рост оборота + сектор в плюсе за 7 дн. → неделя", b2, passed)
    print("\n  --- O3: ширина пробоя ---")
    show("доля монет на 30-дн. максимуме → неделя", breadth(close, r, idx, mkt), passed)
    cycle_order(idx, mkt)
    print("\n  ИТОГ: " + ("прошли — " + "; ".join(passed) if passed else "ни один признак не угадывает следующий сектор "
                                                                       "с запасом на издержки"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 300)
    report()
