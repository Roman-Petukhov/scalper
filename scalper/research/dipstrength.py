"""Сила в падении: ведут ли следующий рост монеты, которые держались лучше рынка во время коррекции.

Идея из акций (О'Нил, Минервини: «лидеры нового роста видны в коррекции — они падают меньше и первыми выходят на
максимум»). У нас относительная сила проверялась только как фильтр сделок бота на 24 / 72 ч (research/rs.py — не
принята); на горизонте недель — нет.

Событие (всё известно на день T, без заглядывания вперёд):
    рынок — равновзвешенный индекс ликвидных альтов без BTC (как в research/sectors.py);
    коррекция — индекс упал на DD_MIN от максимума за PEAK_WIN дней; P — день максимума, L — день минимума;
    разворот T — первый день, когда индекс закрылся на REBOUND выше минимума коррекции; следующее событие — не раньше
    чем через COOLDOWN дней.
Сила монеты — её доходность от P до L минус доходность индекса (лог), среди альтов, ликвидных на T, с ценой на P и L.
    D1 a/b/c  верхняя пятая часть по силе → держим 14 / 28 / 56 дней от закрытия T;
    D2        нижняя пятая часть (упали сильнее всех — «отскок») → 28 дней, справка о знаке;
    D3        монеты, у которых линия «цена / индекс» на T на максимуме за PEAK_WIN дней (уже обновили относительный
              максимум) → 28 дней;
    D4        D1b по режиму BTC на T (выше / ниже недельной EMA50).
Исход — доходность монеты минус средняя доходность всех ликвидных на T альтов за тот же срок, минус издержки taker
11 б.п. за круг; монета, снятая с торгов внутри срока, считается по последней цене. t — по событиям (монеты одного
события — одна группа).
Принятие (задано до запуска): > 0 в каждом периоде IS / VAL / HO (монето-сделок >= 20 в каждом), t по событиям >= 3,
> 0 при издержках ×2. Событий мало (десятки) — без t >= 3 это справка, не сигнал.

    python -m research.prepare 1h-all sect-metrics && python -m research.posflow collect --symbols <1h-all>
    python -m research.dipstrength
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .flows import LLAMA, judge
from .flows import _get as get_json
from .posflow import btc_bull, in_regime, load
from .sectors import COST, panel

PEAK_WIN = 60
DD_MIN = 0.20
REBOUND = 0.10
COOLDOWN = 30
HOLDS = (14, 28, 56)
QUINT = 0.2
MIN_UNIVERSE = 30


@dataclass(frozen=True)
class Dip:
    peak: pd.Timestamp
    low: pd.Timestamp
    turn: pd.Timestamp
    depth: float                # падение индекса от максимума до минимума


def market_index(mkt: pd.Series) -> pd.Series:
    return (1 + mkt.fillna(0.0)).cumprod().where(mkt.notna().cummax())


def find_dips(index: pd.Series, peak_win: int = PEAK_WIN, dd_min: float = DD_MIN, rebound: float = REBOUND,
              cooldown: int = COOLDOWN) -> list[Dip]:
    """Коррекции индекса с подтверждённым разворотом; см. описание модуля."""
    x = index.dropna()
    out: list[Dip] = []
    peak_day = low_day = None
    low = 0.0
    wait_until = None
    for i, (t, v) in enumerate(x.items()):
        if wait_until is not None and t < wait_until:
            continue
        if low_day is None:
            win = x.iloc[max(0, i - peak_win + 1):i + 1]
            if len(win) >= peak_win // 2 and v / win.max() - 1 <= -dd_min:
                peak_day, low_day, low = win.idxmax(), t, v
            continue
        if v < low:
            low_day, low = t, v
        elif v >= low * (1 + rebound):
            out.append(Dip(peak_day, low_day, t, float(low / x[peak_day] - 1)))
            peak_day = low_day = None
            wait_until = t + pd.Timedelta(days=cooldown)
    return out


def universe(r: pd.DataFrame, close: pd.DataFrame, d: Dip) -> list[str]:
    """Альты, ликвидные на день разворота, с ценой на максимуме и минимуме коррекции."""
    if d.turn not in r.index:
        return []
    ok = r.loc[d.turn].notna() & close.loc[d.peak].notna() & close.loc[d.low].notna() & close.loc[d.turn].notna()
    return [c for c in r.columns[ok.to_numpy()] if c != "BTCUSDT"]


def fall_strength(close: pd.DataFrame, index: pd.Series, d: Dip, coins: list[str]) -> pd.Series:
    """Лог-доходность монеты от максимума рынка до минимума минус то же у индекса."""
    coin = np.log(close.loc[d.low, coins] / close.loc[d.peak, coins])
    return (coin - np.log(index[d.low] / index[d.peak])).replace([np.inf, -np.inf], np.nan).dropna()


def rs_at_high(close: pd.DataFrame, index: pd.Series, d: Dip, coins: list[str], win: int = PEAK_WIN) -> list[str]:
    """Монеты, у которых цена / индекс на день разворота — максимум за win дней."""
    seg = close.loc[:d.turn, coins].tail(win)
    line = seg.div(index.reindex(seg.index), axis=0)
    full = line.notna().sum() >= win // 2
    top = line.iloc[-1] >= line.max() - 1e-12
    return [c for c in coins if bool(full[c]) and bool(top[c])]


def forward(close: pd.DataFrame, d: Dip, coins: list[str], hold: int) -> pd.Series | None:
    """Доходность за hold дней от закрытия дня разворота; снятая с торгов монета — по последней цене. Срок ещё не
    прошёл — None."""
    end = d.turn + pd.Timedelta(days=hold)
    if end > close.index[-1]:
        return None
    seg = close.loc[d.turn:end, coins].ffill()
    return (seg.iloc[-1] / seg.iloc[0] - 1).dropna()


def trades(close: pd.DataFrame, r: pd.DataFrame, index: pd.Series, dips: list[Dip], hold: int, pick: str) -> pd.DataFrame:
    """Монето-сделки отбора pick: top / bottom — пятая часть по силе в падении, high — относительный максимум."""
    rows = []
    for d in dips:
        coins = universe(r, close, d)
        fwd = forward(close, d, coins, hold) if len(coins) >= MIN_UNIVERSE else None
        if fwd is None or len(fwd) < MIN_UNIVERSE:
            continue
        base = float(fwd.mean())
        if pick == "high":
            chosen = [c for c in rs_at_high(close, index, d, coins) if c in fwd.index]
        else:
            s = fall_strength(close, index, d, [c for c in coins if c in fwd.index])
            n = max(int(len(s) * QUINT), 1)
            chosen = list((s.nlargest(n) if pick == "top" else s.nsmallest(n)).index)
        for c in chosen:
            rows.append({"t": d.turn, "X": float(fwd[c]) - base - COST, "sector": c.removesuffix("USDT")})
    return pd.DataFrame(rows)


def by_event(z: pd.DataFrame) -> str:
    if z.empty:
        return "—"
    g = z.groupby("t").X
    return "  ".join(f"{t:%Y-%m-%d} {m:+.1%} ({n})" for t, m, n in zip(g.mean().index, g.mean(), g.size()))


def fee_fields() -> None:
    """Справка для скринера панели: какие поля периодов отдаёт /overview/fees (trader/infrastructure/fundamentals.py)."""
    raw = get_json(f"{LLAMA}/overview/fees?excludeTotalDataChart=true&excludeTotalDataChartBreakdown=true")
    protos = raw.get("protocols") if isinstance(raw, dict) else None
    if not isinstance(protos, list) or not protos:
        print("  DefiLlama /overview/fees: нет ответа")
        return
    keys = sorted({k for p in protos[:200] if isinstance(p, dict) for k in p if k.startswith(("total", "change", "average"))})
    want = ("total30d", "total60dto30d", "total7d", "total14dto7d", "total1y")
    have = {k: sum(1 for p in protos if isinstance(p, dict) and isinstance(p.get(k), (int, float))) for k in want}
    print(f"  DefiLlama /overview/fees: протоколов {len(protos)}; поля периодов: {', '.join(keys)}")
    print("    с числом: " + ", ".join(f"{k} {v}" for k, v in have.items()))


def report() -> None:
    print("\n===== DIPSTRENGTH: сила в падении — лидеры после коррекции рынка альтов =====")
    got = load()
    if got is None:
        print("  dipstrength: нет данных")
        return
    close, qv = got["close"], got["qv"]
    r, mkt, _ = panel(close, qv)
    index = market_index(mkt)
    dips = find_dips(index)
    print(f"  монет {close.shape[1]}, дни {close.index[0].date()} … {close.index[-1].date()}; коррекций с разворотом: {len(dips)}")
    for d in dips:
        print(f"    максимум {d.peak:%Y-%m-%d} → минимум {d.low:%Y-%m-%d} ({d.depth:+.0%}) → разворот {d.turn:%Y-%m-%d}, "
              f"монет {len(universe(r, close, d))}")
    passed: list[str] = []
    print("\n  --- D1: держались лучше рынка (верхняя пятая часть по силе в падении) ---")
    z28 = pd.DataFrame()
    for h, tag in zip(HOLDS, "abc"):
        z = trades(close, r, index, dips, h, "top")
        if h == 28:
            z28 = z
        judge(f"{tag}: сильные в падении → {h} дн.", z, passed)
        print(f"        по событиям: {by_event(z)}")
    print("\n  --- D2: упали сильнее всех (нижняя пятая часть) — справка о знаке ---")
    z = trades(close, r, index, dips, 28, "bottom")
    judge("слабые в падении → 28 дн.", z, [])
    print(f"        по событиям: {by_event(z)}")
    print("\n  --- D3: линия «цена / рынок» на максимуме на день разворота ---")
    z = trades(close, r, index, dips, 28, "high")
    judge("относительный максимум → 28 дн.", z, passed)
    print(f"        по событиям: {by_event(z)}")
    print("\n  --- D4: D1b по режиму BTC на день разворота ---")
    if "BTCUSDT" in close and len(z28):
        bull = btc_bull(close["BTCUSDT"])
        r1 = bull.fillna(0.0) > 0.5
        shifted = z28.assign(t=z28.t + pd.Timedelta(days=1))          # in_regime смотрит флаг на вчера
        for name, flag in (("BTC выше недельной EMA50", r1), ("BTC ниже недельной EMA50", ~r1 & bull.notna())):
            sub = in_regime(shifted, flag)
            judge(f"сильные в падении → 28 дн. | {name}", sub.assign(t=sub.t - pd.Timedelta(days=1)) if len(sub) else sub,
                  passed, min_n=10)
    else:
        print("    нет данных")
    if dips:
        d = dips[-1]
        coins = universe(r, close, d)
        if len(coins) >= MIN_UNIVERSE:
            s = fall_strength(close, index, d, coins)
            print(f"\n  последняя коррекция (разворот {d.turn:%Y-%m-%d}): сильнейшие в падении — "
                  + ", ".join(f"{c.removesuffix('USDT')} {v:+.0%}" for c, v in np.expm1(s.nlargest(10)).items()))
    print("\n  --- справка для скринера ---")
    fee_fields()
    print("\n  ИТОГ: " + ("прошли — " + "; ".join(passed) if passed else
                          "сила в падении не даёт отбора монет с запасом на издержки"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 300)
    report()
