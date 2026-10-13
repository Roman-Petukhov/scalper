"""Как увеличить прибыль тренда Дончиана (research/dtrend.py): долив, концентрация в сильнейших, другой выход.

База — ансамбль dtrend: 9 моделей N = 5…360 дней, вход на закрытии не ниже максимума закрытий N прошлых дней, выход на
закрытии не выше середины канала; топ-20 по обороту, размер по волатильности (40% / вол. 90 дней), общий вес до 2,
taker и funding. Варианты (заданы до запуска):
    долив         монета в тренде (≥ половины моделей в лонге): вес × (1 + 0.5 × k), k — сколько новых максимумов
                  закрытий за 20 дней было с начала тренда (без дня входа), не больше 3; тренд кончился — k = 0
    сильнейшие K  из монет топ-20 с сигналом > 0 — только K с лучшим моментумом (доходность за 60 дней / вол. 90 дней),
                  капитал делится на K, а не на 20; K = 5 и 7
    ATR-выход k   модель выходит, когда закрытие ниже максимума закрытий с входа на k × ATR (ATR — среднее |Δ закрытия|
                  за 14 дней); k = 3 и 4; вход тот же
    сочетания     долив + сильнейшие 7, долив + ATR 3, сильнейшие 7 + ATR 3, все три
Принятие: Sharpe выше базы во всех периодах IS / VAL / HO / 2026 и худшая просадка с 2022 года — не больше 1.5 базы при
равной волатильности (вариант масштабирован к волатильности базы за 2022+). Вариантов 9 — один случайный проход
возможен. Для каждого — счёт «бот 4h + тренд» (равная волатильность, помесячно), как в dtrend.

    python -m research.dtrend collect --symbols <1h-qualified>     (части dtrend и dtrend_bot)
    python -m research.dboost report
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from .broad import ADV_MIN
from .dtrend import COIN_VOL, LOOKBACKS, TOP, VOL_DAYS, _models, portfolio
from .oos import PER_ALL
from .shard import all_parts
from .tline import _bot_base

PER = {**PER_ALL, "2026": ("2026-01-01", "2026-10-01")}
JUDGED = ("is", "val", "ho", "2026")
ADD_STEP, ADD_MAX, ADD_HIGH = 0.5, 3, 20
MOM_DAYS = 60
ATR_DAYS = 14
SINCE = "2022-01-01"


def pyramid(close: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    """Множитель веса 1 + 0.5 × (новые максимумы 20 дней с начала тренда, до 3)."""
    hi = close.rolling(ADD_HIGH, min_periods=ADD_HIGH).max().shift(1)
    new_hi = (close >= hi).to_numpy()
    on = (sig >= 0.5).to_numpy()
    k = np.zeros(close.shape[1])
    out = np.ones(close.shape)
    for i in range(len(close)):
        prev = on[i - 1] if i else np.zeros(close.shape[1], bool)
        k = np.where(on[i], np.where(prev, np.minimum(k + new_hi[i], ADD_MAX), 0.0), 0.0)
        out[i] = 1 + ADD_STEP * k
    return pd.DataFrame(out, index=close.index, columns=close.columns)


def strongest(close: pd.DataFrame, adv: pd.DataFrame, sig: pd.DataFrame, k: int) -> pd.DataFrame:
    """Сигнал только у K сильнейших по моментуму среди топ-20 с сигналом > 0; × 20 / K — капитал делится на K."""
    ret = close.pct_change(fill_method=None)
    vol = ret.rolling(VOL_DAYS, min_periods=30).std() * np.sqrt(365)
    mom = (close / close.shift(MOM_DAYS) - 1) / vol
    rk = adv.where(adv >= ADV_MIN).rank(axis=1, ascending=False, method="first")
    cand = mom.where((rk <= TOP) & (sig > 0) & mom.notna())
    keep = cand.rank(axis=1, ascending=False, method="first") <= k
    return sig.where(keep, 0.0) * (TOP / k)


def atr_models(close: pd.DataFrame, n: int, k: float) -> pd.DataFrame:
    """Модель N: вход как в dtrend, выход — закрытие ниже максимума с входа на k × ATR."""
    hi = close.rolling(n, min_periods=n).max().shift(1).to_numpy()
    atr = close.diff().abs().rolling(ATR_DAYS, min_periods=ATR_DAYS).mean().to_numpy()
    x = close.to_numpy()
    cur = np.zeros(x.shape[1])
    peak = np.full(x.shape[1], np.nan)
    st = np.zeros_like(x)
    for i in range(len(x)):
        peak = np.where(cur > 0, np.fmax(peak, x[i]), peak)
        cur = np.where((cur > 0) & (x[i] <= peak - k * atr[i]), 0.0, cur)
        enter = (cur == 0) & (x[i] >= hi[i])
        cur = np.where(enter, 1.0, cur)
        peak = np.where(enter, x[i], np.where(cur > 0, peak, np.nan))
        cur = np.where(np.isnan(x[i]), 0.0, cur)
        st[i] = cur
    return pd.DataFrame(st, index=close.index, columns=close.columns)


def ensemble(close: pd.DataFrame, atr_k: float = 0.0) -> pd.DataFrame:
    if atr_k:
        return sum(atr_models(close, n, atr_k) for n in LOOKBACKS) / len(LOOKBACKS)
    return sum(_models(close, n, False) for n in LOOKBACKS) / len(LOOKBACKS)


def variants(close: pd.DataFrame, adv: pd.DataFrame) -> dict[str, pd.DataFrame]:
    base = ensemble(close)
    atr3, atr4 = ensemble(close, 3.0), ensemble(close, 4.0)
    return {
        "база (dtrend)": base,
        "долив": base * pyramid(close, base),
        "сильнейшие 5": strongest(close, adv, base, 5),
        "сильнейшие 7": strongest(close, adv, base, 7),
        "ATR-выход 3": atr3,
        "ATR-выход 4": atr4,
        "долив + сильнейшие 7": strongest(close, adv, base * pyramid(close, base), 7),
        "долив + ATR 3": atr3 * pyramid(close, atr3),
        "сильнейшие 7 + ATR 3": strongest(close, adv, atr3, 7),
        "все три": strongest(close, adv, atr3 * pyramid(close, atr3), 7),
    }


def _sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std() * np.sqrt(365)) if len(r) >= 30 and r.std() > 0 else np.nan


def _dd(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    return float((1 - eq / eq.cummax()).max())


def judge(r: pd.Series, base: pd.Series) -> tuple[bool, list[bool], float]:
    """(принят, Sharpe выше базы по периодам, просадка 2022+ при волатильности базы / просадка базы)."""
    better = []
    for p in JUDGED:
        a, b = PER[p]
        sel = (r.index >= a) & (r.index < b)
        better.append(bool(_sharpe(r[sel]) > _sharpe(base[sel])))
    rs, bs = r[r.index >= SINCE], base[base.index >= SINCE]
    scaled = rs * (bs.std() / rs.std()) if rs.std() > 0 else rs
    ratio = _dd(scaled) / _dd(bs) if _dd(bs) > 0 else np.nan
    return all(better) and ratio <= 1.5, better, ratio


def _line(r: pd.Series) -> str:
    cells = []
    for p, (a, b) in PER.items():
        x = r[(r.index >= a) & (r.index < b)]
        cells.append(f"{p} {x.mean() * 365:+.0%} / {_sharpe(x):.2f}" if len(x) >= 30 else f"{p} —")
    return "  ".join(cells)


def _bot_months(bparts: list) -> pd.Series:
    bt = pd.concat([pd.read_parquet(p) for p in bparts], ignore_index=True)
    bt["t"] = pd.to_datetime(bt["t"], utc=True)
    bt = _bot_base(bt[bt.adv.isna() | (bt.adv >= ADV_MIN)])
    bt = bt[bt.entry == "retest"]
    return bt.groupby(bt.t.dt.tz_localize(None).dt.to_period("M")).R3.sum() * 0.01      # риск 1% на сделку


def _combo(bot_m: pd.Series, r: pd.Series) -> str:
    tr_m = (1 + r).groupby(r.index.to_period("M")).prod() - 1
    idx = tr_m.index[tr_m.index >= pd.Period("2020-06", "M")]
    b, t = bot_m.reindex(idx, fill_value=0.0), tr_m.reindex(idx)
    c = (b + t * (b.std() / t.std())) / 2
    eq = (1 + c).cumprod()
    since = c[c.index >= pd.Period(SINCE[:7], "M")]
    return (f"корр. с ботом {b.corr(t):+.2f}; бот + тренд: {c.mean() * 12:+.0%} в год, Sharpe "
            f"{c.mean() / c.std() * np.sqrt(12):.2f}, просадка {(1 - eq / eq.cummax()).max():.0%}; с 2022: "
            f"{since.mean() * 12:+.0%} в год, Sharpe {since.mean() / since.std() * np.sqrt(12):.2f}")


def report() -> None:
    parts, bparts = all_parts("dtrend"), all_parts("dtrend_bot")
    print("\n===== DBOOST: тренд Дончиана — долив, сильнейшие монеты, ATR-выход; ячейка — доходность в год / Sharpe =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["day"] = pd.to_datetime(df["day"], utc=True).dt.tz_localize(None)
    close = df.pivot_table(index="day", columns="symbol", values="close")
    fund = df.pivot_table(index="day", columns="symbol", values="funding").reindex_like(close)
    adv = df.pivot_table(index="day", columns="symbol", values="adv").reindex_like(close)
    print(f"дней {len(close)}, монет {close.shape[1]}, с {close.index[0]:%Y-%m-%d} по {close.index[-1]:%Y-%m-%d}")
    res = {nm: portfolio(close, fund, adv, sig=s) for nm, s in variants(close, adv).items()}
    base = res["база (dtrend)"]
    bot_m = _bot_months(bparts) if bparts else None
    for nm, r in res.items():
        x = r[r.index >= SINCE]
        head = (f"\n  {nm}: с 2022 {x.mean() * 365:+.0%} в год, вол {x.std() * np.sqrt(365):.0%}, Sharpe {_sharpe(x):.2f}, "
                f"просадка {_dd(x):.0%}")
        if nm != "база (dtrend)":
            ok, better, ratio = judge(r, base)
            head += (f"  → {'принято' if ok else 'не принято'} (Sharpe выше базы в {sum(better)} из {len(better)}, "
                     f"просадка при равной вол. ×{ratio:.2f} базы)")
        print(head)
        print("    " + _line(r))
        if bot_m is not None:
            print("    " + _combo(bot_m, r))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
