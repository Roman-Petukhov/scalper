"""Как сделать прибыльнее тренд бумажного счёта («сильнейшие 7 + долив + приток», trader/domain/trend.py). Всё, что
проверено раньше (dboost / dboost2 / dboost3), здесь не повторяется. Варианты (заданы до запуска, каждый — одно изменение):
    спот             держать спот вместо перпетуала: funding не платится, комиссия спота 0.1%, общий вес до 1 (без
                     плеча); цена спота принята равной перпу (у топ-20 расхождение — доли процента)
    перп, вес до 1   контроль к «споту»: тот же потолок веса 1, funding и taker перпа — сколько даёт сам потолок
    вход внутри дня  модель входит, как только максимум дня достал уровень канала по закрытиям (стоп-ордер; цена —
                     max(открытие, уровень)), а не на закрытии; выход тот же. Отбор 7 и вес — на закрытии дня, так что
                     прибыль дня входа считается только у монет, взятых в портфель (небольшое забегание вперёд в пользу
                     варианта — его учитываем при решении)
    лимит сектора    не больше 2 монет из одного сектора research/oos.py (мемы, AI, L2, DeFi, игры; L1 и прочие — без
                     лимита), место — следующей по моментуму
    цель волатильности  веса × 30% / волатильность самой стратегии за 30 прошлых дней (множитель 0.25…2), общий вес до 2
Для справки (не решает): база без funding — сколько его платит стратегия.
Принятие (research/dboost.py): Sharpe выше базы во всех периодах IS / VAL / HO / 2026 и просадка с 2022 при равной
волатильности не больше ×1.5 базы. Вариантов 4 (перп с весом до 1 — контроль).

    python -m research.dboost4 collect --symbols <1h-qualified>      (+ python -m research.extreg collect)
    python -m research.dboost4 report
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from trader.domain.trend import LOOKBACKS, TAKER

from . import dtrend
from .dboost import JUDGED, PER, SINCE, _bot_months, _combo, _dd, _sharpe, judge
from .dboost2 import book, inflow_gate
from .shard import all_parts

SPOT_FEE = 0.001
VOL_TARGET, VOL_DAYS, VOL_LO, VOL_HI = 0.30, 30, 0.25, 2.0
GROSS_MAX = 2.0
SECTOR_CAP = 2


def capped(w: pd.DataFrame, gross_max: float) -> pd.DataFrame:
    gross = w.abs().sum(axis=1)
    return w.mul(np.minimum(1.0, gross_max / gross.replace(0, np.nan)).fillna(1.0), axis=0)


def returns(w: pd.DataFrame, close: pd.DataFrame, fund: pd.DataFrame, *, fee: float = TAKER,
            funding: bool = True, extra: pd.Series | None = None) -> pd.Series:
    """Как dboost2.returns, с комиссией fee, без funding по флагу и с добавкой extra (прибыль дня входа)."""
    held = w.shift(1).fillna(0.0)
    ret = close.pct_change(fill_method=None).fillna(0.0)
    pnl = (held * ret).sum(axis=1)
    if funding:
        pnl = pnl - (held * fund.fillna(0.0)).sum(axis=1)
    pnl = pnl - (w - w.shift(1).fillna(0.0)).abs().sum(axis=1).shift(1).fillna(0.0) * fee
    return pnl + (extra if extra is not None else 0.0)


def intraday_models(close: pd.DataFrame, high: pd.DataFrame, opn: pd.DataFrame,
                    n: int) -> tuple[np.ndarray, np.ndarray]:
    """Модель N с входом внутри дня: (состояние 0/1 на закрытии, прибыль дня входа close / цена входа − 1)."""
    lvl = close.rolling(n, min_periods=n).max().shift(1).to_numpy()
    lo = close.rolling(n, min_periods=n).min().shift(1).to_numpy()
    mid = (lvl + lo) / 2
    x, h, o = close.to_numpy(), high.to_numpy(), opn.to_numpy()
    st, gain = np.zeros_like(x), np.zeros_like(x)
    cur = np.zeros(x.shape[1])
    for i in range(len(x)):
        cur = np.where((cur > 0) & (x[i] <= mid[i]), 0.0, cur)
        enter = (cur == 0) & (h[i] >= lvl[i])
        px = np.fmax(o[i], lvl[i])
        gain[i] = np.where(enter & (px > 0), x[i] / px - 1, 0.0)
        cur = np.where(enter, 1.0, cur)
        cur = np.where(np.isnan(x[i]), 0.0, cur)
        st[i] = cur
    return st, np.nan_to_num(gain)


def intraday_signal(close: pd.DataFrame, high: pd.DataFrame, opn: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(доля моделей в лонге, прибыль дня входа на единицу сигнала — среднее по вошедшим сегодня моделям / 9)."""
    sts, gains = zip(*(intraday_models(close, high, opn, n) for n in LOOKBACKS))
    k = len(LOOKBACKS)
    return (pd.DataFrame(sum(sts) / k, index=close.index, columns=close.columns),
            pd.DataFrame(sum(gains) / k, index=close.index, columns=close.columns))


def entry_extra(w: pd.DataFrame, sig: pd.DataFrame, gain: pd.DataFrame) -> pd.Series:
    """Прибыль дня входа: вес монеты на закрытии × доля сигнала, вошедшая сегодня × её ход от цены входа."""
    per_unit = (gain / sig.where(sig > 0)).fillna(0.0)
    return (w * per_unit).sum(axis=1)


def vol_scaled(w: pd.DataFrame, r: pd.Series) -> pd.DataFrame:
    vol = r.rolling(VOL_DAYS, min_periods=20).std().shift(1) * np.sqrt(365)
    k = (VOL_TARGET / vol.replace(0, np.nan)).clip(VOL_LO, VOL_HI).fillna(1.0)
    return capped(w.mul(k, axis=0), GROSS_MAX)


def report() -> None:
    parts, bparts, eparts = all_parts("dtrend"), all_parts("dtrend_bot"), all_parts("extreg")
    print("\n===== DBOOST4: тренд бумажного счёта (сильнейшие 7 + долив + приток) — спот, вход внутри дня, лимит "
          "сектора, цель волатильности; ячейка — доходность в год / Sharpe =====")
    if not parts or not eparts:
        print(f"частей нет (монеты {len(parts)}, стейблкоины {len(eparts)})")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["day"] = pd.to_datetime(df["day"], utc=True).dt.tz_localize(None)
    piv = {k: df.pivot_table(index="day", columns="symbol", values=k) for k in ("close", "open", "high", "funding", "adv")}
    close = piv["close"]
    opn, high, fund, adv = (piv[k].reindex_like(close) for k in ("open", "high", "funding", "adv"))
    ext = pd.concat([pd.read_parquet(p) for p in eparts], ignore_index=True)
    ext["day"] = pd.to_datetime(ext["day"], utc=True).dt.tz_localize(None)
    gate = inflow_gate(ext.groupby("day")["stable"].last().dropna(), close.index).astype(float)
    print(f"дней {len(close)}, монет {close.shape[1]}; приток включён {gate[gate.index >= SINCE].mean():.0%} дней с 2022")

    base_w = book(close, adv).mul(gate, axis=0)
    base = returns(base_w, close, fund)
    sig_i, gain_i = intraday_signal(close, high, opn)
    w_i = book(close, adv, sig=sig_i).mul(gate, axis=0)
    res = {
        "база: бумажный счёт": base,
        "спот (без funding, 0.1%, вес до 1)": returns(capped(base_w, 1.0), close, fund, fee=SPOT_FEE, funding=False),
        "контроль: перп, вес до 1": returns(capped(base_w, 1.0), close, fund),
        "вход внутри дня по уровню канала": returns(w_i, close, fund, extra=entry_extra(w_i, sig_i, gain_i)),
        "лимит сектора 2": returns(book(close, adv, sector_cap=SECTOR_CAP).mul(gate, axis=0), close, fund),
        "цель волатильности 30%": returns(vol_scaled(base_w, base), close, fund),
        "справка: база без funding": returns(base_w, close, fund, funding=False),
    }
    judged = ("спот", "вход", "лимит", "цель")
    bot_m = _bot_months(bparts) if bparts else None
    fpaid = (base_w.shift(1).fillna(0.0) * fund.fillna(0.0)).sum(axis=1)
    print(f"funding, который платит база: с 2022 {fpaid[fpaid.index >= SINCE].mean() * 365:.1%} капитала в год, "
          f"2021 {fpaid[(fpaid.index >= '2021-01-01') & (fpaid.index < '2022-01-01')].mean() * 365:.1%}")
    print(f"средний общий вес базы с 2022: {base_w[base_w.index >= SINCE].abs().sum(axis=1).mean():.2f}")
    for nm, r in res.items():
        x = r[r.index >= SINCE]
        head = (f"\n  {nm}: с 2022 {x.mean() * 365:+.0%} в год, вол {x.std() * np.sqrt(365):.0%}, Sharpe {_sharpe(x):.2f}, "
                f"просадка {_dd(x):.0%}")
        if r is not base:
            ok, better, ratio = judge(r, base)
            verdict = ("принято" if ok else "не принято") if nm.startswith(judged) else "не решает"
            head += (f"  → {verdict} (Sharpe выше базы в {sum(better)} из {len(JUDGED)}, просадка при равной вол. "
                     f"×{ratio:.2f} базы)")
        print(head)
        print("    " + "  ".join(f"{p} {r[(r.index >= a) & (r.index < b)].mean() * 365:+.0%} / "
                                 f"{_sharpe(r[(r.index >= a) & (r.index < b)]):.2f}" for p, (a, b) in PER.items()))
        if bot_m is not None:
            print("    " + _combo(bot_m, r))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        dtrend.collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x])
    else:
        report()
