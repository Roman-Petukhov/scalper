"""Тренд «сильнейшие 7 + долив» (trader/domain/trend.py — бумажный счёт бота): что ещё поднимет прибыль.

База — правило бота: ансамбль Дончиана 9 периодов (5…360 дней, выход на середине канала), из топ-20 по обороту с
сигналом > 0 — 7 с лучшим моментумом (60 дней / вол. 90 дней), долив +0.5 на новом максимуме 20 дней (до 3), вес
сигнал × долив × 40% / вол. / 7, общий вес до 2, taker и funding. Варианты (заданы до запуска), каждый — одно изменение:
    пул топ-50       7 сильнейших из топ-50 по обороту вместо топ-20
    + шорты          зеркальная книга: то же правило на 1 / цена (шорт 7 слабейших в нисходящем тренде), вместе с
                     лонгами, общий вес до 2
    приток           весь портфель только в дни, когда предложение стейблкоинов за 30 дней выросло (research/extreg.py)
    пересчёт на 4h   то же правило на свечах 4h (периоды в днях × 6), решение на закрытии каждой 4h-свечи
    без дорогого funding  монета не берётся, если средний funding за 7 дней выше 0.1% в день (место — следующей)
Принятие (как research/dboost.py): Sharpe выше базы во всех периодах IS / VAL / HO / 2026 и просадка с 2022 при равной
волатильности не больше ×1.5 базы. Вариантов 5. Для каждого — счёт «бот 4h + тренд».

    python -m research.dboost2 collect --symbols <1h-qualified>      (+ python -m research.extreg collect)
    python -m research.dboost2 report
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from trader.domain.trend import LOOKBACKS, TAKER, _mid_models

from . import dtrend
from .broad import ADV_MIN, adv30
from .dboost import JUDGED, PER, SINCE, _bot_months, _combo, _dd, _sharpe, judge
from .shard import all_parts, mine, part_path
from .tline import tf_frame

TOP, K = 20, 7
COIN_VOL, GROSS_MAX = 0.40, 2.0
FUND_CAP = 0.001                    # средний funding в день, выше — монету не берём
ADV_4H = 50e6                       # 4h-свечи собираем только у монет, бывших хоть раз с оборотом от $50M


def book(close: pd.DataFrame, adv: pd.DataFrame, fund: pd.DataFrame | None = None, *, top: int = TOP, k: int = K,
         bpd: int = 1, fund_cap: float | None = None) -> pd.DataFrame:
    """Веса «сильнейшие K + долив» на свечах с bpd свечами в дне (1 — дневки, 6 — 4h)."""
    sig = pd.DataFrame(sum(_mid_models(close, n * bpd) for n in LOOKBACKS) / len(LOOKBACKS),
                       index=close.index, columns=close.columns)
    hi = close.rolling(20 * bpd, min_periods=20 * bpd).max().shift(1)
    new_hi = (close >= hi).to_numpy()
    on = (sig >= 0.5).to_numpy()
    cnt = np.zeros(close.shape[1])
    adds = np.zeros(close.shape)
    for i in range(len(close)):
        prev = on[i - 1] if i else np.zeros(close.shape[1], bool)
        cnt = np.where(on[i], np.where(prev, np.minimum(cnt + new_hi[i], 3), 0.0), 0.0)
        adds[i] = cnt
    sized = sig * (1 + 0.5 * adds)
    ret = close.pct_change(fill_method=None)
    vol = ret.rolling(90 * bpd, min_periods=30 * bpd).std() * np.sqrt(365 * bpd)
    mom = (close / close.shift(60 * bpd) - 1) / vol
    rk = adv.where(adv >= ADV_MIN).rank(axis=1, ascending=False, method="first")
    member = rk <= top
    ok = member & (sig > 0) & mom.notna()
    if fund_cap is not None and fund is not None:
        ok &= ~(fund.rolling(7 * bpd, min_periods=1).sum() / 7 > fund_cap)
    keep = mom.where(ok).rank(axis=1, ascending=False, method="first") <= k
    w = (sized.where(keep, 0.0) * (COIN_VOL / vol) / k).where(member & close.notna(), 0.0).fillna(0.0)
    return _cap(w)


def _cap(w: pd.DataFrame) -> pd.DataFrame:
    gross = w.abs().sum(axis=1)
    return w.mul(np.minimum(1.0, GROSS_MAX / gross.replace(0, np.nan)).fillna(1.0), axis=0)


def returns(w: pd.DataFrame, close: pd.DataFrame, fund: pd.DataFrame) -> pd.Series:
    """Доходность за свечу: держим веса прошлого закрытия; лонг платит funding > 0, шорт получает; taker с оборота."""
    held = w.shift(1).fillna(0.0)
    ret = close.pct_change(fill_method=None).fillna(0.0)
    pnl = (held * ret).sum(axis=1) - (held * fund.fillna(0.0)).sum(axis=1)
    return pnl - (w - w.shift(1).fillna(0.0)).abs().sum(axis=1).shift(1).fillna(0.0) * TAKER


def inflow_gate(stable: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    st = stable.reindex(idx).ffill(limit=3)
    return (st > st.shift(30)).fillna(False)


def to_daily(r: pd.Series) -> pd.Series:
    return (1 + r).groupby(r.index.normalize()).prod() - 1


def collect(root: Path, syms: list[str]) -> None:
    dtrend.collect(root, syms)                                    # дневки и сделки бота (части dtrend, dtrend_bot)
    rows = []
    for s in mine(syms):
        try:
            a = adv30(root, s)
            if not (a.max() >= ADV_4H):
                continue
            d = tf_frame(root, s, "4h")
            if d is None or len(d) < 500:
                continue
            rows.append(pd.DataFrame({"symbol": s, "t": d.index, "close": d["close"].to_numpy(),
                                      "funding": d["funding"].to_numpy()}))
        except Exception as e:
            print(f"  dboost2 {s}: пропуск ({e})", flush=True)
    print(f"  dboost2: монет с 4h {len(rows)}", flush=True)
    if rows:
        pd.concat(rows, ignore_index=True).to_parquet(part_path("dboost2_4h"), index=False)


def report() -> None:
    parts, bparts = all_parts("dtrend"), all_parts("dtrend_bot")
    hparts, eparts = all_parts("dboost2_4h"), all_parts("extreg")
    print("\n===== DBOOST2: «сильнейшие 7 + долив» (правило бумажного счёта) — пул, шорты, приток, 4h, funding; "
          "ячейка — доходность в год / Sharpe =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["day"] = pd.to_datetime(df["day"], utc=True).dt.tz_localize(None)
    close = df.pivot_table(index="day", columns="symbol", values="close")
    fund = df.pivot_table(index="day", columns="symbol", values="funding").reindex_like(close)
    adv = df.pivot_table(index="day", columns="symbol", values="adv").reindex_like(close)
    print(f"дней {len(close)}, монет {close.shape[1]}, с {close.index[0]:%Y-%m-%d} по {close.index[-1]:%Y-%m-%d}")

    base_w = book(close, adv)
    res: dict[str, pd.Series | None] = {"база: сильнейшие 7 + долив": returns(base_w, close, fund)}
    res["пул топ-50"] = returns(book(close, adv, top=50), close, fund)
    inv = 1 / close
    both = _cap(base_w - book(inv, adv))
    res["+ шорты (зеркально)"] = returns(both, close, fund)
    if eparts:
        ext = pd.concat([pd.read_parquet(p) for p in eparts], ignore_index=True)
        ext["day"] = pd.to_datetime(ext["day"], utc=True).dt.tz_localize(None)
        g = inflow_gate(ext.groupby("day")["stable"].last().dropna(), close.index)
        print(f"приток стейблкоинов: включён {g[g.index >= SINCE].mean():.0%} дней с 2022")
        res["включатель «приток»"] = returns(base_w.mul(g.astype(float), axis=0), close, fund)
    else:
        res["включатель «приток»"] = None
    if hparts:
        h = pd.concat([pd.read_parquet(p) for p in hparts], ignore_index=True)
        h["t"] = pd.to_datetime(h["t"], utc=True).dt.tz_localize(None)
        c4 = h.pivot_table(index="t", columns="symbol", values="close")
        f4 = h.pivot_table(index="t", columns="symbol", values="funding").reindex_like(c4)
        a4 = adv.reindex(columns=c4.columns).reindex(c4.index.normalize())
        a4.index = c4.index
        r4 = returns(book(c4, a4, bpd=6), c4, f4)
        res["пересчёт на 4h"] = to_daily(r4).reindex(close.index).fillna(0.0)
        print(f"4h: монет {c4.shape[1]}, свечей {len(c4)}")
    else:
        res["пересчёт на 4h"] = None
    res["без дорогого funding (> 0.1% в день)"] = returns(book(close, adv, fund, fund_cap=FUND_CAP), close, fund)

    base = res["база: сильнейшие 7 + долив"]
    bot_m = _bot_months(bparts) if bparts else None
    for nm, r in res.items():
        if r is None:
            print(f"\n  {nm}: нет данных")
            continue
        x = r[r.index >= SINCE]
        head = (f"\n  {nm}: с 2022 {x.mean() * 365:+.0%} в год, вол {x.std() * np.sqrt(365):.0%}, Sharpe {_sharpe(x):.2f}, "
                f"просадка {_dd(x):.0%}")
        if r is not base:
            ok, better, ratio = judge(r, base)
            head += (f"  → {'принято' if ok else 'не принято'} (Sharpe выше базы в {sum(better)} из {len(JUDGED)}, "
                     f"просадка при равной вол. ×{ratio:.2f} базы)")
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
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x])
    else:
        report()
