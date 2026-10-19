"""15m (правило бота: пологие шорты, топ-150): какой выход нужен для прибыльности — цели дальше 3R и скользящий стоп.

Сигнал и стоп — как в research/retest15.py (правило бота). Входы: по рынку на закрытии и ретест 12 свечей (лимитка на
линии, как торгует владелец). Выходы (заданы до запуска), срок сделки 200 свечей:
    цель 2 / 3 (бот) / 4 / 5 / 6 / 8R
    трейл 2 / 3 ATR    без цели, стоп на 2 / 3 ATR сигнальной свечи за лучшим закрытием
    б/у 1R + трейл 3   без цели, на +1R стоп в безубыток, дальше трейл 3 ATR
С лимитом 3 позиции (как в боте; добавлено после первого прогона): место занято от сигнала до выхода, у ретеста —
и пока висит лимитка (12 свечей); монета — одна позиция; сигналы одной свечи — по обороту. Решение то же.
Издержки — maker на лимитку и тейк, taker на рыночный вход, стоп и выход по сроку, funding; «x2» — ещё раз комиссии
входа и выхода (taker) в R. Решение: выход лучше бота (3R), если R в месяц на сигнал выше во всех периодах IS / VAL / HO /
2026 и при издержках x2 в плюсе во всех периодах; выбираем плато (соседние выходы тоже лучше), а не пик.

    python -m research.exits15 collect --symbols <15m-wide>
    python -m research.exits15 report
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30
from .dconf import Exit, simulate_bar
from .retest15 import MAX_SLOPE, PER, TF, TOP
from .shard import all_parts, mine, part_path
from .smc import _atr
from .tline import MAKER, PIV, STOP_BUF, TAKER, htf_trend, last_confirmed, pivots, retest_fill, tf_frame, zz_lines

HOLD = 200
RETEST = 12
EXITS: dict[str, Exit] = {
    **{f"R{k:g}": Exit(float(k), (), 0.0, HOLD) for k in (2, 3, 4, 5, 6, 8)},
    "trail2": Exit(None, (), 0.0, HOLD, 2.0),
    "trail3": Exit(None, (), 0.0, HOLD, 3.0),
    "be1_trail3": Exit(None, (1.0,), 0.0, HOLD, 3.0),
}
BASE = "R3"
SLOTS = 3                      # бот 15m: не больше 3 позиций сразу (trader/domain/models.py)
NAMES = {"R2": "цель 2R", "R3": "цель 3R (бот)", "R4": "цель 4R", "R5": "цель 5R", "R6": "цель 6R", "R8": "цель 8R",
         "trail2": "трейл 2 ATR", "trail3": "трейл 3 ATR", "be1_trail3": "б/у на 1R + трейл 3 ATR"}


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str) -> list[dict]:
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64")
    a = _atr(d).to_numpy()
    trend = htf_trend(d, TF)
    buy = (d["taker_buy_volume"] / d["volume"].replace(0, np.nan)).to_numpy()
    rng = hi - lo
    sw_hi = last_confirmed(pivots(hi, PIV, True), len(c))
    rows = []
    bar = pd.Timedelta(minutes=15)

    def end(j: int) -> pd.Timestamp:
        return d.index[j] + bar                                          # закрытие бара выхода

    for r in zz_lines(d):
        t, side = int(r["t"]), int(r["side"])
        if side > 0 or t <= 0 or t >= len(c) - 1 or not (a[t] > 0) or not (liq[t] >= ADV_MIN):
            continue
        if abs(float(r["slope"])) / a[t] > MAX_SLOPE or trend[t] != -1 or not (1 - buy[t] >= 0.55):
            continue
        if not (rng[t] > 0 and (hi[t] - c[t]) / rng[t] >= 0.5) or sw_hi[t] < 0:
            continue
        level, stop = float(r["line_t"]), hi[sw_hi[t]] + STOP_BUF * a[t]
        row: dict = {"symbol": sym, "t": d.index[t]}
        risk = stop - c[t]
        ok = 0.3 * a[t] <= risk <= 4 * a[t]
        row["cost_market"] = 2 * TAKER * c[t] / risk if ok else np.nan
        for k, ex in EXITS.items():
            r_, j = simulate_bar(o, hi, lo, c, f, t, -1, c[t], stop, a[t], ex, TAKER) if ok else (np.nan, -1)
            row[f"market_{k}"], row[f"market_{k}_end"] = r_, (end(j) if j >= 0 else pd.NaT)
        rr = stop - level
        ok = 0.3 * a[t] <= rr <= 4 * a[t]
        fill = retest_fill(hi, lo, t, -1, level, 3.0 * rr, RETEST) if ok else -1
        row["cost_retest"] = (MAKER + TAKER) * level / rr if ok else np.nan
        for k, ex in EXITS.items():
            j = -1
            if not ok:
                row[f"retest_{k}"] = np.nan
            elif fill < 0:
                row[f"retest_{k}"] = np.inf                              # лимитка не исполнилась
                j = min(t + RETEST, len(c) - 1)                          # место занято, пока висит лимитка
            elif hi[fill] >= stop:
                row[f"retest_{k}"], j = (level - stop - (MAKER + TAKER) * level) / rr, fill
            else:
                row[f"retest_{k}"], j = simulate_bar(o, hi, lo, c, f, fill, -1, level, stop, a[t], ex, MAKER)
            row[f"retest_{k}_end"] = end(j) if j >= 0 else pd.NaT
        rows.append(row)
    return rows


def collect(root: Path, syms: list[str]) -> None:
    rows, advs = [], []
    for s in mine(syms):
        try:
            adv = adv30(root, s)
            a = adv.dropna()
            advs.append(pd.DataFrame({"symbol": s, "day": a.index, "adv": a.to_numpy()}))
            d = tf_frame(root, s, TF)
            if d is None or len(d) < 1000:
                continue
            rows += coin_rows(d, adv.reindex(d.index.floor("D")).to_numpy(), s)
        except Exception as e:
            print(f"  exits15 {s}: пропуск ({e})", flush=True)
    print(f"  exits15: сигналов {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("exits15"), index=False)
    if advs:
        pd.concat(advs, ignore_index=True).to_parquet(part_path("exits15_adv"), index=False)


def month_r(df: pd.DataFrame, col: str, cost: str | None) -> dict[str, float]:
    """R в месяц на сигнал (неисполненная лимитка = 0) по периодам; cost — вычесть ещё раз издержки (x2)."""
    out = {}
    for p, (a, b) in PER.items():
        x = df[(df.t >= a) & (df.t < b)]
        r = x[col].where(np.isfinite(x[col]), 0.0)
        if cost is not None:
            r = r - x[cost].where(np.isfinite(x[col]) & x[col].notna(), 0.0)
        out[p] = float(r.sum() / ((pd.Timestamp(b) - pd.Timestamp(a)).days / 30.44))
    return out


def slots(df: pd.DataFrame, col: str, limit: int = SLOTS) -> pd.Series:
    """Какие сигналы взял бы бот с лимитом limit позиций 15m: место занято от закрытия сигнальной свечи до выхода
    (у ретеста — и пока висит лимитка), одна монета — не больше одной позиции. Сигналы одной свечи — по обороту."""
    end = df[f"{col}_end"]
    order = df.assign(_start=df.t + pd.Timedelta(minutes=15)).sort_values(["_start", "rank"]).index
    taken = pd.Series(False, index=df.index)
    busy: list[tuple[pd.Timestamp, str]] = []
    for i in order:
        if not pd.notna(end[i]) or not pd.notna(df.at[i, col]):
            continue
        start = df.at[i, "t"] + pd.Timedelta(minutes=15)
        busy = [(e, s_) for e, s_ in busy if e > start]
        if len(busy) >= limit or any(s_ == df.at[i, "symbol"] for _, s_ in busy):
            continue
        taken[i] = True
        busy.append((end[i], df.at[i, "symbol"]))
    return taken


def report() -> None:
    tp, ap = all_parts("exits15"), all_parts("exits15_adv")
    print("\n===== EXITS15: 15m, правило бота (пологие шорты, топ-150) — цели 2…8R и трейл, по рынку и на ретесте =====")
    if not tp or not ap:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in tp], ignore_index=True)
    adv = pd.concat([pd.read_parquet(p) for p in ap], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    adv["day"] = pd.to_datetime(adv["day"], utc=True)
    adv["rank"] = adv.groupby("day")["adv"].rank(ascending=False, method="first")
    df["day"] = df.t.dt.floor("D")
    df = df.merge(adv[["symbol", "day", "rank"]], on=["symbol", "day"], how="left")
    df = df[df["rank"] <= TOP]
    print(f"сигналов (топ-{TOP}) {len(df)}")
    for entry in ("market", "retest"):
        print(f"\n  вход {'по рынку' if entry == 'market' else f'на ретесте {RETEST} свечей'} — R в месяц на сигнал "
              f"(IS / VAL / HO / 2026); R на исполненную сделку; при издержках x2")
        base = month_r(df, f"{entry}_{BASE}", None)
        for k in EXITS:
            col = f"{entry}_{k}"
            pm, pm2 = month_r(df, col, None), month_r(df, col, f"cost_{entry}")
            filled = df[np.isfinite(df[col])]
            per_trade = " / ".join(f"{filled[(filled.t >= a) & (filled.t < b)][col].mean():+.3f}"
                                   for a, b in PER.values())
            better = [pm[p] > base[p] for p in PER]
            verdict = ("" if k == BASE else
                       f" → {'ЛУЧШЕ' if all(better) and all(v > 0 for v in pm2.values()) else 'нет'} "
                       f"(выше 3R в {sum(better)} из {len(better)}, x2 > 0 в {sum(v > 0 for v in pm2.values())})")
            print(f"    {NAMES[k]:<26} " + " / ".join(f"{pm[p]:+5.1f}" for p in PER) + f"  | на сделку {per_trade}"
                  f"  | x2 " + " / ".join(f"{pm2[p]:+5.1f}" for p in PER) + verdict)
        if f"{entry}_{BASE}_end" not in df:
            continue
        print(f"\n  то же с лимитом {SLOTS} позиции, как в боте (место занято до выхода; у ретеста — и пока висит лимитка):")
        taken = {k: slots(df, f"{entry}_{k}") for k in EXITS}
        base_l = month_r(df[taken[BASE]], f"{entry}_{BASE}", None)
        for k in EXITS:
            col, sub = f"{entry}_{k}", df[taken[k]]
            pm, pm2 = month_r(sub, col, None), month_r(sub, col, f"cost_{entry}")
            better = [pm[p] > base_l[p] for p in PER]
            verdict = ("" if k == BASE else
                       f" → {'ЛУЧШЕ' if all(better) and all(v > 0 for v in pm2.values()) else 'нет'} "
                       f"(выше 3R в {sum(better)} из {len(better)}, x2 > 0 в {sum(v > 0 for v in pm2.values())})")
            print(f"    {NAMES[k]:<26} " + " / ".join(f"{pm[p]:+5.1f}" for p in PER) +
                  f"  | взято {len(sub) / max(len(df[df[col].notna()]), 1):.0%} сигналов  | x2 " +
                  " / ".join(f"{pm2[p]:+5.1f}" for p in PER) + verdict)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("mode", choices=["collect", "report"])
    ap_.add_argument("--root", default="~/bn")
    ap_.add_argument("--symbols", default="")
    a_ = ap_.parse_args()
    if a_.mode == "collect":
        collect(Path(a_.root).expanduser(), [x for x in a_.symbols.split(",") if x])
    else:
        report()
