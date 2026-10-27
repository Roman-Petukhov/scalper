"""Дневной разворот на малоликвидных альтах (Zaremba и др., IRFA 2021: ликвидные монеты — дневной моментум, неликвидные —
дневной разворот). Межмонетный моментум 7–28 дней у нас не прошёл (docs/research_report.md); дневной горизонт у мелочи
не проверялся.

Все USDT-перпетуалы Binance (с делистнутыми), 1h. Каждый день в 00:00 UTC — ранжирование монет из полосы по доходности
прошлых суток; держим сутки равными весами:
    L    купить 10% худших
    S    продать 10% лучших
    L−S  обе ноги (половина капитала на ногу)
Полосы: $5–50M оборота за 30 прошлых дней — главная; от $50M — контроль (там ждём моментум, то есть минус). Вход:
на закрытии суток (справка) и через 1 ч (главный — без отскока цены закрытия, частого у неликвидных). Издержки за круг
25 б.п. (спред мелочи шире, чем у бота) и ×2 — 50 б.п.; оборот считается полным каждый день. Funding не учтён (сутки,
обе стороны). Принятие (полоса $5–50M, вход через 1 ч, L / S / L−S): Sharpe > 0 в каждом периоде 2021 / IS / VAL / HO,
t >= 3 за всё время и плюс при издержках ×2.

    python -m research.dayrev collect --symbols <1h-all>
    python -m research.dayrev report
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .spikes_full import adv_daily
from .tline import tf_frame

BANDS = {"$5–50M": (5e6, 50e6), "от $50M": (50e6, np.inf)}
DECILE = 0.10
MIN_COINS = 20
COST = 0.0025
LAGS = {"через 1 ч": 1, "на закрытии": 0}


def coin_days(d: pd.DataFrame, adv: pd.Series) -> pd.DataFrame:
    """Цена на 00:00 и 01:00 каждого дня (закрытие часа 23 и 0) и оборот за 30 прошлых дней."""
    c = d["close"]
    at = c.copy()
    at.index = at.index + pd.Timedelta(hours=1)                 # закрытие часа — на его конце
    p0 = at[at.index.hour == 0]
    p1 = at[at.index.hour == 1]
    p0.index, p1.index = p0.index.normalize(), p1.index.normalize()
    out = pd.DataFrame({"p0": p0, "p1": p1})
    out["adv"] = adv.reindex(out.index)
    return out


def collect(root: Path, syms: list[str]) -> None:
    rows = []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, "1h")
            a = adv_daily(root, s)
            if d is None or a is None or len(d) < 24 * 60:
                continue
            x = coin_days(d, a)
            rows.append(x.assign(symbol=s).reset_index(names="day"))
        except Exception as e:
            print(f"  dayrev {s}: пропуск ({e})", flush=True)
    print(f"  dayrev: монет {len(rows)}", flush=True)
    if rows:
        pd.concat(rows, ignore_index=True).to_parquet(part_path("dayrev"), index=False)


def portfolio(p0: pd.DataFrame, p1: pd.DataFrame, adv: pd.DataFrame, band: tuple[float, float],
              lag: int) -> pd.DataFrame:
    """Дневная доходность ног L, S и L−S (до издержек) и число монет в полосе."""
    sig = p0 / p0.shift(1) - 1                                  # прошлые сутки, известны в 00:00 дня
    px = p1 if lag else p0
    fwd = px.shift(-1) / px - 1                                 # держим сутки от входа
    inb = (adv >= band[0]) & (adv < band[1]) & sig.notna() & fwd.notna()
    s = sig.where(inb)
    n = inb.sum(axis=1)
    lo = s.rank(axis=1, pct=True) <= DECILE
    hi = s.rank(axis=1, pct=True, ascending=False) <= DECILE
    long_ = fwd.where(lo).mean(axis=1)
    short = -fwd.where(hi).mean(axis=1)
    out = pd.DataFrame({"L": long_, "S": short, "L−S": (long_ + short) / 2, "n": n})
    return out[out.n >= MIN_COINS]


def stats(r: pd.Series) -> dict:
    if len(r) < 30 or r.std() == 0:
        return {"ann": np.nan, "sh": np.nan, "t": np.nan}
    return {"ann": r.mean() * 365, "sh": r.mean() / r.std() * np.sqrt(365), "t": r.mean() / r.std() * np.sqrt(len(r))}


def report() -> None:
    print("\n===== DAYREV: дневной разворот на малоликвидных альтах (все перпетуалы Binance; в год / Sharpe; издержки "
          f"{COST * 1e4:.0f} б.п. за круг каждый день) =====")
    parts = all_parts("dayrev")
    if not parts:
        print("  dayrev: нет данных")
        return
    x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    x["day"] = pd.to_datetime(x["day"], utc=True).dt.tz_localize(None)
    p0, p1, adv = (x.pivot_table(index="day", columns="symbol", values=k) for k in ("p0", "p1", "adv"))
    p1, adv = p1.reindex_like(p0), adv.reindex_like(p0)
    passed = []
    for bname, band in BANDS.items():
        for lname, lag in LAGS.items():
            pf = portfolio(p0, p1, adv, band, lag)
            print(f"\n  --- {bname}, вход {lname}: дней {len(pf)}, монет в полосе медиана {pf.n.median():.0f} ---")
            for leg in ("L", "S", "L−S"):
                r1, r2 = pf[leg] - COST, pf[leg] - 2 * COST          # у L−S ноги по половине капитала — тот же круг
                cells, ok = [], []
                for p, (a, b) in PER_ALL.items():
                    st = stats(r1[(r1.index >= a) & (r1.index < b)])
                    if not np.isfinite(st["sh"]):
                        continue
                    cells.append(f"{p} {st['ann']:+.0%} / {st['sh']:.2f}")
                    if p != "2020":
                        ok.append(st["sh"] > 0)
                al, a2 = stats(r1), stats(r2)
                gross = pf[leg].mean() * 1e4
                main = bname == "$5–50M" and lag == 1
                good = main and bool(ok) and all(ok) and al["t"] >= 3 and a2["ann"] > 0
                print(f"    {leg}: {('принято' if good else 'нет') if main else 'справка'} — до издержек {gross:+.1f} б.п. "
                      f"в день\n      {'  '.join(cells)}\n      всё время {al['ann']:+.0%} в год, Sharpe {al['sh']:.2f}, "
                      f"t {al['t']:.1f}; издержки ×2 {a2['ann']:+.0%}")
                if good:
                    passed.append(leg)
    print("\n  ИТОГ: " + ("прошли — " + ", ".join(passed) if passed else "дневной разворот у мелочи не окупает издержки"))


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
