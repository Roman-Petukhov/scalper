"""Шорт после пампа монеты вне топ-50 по обороту: мелкие монеты после резкого роста чаще откатываются, чем растут дальше.

Повод: дневной тренд на монетах 51–200 хуже, чем на топ-20 (research/dtiers.py) — рост мелочи это короткие пампы. Вопрос:
заработать на сдувании пампа. Уже было: шорт после листинга (research/newlist.py) +1.7% на сделку, но эффект затухает,
максимальный рост против шорта +475%.

Правило задано до запуска (дневные свечи UTC, решение на закрытии дня t):
    вселенная   все USDT-перпетуалы Binance, включая делистнутые; оборот за 30 прошлых дней от $5M и место по обороту
                ниже 50 (контроль — топ-50)
    памп        рост за 3 дня r3 = close_t / close_{t-3} − 1 не меньше 25% и не меньше 3 ATR14 (ATR до пампа, на t−3,
                в долях цены) и в верхних 5% рынка в этот день
    вход        шорт по open дня t+1; стоп — над максимумом пампа (high за t−2…t) + 0.5 ATR; если open уже выше стопа —
                пропуск. Одна позиция на монету, новый сигнал — после выхода
    выход       стоп (taker + проскальзывание 0.5% — сквиз), цель — середина пампа (close_{t−3} + максимум) / 2 лимиткой
                (maker), иначе по close через H = 5 дней (taker). Стоп и цель в один день — считаем стоп
    funding     шорт получает положительную ставку (сумма за дни сделки); делистинг в сделке — выход по последнему close
Фильтры, заданные заранее: F — funding дня t выше 0.06% (вдвое выше базовых 0.01% × 3), C — день t закрылся в нижней
половине своего диапазона (выдох). Проверки: H = 3 / 7, без цели, издержки ×2, контроль на топ-50.
Метрики: R на сделку (1R — от входа до стопа) по периодам 2020 / 2021 / IS / VAL / HO / 2026, худшая сделка, среднее без 5%
лучших, рост против шорта; счёт — риск 0.5% на сделку, не больше 3 позиций сразу.
Принятие: плюс во всех шести периодах и без 5% лучших сделок.

    python -m research.pumpfade collect --symbols <все перпетуалы>   (по частям)
    python -m research.pumpfade report
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import adv30
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .tline import MAKER, TAKER, tf_frame

ADV_FLOOR = 5e6
TOP_EXCLUDE = 50
R3_MIN = 0.25
R3_ATR = 3.0
R3_PCT = 0.95
STOP_ATR = 0.5
SLIP = 0.005
FUND_HIGH = 0.0006
RISK = 0.005
MAX_OPEN = 3


@dataclass(frozen=True)
class Rule:
    hold: int = 5
    target: bool = True
    fund: bool = False
    exhaust: bool = False
    top: bool = False             # контроль: топ-50 вместо монет ниже 50-го места
    cost_k: float = 1.0


def collect(root: Path, syms: list[str]) -> None:
    out = []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, "1d")
            if d is None or len(d) < 30:
                continue
            a = adv30(root, s).reindex(d.index)
            out.append(pd.DataFrame({"symbol": s, "day": d.index, "open": d["open"].to_numpy(),
                                     "high": d["high"].to_numpy(), "low": d["low"].to_numpy(),
                                     "close": d["close"].to_numpy(), "funding": d["funding"].to_numpy(),
                                     "adv": a.to_numpy()}))
        except Exception as e:
            print(f"  pumpfade {s}: пропуск ({e})", flush=True)
    print(f"  pumpfade: монет {len(out)}", flush=True)
    if out:
        pd.concat(out, ignore_index=True).to_parquet(part_path("pumpfade"), index=False)


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Признаки дня t по каждой монете (только прошлое и сам день t) и место по обороту среди всех монет."""
    df = df.sort_values(["symbol", "day"]).reset_index(drop=True)
    g = df.groupby("symbol", sort=False)
    pc = g["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    atr = tr.groupby(df["symbol"]).transform(lambda x: x.rolling(14, min_periods=14).mean())
    c3 = g["close"].shift(3)
    df["c3"] = c3
    df["atr_pre"] = atr.groupby(df["symbol"]).shift(3)
    df["r3"] = df["close"] / c3 - 1
    df["pump_high"] = g["high"].transform(lambda x: x.rolling(3, min_periods=3).max())
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["close_loc"] = (df["close"] - df["low"]) / rng
    df["adv_rank"] = df.groupby("day")["adv"].rank(ascending=False, method="first")
    ok = df["adv"] >= ADV_FLOOR
    df["r3_pct"] = df["r3"].where(ok).groupby(df["day"]).rank(pct=True)
    return df


def signals(f: pd.DataFrame, rule: Rule) -> pd.Series:
    small = f["adv_rank"] <= TOP_EXCLUDE if rule.top else f["adv_rank"] > TOP_EXCLUDE
    m = ((f["adv"] >= ADV_FLOOR) & small & (f["r3"] >= R3_MIN) & (f["r3"] >= R3_ATR * f["atr_pre"] / f["c3"])
         & (f["r3_pct"] >= R3_PCT))
    if rule.fund:
        m &= f["funding"] > FUND_HIGH
    if rule.exhaust:
        m &= f["close_loc"] < 0.5
    return m.fillna(False)


def trades(f: pd.DataFrame, rule: Rule) -> pd.DataFrame:
    sig = signals(f, rule).to_numpy()
    rows = []
    for sym, idx in f.groupby("symbol", sort=False).indices.items():
        o, h, l_, c = (f[k].to_numpy()[idx] for k in ("open", "high", "low", "close"))
        fu, days = f["funding"].to_numpy()[idx], f["day"].to_numpy()[idx]
        c3, atr, ph = (f[k].to_numpy()[idx] for k in ("c3", "atr_pre", "pump_high"))
        s, n, i = sig[idx], len(idx), 0
        while i < n - 1:
            if not s[i]:
                i += 1
                continue
            e = i + 1
            entry, stop, mid = o[e], ph[i] + STOP_ATR * atr[i], (c3[i] + ph[i]) / 2
            if not entry < stop:
                i += 1
                continue
            use_tgt = rule.target and mid < entry
            last = min(n - 1, e + rule.hold - 1)
            j, kind, px = e, "time", c[last]
            while True:
                if h[j] >= stop:
                    kind, px = "stop", stop * (1 + SLIP)
                    break
                if use_tgt and l_[j] <= mid:
                    kind, px = "target", mid
                    break
                if j == last:
                    px = c[j]
                    break
                j += 1
            fee = TAKER + {"stop": TAKER, "target": MAKER, "time": TAKER}[kind]
            pnl = (entry - px) / entry - fee * rule.cost_k + fu[e:j + 1].sum()
            risk = (stop - entry) / entry
            rows.append({"symbol": sym, "t": pd.Timestamp(days[e]), "exit": pd.Timestamp(days[j]), "kind": kind,
                         "pnl": pnl, "risk": risk, "R": pnl / risk, "r3": (c[i] / c3[i] - 1),
                         "mae": h[e:j + 1].max() / entry - 1})
            i = j + 1
    return pd.DataFrame(rows, columns=["symbol", "t", "exit", "kind", "pnl", "risk", "R", "r3", "mae"])


def account(t: pd.DataFrame, risk: float = RISK, max_open: int = MAX_OPEN) -> pd.Series:
    """Месячный результат счёта: риск на сделку, не больше max_open позиций (кто раньше — тот и вошёл)."""
    taken, open_exits = [], []
    for r in t.sort_values("t").itertuples():
        open_exits = [x for x in open_exits if x >= r.t]
        if len(open_exits) < max_open:
            open_exits.append(r.exit)
            taken.append((r.exit, r.R * risk))
    if not taken:
        return pd.Series(dtype=float)
    s = pd.DataFrame(taken, columns=["exit", "ret"])
    return s.groupby(pd.to_datetime(s["exit"]).dt.to_period("M"))["ret"].sum()


def _cell(z: pd.DataFrame) -> str:
    if len(z) < 10:
        return f"n={len(z)}"
    m = z.groupby(z.t.dt.to_period("M")).R.sum()
    months = (z.t.max().to_period("M") - z.t.min().to_period("M")).n + 1
    t = m.mean() / (m.std() / np.sqrt(len(m))) if len(m) > 2 and m.std() > 0 else np.nan
    return f"{z.R.mean():+.2f}R (t_мес {t:.1f}, плюс {np.mean(z.R > 0):.0%}, n={len(z)}, {len(z) / months:.1f}/мес)"


def _tails(z: pd.DataFrame) -> str:
    if len(z) < 20:
        return ""
    cut = z.R.quantile(0.95)
    return (f"худшая {z.R.min():+.1f}R, без 5% лучших {z.R[z.R < cut].mean():+.2f}R, рост против шорта p95 "
            f"{z.mae.quantile(0.95):+.0%} / макс {z.mae.max():+.0%}, стопом {np.mean(z.kind == 'stop'):.0%}, "
            f"целью {np.mean(z.kind == 'target'):.0%}, риск (до стопа) медиана {z.risk.median():.0%}")


def _mstats(m: pd.Series) -> str:
    if len(m) < 6 or m.std() == 0:
        return "—"
    eq = (1 + m).cumprod()
    return (f"{m.mean() * 12:+.0%} год, Sharpe {m.mean() / m.std() * np.sqrt(12):.2f}, просадка "
            f"{(1 - eq / eq.cummax()).max():.0%}, худший мес. {m.min():+.1%}, убыт. мес. {(m < 0).mean():.0%}")


VARIANTS = {
    "A: памп, стоп над максимумом, цель — середина, 5 дней (основной)": Rule(),
    "A + F (funding высокий)": Rule(fund=True),
    "A + C (закрытие в нижней половине)": Rule(exhaust=True),
    "A + F + C": Rule(fund=True, exhaust=True),
    "  A, держать 3 дня": Rule(hold=3),
    "  A, держать 7 дней": Rule(hold=7),
    "  A, без цели": Rule(target=False),
    "  A, издержки x2": Rule(cost_k=2.0),
    "контроль: A на топ-50": Rule(top=True),
}


def report() -> None:
    parts = all_parts("pumpfade")
    print("\n===== PUMPFADE: шорт после пампа (3 дня +25% и 3 ATR, верхние 5% рынка) на монетах ниже 50-го места по "
          "обороту; правило задано до запуска =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["day"] = pd.to_datetime(df["day"], utc=True).dt.tz_localize(None)
    df = df.drop_duplicates(["symbol", "day"])
    f = features(df)
    print(f"монет {f.symbol.nunique()}, дней {f.day.nunique()}, с {f.day.min():%Y-%m-%d}")
    for nm, rule in VARIANTS.items():
        z = trades(f, rule)
        print(f"\n  {nm}")
        if z.empty:
            print("    сделок нет")
            continue
        for p, (a, b) in PER_ALL.items():
            print(f"    {p:>4}: {_cell(z[(z.t >= a) & (z.t < b)])}")
        print(f"    всё: {_cell(z)}; {_tails(z)}")
        y = z.groupby(z.t.dt.year).R.mean()
        print("    по годам R: " + ", ".join(f"{k} {v:+.2f}" for k, v in y.items()))
        if nm.startswith("A") or nm.startswith("контроль"):
            m = account(z)
            print(f"    счёт (риск {RISK:.1%}, до {MAX_OPEN} позиций): {_mstats(m)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 300)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x])
    else:
        report()
