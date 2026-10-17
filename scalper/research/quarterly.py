"""Квартальная пересдача: держатся ли стратегии на свежих данных. Одна команда — TASKS=quarterly (research/job.sh).

Данные — по последний закрытый месяц (RESEARCH_END), окно проверки — последние 12 месяцев. Проверки (правила бота):
    бот 4h            пробой линии зигзага, все фильтры, агрессоры ≥ 55%, ретест, 3R, долив (research/topcrowd.py)
    фильтр агрессоров 4h: разница R сделок с агрессорами ≥ 55% и без них
    бот 15m           пологие шорты, топ-150, ретест 12 свечей и по рынку (research/retest15.py)
    тренд             «сильнейшие 7 + долив» при притоке стейблкоинов (research/dboost2.py, бумажный счёт бота)
Норма — половина бэктеста (на реале вне выборки остаётся 50–57%, docs/knowledge.md; та же норма, что у детектора
здоровья в журнале, trader/domain/journal.py). z — насколько окно ниже нормы в стандартных ошибках:
    держится              z ≥ −1 и само окно в плюсе (иначе — не лучше «под наблюдением»: у года мало точности)
    под наблюдением       −2 ≤ z < −1
    стареет               z < −2 (окно значимо хуже нормы)
Итог — худший из проверок. Пишется в лог и в QUARTERLY_OUT (markdown, по умолчанию ../out/quarterly.md).

    python -m research.quarterly collect --symbols <1h-qualified> --symbols15 <15m-wide>
    python -m research.quarterly report
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trader.domain.journal import BACKTEST_R, EXPECT_SHARE
from trader.domain.models import Timeframe

from . import retest15
from .broad import adv30
from .dboost2 import book, inflow_gate, returns
from .shard import all_parts, mine, part_path
from .tline import tf_frame
from .topcrowd import coin_rows

WINDOW_DAYS = 365
TREND_SHARPE = 1.27          # бэктест книги с 2022 (trader/domain/trend.py RULES, research/dboost2.py «приток»)
LEVELS = ("держится", "под наблюдением", "стареет")
OUT = os.environ.get("QUARTERLY_OUT", "../out/quarterly.md")


@dataclass(frozen=True)
class Check:
    name: str
    live: str                 # что показало окно
    norm: str                 # с чем сравниваем
    z: float
    value: float              # сама метрика окна (R, разница R, Sharpe): ниже нуля — не лучше «под наблюдением»
    history: str = ""         # для контекста: по годам

    @property
    def level(self) -> str:
        v = verdict(self.z)
        return LEVELS[1] if v == LEVELS[0] and not self.value > 0 else v


def verdict(z: float) -> str:
    if not np.isfinite(z):
        return "под наблюдением"
    return LEVELS[0] if z >= -1 else LEVELS[1] if z >= -2 else LEVELS[2]


def mean_se(r: np.ndarray, days: np.ndarray) -> tuple[float, float]:
    """Среднее и его стандартная ошибка с группировкой по дням (сделки одного дня связаны рынком)."""
    r = np.asarray(r, dtype="float64")
    if len(r) < 2:
        return (float(r.mean()) if len(r) else np.nan), np.nan
    m = float(r.mean())
    g = pd.Series(r - m).groupby(np.asarray(days)).sum().to_numpy()
    return m, float(np.sqrt((g ** 2).sum()) / len(r))


def trades_check(name: str, t: pd.Series, r: pd.Series, start: pd.Timestamp, norm: float) -> Check:
    ok = r.notna() & np.isfinite(r)
    t, r = t[ok], r[ok].astype("float64")
    w = t >= start
    m, se = mean_se(r[w].to_numpy(), t[w].dt.floor("D").to_numpy())
    z = (m - norm) / se if se and se > 0 else np.nan
    years = " · ".join(f"{y}: {g.mean():+.2f}R ({len(g)})" for y, g in r.groupby(t.dt.year))
    return Check(name, f"{m:+.2f}R на сделку, сделок {int(w.sum())}", f"{norm:+.2f}R", z, m, years)


def diff_check(name: str, df: pd.DataFrame, start: pd.Timestamp) -> Check:
    """Разница R «агрессоры ≥ 55%» минус «< 55%»; норма — половина разницы до окна."""
    d = df[df.R.notna()]
    past, win = d[d.t < start], d[d.t >= start]

    def gap(x: pd.DataFrame) -> tuple[float, float]:
        a, b = x[x.aggr], x[~x.aggr]
        ma, sa = mean_se(a.R.to_numpy(), a.t.dt.floor("D").to_numpy())
        mb, sb = mean_se(b.R.to_numpy(), b.t.dt.floor("D").to_numpy())
        return ma - mb, float(np.hypot(sa, sb))

    hist, _ = gap(past)
    live, se = gap(win)
    norm = EXPECT_SHARE * hist
    z = (live - norm) / se if se > 0 else np.nan
    return Check(name, f"{live:+.2f}R разницы", f"{norm:+.2f}R (половина {hist:+.2f}R до окна)", z, live)


def sharpe_check(name: str, r: pd.Series, start: pd.Timestamp, norm: float) -> Check:
    """Sharpe дневных доходностей в окне против нормы; ошибка Sharpe за T лет ≈ 1 / √T."""
    w = r[r.index >= start]
    years = len(w) / 365
    sh = float(w.mean() / w.std() * np.sqrt(365)) if w.std() > 0 else 0.0
    eq = (1 + w).cumprod()
    dd = float((1 - eq / eq.cummax()).max()) if len(eq) else 0.0
    z = (sh - norm) * np.sqrt(years) if years > 0 else np.nan
    hist = " · ".join(f"{y}: {(1 + g).prod() - 1:+.0%}" for y, g in r.groupby(r.index.year))
    return Check(name, f"Sharpe {sh:.2f}, {(1 + w).prod() - 1:+.0%} за окно, просадка {dd:.0%}", f"Sharpe {norm:.2f}",
                 z, sh, hist)


def render(checks: list[Check], end: pd.Timestamp) -> str:
    worst = max((LEVELS.index(c.level) for c in checks), default=1)
    lines = [f"# Квартальная пересдача: {LEVELS[worst].upper()}",
             f"Данные по {end:%Y-%m-%d}, окно — 12 месяцев. Норма — половина бэктеста.", "",
             "| Проверка | Окно | Норма | z | Итог |", "|---|---|---|---|---|"]
    lines += [f"| {c.name} | {c.live} | {c.norm} | {c.z:+.1f} | {c.level} |" for c in checks]
    lines += ["", "По годам:"] + [f"- {c.name}: {c.history}" for c in checks if c.history]
    lines += ["", "Правило: z ≥ −1 и окно в плюсе — держится; −2 ≤ z < −1 или окно в минусе — под наблюдением; "
              "z < −2 — стареет (значимо хуже нормы)."]
    return "\n".join(lines)


def collect(root: Path, syms: list[str], syms15: list[str]) -> None:
    rows, days = [], []
    for s in mine(syms):
        try:
            a = adv30(root, s)
            dd = tf_frame(root, s, "1d")
            if dd is not None and len(dd):
                days.append(pd.DataFrame({"symbol": s, "day": dd.index, "close": dd["close"].to_numpy(),
                                          "funding": dd["funding"].to_numpy(), "adv": a.reindex(dd.index).to_numpy()}))
            d = tf_frame(root, s, "4h")
            if d is None or len(d) < 500:
                continue
            rows += coin_rows(d, a.reindex(d.index.floor("D")).to_numpy(), s, "4h", np.zeros(len(d)))
        except Exception as e:
            print(f"  quarterly {s}: пропуск ({e})", flush=True)
    print(f"  quarterly: пробоев 4h {len(rows)}, монет с дневками {len(days)}", flush=True)
    if rows:
        pd.DataFrame(rows).drop(columns=["dev", "tf"]).to_parquet(part_path("quarterly4h"), index=False)
    if days:
        pd.concat(days, ignore_index=True).to_parquet(part_path("quarterly_daily"), index=False)
    if syms15:
        retest15.collect(root, syms15)


def _read(name: str) -> pd.DataFrame | None:
    parts = all_parts(name)
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True) if parts else None


def report() -> None:
    print("\n===== QUARTERLY: держатся ли бот 4h, фильтр агрессоров, бот 15m и тренд на последних 12 месяцах =====")
    dd = _read("quarterly_daily")
    if dd is None:
        print("частей нет")
        return
    dd["day"] = pd.to_datetime(dd["day"], utc=True).dt.tz_localize(None)
    close = dd.pivot_table(index="day", columns="symbol", values="close").sort_index()
    end = close.index[-1]
    start = end - pd.Timedelta(days=WINDOW_DAYS - 1)
    start_utc = start.tz_localize("UTC")
    checks: list[Check] = []

    h4 = _read("quarterly4h")
    if h4 is not None:
        h4["t"] = pd.to_datetime(h4["t"], utc=True)
        bot = h4[h4.aggr]
        checks.append(trades_check("бот 4h", bot.t, bot.R, start_utc, EXPECT_SHARE * BACKTEST_R[Timeframe.H4]))
        checks.append(diff_check("фильтр агрессоров 4h", h4, start_utc))

    m15, adv15 = _read("retest15"), _read("retest15_adv")
    if m15 is not None and adv15 is not None:
        m15["t"] = pd.to_datetime(m15["t"], utc=True)
        adv15["day"] = pd.to_datetime(adv15["day"], utc=True)
        adv15["rank"] = adv15.groupby("day")["adv"].rank(ascending=False, method="first")
        m15["day"] = m15.t.dt.floor("D")
        m15 = m15.merge(adv15[["symbol", "day", "rank"]], on=["symbol", "day"], how="left")
        m15 = m15[m15["rank"] <= retest15.TOP]
        norm15 = EXPECT_SHARE * BACKTEST_R[Timeframe.M15]
        checks.append(trades_check("бот 15m, ретест 12", m15.t, m15["retest12"], start_utc, norm15))
        checks.append(trades_check("бот 15m, по рынку", m15.t, m15["market"], start_utc, norm15))

    fund = dd.pivot_table(index="day", columns="symbol", values="funding").reindex_like(close)
    adv = dd.pivot_table(index="day", columns="symbol", values="adv").reindex_like(close)
    w = book(close, adv)
    ext = _read("extreg")
    if ext is not None:
        ext["day"] = pd.to_datetime(ext["day"], utc=True).dt.tz_localize(None)
        w = w.mul(inflow_gate(ext.groupby("day")["stable"].last().dropna(), close.index).astype(float), axis=0)
        name = "тренд (приток)"
    else:
        name = "тренд (без данных притока — без включателя)"
    r = returns(w, close, fund)
    checks.append(sharpe_check(name, r[r.index >= "2022-01-01"], start, EXPECT_SHARE * TREND_SHARPE))

    text = render(checks, end)
    print(text)
    try:
        Path(OUT).parent.mkdir(parents=True, exist_ok=True)
        Path(OUT).write_text(text + "\n", encoding="utf-8")
    except OSError as e:
        print(f"  quarterly: не записал {OUT} ({e})")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--symbols15", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x],
                [x for x in a.symbols15.split(",") if x])
    else:
        report()
