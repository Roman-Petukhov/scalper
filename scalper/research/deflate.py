"""Поправка на перебор для бота 4h: дефлированный Sharpe (Bailey, López de Prado 2014) и вероятность переподгонки
PBO методом CSCV (Bailey, Borwein, López de Prado, Zhu 2015).

Бот выбран из соседних вариантов правила, которые мы смотрели. Здесь эти варианты пересчитаны на одних и тех же
сигналах (первые пробои линии зигзага 3 ATR, 4h, оборот от $20M, стоп за свингом, 3R — research/rebreak.py):
    агрессоры ≥ 50 / 52.5 / 55 / 57.5 / 60%
    закрытие в своей половине свечи ≥ 0 / 50 / 60%
    тренд дневок (close против EMA50) — нужен / не нужен
    вход: по рынку, ретест, гибрид «свеча пробоя от k ATR — по рынку, короче — ретест», k = 1.5 / 2.0 / 2.15 / 2.5 / 3.0
Итого 5 × 3 × 2 × 7 = 210 вариантов. Бот: 55%, 50%, тренд нужен, гибрид 2.15; для сравнения — «ретест» (до гибрида).
Ряд варианта — сумма R за месяц при риске 1% на сделку (месяцы без сделок — 0), 2020-06 … 2026-09.

    PSR      вероятность, что истинный Sharpe > 0, с поправкой на асимметрию и хвосты
    DSR      то же, но порог — ожидаемый максимум Sharpe среди N случайных вариантов с разбросом Sharpe как у сетки;
             N = 210 (сетка) и 1000 (грубо: всё, что мы пробовали за проект, включая отвергнутые идеи)
    PBO      CSCV: 16 блоков месяцев, все 12870 разбиений пополам; доля разбиений, где лучший на одной половине вариант
             на другой оказывается ниже медианы. < 0.1 — подгонки почти нет, около 0.5 — выбор лучшего ничего не значит
    t        t-статистика среднего месяца и порог Бонферрони для N проверок (Harvey, Liu 2015)

    python -m research.deflate collect --symbols <1h-qualified>
    python -m research.deflate report
"""
from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30
from .rebreak import coin_rows
from .shard import all_parts, mine, part_path
from .smc import _atr
from .tline import tf_frame

AGGR = (0.50, 0.525, 0.55, 0.575, 0.60)
LOC = (0.0, 0.5, 0.6)
TREND = (True, False)
ENTRY = ("market", "retest", 1.5, 2.0, 2.15, 2.5, 3.0)
BOT = (0.55, 0.5, True, 2.15)
RETEST = (0.55, 0.5, True, "retest")
START, END = "2020-06", "2026-09"
RISK = 0.01
BLOCKS = 16
EULER = 0.5772156649
N = NormalDist()


def collect(root: Path, syms: list[str]) -> None:
    parts = []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, "4h")
            if d is None or len(d) < 500:
                continue
            liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
            x = coin_rows(d, liq, s, "4h", ("market", "retest"))
            if not len(x):
                continue
            x = x[x.kind == "first"].copy()
            rng = ((d["high"] - d["low"]) / _atr(d)).to_numpy()
            x["rng_atr"] = rng[d.index.get_indexer(x["t"])]
            parts.append(x[["symbol", "t", "side", "adv", "with_trend", "aggr", "close_loc", "rng_atr", "entry", "R3"]])
        except Exception as e:
            print(f"  deflate {s}: пропуск ({e})", flush=True)
    print(f"  deflate: монет со сделками {len(parts)}", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path("deflate"), index=False)


def variant_trades(df: pd.DataFrame, aggr: float, loc: float, trend: bool, entry) -> pd.DataFrame:
    """Сделки варианта: по сигналу одна строка (рынок или ретест; неисполненный ретест — сделки нет)."""
    m = (df.aggr >= aggr) & (df.close_loc >= loc)
    if trend:
        m &= df.with_trend
    x = df[m]
    if entry in ("market", "retest"):
        return x[x.entry == entry]
    use = np.where(x.rng_atr >= entry, "market", "retest")
    return x[x.entry.to_numpy() == use]


def _key(v: tuple) -> str:
    aggr, loc, trend, entry = v
    return f"агр {aggr:.1%} / закр {loc:.0%} / {'тренд' if trend else 'без тренда'} / {entry}"


def monthly(t: pd.DataFrame, idx: pd.PeriodIndex) -> pd.Series:
    return (t.groupby(t.t.dt.tz_localize(None).dt.to_period("M")).R3.sum() * RISK).reindex(idx, fill_value=0.0)


def _moments(r: np.ndarray) -> tuple[float, float, float]:
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0, 0.0, 3.0
    z = (r - r.mean()) / r.std(ddof=0)
    return r.mean() / sd, float((z ** 3).mean()), float((z ** 4).mean())


def psr(r: np.ndarray, sr0: float) -> float:
    sr, sk, ku = _moments(r)
    den = np.sqrt(max(1e-12, 1 - sk * sr + (ku - 1) / 4 * sr ** 2))
    return N.cdf((sr - sr0) * np.sqrt(len(r) - 1) / den)


def expected_max_sr(var_sr: float, n: int) -> float:
    """Ожидаемый максимум Sharpe среди n независимых вариантов с нулевым истинным Sharpe и дисперсией var_sr."""
    return np.sqrt(var_sr) * ((1 - EULER) * N.inv_cdf(1 - 1 / n) + EULER * N.inv_cdf(1 - 1 / (n * np.e)))


def pbo(mat: np.ndarray, blocks: int = BLOCKS) -> tuple[float, float]:
    """CSCV: mat — месяцы × варианты. Возвращает PBO и медиану относительного места лучшего IS-варианта вне выборки."""
    t = mat.shape[0] - mat.shape[0] % blocks
    parts = np.array_split(np.arange(t), blocks)
    logits = []
    for comb in itertools.combinations(range(blocks), blocks // 2):
        is_i = np.concatenate([parts[i] for i in comb])
        oos_i = np.concatenate([parts[i] for i in range(blocks) if i not in comb])
        sr_is = _sr_cols(mat[is_i])
        sr_oos = _sr_cols(mat[oos_i])
        best = int(np.argmax(sr_is))
        rank = (sr_oos < sr_oos[best]).sum() + 0.5 * ((sr_oos == sr_oos[best]).sum() - 1)
        w = (rank + 1) / (mat.shape[1] + 1)
        logits.append(np.log(w / (1 - w)))
    lg = np.array(logits)
    return float((lg <= 0).mean()), float(1 / (1 + np.exp(-np.median(lg))))


def _sr_cols(m: np.ndarray) -> np.ndarray:
    sd = m.std(axis=0, ddof=1)
    return np.where(sd > 0, m.mean(axis=0) / np.where(sd > 0, sd, 1), 0.0)


def _line(name: str, r: np.ndarray, var_sr: float) -> str:
    sr, sk, ku = _moments(r)
    t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
    out = (f"  {name}: {r.mean() * 100:+.2f}% в месяц, Sharpe год {sr * np.sqrt(12):.2f}, асимметрия {sk:+.2f}, "
           f"эксцесс {ku:.1f}, t {t:.1f}; PSR(>0) {psr(r, 0.0):.3f}")
    for n in (210, 1000):
        bonf = N.inv_cdf(1 - 0.05 / (2 * n))
        out += (f"; N={n}: порог Sharpe год {expected_max_sr(var_sr, n) * np.sqrt(12):.2f}, DSR "
                f"{psr(r, expected_max_sr(var_sr, n)):.3f}, t Бонферрони {bonf:.1f}")
    return out


def report() -> None:
    parts = all_parts("deflate")
    print("\n===== DEFLATE: поправка на перебор для бота 4h — дефлированный Sharpe и PBO (CSCV) по 210 соседним "
          "вариантам правила; месяцы при риске 1% на сделку =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df[(df.adv >= ADV_MIN) & (df.t >= pd.Timestamp(START, tz="UTC"))].dropna(subset=["R3"])
    idx = pd.period_range(START, END, freq="M")
    grid = list(itertools.product(AGGR, LOC, TREND, ENTRY))
    cols, n_tr = {}, {}
    for v in grid:
        tr = variant_trades(df, *v)
        cols[_key(v)] = monthly(tr, idx).to_numpy()
        n_tr[v] = len(tr)
    mat = pd.DataFrame(cols, index=idx)
    srs = _sr_cols(mat.to_numpy())
    var_sr = float(np.var(srs, ddof=1))
    print(f"месяцев {len(idx)}, вариантов {len(grid)}; Sharpe год по сетке: медиана {np.median(srs) * np.sqrt(12):.2f}, "
          f"p10 {np.quantile(srs, 0.1) * np.sqrt(12):.2f}, p90 {np.quantile(srs, 0.9) * np.sqrt(12):.2f}, "
          f"макс {srs.max() * np.sqrt(12):.2f}; место бота {int((srs > srs[grid.index(BOT)]).sum()) + 1} из {len(grid)}")
    for nm, v in (("бот (гибрид 2.15)", BOT), ("бот с ретестом", RETEST)):
        print(_line(f"{nm}, {n_tr[v]} сделок", mat[_key(v)].to_numpy(), var_sr))
    best = grid[int(np.argmax(srs))]
    print(_line(f"лучший в сетке {_key(best)}, {n_tr[best]} сделок", mat[_key(best)].to_numpy(), var_sr))
    p, w = pbo(mat.to_numpy())
    print(f"  PBO по всей сетке: {p:.2f} (медианное место лучшего на одной половине — на другой: {w:.0%} снизу)")
    ent = [v for v in grid if v[:3] == BOT[:3]]
    p, w = pbo(mat[[_key(v) for v in ent]].to_numpy())
    print(f"  PBO только по способу входа (7 вариантов при фильтрах бота): {p:.2f} ({w:.0%})")
    sub = [v for v in grid if v[3] == BOT[3]]
    p, w = pbo(mat[[_key(v) for v in sub]].to_numpy())
    print(f"  PBO только по фильтрам (30 вариантов при гибриде 2.15): {p:.2f} ({w:.0%})")
    print("  по способу входа при фильтрах бота, Sharpe год / % в месяц: " + "; ".join(
        f"{v[3]} {srs[grid.index(v)] * np.sqrt(12):.2f} / {mat[_key(v)].mean() * 100:+.2f}" for v in ent))
    yy = mat[_key(BOT)].groupby(mat.index.year).sum()
    print("  бот по годам, % при риске 1%: " + ", ".join(f"{k} {v * 100:+.0f}" for k, v in yy.items()))


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
