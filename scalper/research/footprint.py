"""Профиль агрессоров внутри свечи пробоя бота 4h — «кластеры» по 15-минутным свечам (по просьбе владельца: как
используют стакан и кластеры в TigerTrade / ATAS и есть ли там что-то для фильтра агрессоров).

Сделки — правило бота 4h без изменений (research/noline.py: линия зигзага 3 ATR, агрессоры от 55%, закрытие в своей
половине, тренд, ретест, стоп за свингом, 3R). Свеча пробоя делится на 16 свечей 15m (нет 15m — на 4 свечи 1h).
«Наша сторона» — покупатели для лонга, продавцы для шорта; уровень — значение линии на свече пробоя (цена ретеста).

Признаки и ожидаемое направление заданы до запуска:
  F1 конец свечи      доля агрессоров нашей стороны в последней четверти свечи пробоя: < 50% / 50–55% / ≥ 55%;
                      ждём, что «≥ 55%» (давят к закрытию) лучше
  F2 выдохлись        доля в последней четверти ниже, чем в первых трёх четвертях; ждём, что хуже
  F3 давили у линии   до первой подсвечи, закрывшейся за линией, прошло ≥ 50% объёма свечи и доля нашей стороны в
                      этой части ≥ 55% — агрессоры давили, лимитки держали, потом уровень сдался; ждём, что лучше
  F4 POC за линией    подсвеча с наибольшим объёмом закрылась за линией (типичная цена (H+L+C)/3 за уровнем);
                      ждём, что лучше
Принятие: ожидаемая группа лучше остальных в каждом периоде (2020 / 2021 / IS / VAL / HO), где в обеих группах
≥ 20 сделок, и разница за всё время с t ≥ 2. Тогда — кандидат в фильтр (проверка на 15m-боте и живьём отдельно).

    python -m research.footprint collect --symbols <1h-qualified>
    python -m research.footprint report
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import adv30
from .noline import TF, coin_trades
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _cell
from .tline import tf_frame

MIN_N = 20
SPLITS = {
    "F1": ("конец свечи: агрессоры нашей стороны в последней четверти", "≥ 55%"),
    "F2": ("выдохлись к закрытию: последняя четверть слабее первых трёх", "нет"),
    "F3": ("давили у линии: ≥ 50% объёма до прохода, нашей стороны ≥ 55%", "да"),
    "F4": ("наибольший объём свечи — за линией", "да"),
}


def profile(sub: pd.DataFrame, side: int, level: float) -> dict:
    """Признаки F1–F4 по подсвечам свечи пробоя (open/high/low/close/volume/taker_buy_volume)."""
    v = sub["volume"].to_numpy(dtype="float64")
    buy = sub["taker_buy_volume"].to_numpy(dtype="float64")
    own = buy if side > 0 else v - buy
    hi, lo, c = (sub[k].to_numpy(dtype="float64") for k in ("high", "low", "close"))
    n = len(v)
    q = max(1, n // 4)
    out: dict = {"n_sub": n}
    if n < 4 or not v.sum() > 0:
        return out
    last = own[-q:].sum() / v[-q:].sum() if v[-q:].sum() > 0 else np.nan
    early = own[:-q].sum() / v[:-q].sum() if v[:-q].sum() > 0 else np.nan
    out["own_last"] = last
    out["F1"] = "< 50%" if last < 0.50 else "50–55%" if last < 0.55 else "≥ 55%"
    out["F2"] = "да" if last < early else "нет"
    beyond = side * (c - level) > 0
    k = int(np.argmax(beyond)) if beyond.any() else n
    pre_v = v[:k].sum()
    pre_own = own[:k].sum() / pre_v if pre_v > 0 else np.nan
    out["F3"] = "да" if pre_v >= 0.5 * v.sum() and pre_own >= 0.55 else "нет"
    m = int(np.argmax(v))
    out["F4"] = "да" if side * ((hi[m] + lo[m] + c[m]) / 3 - level) > 0 else "нет"
    return out


def coin_rows(d4: pd.DataFrame, sub: pd.DataFrame, liq: np.ndarray, sym: str, tf_sub: str) -> list[dict]:
    x = coin_trades(d4, liq, sym)
    if not len(x):
        return []
    x = x[(x.trig == "line") & (x.entry == "retest")]
    rows = []
    step = pd.Timedelta(TF)
    for r in x.itertuples(index=False):
        s = sub[(sub.index >= r.t) & (sub.index < r.t + step)]
        rows.append({"symbol": sym, "t": r.t, "side": int(r.side), "R": float(r.R3), "sub_tf": tf_sub,
                     **profile(s, int(r.side), float(r.px))})
    return rows


def collect(root: Path, syms: list[str]) -> None:
    rows = []
    for s in mine(syms):
        try:
            d4 = tf_frame(root, s, TF)
            if d4 is None or len(d4) < 500:
                continue
            sub, tf_sub = (tf_frame(root, s, "15m"), "15m") if (root / f"{s}-15m.parquet").exists() else \
                (tf_frame(root, s, "1h"), "1h")
            liq = adv30(root, s).reindex(d4.index.floor("D")).to_numpy()
            rows += coin_rows(d4, sub, liq, s, tf_sub)
        except Exception as e:
            print(f"  footprint {s}: пропуск ({e})", flush=True)
    print(f"  footprint: сделок {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("footprint"), index=False)


def diff_t(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return np.nan
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se if se > 0 else np.nan


def verdict(x: pd.DataFrame, key: str, good: str) -> tuple[str, bool]:
    """Ожидаемая группа против остальных по периодам; (строка, принято)."""
    cells, ok = [], []
    for p, (a, b) in PER_ALL.items():
        m = x[(x.t >= a) & (x.t < b) & x[key].notna()]
        g, r = m[m[key] == good].R.to_numpy(), m[m[key] != good].R.to_numpy()
        if len(g) >= MIN_N and len(r) >= MIN_N:
            ok.append(g.mean() > r.mean())
            cells.append(f"{p}: {g.mean() - r.mean():+.2f}R")
        else:
            cells.append(f"{p}: n={len(g)}/{len(r)}")
    m = x[x[key].notna()]
    t = diff_t(m[m[key] == good].R.to_numpy(), m[m[key] != good].R.to_numpy())
    passed = bool(ok) and all(ok) and t >= 2
    return f"{'  '.join(cells)}  всё время t {t:.1f}  → {'принято' if passed else 'нет'}", passed


def report() -> None:
    parts = all_parts("footprint")
    if not parts:
        print("  footprint: нет данных")
        return
    x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
    print("===== FOOTPRINT: профиль агрессоров внутри свечи пробоя бота 4h (подсвечи 15m / 1h; признаки до запуска) =====")
    print(f"     сделок {len(x)}, подсвечи 15m у {(x.sub_tf == '15m').mean():.0%}; R на сделку — 3R, ретест, как бот")
    for nm, z in (("все", x), ("только 15m", x[x.sub_tf == "15m"])):
        print(f"  {nm}:")
        for key, (title, good) in SPLITS.items():
            print(f"   {key} {title} (ждём: «{good}» лучше):")
            for g in sorted(z[key].dropna().unique()):
                m = z[z[key] == g]
                cells = "  ".join(f"{p}: {_cell(m[(m.t >= a) & (m.t < b)])}" for p, (a, b) in PER_ALL.items())
                print(f"     {g}: доля {len(m) / max(1, z[key].notna().sum()):.0%}  {cells}  всё время: {_cell(m)}")
            line, _ = verdict(z, key, good)
            print(f"     «{good}» минус остальные: {line}")


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
