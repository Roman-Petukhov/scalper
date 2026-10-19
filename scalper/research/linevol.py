"""Объём за всю консолидацию до пробоя линии — по канону Эдвардса и Мэги (по вопросу владельца: проверяли ли объём на
протяжении всей консолидации перед выходом за трендовую).

Сделки — правило бота 4h без изменений (research/noline.py: линия зигзага 3 ATR, агрессоры от 55%, закрытие в своей
половине, тренд, ретест, стоп за свингом, 3R). Консолидация — свечи от первой опоры линии (не включая) до свечи
пробоя (не включая). «Наша сторона» — покупатели для лонга, продавцы для шорта.

Признаки и ожидаемое направление заданы до запуска:
  V1 тренд объёма        наклон log(объёма) по времени за консолидацию < 0 («затухает»); ждём, что лучше
  V2 всплеск на пробое   объём свечи пробоя ≥ 2 средних за консолидацию; ждём, что лучше
  V3 пересохло к концу   средний объём последней трети консолидации < 0.7 первой трети; ждём, что лучше
  V4 дельта за фигуру    доля агрессоров нашей стороны за всю консолидацию ≥ 50%; ждём, что лучше
Принятие — как research/footprint.py: ожидаемая группа лучше остальных в каждом периоде, где в обеих ≥ 20 сделок,
и разница за всё время с t ≥ 2.

    python -m research.linevol collect --symbols <1h-qualified>
    python -m research.linevol report
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import adv30
from .footprint import verdict
from .noline import TF, coin_trades
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _cell
from .tline import tf_frame, zz_lines

MIN_SPAN = 6
SPLITS = {
    "V1": ("тренд объёма за консолидацию", "затухает"),
    "V2": ("объём свечи пробоя ≥ 2 средних за консолидацию", "да"),
    "V3": ("последняя треть консолидации < 0.7 первой по объёму", "да"),
    "V4": ("агрессоры нашей стороны за всю консолидацию ≥ 50%", "да"),
}


def features(vol: np.ndarray, buy: np.ndarray, a: int, t: int, side: int) -> dict:
    """Признаки V1–V4 по свечам a+1 … t−1 (консолидация) и свече пробоя t."""
    v, b = vol[a + 1:t], buy[a + 1:t]
    out: dict = {"span": len(v)}
    if len(v) < MIN_SPAN or not v.sum() > 0 or not np.all(v > 0):
        return out
    x = np.arange(len(v), dtype="float64")
    slope = np.polyfit(x, np.log(v), 1)[0]
    third = len(v) // 3
    own = b.sum() / v.sum() if side > 0 else 1 - b.sum() / v.sum()
    out.update({"vol_slope": float(slope), "spike": float(vol[t] / v.mean()),
                "dry": float(v[-third:].mean() / v[:third].mean()), "own_span": float(own)})
    out["V1"] = "затухает" if slope < 0 else "растёт"
    out["V2"] = "да" if out["spike"] >= 2 else "нет"
    out["V3"] = "да" if out["dry"] < 0.7 else "нет"
    out["V4"] = "да" if own >= 0.5 else "нет"
    return out


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str) -> list[dict]:
    x = coin_trades(d, liq, sym)
    if not len(x):
        return []
    x = x[(x.trig == "line") & (x.entry == "retest")]
    anchor = {(int(r["t"]), int(r["side"])): int(r["a"]) for r in zz_lines(d) if r["t"] > 0}
    vol = d["volume"].to_numpy(dtype="float64")
    buy = d["taker_buy_volume"].to_numpy(dtype="float64")
    rows = []
    for r in x.itertuples(index=False):
        t, side = d.index.get_loc(r.t), int(r.side)
        a = anchor.get((t, side))
        if a is None:
            continue
        rows.append({"symbol": sym, "t": r.t, "side": side, "R": float(r.R3), **features(vol, buy, a, t, side)})
    return rows


def collect(root: Path, syms: list[str]) -> None:
    rows = []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, TF)
            if d is None or len(d) < 500:
                continue
            liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
            rows += coin_rows(d, liq, s)
        except Exception as e:
            print(f"  linevol {s}: пропуск ({e})", flush=True)
    print(f"  linevol: сделок {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("linevol"), index=False)


def report() -> None:
    parts = all_parts("linevol")
    if not parts:
        print("  linevol: нет данных")
        return
    x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
    print("===== LINEVOL: объём за всю консолидацию до пробоя линии, бот 4h (признаки и направления до запуска) =====")
    print(f"     сделок {len(x)}, с признаками {x['V1'].notna().sum()}; консолидация в медиане {x.span.median():.0f} свечей")
    for nm, z in (("все", x), ("шорты", x[x.side < 0]), ("лонги", x[x.side > 0])):
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
