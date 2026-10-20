"""Срок сделки на 15m и 5m (по вопросу владельца: закрытие через 200 свечей — хорошая ли практика). 200 свечей
перенесены в бота по аналогии, отдельно не подбирались.

Сделки — правило бота (research/noline.py: линия зигзага 3 ATR, агрессоры ≥ 55%, закрытие в своей половине, тренд
старшего ТФ, стоп за свингом, 3R), все ~725 монет с 2022 (как research/hotcoins.py). Меняется только срок: позиция
закрывается по рынку через H свечей после входа, H = 50 / 100 / 200 (бот) / 400 / 800. Срез «как бот»: шорты, линия
не круче 0.025 ATR за свечу, вход по рынку; справочно — все сделки и вход на ретесте.
Принятие (задано до запуска): срок H заменяет 200, если R на сделку «как бот» выше в каждом периоде IS / VAL / HO
и парная разница по тем же сделкам за всё время с t ≥ 2 (по соседним срокам видно, плато это или пик).

    python -m research.holdlen collect --symbols <hot-wide>
    python -m research.holdlen report
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import noline
from .broad import adv30
from .oos import PER
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import MAKER, TAKER, tf_frame, two_targets, zz_lines

TFS = tuple(os.environ.get("HOLDLEN_TFS", "15m,5m").split(","))
HOLDS = (50, 100, 200, 400, 800)
BOT_HOLD, MAX_SLOPE = 200, 0.025


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str) -> list[dict]:
    noline.TF = tf
    x = noline.coin_trades(d, liq, sym, lines_only=True)
    if not len(x):
        return []
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    atr = _atr(d).to_numpy()
    slope = {(int(r["t"]), int(r["side"])): abs(float(r["slope"])) for r in zz_lines(d) if r["t"] > 0}
    rows = []
    for r in x.itertuples(index=False):
        t, side, fill, px, stop = d.index.get_loc(r.t), int(r.side), int(r.fill_i), float(r.px), float(r.stop)
        fee = MAKER if r.entry == "retest" else TAKER
        out = {"symbol": sym, "tf": tf, "t": r.t, "side": side, "entry": r.entry,
               "slope_atr": slope.get((t, side), np.nan) / atr[t] if atr[t] > 0 else np.nan}
        stopped_on_fill = r.entry == "retest" and ((side > 0 and lo[fill] <= stop) or (side < 0 and hi[fill] >= stop))
        for h in HOLDS:
            if stopped_on_fill:
                out[f"R{h}"], out[f"to{h}"] = float(r.R3), False
                continue
            v, ex = two_targets(o, hi, lo, c, f, fill, side, px, stop, 3.0, 3.0, False, h, fee)
            out[f"R{h}"], out[f"to{h}"] = float(v), bool(ex - fill >= h)
        rows.append(out)
    return rows


def collect(root: Path, syms: list[str]) -> None:
    rows = []
    for s in mine(syms):
        for tf in TFS:
            try:
                d = tf_frame(root, s, tf)
                if d is None or len(d) < 2000:
                    continue
                liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
                rows += coin_rows(d, liq, s, tf)
            except Exception as e:
                print(f"  holdlen {s} {tf}: пропуск ({e})", flush=True)
    print(f"  holdlen: сделок {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("holdlen"), index=False)


def paired(z: pd.DataFrame, h: int) -> tuple[str, bool]:
    cells, ok = [], []
    for p, (a, b) in PER.items():
        m = z[(z.t >= a) & (z.t < b)]
        if len(m) >= 20:
            dlt = m[f"R{h}"].mean() - m[f"R{BOT_HOLD}"].mean()
            ok.append(dlt > 0)
            cells.append(f"{p}: {dlt:+.3f}")
    dd = (z[f"R{h}"] - z[f"R{BOT_HOLD}"]).to_numpy()
    sd = dd.std(ddof=1) if len(dd) > 2 else 0.0
    t = dd.mean() / (sd / np.sqrt(len(dd))) if sd > 0 else np.nan
    return f"{'  '.join(cells)}  всё время t {t:.1f}", bool(ok) and all(ok) and t >= 2


def report() -> None:
    parts = all_parts("holdlen")
    print("\n===== HOLDLEN: срок сделки 50 / 100 / 200 (бот) / 400 / 800 свечей на 15m и 5m; R на сделку (t по дням, "
          "прибыльных, сделок в месяц) =====")
    if not parts:
        print("  holdlen: нет данных")
        return
    x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
    for tf in TFS:
        z0 = x[x.tf == tf]
        if z0.empty:
            continue
        bot = z0[(z0.side < 0) & (z0.slope_atr <= MAX_SLOPE) & (z0.entry == "market")]
        for nm, z in (("как бот: шорты, пологие, по рынку", bot), ("все, по рынку", z0[z0.entry == "market"]),
                      ("все, ретест", z0[z0.entry == "retest"])):
            if len(z) < 20:
                continue
            print(f"\n  --- {tf}, {nm}: сделок {len(z)} ---")
            passed = {}
            for h in HOLDS:
                zz = z.assign(R=z[f"R{h}"])
                to = z[z[f"to{h}"]]
                cells = "  ".join(f"{p}: {_cell(zz[(zz.t >= a) & (zz.t < b)])}" for p, (a, b) in PER.items())
                line = (f"    {h:4d} свечей: {cells}  всё время: {_cell(zz)}; по сроку {len(to) / len(z):.0%} "
                        f"(в среднем {to[f'R{h}'].mean():+.2f}R)" if len(to) else
                        f"    {h:4d} свечей: {cells}  всё время: {_cell(zz)}; по сроку 0%")
                print(line)
                if h != BOT_HOLD:
                    txt, ok = paired(z, h)
                    passed[h] = ok
                    print(f"           минус 200: {txt}")
            if nm.startswith("как бот"):
                good = [h for h, ok in passed.items() if ok]
                print(f"    ИТОГ {tf}: " + (f"принято — срок {good}" if good else "200 свечей оставляем (ни один срок не "
                                                                                   "лучше во всех периодах с t ≥ 2)"))


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
