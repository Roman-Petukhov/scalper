"""Круглые числа как «топливо» пробоя и стоп «за толпой» — по Osler (NY Fed SR 125 / 150; Journal of Finance 2003):
стопы толпы стоят сразу за круглыми числами (на покупку — чуть выше, на продажу — чуть ниже), тейки — на самих круглых
числах; пройдя круглый уровень, цена ускоряется (каскад стопов), у самого уровня чаще разворачивается.

Сделки — правило бота без изменений (research/noline.py; 4h — ретест, основное; 15m — по рынку, справочно).
Круглые уровни: k = floor(log10(цена)); «крупные» — шаг 10^k / 2 (для 2.3 — 2.0, 2.5, 3.0; для 67 000 — 65 000, 70 000),
«мелкие» — шаг 10^(k−1) (для 2.3 — 2.2, 2.3, 2.4). Признаки и направления заданы до запуска:
  F1 пробой прошёл крупный круглый уровень   закрытие свечи пробоя за уровнем, прошлое — нет; ждём, что лучше
  F2 то же для мелкого (справочно)
  F3 крупный уровень впереди в пределах 1R    между ценой входа и +1R стоит крупный круглый (там тейки); ждём, что хуже
  S1 отступ стопа 0.25 ATR вместо 0.1         тот же вход; ждём, что лучше (на старых данных HO +0.50 против +0.43R)
  S2 стоп «за толпой»                         если в пределах 0.5 ATR от нашего стопа внутрь сделки стоит мелкий
                                              круглый уровень (наш стоп — в скоплении стопов толпы под ним), стоп
                                              переносится на 0.5 ATR за этот уровень; ждём, что лучше
Принятие: F — «да» лучше остальных в каждом периоде (2020 / 2021 / IS / VAL / HO), где в обеих группах ≥ 20 сделок,
и t ≥ 2; S — R на сделку выше, чем у бота, в каждом периоде и парная разница с t ≥ 2. Вход на ретесте при другом стопе
не пересчитывается (та же свеча исполнения), меняются только риск и выход.

    python -m research.roundnum collect --symbols <1h-qualified>
    python -m research.roundnum report
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import noline
from .broad import adv30
from .footprint import verdict
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import HOLD, MAKER, STOP_BUF, TAKER, crossed_round, round_grid, tf_frame, two_targets

TFS = tuple(os.environ.get("ROUNDNUM_TFS", "4h,15m").split(","))
BOT_ENTRY = {"4h": "retest", "15m": "market"}
CLUSTER_ATR, S1_BUF = 0.5, 0.25


step, crossed = round_grid, crossed_round


def level_ahead(px: float, side: int, dist: float, major: bool) -> bool:
    """Круглый уровень строго между ценой входа и px + side·dist."""
    s = step(px, major)
    if side > 0:
        nxt = (math.floor(px / s) + 1) * s
        return nxt < px + dist
    nxt = (math.ceil(px / s) - 1) * s
    return nxt > px - dist


def crowd_stop(stop: float, side: int, atr: float) -> float | None:
    """Стоп «за толпой»: если внутрь сделки от стопа в пределах CLUSTER_ATR ATR есть мелкий круглый уровень — новый стоп
    на CLUSTER_ATR ATR за этим уровнем; иначе None."""
    s = step(stop, False)
    if side > 0:
        r = math.ceil(stop / s) * s
        return r - CLUSTER_ATR * atr if r - stop <= CLUSTER_ATR * atr else None
    r = math.floor(stop / s) * s
    return r + CLUSTER_ATR * atr if stop - r <= CLUSTER_ATR * atr else None


def resim(o, hi, lo, c, f, fill, side, px, stop, fee, tf) -> float:
    risk = side * (px - stop)
    if not (risk > 0):
        return np.nan
    if fee == MAKER and ((side > 0 and lo[fill] <= stop) or (side < 0 and hi[fill] >= stop)):
        return (side * (stop - px) - (fee + TAKER) * px) / risk
    return two_targets(o, hi, lo, c, f, fill, side, px, stop, 3.0, 3.0, False, HOLD[tf], fee)[0]


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str) -> list[dict]:
    noline.TF = tf
    x = noline.coin_trades(d, liq, sym, lines_only=True)
    if not len(x):
        return []
    x = x[x.entry == BOT_ENTRY[tf]]
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    atr = _atr(d).to_numpy()
    rows = []
    for r in x.itertuples(index=False):
        t, side, fill, px, stop = d.index.get_loc(r.t), int(r.side), int(r.fill_i), float(r.px), float(r.stop)
        if not (c[t] > 0 and c[t - 1] > 0 and atr[t] > 0 and stop > 0):
            continue
        fee = MAKER if r.entry == "retest" else TAKER
        risk = side * (px - stop)
        s1 = stop - side * (S1_BUF - STOP_BUF) * atr[t]
        cs = crowd_stop(stop, side, atr[t])
        rows.append({"symbol": sym, "tf": tf, "t": r.t, "side": side, "R": float(r.R3),
                     "F1": "да" if crossed(c[t - 1], c[t], True) else "нет",
                     "F2": "да" if crossed(c[t - 1], c[t], False) else "нет",
                     "F3": "да" if level_ahead(px, side, risk, True) else "нет",
                     "R_s1": resim(o, hi, lo, c, f, fill, side, px, s1, fee, tf),
                     "crowd": cs is not None,
                     "R_s2": resim(o, hi, lo, c, f, fill, side, px, cs, fee, tf) if cs is not None else float(r.R3)})
    return rows


def collect(root: Path, syms: list[str]) -> None:
    rows = []
    for s in mine(syms):
        for tf in TFS:
            try:
                if tf == "15m" and not (root / f"{s}-15m.parquet").exists():
                    continue
                d = tf_frame(root, s, tf)
                if d is None or len(d) < 500:
                    continue
                liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
                rows += coin_rows(d, liq, s, tf)
            except Exception as e:
                print(f"  roundnum {s} {tf}: пропуск ({e})", flush=True)
    print(f"  roundnum: сделок {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("roundnum"), index=False)


def paired(z: pd.DataFrame, col: str) -> str:
    cells, ok = [], []
    for p, (a, b) in PER_ALL.items():
        m = z[(z.t >= a) & (z.t < b)]
        if len(m) >= 20:
            dlt = m[col].mean() - m["R"].mean()
            ok.append(dlt > 0)
            cells.append(f"{p}: {dlt:+.3f}")
    dd = (z[col] - z["R"]).to_numpy()
    sd = dd.std(ddof=1) if len(dd) > 2 else 0.0
    t = dd.mean() / (sd / np.sqrt(len(dd))) if sd > 0 else np.nan
    passed = bool(ok) and all(ok) and t >= 2
    return f"{'  '.join(cells)}  всё время t {t:.1f}  → {'принято' if passed else 'нет'}"


def report() -> None:
    parts = all_parts("roundnum")
    print("\n===== ROUNDNUM: круглые числа (Osler) — пробой через круглый уровень, уровень впереди, стоп «за толпой»; "
          "R на сделку (t по дням, прибыльных, сделок в месяц) =====")
    if not parts:
        print("  roundnum: нет данных")
        return
    x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
    x = x.dropna(subset=["R", "R_s1", "R_s2"])
    titles = {"F1": ("пробой прошёл крупный круглый уровень", "да"), "F2": ("то же, мелкий (справочно)", "да"),
              "F3": ("крупный уровень впереди в пределах 1R — ждём хуже", "нет")}
    for tf in TFS:
        for nm, z in (("все", x[x.tf == tf]), ("шорты", x[(x.tf == tf) & (x.side < 0)])):
            if len(z) < 20:
                continue
            print(f"\n  --- {tf}, вход {BOT_ENTRY[tf]}, {nm}: сделок {len(z)} ---")
            for key, (title, good) in titles.items():
                print(f"    {key} {title}:")
                for g in ("да", "нет"):
                    m = z[z[key] == g]
                    cells = "  ".join(f"{p}: {_cell(m[(m.t >= a) & (m.t < b)])}" for p, (a, b) in PER_ALL.items()
                                      if ((m.t >= a) & (m.t < b)).any())
                    print(f"       {g}: доля {len(m) / len(z):.0%}  {cells}  всё время: {_cell(m)}")
                print(f"       «{good}» минус остальные: {verdict(z, key, good)[0]}")
            for col, title in (("R_s1", "S1 отступ стопа 0.25 ATR"), ("R_s2", "S2 стоп «за толпой»")):
                zz = z.assign(R=z[col])
                cells = "  ".join(f"{p}: {_cell(zz[(zz.t >= a) & (zz.t < b)])}" for p, (a, b) in PER_ALL.items()
                                  if ((zz.t >= a) & (zz.t < b)).any())
                extra = f" (стоп перенесён у {z.crowd.mean():.0%})" if col == "R_s2" else ""
                print(f"    {title}{extra}: {cells}  всё время: {_cell(zz)}")
                print(f"       минус бот: {paired(z, col)}")
            zc = z[z.crowd]
            if len(zc) >= 20:
                print(f"       только перенесённые ({len(zc)}): бот {_cell(zc)}; «за толпой» {_cell(zc.assign(R=zc.R_s2))}")


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
