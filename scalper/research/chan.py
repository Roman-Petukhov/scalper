"""Торговля внутри наклонного канала: отскок от линии по тренду (по вопросу владельца про каналы). Горизонтальный
боковик проверен в research/range4.py (минус каждый год) — здесь наклонный канал.

Канал (задан до запуска; для лонга, шорт зеркально): четыре подряд точки зигзага по закрытиям 3 ATR, как у линий бота —
впадина L1, вершина H1, впадина L2, вершина H2. L2 выше L1 и не ближе 10 свечей; нижняя линия — через закрытия L1 и L2,
верхняя — параллельно через H1; H2 в пределах 0.5 ATR от верхней линии (второе касание); ширина ≥ 2 ATR. Канал известен
на баре подтверждения H2 (зигзаг ушёл от неё на 3 ATR); если закрытие там уже ниже нижней линии — канал сломан, пропуск.
Вход: лимитка едет по нижней линии (исполнение, когда low ≤ линии; при открытии ниже — по открытию), живёт столько
свечей, сколько длился канал от L1 до H2; отмена, если раньше закрытие выше верхней линии. Стоп — 1 ATR ниже линии на
свече исполнения (касание стопа в свече исполнения — убыток). Цели: «граница» — верхняя линия на свече исполнения, «3R».
Удержание и издержки — как у бота (maker на вход, taker на выход по стопу, funding).
Вход с подтверждением (по просьбе владельца — «обратный фильтр агрессоров на отскок»): свеча первого касания закрылась
обратно внутри канала → вход по рынку на её закрытии (taker), стоп и цели те же. Справочно три группы по агрессорам
свечи касания: «выкупили» — нашей стороны ≥ 55%; «поглощение» — против нас ≥ 55%, а линия устояла; остальные. Тренд — старший ТФ, как у бота
(4h — дневной, 15m — часовой): «по тренду» — лонг в восходящем канале при восходящем тренде; «против» — справочно.
Оборот монеты от $20M. Один вход на канал.

Принятие: «по тренду» с целью «граница» или «3R» — R на сделку в плюсе в каждом периоде (2020 / 2021 / IS / VAL / HO),
где ≥ 20 сделок, и t по дням за всё время ≥ 2. 4h — основное, 15m — справочно.

    python -m research.chan collect --symbols <1h-qualified>
    python -m research.chan report
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import HOLD, MAKER, TAKER, ZZ_K, htf_trend, tf_frame, two_targets, zigzag

TFS = tuple(os.environ.get("CHAN_TFS", "4h,15m").split(","))
MIN_GAP, TOUCH, MIN_WIDTH, STOP_ATR, MIN_K = 10, 0.5, 2.0, 1.0, 0.5


def pivots_seq(c: np.ndarray, atr: np.ndarray) -> list[tuple[int, int, int]]:
    """Точки зигзага по порядку: (индекс, бар подтверждения, +1 вершина / −1 впадина)."""
    hs, ls = zigzag(c, atr, ZZ_K)
    seq = [(int(p), int(q), 1) for p, q in hs] + [(int(p), int(q), -1) for p, q in ls]
    return sorted(seq)


def channels(c: np.ndarray, atr: np.ndarray) -> list[dict]:
    """Каналы по четвёркам точек; всё в «пространстве стороны» y = side·close (шорт = лонг по −close)."""
    seq = pivots_seq(c, atr)
    out = []
    for i in range(len(seq) - 3):
        (a1, _, t1), (b1, _, t2), (a2, _, t3), (b2, cb2, t4) = seq[i:i + 4]
        if not (t1 == t3 == -t2 == -t4):
            continue
        side = 1 if t1 < 0 else -1                     # впадина–вершина–впадина–вершина → лонг; наоборот → шорт
        y = side * c
        if a2 - a1 < MIN_GAP or not (y[a2] > y[a1]):
            continue
        s = (y[a2] - y[a1]) / (a2 - a1)
        w = y[b1] - (y[a1] + s * (b1 - a1))
        at = atr[cb2]
        if not (at > 0) or w < MIN_WIDTH * at or abs(y[b2] - (y[a1] + s * (b2 - a1) + w)) > TOUCH * at:
            continue
        out.append({"side": side, "a1": a1, "slope": s, "base": float(y[a1]), "width": float(w), "conf": cb2,
                    "life": b2 - a1, "atr": float(at)})
    return out


def confirmed(o, hi, lo, c, f, j, side, lv, w, at, tf) -> tuple[float, float]:
    """Вход по рынку на закрытии свечи касания j (уже известно, что закрылась внутри канала). (R граница, R 3R)."""
    px_y, stop_y = side * c[j], lv - STOP_ATR * at
    risk = px_y - stop_y
    if not (risk > 0):
        return np.nan, np.nan
    px, stop = side * px_y, side * stop_y
    k_bound = (lv + w - px_y) / risk
    rb = two_targets(o, hi, lo, c, f, j, side, px, stop, k_bound, k_bound, False, HOLD[tf], TAKER)[0] \
        if k_bound >= MIN_K else np.nan
    return rb, two_targets(o, hi, lo, c, f, j, side, px, stop, 3.0, 3.0, False, HOLD[tf], TAKER)[0]


def trade(o, hi, lo, c, f, ch: dict, tf: str, buy: np.ndarray | None = None) -> dict | None:
    side, a1, s, base, w, conf, life, at = (ch[k] for k in ("side", "a1", "slope", "base", "width", "conf", "life",
                                                             "atr"))
    n = len(c)
    line = lambda j: base + s * (j - a1)
    oy, ly, cy = side * o, side * (lo if side > 0 else hi), side * c
    if conf >= n - 2 or cy[conf] < line(conf):
        return None
    for j in range(conf + 1, min(conf + 1 + life, n - 1)):
        lv = line(j)
        if ly[j] <= lv:
            px_y = min(oy[j], lv)
            stop_y = lv - STOP_ATR * at
            risk = px_y - stop_y
            if not (risk > 0):
                return None
            k_bound = (lv + w - px_y) / risk
            px, stop = side * px_y, side * stop_y
            row = {"fill": j, "k_bound": k_bound, "c_bound": np.nan, "c_r3": np.nan, "aggr": np.nan}
            if cy[j] > lv:                             # закрылись обратно внутри — вариант с подтверждением
                row["c_bound"], row["c_r3"] = confirmed(o, hi, lo, c, f, j, side, lv, w, at, tf)
                if buy is not None:
                    row["aggr"] = buy[j] if side > 0 else 1 - buy[j]
            if ly[j] <= stop_y:
                loss = (-risk - (MAKER + TAKER) * abs(px)) / risk
                row.update({"bound": loss, "r3": loss})
                return row
            row["bound"] = two_targets(o, hi, lo, c, f, j, side, px, stop, k_bound, k_bound, False, HOLD[tf],
                                       MAKER)[0] if k_bound >= MIN_K else np.nan
            row["r3"] = two_targets(o, hi, lo, c, f, j, side, px, stop, 3.0, 3.0, False, HOLD[tf], MAKER)[0]
            return row
        if cy[j] > lv + w:
            return None                                # ушли вверх из канала — отката не дождались
    return None


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str) -> list[dict]:
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    atr = _atr(d).to_numpy()
    trend = htf_trend(d, tf)
    buy = (d["taker_buy_volume"] / d["volume"].replace(0, np.nan)).to_numpy()
    rows = []
    for ch in channels(c, atr):
        if not (liq[ch["conf"]] >= ADV_MIN):
            continue
        r = trade(o, hi, lo, c, f, ch, tf, buy)
        if r is None:
            continue
        rows.append({"symbol": sym, "tf": tf, "t": d.index[r["fill"]], "side": ch["side"],
                     "with_trend": "по тренду" if trend[ch["conf"]] == ch["side"] else "против",
                     "width_atr": ch["width"] / ch["atr"], "k_bound": r["k_bound"], "bound": r["bound"],
                     "r3": r["r3"], "c_bound": r["c_bound"], "c_r3": r["c_r3"], "aggr": r["aggr"]})
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
                print(f"  chan {s} {tf}: пропуск ({e})", flush=True)
    print(f"  chan: сделок {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("chan"), index=False)


def accepted(z: pd.DataFrame) -> tuple[bool, str]:
    pos = []
    for a, b in PER_ALL.values():
        m = z[(z.t >= a) & (z.t < b)]
        if len(m) >= 20:
            pos.append(m.R.mean() > 0)
    r = z.R.to_numpy()
    g = pd.Series(r - r.mean()).groupby(z.t.dt.floor("D").to_numpy()).sum()
    t = r.sum() / np.sqrt((g ** 2).sum()) if (g ** 2).sum() > 0 else np.nan
    ok = bool(pos) and all(pos) and t >= 2
    return ok, f"плюс во всех периодах: {'да' if pos and all(pos) else 'нет'}; t {t:.1f}"


def report() -> None:
    parts = all_parts("chan")
    print("\n===== CHAN: отскок от линии наклонного канала (зигзаг 3 ATR, 2+2 касания), лимитка по линии, стоп 1 ATR; "
          "R на сделку (t по дням, прибыльных, сделок в месяц) =====")
    if not parts:
        print("  chan: нет данных")
        return
    x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
    for tf in TFS:
        z0 = x[x.tf == tf]
        if z0.empty:
            continue
        print(f"\n  --- {tf}: сделок {len(z0)} на {z0.symbol.nunique()} монетах; шортов {(z0.side < 0).mean():.0%}; "
              f"ширина канала медиана {z0.width_atr.median():.1f} ATR, до границы медиана {z0.k_bound.median():.1f}R ---")
        for trend in ("по тренду", "против"):
            for side, nm in ((0, "обе стороны"), (1, "лонги"), (-1, "шорты")):
                z = z0[(z0.with_trend == trend) & ((z0.side == side) if side else True)]
                if len(z) < 10:
                    continue
                print(f"    {trend}, {nm} ({len(z)}):")
                for col, title in (("bound", "цель — граница"), ("r3", "цель 3R"),
                                   ("c_bound", "подтверждение, граница"), ("c_r3", "подтверждение, 3R")):
                    zz = z.dropna(subset=[col]).assign(R=lambda q, c=col: q[c])
                    cells = "  ".join(f"{p}: {_cell(zz[(zz.t >= a) & (zz.t < b)])}" for p, (a, b) in PER_ALL.items()
                                      if ((zz.t >= a) & (zz.t < b)).any())
                    line = f"      {title}: {cells}  всё время: {_cell(zz)}"
                    if trend == "по тренду" and side == 0 and len(zz):
                        ok, why = accepted(zz)
                        line += f"  → {'принято' if ok else 'не принято'} ({why})"
                    print(line)
                zc = z.dropna(subset=["c_r3"])
                groups = (("выкупили (наши ≥ 55%)", zc.aggr >= 0.55), ("поглощение (против ≥ 55%)", zc.aggr <= 0.45),
                          ("остальные", (zc.aggr > 0.45) & (zc.aggr < 0.55)))
                for g, mask in groups:
                    zz = zc[mask].assign(R=lambda q: q["c_r3"])
                    cells = "  ".join(f"{p}: {_cell(zz[(zz.t >= a) & (zz.t < b)])}" for p, (a, b) in PER_ALL.items()
                                      if ((zz.t >= a) & (zz.t < b)).any())
                    print(f"      подтверждение 3R, {g}: {cells}  всё время: {_cell(zz)}")


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
