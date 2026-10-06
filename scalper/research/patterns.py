"""Классические фигуры на 4h и 15m: голова-плечи, двойная вершина / дно, треугольники, прямоугольник.

Определения — по Lo, Mamaysky, Wang (2000, «Foundations of Technical Analysis»), экстремумы — вершины зигзага по
закрытиям (разворот ZZ ATR, известны с бара подтверждения, без заглядывания вперёд). Допуски — в ATR последнего
экстремума. Правило задано заранее и ни на одном периоде не подбирается:
    ГиП / перевёрнутая  E1..E5 = В Н В Н В: E3 выше E1 и E5, |E1−E5| и |E2−E4| <= 1 ATR; шорт — закрытие ниже линии
                        шеи через E2 и E4. Перевёрнутая — зеркально, лонг
    двойная вершина/дно E1..E3 = В Н В, |E1−E3| <= 0.5 ATR; шорт — закрытие ниже E2. Дно — зеркально
    треугольник         сходящийся: вершины ниже одна другой, впадины выше; пробой линии через две последние
                        вершины вверх — лонг, через две последние впадины вниз — шорт. Восходящий: вершины ровные
                        (<= 0.5 ATR), впадины растут — лонг выше вершин; нисходящий — зеркально, шорт
    прямоугольник       две вершины и две впадины, каждая пара в пределах 0.75 ATR; пробой границы в любую сторону
Пробой — первое закрытие за уровнем не позже BREAK_WIN свечей после подтверждения последнего экстремума и до
подтверждения следующего; отмена — закрытие за противоположным краем фигуры раньше пробоя.
Вход по рынку на закрытии свечи пробоя; стоп — за крайней тенью от последнего экстремума до пробоя ∓ 0.1 ATR
(0.3–4 ATR); выход — 2R, 3R или «мера фигуры» (высота от вершины до линии, отложенная от уровня пробоя);
срок — 60 свечей на 4h, 200 на 15m; комиссия taker на входе и выходе, funding. Срезы: все пробои / с фильтром бота
(агрессоры >= 55% и тренд старшего ТФ). Оборот монеты от $20M за 30 прошлых дней.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import HOLD, PER, TAKER, _per_month, _years, htf_trend, tf_frame, two_targets, zigzag

ZZ = {"4h": 2.0, "15m": 2.0}
BREAK_WIN = 20
STOP_BUF = 0.1
STOP_ATR = (0.3, 4.0)


def _ext(c: np.ndarray, a: np.ndarray, k: float) -> list[tuple[int, int, int]]:
    """Экстремумы по порядку: (индекс, бар подтверждения, +1 вершина / −1 впадина)."""
    hs, ls = zigzag(c, a, k)
    ev = [(int(i), int(cf), 1) for i, cf in hs] + [(int(i), int(cf), -1) for i, cf in ls]
    return sorted(ev)


def _line(p1: tuple[int, float], p2: tuple[int, float], t: int) -> float:
    (x1, y1), (x2, y2) = p1, p2
    return y1 + (y2 - y1) / (x2 - x1) * (t - x1)


def _patterns(seq: list[tuple[int, int, int]], c: np.ndarray, at: float) -> list[dict]:
    """Фигуры, которые завершает последний экстремум seq. Уровень пробоя — функция бара (наклонные линии)."""
    out = []
    v = [c[i] for i, _, _ in seq]
    kinds = [s for _, _, s in seq]
    x = [i for i, _, _ in seq]
    if len(seq) >= 5:
        e1, e2, e3, e4, e5 = v[-5:]
        top = kinds[-1] == 1
        s = 1 if top else -1                                     # s = +1: последняя — вершина
        if s * (e3 - e1) > 0 and s * (e3 - e5) > 0 and abs(e1 - e5) <= 1.0 * at and abs(e2 - e4) <= 1.0 * at:
            neck = ((x[-4], e2), (x[-2], e4))
            out.append({"name": "ГиП" if top else "перевёрнутая ГиП", "side": -s,
                        "level": lambda t, n=neck: _line(n[0], n[1], t), "cancel": e3, "height": abs(e3 - (e2 + e4) / 2)})
        hi_pts = [(x[j], v[j]) for j in range(-5, 0) if kinds[j] == 1][-2:]
        lo_pts = [(x[j], v[j]) for j in range(-5, 0) if kinds[j] == -1][-2:]
        if len(hi_pts) == 2 and len(lo_pts) == 2:
            h1, h2, l1, l2 = hi_pts[0][1], hi_pts[1][1], lo_pts[0][1], lo_pts[1][1]
            height = max(h1, h2) - min(l1, l2)
            if h2 < h1 - 0.25 * at and l2 > l1 + 0.25 * at:
                out.append({"name": "треугольник ↑", "side": 1, "level": lambda t, p=hi_pts: _line(p[0], p[1], t),
                            "cancel_fn": lambda t, p=lo_pts: _line(p[0], p[1], t), "height": height})
                out.append({"name": "треугольник ↓", "side": -1, "level": lambda t, p=lo_pts: _line(p[0], p[1], t),
                            "cancel_fn": lambda t, p=hi_pts: _line(p[0], p[1], t), "height": height})
            elif abs(h1 - h2) <= 0.5 * at and l2 > l1 + 0.25 * at:
                out.append({"name": "восходящий треугольник", "side": 1, "level": lambda t, y=max(h1, h2): y,
                            "cancel_fn": lambda t, p=lo_pts: _line(p[0], p[1], t), "height": height})
            elif abs(l1 - l2) <= 0.5 * at and h2 < h1 - 0.25 * at:
                out.append({"name": "нисходящий треугольник", "side": -1, "level": lambda t, y=min(l1, l2): y,
                            "cancel_fn": lambda t, p=hi_pts: _line(p[0], p[1], t), "height": height})
            elif abs(h1 - h2) <= 0.75 * at and abs(l1 - l2) <= 0.75 * at:
                up, dn = max(h1, h2), min(l1, l2)
                out.append({"name": "прямоугольник ↑", "side": 1, "level": lambda t, y=up: y, "cancel": dn,
                            "height": up - dn})
                out.append({"name": "прямоугольник ↓", "side": -1, "level": lambda t, y=dn: y, "cancel": up,
                            "height": up - dn})
    if len(seq) >= 3:
        e1, e2, e3 = v[-3:]
        top = kinds[-1] == 1
        if abs(e1 - e3) <= 0.5 * at:
            out.append({"name": "двойная вершина" if top else "двойное дно", "side": -1 if top else 1,
                        "level": lambda t, y=e2: y, "cancel": max(e1, e3) if top else min(e1, e3),
                        "height": abs(max(e1, e3) - e2) if top else abs(e2 - min(e1, e3))})
    return out


def coin_patterns(d: pd.DataFrame, tf: str, liq: np.ndarray, sym: str) -> pd.DataFrame:
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    a = _atr(d).to_numpy()
    trend = htf_trend(d, tf)
    buy = (d["taker_buy_volume"] / d["volume"].replace(0, np.nan)).to_numpy()
    ev = _ext(c, a, ZZ[tf])
    n = len(c)
    rows = []
    for k in range(2, len(ev)):
        last_i, conf, _ = ev[k]
        nxt = ev[k + 1][1] if k + 1 < len(ev) else n
        at = a[last_i]
        if not (at > 0):
            continue
        for p in _patterns(ev[max(0, k - 4): k + 1], c, at):
            side = p["side"]
            j = -1
            for t in range(conf, min(conf + BREAK_WIN, nxt, n - 1)):
                lv = p["level"](t)
                cancel = p["cancel_fn"](t) if "cancel_fn" in p else p.get("cancel")
                if cancel is not None and side * (cancel - c[t]) > 0:
                    break                                        # закрылась за противоположным краем
                if side * (c[t] - lv) > 0:
                    if t > conf and side * (c[t - 1] - p["level"](t - 1)) > 0:
                        break                                    # уже была за уровнем при подтверждении
                    j = t
                    break
            if j < 0 or not (liq[j] >= ADV_MIN) or not (a[j] > 0):
                continue
            ext = lo[last_i: j + 1].min() if side > 0 else hi[last_i: j + 1].max()
            stop = ext - STOP_BUF * a[j] if side > 0 else ext + STOP_BUF * a[j]
            risk = side * (c[j] - stop)
            if not (STOP_ATR[0] * a[j] <= risk <= STOP_ATR[1] * a[j]):
                continue
            aggr = buy[j] if side > 0 else 1 - buy[j]
            row = {"symbol": sym, "tf": tf, "t": d.index[j], "pattern": p["name"], "side": side,
                   "bot": bool(aggr >= 0.55 and trend[j] == side), "risk_pct": risk / c[j]}
            for tgt in (2.0, 3.0):
                row[f"R{tgt:g}"] = two_targets(o, hi, lo, c, f, j, side, c[j], stop, tgt, tgt, False, HOLD[tf], TAKER)[0]
            mm = min(max((p["height"] - abs(c[j] - p["level"](j))) / risk, 0.5), 6.0)
            row["Rmm"] = two_targets(o, hi, lo, c, f, j, side, c[j], stop, mm, mm, False, HOLD[tf], TAKER)[0]
            rows.append(row)
    return pd.DataFrame(rows)


def collect(root: Path, syms4: list[str], syms15: list[str]) -> None:
    parts = []
    for tf, syms in (("4h", syms4), ("15m", syms15)):
        for s in mine(syms):
            try:
                d = tf_frame(root, s, tf)
                if d is None or len(d) < 500:
                    continue
                liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
                x = coin_patterns(d, tf, liq, s)
                if len(x):
                    parts.append(x)
            except Exception as e:
                print(f"  patterns {tf} {s}: пропуск ({e})", flush=True)
    print(f"  patterns: частей {len(parts)}", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path("patterns"), index=False)


def report() -> None:
    parts = all_parts("patterns")
    print("\n===== PATTERNS: классические фигуры (Lo–Mamaysky–Wang), вход по рынку на закрытии пробоя, стоп за "
          "крайней тенью с последнего экстремума; ячейка — R на сделку (t по дням, прибыльных, сделок в месяц) =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["per"] = ""
    for p, (a, b) in PER.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    for tf in ("4h", "15m"):
        g0 = df[df.tf == tf]
        if not len(g0):
            continue
        for col, nm in (("R2", "2R"), ("R3", "3R"), ("Rmm", "мера фигуры")):
            rows = []
            for pat in ["все фигуры", *sorted(g0.pattern.unique())]:
                g = g0 if pat == "все фигуры" else g0[g0.pattern == pat]
                for fl, z in (("", g), (" + фильтр бота", g[g.bot])):
                    rows.append({"фигура": pat + fl, **{p: _cell(z[z.per == p].assign(R=z[col])) for p in PER},
                                 "R/мес IS / VAL / HO": _per_month(z, col)})
            print(f"\n  --- {tf}, выход {nm} ---")
            print(pd.DataFrame(rows).to_string(index=False))
        g = g0[g0.bot]
        print(f"  {tf}, все фигуры + фильтр бота, 3R, по годам: {_years(g, 'R3')}; издержки x2: " + ", ".join(
            f"{p}: {(z.R3 - 2 * TAKER / z.risk_pct).mean():+.3f}" for p, z in g.groupby('per') if p))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 360)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols4", default="")
    ap.add_argument("--symbols15", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols4.split(",") if x],
                [x for x in a.symbols15.split(",") if x])
    else:
        report()
