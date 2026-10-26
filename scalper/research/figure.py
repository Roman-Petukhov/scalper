"""Вид фигуры у пробоя трендовой линии: треугольник, клин, канал, коробка, расширение; ритм касаний; свечи внутри.
По просьбе владельца после research/formation.py (там — только размеры фигуры): «как по графику видно».

Вторая граница фигуры — прямая МНК через крупные развороты закрытий по другую сторону от линии (для пробоя вверх —
впадины, вниз — вершины): самое далёкое от линии пробоя закрытие в окне ±PIV свечей, отошедшее от неё хотя бы на MAJOR
высоты фигуры — так глаз ведёт границу по главным экстремумам, а не по шуму; если таких меньше двух — через крайние
закрытия первой и второй половины отрезка A → пробой. Класс — по наклонам двух границ и сходимости
conv = ширина у пробоя / ширина у опоры A; граница «ровная», если за всю фигуру сдвинулась меньше чем на FLAT средней
ширины фигуры (порог в ATR не годится: на длинной 4h фигуре наклон в 1 ATR — уже заметный канал):
  box             — коробка: обе границы ровные (или средний наклон ровный при параллельных);
  channel_counter — канал против пробоя (флаг: падающий канал, пробитый вверх); channel_with — канал по пробою;
  tri_flat_break  — треугольник с ровной линией пробоя (восходящий для пробоя вверх);
  tri_flat_opp    — треугольник с ровной другой границей; tri_sym — симметричный треугольник;
  wedge_counter   — клин против пробоя (падающий клин вверх); wedge_with — клин по пробою;
  expanding       — расширение (conv > 1 / CONV).
Касания (как метка бота: в пределах TOUCH_ATR, подходы через TOUCH_GAP+ свечей): touch_gap — средний шаг между
касаниями, свечей; touch_cv — неравномерность шага (σ / среднее); touch_err — точность касаний, ATR (без опор A и B);
bounce — средний отскок от линии между соседними касаниями, ATR свечи пробоя.
Свечи фигуры (A … пробой): body — среднее тело, ATR; long_share — доля длинных свечей (размах > 1.5 ATR);
wick — доля теней в размахе; contraction — средний размах последней трети / первой трети.

Принятие (задано до запуска, как в formation.py): числовой признак — срезаем нижний или верхний квинтиль (границы по IS);
класс фигуры — убираем класс или оставляем только его. Отрезанные сделки убыточны в каждом периоде, где их >= 20,
t по дням <= −3, R в месяц с фильтром не ниже. Решают срезы 4h (обе стороны) и 15m шорты; журнал — только иллюстрация.

    python -m research.figure collect --symbols <монеты>      # FORM_TFS=4h или 15m
    python -m research.figure report
    python -m research.figure journal <journal.json> [--png out.png]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import formation as F
from .oos import PER_ALL
from .shard import all_parts
from .smc import _atr, _cell
from .spec import cut_verdicts
from .tline import TOUCH_ATR, TOUCH_GAP

FLAT, CONV, PIV, MAJOR = 0.25, 0.6, 5, 0.5
SHAPES = {"box": "коробка", "channel_counter": "канал против пробоя (флаг)", "channel_with": "канал по пробою",
          "tri_flat_break": "треугольник, линия пробоя ровная", "tri_flat_opp": "треугольник, другая граница ровная",
          "tri_sym": "симметричный треугольник", "wedge_counter": "клин против пробоя",
          "wedge_with": "клин по пробою", "expanding": "расширение"}
FEATS = ("conv", "touch_gap", "touch_cv", "touch_err", "bounce", "body", "long_share", "wick", "contraction")
NAMES = {"conv": "сходимость (ширина у пробоя / у опоры A)", "touch_gap": "шаг между касаниями, свечей",
         "touch_cv": "неравномерность касаний (σ/среднее)", "touch_err": "точность касаний, ATR",
         "bounce": "отскок между касаниями, ATR", "body": "тело свечи, ATR", "long_share": "доля длинных свечей",
         "wick": "доля теней", "contraction": "размах свечей: конец / начало"}


def opposite(c: np.ndarray, a: int, t: int, side: int, sb: float) -> tuple[int, int, float, float]:
    """Вторая граница на A … t−1 (линия пробоя c[a] + sb·(j − a)) → (первый и последний разворот, наклон, значение
    прямой в первом развороте)."""
    y = side * c
    dist = side * (c[a] + sb * (np.arange(len(c)) - a)) - y          # насколько закрытие отошло от линии внутрь
    top = dist[a:t].max()
    piv = [j for j in range(a + 1, t - 1)
           if dist[j] == dist[max(a, j - PIV):min(t, j + PIV + 1)].max() and dist[j] >= MAJOR * top > 0]
    if len(piv) >= 2:
        k = np.array(piv)
        so, b0 = np.polyfit(k - k[0], c[k], 1)
        return int(k[0]), int(k[-1]), float(so), float(b0)
    m = a + max((t - a) // 2, 1)
    j1 = a + int(np.argmax(-y[a:m]))
    j2 = m + int(np.argmax(-y[m:t])) if t > m else j1
    return j1, j2, (c[j2] - c[j1]) / (j2 - j1) if j2 != j1 else 0.0, float(c[j1])


def classify(sb: float, so: float, n: int, width: float, conv: float, side: int) -> str:
    """Класс по наклонам линии пробоя sb и второй границы so (цена за свечу), длине n, средней ширине и сходимости."""
    flat = lambda s: abs(s) * n < FLAT * width
    counter = lambda s: np.sign(s) == -side
    if conv < CONV:
        if flat(sb) and flat(so):
            return "box"
        if flat(sb):
            return "tri_flat_break"
        if flat(so):
            return "tri_flat_opp"
        if np.sign(sb) != np.sign(so):
            return "tri_sym"
        return "wedge_counter" if counter(sb) else "wedge_with"
    if conv > 1 / CONV:
        return "expanding"
    mid = (sb + so) / 2
    if flat(mid):
        return "box"
    return "channel_counter" if counter(mid) else "channel_with"


def touch_points(c: np.ndarray, atr: np.ndarray, a: int, t: int, side: int, slope: float) -> list[tuple[int, float]]:
    """Касания линии (как tline.line_touches): лучшая свеча каждого подхода → (индекс, расстояние в ATR)."""
    j = np.arange(a, t)
    dist = np.abs(side * (c[a] + slope * (j - a) - c[a:t]))
    unit = np.where(atr[a:t] > 0, atr[a:t], np.inf)
    out: list[tuple[int, float]] = []
    last = -10 ** 9
    for k in np.flatnonzero(dist <= TOUCH_ATR * unit):
        e = float(dist[k] / unit[k])
        if k - last >= TOUCH_GAP:
            out.append((a + int(k), e))
        elif e < out[-1][1]:
            out[-1] = (a + int(k), e)
        last = k
    return out


def figure_features(d: pd.DataFrame, atr: np.ndarray, rec: dict, t: int) -> dict:
    o, h, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    a, b, side, sb = int(rec["a"]), int(rec["b"]), int(rec["side"]), float(rec["slope"])
    j1, j2, so, o1 = opposite(c, a, t, side, sb)
    line = lambda j: c[a] + sb * (j - a)
    opp = lambda j: o1 + so * (j - j1)
    w0, w1 = side * (line(a) - opp(a)), side * (line(t - 1) - opp(t - 1))
    if not w0 > 0:                                   # прямые пересеклись левее A — ширина в первом развороте
        w0 = side * (line(j1) - opp(j1))
    w0 = w0 if w0 > 0 else np.nan
    conv = w1 / w0 if w0 > 0 else np.nan
    tp = touch_points(c, atr, a, t, side, sb)
    js = np.array([p[0] for p in tp])
    gaps = np.diff(js) if len(js) >= 2 else np.array([])
    errs = [e for jj, e in tp if jj not in (a, b)]
    inside = -side * (c - line(np.arange(len(c))))
    bounce = [inside[x:y].max() / atr[t] for x, y in zip(js[:-1], js[1:]) if y > x] if len(js) >= 2 else []
    ak = atr[a:t]
    ok = ak > 0
    rng, body = (h - lo)[a:t], np.abs(c - o)[a:t]
    third = max((t - a) // 3, 1)
    return {"shape": classify(sb, so, t - a, max((w0 + w1) / 2, atr[t]), conv if np.isfinite(conv) else 1.0, side), "conv": conv,
            "opp_j1": j1, "opp_j2": j2, "opp_slope": so, "opp_v1": o1,
            "touch_gap": float(gaps.mean()) if len(gaps) else np.nan,
            "touch_cv": float(gaps.std() / gaps.mean()) if len(gaps) >= 2 and gaps.mean() > 0 else np.nan,
            "touch_err": float(np.mean(errs)) if errs else np.nan,
            "bounce": float(np.mean(bounce)) if bounce else np.nan,
            "body": float(np.mean(body[ok] / ak[ok])) if ok.any() else np.nan,
            "long_share": float(np.mean(rng[ok] > 1.5 * ak[ok])) if ok.any() else np.nan,
            "wick": float(np.mean(1 - body[rng > 0] / rng[rng > 0])) if (rng > 0).any() else np.nan,
            "contraction": float(rng[-third:].mean() / rng[:third].mean()) if rng[:third].mean() > 0 else np.nan}


def _load(tf: str) -> pd.DataFrame | None:
    parts = all_parts(f"figure_{tf}")
    if not parts:
        return None
    x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
    return x.dropna(subset=["R"])


def _periods(m: pd.DataFrame) -> str:
    return "  ".join(f"{p}: {_cell(m[(m.t >= a) & (m.t < b)])}" for p, (a, b) in PER_ALL.items()
                     if ((m.t >= a) & (m.t < b)).any())


def report() -> None:
    print("\n===== FIGURE: вид фигуры у пробоя линии — треугольник, клин, канал, коробка, расширение; ритм касаний; "
          "свечи внутри (t по дням, прибыльных, сделок в месяц) =====")
    passed = []
    for tf in F.TFS:
        x = _load(tf)
        if x is None:
            print(f"  figure {tf}: нет данных")
            continue
        slices = (("обе стороны", x),) if tf == "4h" else (("шорты", x[x.side < 0]), ("обе стороны", x))
        for label, z in slices:
            primary = tf == "4h" or label == "шорты"
            print(f"\n  --- {tf}, вход {F.ENTRY[tf]}, {label}: сделок {len(z)}, {_cell(z)} ---")
            print("    вид фигуры:")
            for k, name in SHAPES.items():
                m = z[z["shape"] == k]
                print(f"      {name} ({len(m) / max(len(z), 1):.0%}): {_periods(m)}  всё время: {_cell(m)}")
                if len(m) < 30:
                    continue
                zz = z.assign(is_k=(z["shape"] == k).astype(float))
                for what, col, thr in (("убираем", "neg", -0.5), ("оставляем только", "is_k", 0.5)):
                    for _, (ok, txt) in cut_verdicts(zz.assign(neg=-zz.is_k), col, (thr,)).items():
                        print(f"        {what}: {'принято' if ok and primary else 'нет'}\n{txt}")
                        if ok and primary:
                            passed.append(f"{tf} {label}: {what} «{name}»")
            for f in FEATS:
                e = F.is_edges(z, f)
                if not len(e):
                    continue
                print(f"    {NAMES[f]} (границы IS: {', '.join(f'{v:.3g}' for v in e)}):")
                q = np.searchsorted(e, z[f].to_numpy(), side="right")
                for i in range(len(e) + 1):
                    m = z[(q == i) & z[f].notna()]
                    print(f"      квинтиль {i + 1}: {_periods(m)}  всё время: {_cell(m)}")
                for tail, col, thr in (("нижний", f, e[0]), ("верхний", "neg", -e[-1])):
                    zz = z[z[f].notna()].assign(neg=-z[f])
                    for _, (ok, txt) in cut_verdicts(zz, col, (thr,)).items():
                        print(f"      режем {tail} квинтиль: {'принято' if ok and primary else 'нет'}\n{txt}")
                        if ok and primary:
                            passed.append(f"{tf} {label}: {NAMES[f]}, режем {tail}")
    print("\n  ИТОГ: " + ("прошли — " + "; ".join(passed) if passed else "ни вид фигуры, ни ритм касаний, ни свечи "
                                                                          "внутри не приняты как фильтр"))


def journal_frames(path: Path) -> list[tuple[dict, pd.DataFrame, dict, int]]:
    """Сигналы журнала бота со свечами: (сигнал, свечи, линия {a, b, side, slope}, индекс свечи пробоя)."""
    out = []
    for s in json.loads(path.read_text(encoding="utf-8"))["signals"]:
        b = s.get("bars") or {}
        if not b or s.get("x_log_line"):
            continue
        idx = pd.to_datetime(b["t"], unit="s", utc=True)
        d = pd.DataFrame({"open": b["o"], "high": b["h"], "low": b["l"], "close": b["c"]}, index=idx, dtype="float64")
        pos = {ts: i for i, ts in enumerate(idx)}
        at, bt, tt = (pd.Timestamp(datetime.fromisoformat(s[k])) for k in ("line_a_time", "line_b_time", "bar_time"))
        if at not in pos or bt not in pos or tt not in pos:
            continue
        a, bb, t = pos[at], pos[bt], pos[tt]
        out.append((s, d, {"a": a, "b": bb, "side": int(s["side"]), "slope": (s["line_b"] - s["line_a"]) / (bb - a)}, t))
    return out


def journal(path: Path, png: Path | None) -> None:
    rows, cases = [], {}
    for s, d, rec, t in journal_frames(path):
        atr = _atr(d).to_numpy()
        if not (atr[t] > 0):
            continue
        tr = s.get("trade") or {}
        f = figure_features(d, atr, rec, t)
        rows.append({"tf": s["tf"], "symbol": s["symbol"], "side": s["side"], "R": tr.get("r"), **f})
        cases.setdefault(f["shape"], (s, d, rec, t, f))
    z = pd.DataFrame(rows)
    print(f"Журнал {path.name}: сигналов с фигурой {len(z)}, с закрытой сделкой {int(z.R.notna().sum())}")
    for tf, g in z.groupby("tf"):
        print(f"\n  {tf}: вид фигуры — сигналов / закрытых / R на сделку / сумма R")
        for k, name in SHAPES.items():
            m = g[g["shape"] == k]
            if len(m):
                r = m.R.dropna()
                print(f"    {name:<36} {len(m):>3} / {len(r):>3} / "
                      f"{(f'{r.mean():+.2f}R' if len(r) else '—'):>7} / {r.sum():+.1f}R")
        for f in FEATS:
            m = g.dropna(subset=["R", f])
            if len(m) >= 6:
                med = m[f].median()
                lo_, hi_ = m[m[f] <= med].R, m[m[f] > med].R
                print(f"    {NAMES[f]}: ниже медианы {med:.3g} — {lo_.mean():+.2f}R на {len(lo_)}, "
                      f"выше — {hi_.mean():+.2f}R на {len(hi_)}")
    if png is not None and cases:
        draw(cases, png)


def draw(cases: dict, png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [k for k in SHAPES if k in cases]
    cols = 3
    fig, axes = plt.subplots((len(keys) + cols - 1) // cols, cols, figsize=(5.2 * cols, 3.4 * ((len(keys) + 2) // 3)),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, k in zip(axes.ravel(), keys):
        s, d, rec, t, f = cases[k]
        a, side, sb = rec["a"], rec["side"], rec["slope"]
        lo_ = max(a - 10, 0)
        hi_ = min(t + 6, len(d) - 1)
        c = d["close"].to_numpy()
        j = np.arange(lo_, hi_ + 1)
        ax.axis("on")
        ax.vlines(j, d["low"].to_numpy()[lo_:hi_ + 1], d["high"].to_numpy()[lo_:hi_ + 1], color="#9aa0a6", lw=0.6)
        ax.plot(j, c[lo_:hi_ + 1], color="#202124", lw=1)
        jl = np.arange(a, t + 1)
        ax.plot(jl, c[a] + sb * (jl - a), color="#d93025", lw=1.6, label="линия пробоя")
        j1 = f["opp_j1"]
        ax.plot(jl, f["opp_v1"] + f["opp_slope"] * (jl - j1), color="#1a73e8", lw=1.2, ls="--", label="вторая граница")
        ax.scatter([t], [c[t]], color="#d93025", zorder=5, s=18)
        r = (s.get("trade") or {}).get("r")
        ax.set_title(f"{SHAPES[k]}\n{s['symbol']} {s['tf']} {'вверх' if side > 0 else 'вниз'}"
                     f"{'' if r is None else f', {r:+.2f}R'}", fontsize=9)
        ax.tick_params(labelsize=7)
    axes.ravel()[0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(png, dpi=110)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report", "journal"])
    ap.add_argument("path", nargs="?")
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--png")
    a = ap.parse_args()
    if a.mode == "collect":
        F.collect(Path(a.root).expanduser(), [s for s in a.symbols.split(",") if s], "figure", figure_features)
    elif a.mode == "journal":
        journal(Path(a.path), Path(a.png) if a.png else None)
    else:
        report()
