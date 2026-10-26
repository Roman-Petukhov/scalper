"""Вид фигуры у пробоя трендовой линии бота — по канону: Bulkowski («Encyclopedia of Chart Patterns», thepatternsite.com)
и Lo, Mamaysky, Wang (2000, «Foundations of Technical Analysis»). По просьбе владельца: «как по графику видно».

Развороты (Bulkowski, minor high / low): вершина, выше которой нет в окне ±k свечей, впадина — так же; только
подтверждённые до свечи пробоя. У Bulkowski k = 2 дня на фигурах от 3 недель (~15 баров); у нас фигуры по 50–300 баров,
и ±2 ловит шум — окно масштабируем к длине фигуры: k = max(PIV, длина / SCALE), на фигуре в 15 баров это его ±2.
Границы — по закрытиям (как линия бота; линии по теням в проекте хуже, docs/knowledge.md, research/tlcanon.py K4, и
одиночная шпилька задирает границу): прямая через два закрытия фигуры A … пробой, за которую не закрылась ни одна
свеча (ребро оболочки); из таких — с наибольшим числом касаний (отдельные развороты ближе TOUCH_ATR ATR), при
равенстве самая длинная. Допуски LMW (0.75% у дневных акций) переведены в ATR, как в research/patterns.py: TOL = 0.75 ATR.
Класс (названия Bulkowski; граница «горизонтальна», если за фигуру сдвинулась не больше TOL; «сходятся / расходятся» —
ширина у пробоя меньше / больше ширины у A больше чем на max(TOL, PAR средней ширины)):
  rect        прямоугольник — обе горизонтальны (или параллельны со средним наклоном в пределах TOL);
  tri_asc     восходящий треугольник — верх горизонтален, низ растёт;  tri_desc — нисходящий, зеркально;
  tri_sym     симметричный треугольник — верх падает, низ растёт (LMW: E1 > E3 > E5, E2 < E4);
  wedge_rise  восходящий клин — обе растут и сходятся;  wedge_fall — нисходящий клин;
  chan_up     восходящий канал — обе растут, параллельны;  chan_down — нисходящий канал;
  broad       расширение (мегафон, расширяющиеся клинья, прямоугольное расширение; LMW: E1 < E3 < E5, E2 > E4);
  none        фигуры по канону нет — касаний мало: треугольникам, клиньям, прямоугольнику и расширению Bulkowski
              требует >= 5 касаний (3 + 2, на каждой границе >= 2), каналу — >= 2 на каждой.
classic — пробой в типичную по Bulkowski сторону фигуры (+1), против (−1), у фигуры без типичной стороны — 0:
  восходящий треугольник и нисходящий клин — вверх; нисходящий треугольник и восходящий клин — вниз.
apex_pos — у сходящихся фигур доля пути от A до вершины в момент пробоя (Bulkowski: обычно 64–74%).
Флаги и вымпелы отдельно не выделяем: по Bulkowski нужен крутой «флагшток», а порога для него в барах канон не даёт.
Касания линии бота (как метка бота: в пределах TOUCH_ATR, подходы через TOUCH_GAP+ свечей): touch_gap — средний шаг,
свечей; touch_cv — неравномерность шага (σ / среднее); touch_err — точность касаний, ATR (без опор A и B);
bounce — средний отскок от линии между соседними касаниями, ATR свечи пробоя.
Свечи фигуры (A … пробой): body — среднее тело, ATR; long_share — доля длинных свечей (размах > 1.5 ATR);
wick — доля теней в размахе; contraction — средний размах последней трети / первой трети.

Принятие (задано до запуска, как в formation.py): числовой признак — срезаем нижний или верхний квинтиль (границы по IS);
класс фигуры — убираем класс или оставляем только его; classic — убираем пробои против типичной стороны или оставляем
только типичные. Отрезанные сделки убыточны в каждом периоде, где их >= 20, t по дням <= −3, R в месяц с фильтром не
ниже. Решают срезы 4h (обе стороны) и 15m шорты; журнал — только иллюстрация.

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

PIV, SCALE, TOL, PAR = 2, 15, 0.75, 0.2
SHAPES = {"rect": "прямоугольник", "tri_asc": "восходящий треугольник", "tri_desc": "нисходящий треугольник",
          "tri_sym": "симметричный треугольник", "wedge_rise": "восходящий клин", "wedge_fall": "нисходящий клин",
          "chan_up": "восходящий канал", "chan_down": "нисходящий канал", "broad": "расширение",
          "none": "нет фигуры по канону (мало касаний)"}
CLASSIC = {"tri_asc": 1, "wedge_fall": 1, "tri_desc": -1, "wedge_rise": -1}
FEATS = ("apex_pos", "conv", "touch_gap", "touch_cv", "touch_err", "bounce", "body", "long_share", "wick", "contraction")
NAMES = {"apex_pos": "пробой: доля пути до вершины", "conv": "сходимость (ширина у пробоя / у опоры A)",
         "touch_gap": "шаг между касаниями, свечей", "touch_cv": "неравномерность касаний (σ/среднее)",
         "touch_err": "точность касаний, ATR", "bounce": "отскок между касаниями, ATR", "body": "тело свечи, ATR",
         "long_share": "доля длинных свечей", "wick": "доля теней", "contraction": "размах свечей: конец / начало"}


def minor_pivots(y: np.ndarray, a: int, t: int, top: bool, k: int = PIV) -> np.ndarray:
    """Развороты Bulkowski на a … t−1: вершина (top) без более высокой в окне ±k свечей, впадина — зеркально; окно не
    заходит за пробой (разворот подтверждён до t), у начала фигуры обрезается по A."""
    s = y if top else -y
    out = [j for j in range(a, t - k) if s[j] >= s[max(a, j - k):j + k + 1].max()]
    return np.array(out, dtype=int)


def boundary(c: np.ndarray, a: int, t: int, top: bool, tol: float) -> tuple[float, float, int] | None:
    """Верхняя (top) или нижняя граница фигуры по закрытиям A … t−1 → (наклон, значение в A, касаний) или None."""
    k = max(PIV, round((t - a) / SCALE))
    piv = minor_pivots(c, a, t, top, k)
    sg = 1.0 if top else -1.0
    z = sg * c
    hull: list[int] = []                                              # верхняя оболочка закрытий (j, z)
    for q in range(a, t):
        while len(hull) >= 2 and ((hull[-1] - hull[-2]) * (z[q] - z[hull[-2]])
                                  - (z[hull[-1]] - z[hull[-2]]) * (q - hull[-2])) >= 0:
            hull.pop()
        hull.append(q)
    best = None
    for p, q in zip(hull[:-1], hull[1:]):
        sl = (z[q] - z[p]) / (q - p)
        near = piv[np.abs(z[piv] - (z[p] + sl * (piv - p))) <= tol]
        touch, last = 0, -10 ** 9
        for j in near:                                               # касания — отдельные развороты
            if j - last >= k:
                touch += 1
                last = j
        if best is None or (touch, q - p) > best[0]:
            best = ((touch, q - p), sg * sl, sg * (z[p] + sl * (a - p)), touch)
    return None if best is None else best[1:]


def classify(su: float, sl: float, n: int, w0: float, w1: float, at: float, tu: int, tl: int) -> str:
    """Класс по наклонам верхней su и нижней sl границ (цена за свечу), длине n, ширине у A (w0) и у пробоя (w1),
    ATR и касаниям границ."""
    tol = TOL * at
    flat = lambda s: abs(s) * n <= tol
    band = max(tol, PAR * (abs(w0) + abs(w1)) / 2)
    if flat(su) and flat(sl):
        k = "rect"
    elif w1 - w0 < -band:
        k = ("tri_asc" if flat(su) else "tri_desc" if flat(sl) else "tri_sym" if su < 0 < sl
             else "wedge_rise" if su > 0 else "wedge_fall")
    elif w1 - w0 > band:
        k = "broad"
    else:
        mid = (su + sl) / 2
        k = "rect" if flat(mid) else "chan_up" if mid > 0 else "chan_down"
    if tu < 2 or tl < 2 or (not k.startswith("chan") and tu + tl < 5):
        return "none"
    return k


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
    tol = TOUCH_ATR * atr[t]
    up, dn = boundary(c, a, t, True, tol), boundary(c, a, t, False, tol)
    shape, conv, apex = "none", np.nan, np.nan
    geo = {"up_slope": np.nan, "up_a": np.nan, "up_n": 0, "dn_slope": np.nan, "dn_a": np.nan, "dn_n": 0}
    if up is not None and dn is not None:
        (su, ua, tu), (sl, la, tl) = up, dn
        n = t - 1 - a
        w0, w1 = ua - la, ua + su * n - (la + sl * n)
        shape = classify(su, sl, n, w0, w1, atr[t], tu, tl)
        conv = w1 / w0 if w0 > 0 else np.nan
        if w0 > 0 and sl - su > 0:
            apex = (t - a) / (w0 / (sl - su))                      # вершина: верх и низ встречаются
        geo = {"up_slope": su, "up_a": ua, "up_n": tu, "dn_slope": sl, "dn_a": la, "dn_n": tl}
    tp = touch_points(c, atr, a, t, side, sb)
    js = np.array([p[0] for p in tp])
    gaps = np.diff(js) if len(js) >= 2 else np.array([])
    errs = [e for jj, e in tp if jj not in (a, b)]
    inside = -side * (c - (c[a] + sb * (np.arange(len(c)) - a)))
    bounce = [inside[x:y].max() / atr[t] for x, y in zip(js[:-1], js[1:]) if y > x] if len(js) >= 2 else []
    ak = atr[a:t]
    ok = ak > 0
    rng, body = (h - lo)[a:t], np.abs(c - o)[a:t]
    third = max((t - a) // 3, 1)
    return {"shape": shape, "classic": side * CLASSIC.get(shape, 0), "apex_pos": apex, "conv": conv, **geo,
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
            print("    пробой в типичную сторону фигуры (Bulkowski) / против / фигура без типичной стороны:")
            for v, name in ((1, "типичная"), (-1, "против"), (0, "нет типичной")):
                m = z[z.classic == v]
                print(f"      {name} ({len(m) / max(len(z), 1):.0%}): {_periods(m)}  всё время: {_cell(m)}")
            for what, col, thr in (("убираем пробои против", "classic", -0.5), ("оставляем только типичные", "classic", 0.5)):
                if (z[col] < thr).sum() >= 30 and (z[col] >= thr).sum() >= 30:
                    for _, (ok, txt) in cut_verdicts(z, col, (thr,)).items():
                        print(f"      {what}: {'принято' if ok and primary else 'нет'}\n{txt}")
                        if ok and primary:
                            passed.append(f"{tf} {label}: {what}")
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
    rows = (len(keys) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 3.6 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, k in zip(axes.ravel(), keys):
        s, d, rec, t, f = cases[k]
        a, side, sb = rec["a"], rec["side"], rec["slope"]
        lo_, hi_ = max(a - 10, 0), min(t + 6, len(d) - 1)
        o, h, lo, c = (d[x].to_numpy()[lo_:hi_ + 1] for x in ("open", "high", "low", "close"))
        j = np.arange(lo_, hi_ + 1)
        ax.axis("on")
        ax.vlines(j, lo, h, color="#80868b", lw=0.6)
        up = c >= o
        ax.vlines(j[up], o[up], c[up], color="#188038", lw=2.2)
        ax.vlines(j[~up], c[~up], o[~up], color="#d93025", lw=2.2)
        jl = np.arange(a, t + 1)
        cc = d["close"].to_numpy()
        ax.plot(jl, cc[a] + sb * (jl - a), color="#202124", lw=1.6, label="линия бота (закрытия)")
        if np.isfinite(f["up_slope"]):
            ax.plot(jl, f["up_a"] + f["up_slope"] * (jl - a), color="#1a73e8", lw=1.2, ls="--",
                    label="границы фигуры")
            ax.plot(jl, f["dn_a"] + f["dn_slope"] * (jl - a), color="#1a73e8", lw=1.2, ls="--")
        ax.scatter([t], [cc[t]], color="#f29900", zorder=5, s=22)
        r = (s.get("trade") or {}).get("r")
        ax.set_title(f"{SHAPES[k]} (касаний {f['up_n']} + {f['dn_n']})\n{s['symbol']} {s['tf']} пробой "
                     f"{'вверх' if side > 0 else 'вниз'}{'' if r is None else f', {r:+.2f}R'}", fontsize=9)
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
