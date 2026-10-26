"""Стратегии чемпионов (документ «стратегии чемпионов мирового трейдинга»), перенесённые на крипту.

Правила заданы до запуска, по истории не подбираются. Издержки: вход и выход taker, тейк maker, funding перпетуалов;
стоп проверяется раньше тейка; если в свече входа достигнут и стоп, первым считается уровень, ближний к открытию.
Оборот монеты — от $20M в день (перпетуалы) / $5M (спот). Одна позиция на монету в варианте системы.

  W  Ларри Вильямс, пробой волатильности (перпетуалы, путь внутри дня по 1h). День t: уровни Open_t ± k·Range_{t−1}
     (k = 0.5 / 0.7), вход стоп-ордером на первом касании (оба — первым ближний к открытию часа), стоп — открытие дня.
     Выходы: «закрытие дня»; «bailout» — первое открытие следующих дней в плюсе, не дольше 4 дней. COT у крипты нет —
     вместо него тренд недели (закрытие недели против EMA50, как у бота): строки «обе стороны» и «по тренду».
  K  Оливер Келл, ценовой цикл (дневки перпетуалов и спота, 4h перпетуалов). «Разворот»: закрытие пересекло EMA20
     и выше EMA10 на объёме от 1.5× среднего за 20; «первый откат» в следующие 20 свечей: минимум коснулся EMA10,
     закрытие не ниже EMA20, EMA10 > EMA20; вход следующей свечой стоп-ордером над максимумом касания, стоп — его
     минимум. Выходы: «нарушение EMA20» (закрытие ниже, до 120 свечей); «3R» (до 60). Шорт зеркально.
  F  High-tight flag (Люк, Лай), дневки. Флагшток: максимум за 15 дней — от +90% к минимуму 40 дней до него; флаг
     3–15 дней с откатом не глубже 25%; вход на первом закрытии выше максимума при объёме от 1.5× среднего; стоп —
     минимум флага. Выходы: «48 ч, затем EMA10» — выход на закрытии 2-го дня, если ход меньше 1R, иначе по закрытию
     ниже EMA10 (до 60 дней); «3R». Шорт зеркально (справочно).
  O  Opening Range Breakout (Унгер, Смолен), 15m перпетуалов. Сессии: 00:00 UTC (до конца суток) и 13:30 UTC
     (открытие США, до 20:00); диапазон первых 30 / 60 минут; вход стоп-ордером на первом пробое, стоп — другая
     граница (риск от 0.2% цены); выходы: «конец сессии», «2R». Строки «обе стороны» и «по дневному тренду».

Принятие (на строку): за всё время t по дням >= 3 (около 50 строк — Harvey, Liu, Zhu), плюс в каждом периоде
2020 / 2021 / IS / VAL / HO, где сделок >= 20, плюс при издержках ×2; для дневных K и F — ещё и плюс на споте.

    python -m research.champs collect --symbols <1h-qualified> [--spot]     # CHAMPS=W,K,F,O
    python -m research.champs report
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30
from .htftrend import PER as SPOT_PER, list_spot, load_daily
from .masters import ema, sma, walk
from .oos import PER_ALL
from .ppr import day_t
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import MAKER, TAKER, htf_trend, tf_frame

SYSTEMS = tuple(os.environ.get("CHAMPS", "W,K,F,O").split(","))
SPOT_LIQ = 5e6
W_K = (0.5, 0.7)
W_DAYS = 4
POP_VOL, PULL_BARS = 1.5, 20
F_POLE, F_LOOK, F_BASE, F_DEPTH, F_MIN, F_MAX = 1.9, 40, 15, 0.25, 3, 15
ORB_SESS = {"00:00": ("00:00", None), "13:30": ("13:30", "20:00")}
ORB_MIN = (30, 60)
ORB_RISK_MIN = 0.002
NAMES = {"W": "Вильямс: пробой волатильности", "K": "Келл: первый откат после разворота",
         "F": "High-tight flag", "O": "Opening Range Breakout"}


def walk_bars(o, hi, lo, c, f, e: int, side: int, entry: float, stop: float, last: int, *, tp: float | None = None,
              open_exit: np.ndarray | None = None, entry_fee: float = TAKER) -> tuple[float, int]:
    """Сделка внутри бара e (стоп-ордер) до бара last включительно: (R с издержками и funding, бар выхода).
    open_exit[j] — на открытии бара j выйти, если оно в плюсе (bailout Вильямса)."""
    risk = side * (entry - stop)
    if not risk > 0:
        return np.nan, -1

    def r_at(px: float, fee: float, paid: float) -> float:
        return (side * (px - entry) - entry_fee * entry - fee * px - side * paid * entry) / risk

    if ((side > 0 and lo[e] <= stop) or (side < 0 and hi[e] >= stop)) and abs(o[e] - entry) <= abs(o[e] - stop):
        return r_at(stop, TAKER, 0.0), e
    paid = 0.0
    for j in range(e + 1, last + 1):
        paid += f[j]
        if open_exit is not None and open_exit[j] and side * (o[j] - entry) > 0:
            return r_at(o[j], TAKER, paid), j
        if (side > 0 and lo[j] <= stop) or (side < 0 and hi[j] >= stop):
            return r_at(min(stop, o[j]) if side > 0 else max(stop, o[j]), TAKER, paid), j
        if tp is not None and ((side > 0 and hi[j] >= tp) or (side < 0 and lo[j] <= tp)):
            return r_at(tp, MAKER, paid), j
    return r_at(c[last], TAKER, paid), last


def _row(sys_: str, var: str, ex: str, src: str, tf: str, sym: str, t, side: int, r: float, entry: float,
         stop: float, trend: int = 0) -> dict:
    return {"sys": sys_, "var": var, "exit": ex, "src": src, "tf": tf, "symbol": sym, "t": t, "side": side,
            "R": float(r), "c2": 2 * TAKER * entry / abs(entry - stop), "trend": int(trend)}


def williams_rows(h: pd.DataFrame, liq_day: pd.Series, sym: str) -> list[dict]:
    """W по часовым свечам перпетуала; дневные уровни — из тех же часов (сутки UTC)."""
    o, hi, lo, c = (h[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = h["funding"].fillna(0.0).to_numpy(dtype="float64") if "funding" in h else np.zeros(len(c))
    day = h.index.floor("D")
    dd = h.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    week = pd.Series(htf_trend(dd, "1d"), index=dd.index)
    datr = _atr(dd)
    starts = np.r_[True, day[1:] != day[:-1]]
    first = np.flatnonzero(starts)
    bounds = list(zip(first, np.r_[first[1:], len(c)]))
    rows: list[dict] = []
    busy: dict[str, int] = {}
    for k_i in range(1, len(bounds)):
        a, b = bounds[k_i]
        pa, pb = bounds[k_i - 1]
        dname = day[a]
        if b - a < 20 or pb - pa < 20 or not (liq_day.get(dname, 0) >= ADV_MIN):
            continue
        rng = hi[pa:pb].max() - lo[pa:pb].min()
        atr = datr.get(dname - pd.Timedelta("1D"), np.nan)
        if not (rng > 0 and atr > 0):
            continue
        tr = int(week.get(dname - pd.Timedelta("1D"), 0) or 0)
        for k in W_K:
            up, dn = o[a] + k * rng, o[a] - k * rng
            for j in range(a, b):
                hu, hd = hi[j] >= up, lo[j] <= dn
                if not (hu or hd):
                    continue
                side = (1 if up - o[j] <= o[j] - dn else -1) if (hu and hd) else (1 if hu else -1)
                entry = max(up, o[j]) if side > 0 else min(dn, o[j])
                stop = o[a]
                if not 0.3 * atr <= side * (entry - stop) <= 4 * atr:
                    break
                end_k = min(k_i + W_DAYS, len(bounds) - 1)
                exits = {"закрытие дня": walk_bars(o, hi, lo, c, f, j, side, entry, stop, b - 1)}
                key = f"bail{k}"
                if busy.get(key, -1) < j:
                    r, ex = walk_bars(o, hi, lo, c, f, j, side, entry, stop, bounds[end_k][1] - 1, open_exit=starts)
                    exits["bailout"] = (r, ex)
                    busy[key] = ex
                for name, (r, _) in exits.items():
                    if np.isfinite(r):
                        rows.append(_row("W", f"k={k}", name, "perp", "1h", sym, h.index[j], side, r, entry, stop, tr))
                break
    return rows


def kell_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, src: str, tf: str) -> list[dict]:
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    v = d["volume"].to_numpy(dtype="float64")
    f = d["funding"].fillna(0.0).to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    a = _atr(d).to_numpy()
    e10, e20, vavg = ema(c, 10), ema(c, 20), sma(v, 20)
    lim = SPOT_LIQ if src == "spot" else ADV_MIN
    rows: list[dict] = []
    armed = {1: -1, -1: -1}
    busy: dict[str, int] = {}
    for t in range(25, len(c) - 1):
        for side in (1, -1):
            if (side * (c[t - 1] - e20[t - 1]) <= 0 < side * (c[t] - e20[t]) and side * (c[t] - e10[t]) > 0
                    and v[t] >= POP_VOL * vavg[t - 1]):
                armed[side] = t
                continue
            if armed[side] < 0 or t - armed[side] > PULL_BARS:
                armed[side] = -1
                continue
            touch = (lo[t] <= e10[t]) if side > 0 else (hi[t] >= e10[t])
            if not (touch and side * (c[t] - e20[t]) >= 0 and side * (e10[t] - e20[t]) > 0):
                continue
            armed[side] = -1                                   # только первый откат
            if not (liq[t] >= lim and a[t] > 0):
                continue
            lvl, stop = (hi[t], lo[t]) if side > 0 else (lo[t], hi[t])
            if not side * ((hi[t + 1] if side > 0 else lo[t + 1]) - lvl) > 0:
                continue
            entry = max(lvl, o[t + 1]) if side > 0 else min(lvl, o[t + 1])
            if side * (entry - stop) < 0.3 * a[t]:
                stop = entry - side * 0.3 * a[t]
            risk = side * (entry - stop)
            if risk > 4 * a[t]:
                continue
            viol: Callable[[int], bool] = lambda j, s=side: s * (c[j] - e20[j]) < 0
            for name, kw, hold in (("нарушение EMA20", {"close_exit": viol}, 120),
                                   ("3R", {"tp": entry + side * 3 * risk}, 60)):
                key = f"{name}{side}"
                if busy.get(key, -1) >= t + 1:
                    continue
                r, ex = walk(o, hi, lo, c, f, t + 1, side, entry, stop, hold, stop_entry_bar=True, **kw)
                if np.isfinite(r):
                    busy[key] = ex
                    rows.append(_row("K", tf, name, src, tf, sym, d.index[t + 1], side, r, entry, stop))
    return rows


def flag_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, src: str) -> list[dict]:
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    v = d["volume"].to_numpy(dtype="float64")
    f = d["funding"].fillna(0.0).to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    a = _atr(d).to_numpy()
    e10, vavg = ema(c, 10), sma(v, 20)
    lim = SPOT_LIQ if src == "spot" else ADV_MIN
    rows: list[dict] = []
    busy: dict[str, int] = {}
    for t in range(F_LOOK + F_BASE + 1, len(c) - 1):
        if not (liq[t] >= lim and a[t] > 0 and v[t] >= POP_VOL * vavg[t - 1]):
            continue
        for side in (1, -1):
            w = hi[t - F_BASE:t] if side > 0 else lo[t - F_BASE:t]
            p = t - F_BASE + int(np.argmax(w) if side > 0 else np.argmin(w))
            if not F_MIN <= t - p <= F_MAX:
                continue
            ext = hi[p] if side > 0 else lo[p]
            base = lo[p - F_LOOK:p].min() if side > 0 else hi[p - F_LOOK:p].max()
            pole = ext / base if side > 0 else base / ext
            flag = lo[p + 1:t + 1].min() if side > 0 else hi[p + 1:t + 1].max()
            depth = (1 - flag / ext) if side > 0 else (1 - ext / flag)
            if not (pole >= F_POLE and depth <= F_DEPTH and side * (c[t] - ext) > 0):
                continue
            entry, stop = c[t], flag
            if side * (entry - stop) < 0.3 * a[t]:
                stop = entry - side * 0.3 * a[t]
            risk = side * (entry - stop)
            if risk > 4 * a[t]:
                continue
            stall: Callable[[int], bool] = lambda j, s=side, e0=entry, rk=risk, t0=t: (
                (j - t0 == 2 and s * (c[j] - e0) < rk) or (j - t0 > 2 and s * (c[j] - e10[j]) < 0))
            for name, kw, hold in (("48 ч, затем EMA10", {"close_exit": stall}, 60),
                                   ("3R", {"tp": entry + side * 3 * risk}, 60)):
                key = f"{name}{side}"
                if busy.get(key, -1) >= t:
                    continue
                r, ex = walk(o, hi, lo, c, f, t, side, entry, stop, hold, **kw)
                if np.isfinite(r):
                    busy[key] = ex
                    rows.append(_row("F", "1d", name, src, "1d", sym, d.index[t], side, r, entry, stop))
    return rows


def orb_rows(d: pd.DataFrame, liq_day: pd.Series, sym: str) -> list[dict]:
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].fillna(0.0).to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    trend = htf_trend(d, "15m", rule="1D")
    pos = pd.Series(np.arange(len(c)), index=d.index)
    rows: list[dict] = []
    for dname in d.index.floor("D").unique():
        if not (liq_day.get(dname, 0) >= ADV_MIN):
            continue
        for sess, (beg, end) in ORB_SESS.items():
            t0 = dname + pd.Timedelta(beg + ":00")
            t_end = dname + (pd.Timedelta(end + ":00") if end else pd.Timedelta("1D")) - pd.Timedelta("15min")
            if t0 not in pos.index or t_end not in pos.index:
                continue
            s0, s1 = int(pos[t0]), int(pos[t_end])
            if s1 - s0 + 1 != int((t_end - t0) / pd.Timedelta("15min")) + 1:
                continue                                       # дыра в свечах внутри сессии
            for mins in ORB_MIN:
                nb = mins // 15
                orh, orl = hi[s0:s0 + nb].max(), lo[s0:s0 + nb].min()
                for j in range(s0 + nb, s1 + 1):
                    hu, hd = hi[j] >= orh, lo[j] <= orl
                    if not (hu or hd):
                        continue
                    side = (1 if orh - o[j] <= o[j] - orl else -1) if (hu and hd) else (1 if hu else -1)
                    entry = max(orh, o[j]) if side > 0 else min(orl, o[j])
                    stop = orl if side > 0 else orh
                    risk = side * (entry - stop)
                    if risk < ORB_RISK_MIN * entry:
                        break
                    tr = int(trend[j - 1]) if np.isfinite(trend[j - 1]) else 0
                    for name, tp in (("конец сессии", None), ("2R", entry + side * 2 * risk)):
                        r, _ = walk_bars(o, hi, lo, c, f, j, side, entry, stop, s1, tp=tp)
                        if np.isfinite(r):
                            rows.append(_row("O", f"{sess}, {mins} мин", name, "perp", "15m", sym, d.index[j], side,
                                             r, entry, stop, tr))
                    break
    return rows


def _liq_day(root: Path, s: str) -> pd.Series:
    x = adv30(root, s)
    x.index = pd.to_datetime(x.index, utc=True)
    return x


def collect(root: Path, syms: list[str], spot: bool) -> None:
    rows: list[dict] = []
    for s in mine(syms):
        try:
            liq = _liq_day(root, s)
            if "W" in SYSTEMS:
                h = tf_frame(root, s, "1h")
                if h is not None and len(h) > 24 * 60:
                    rows += williams_rows(h, liq, s)
            for tf in ("1d", "4h"):
                if not ({"K", "F"} & set(SYSTEMS)):
                    break
                d = tf_frame(root, s, tf)
                if d is None or len(d) < 120:
                    continue
                lv = liq.reindex(d.index.floor("D")).to_numpy()
                if "K" in SYSTEMS:
                    rows += kell_rows(d, lv, s, "perp", tf)
                if "F" in SYSTEMS and tf == "1d":
                    rows += flag_rows(d, lv, s, "perp")
            if "O" in SYSTEMS and (root / f"{s}-15m.parquet").exists():
                d = tf_frame(root, s, "15m")
                if d is not None and len(d) > 96 * 60:
                    rows += orb_rows(d, liq, s)
        except Exception as e:
            print(f"  champs {s}: пропуск ({e})", flush=True)
    if spot and ({"K", "F"} & set(SYSTEMS)):
        names = mine(list_spot())
        cache = root / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        end = os.environ.get("RESEARCH_END", "2026-09")
        with ThreadPoolExecutor(16) as pool:
            frames = dict(zip(names, pool.map(lambda x: load_daily(x, cache, end), names)))
        for s, d in frames.items():
            try:
                if d is None or len(d) < 120:
                    continue
                d = d.assign(funding=0.0)
                lv = d["quote_volume"].rolling(30, min_periods=20).mean().shift(1).to_numpy()
                if "K" in SYSTEMS:
                    rows += kell_rows(d, lv, s, "spot", "1d")
                if "F" in SYSTEMS:
                    rows += flag_rows(d, lv, s, "spot")
            except Exception as e:
                print(f"  champs spot {s}: пропуск ({e})", flush=True)
    print(f"  champs: сделок {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("champs"), index=False)


def line(z: pd.DataFrame, per: dict[str, tuple[str, str]]) -> tuple[str, bool, float]:
    """Строка по периодам и вердикт по критерию модуля (без условия спота)."""
    cells, pos = [], []
    for p, (a, b) in per.items():
        m = z[(z.t >= a) & (z.t < b)]
        if m.empty:
            continue
        cells.append(f"{p}: {_cell(m)}")
        if len(m) >= 20:
            pos.append(m.R.mean() > 0)
    t = day_t(z, "R") if len(z) > 10 else float("nan")
    x2 = (z.R - z.c2).mean() if len(z) else float("nan")
    ok = bool(pos) and all(pos) and t >= 3 and x2 > 0
    return "  ".join(cells) + f"  всё время: {_cell(z)}  x2: {x2:+.3f}R", ok, t


def report() -> None:
    parts = all_parts("champs")
    print("\n===== CHAMPS: стратегии чемпионов на крипте — Вильямс, Келл, High-tight flag, ORB (t по дням, прибыльных, "
          "сделок в месяц; x2 — издержки вдвое) =====")
    if not parts:
        print("  champs: нет данных")
        return
    x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
    x = x.dropna(subset=["R"])
    passed = []
    for sys_ in SYSTEMS:
        xs = x[x.sys == sys_]
        if xs.empty:
            print(f"\n  {NAMES[sys_]}: сделок нет")
            continue
        print(f"\n  --- {NAMES[sys_]} ---")
        for (var, ex), g in xs[xs.src == "perp"].groupby(["var", "exit"], sort=True):
            if sys_ in ("W", "O"):
                modes = (("обе стороны", g), ("по тренду", g[g.side == g.trend]))
            else:
                modes = (("лонг", g[g.side > 0]), ("шорт", g[g.side < 0]))
            for mode, z in modes:
                if len(z) < 20:
                    continue
                txt, ok, t = line(z, PER_ALL)
                spot_note = ""
                if sys_ in ("K", "F") and var == "1d":
                    sp = xs[(xs.src == "spot") & (xs["var"] == var) & (xs.exit == ex)
                            & (xs.side == (1 if mode == "лонг" else -1))]
                    stxt, _, _ = line(sp, SPOT_PER) if len(sp) >= 20 else ("мало сделок", False, np.nan)
                    spot_note = f"\n        спот с 2017: {stxt}"
                    ok = ok and len(sp) >= 20 and sp.R.mean() > 0
                print(f"    {var}, {ex}, {mode}: {txt}{spot_note}  → {'принято' if ok else 'нет'}")
                if ok:
                    passed.append(f"{sys_} {var} {ex} {mode} (t {t:.1f})")
    print("\n  ИТОГ: " + ("прошли — " + "; ".join(passed) if passed else "ни одна стратегия чемпионов не принята"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--spot", action="store_true")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [s for s in a.symbols.split(",") if s], a.spot)
    else:
        report()
