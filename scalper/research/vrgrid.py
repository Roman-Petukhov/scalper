"""Сетка с отбором монет по статистике возврата цены — последняя непроверенная ось сетки (просьба владельца). Раньше:
адаптивная сетка на 5m (research/grid.py; фильтр «боковик» ER < 0.3 — провал на VAL), сетка вокруг EMA200
(research/emagrid.py; 864 варианта, 0 принято). Фильтр по виду графика запаздывает; здесь — по тому, возвращается ли
цена на горизонте шага: variance ratio (Lo, MacKinlay 1988) и автокорреляция часовых доходностей за 30 дней.

Сетка — research.grid.grid_machine без изменений: центр — EMA за сутки, 5 ступеней на сторону, каждая 1/5 капитала
монеты, тейк — на соседнем уровне, стоп за последним уровнем, худший порядок исполнения внутри свечи; maker 2 б.п. на
лимиты, taker 5.5 б.п. на стоп, funding. Шаг = max(m × ATR в часовом масштабе, 0.6% цены) — не меньше 15 кругов
комиссий. Свечи 15m и 1h (собраны из 5m), 70 монет core70, капитал поровну.
Фильтры (известны на закрытии прошлой свечи; заданы до запуска):
    нет            сетка всегда
    VR < 0.9       VR(12) часовых доходностей за 30 дней ниже 0.9 (возврат на горизонте 12 ч)
    VR < 0.8       то же, строже
    VR < 0.8, AC   и автокорреляция часовых доходностей за 30 дней < −0.05
Варианты: 2 ТФ × m 1 / 2 × 4 фильтра = 16. Принятие: Sharpe > 0 в каждом периоде 2021 / IS / VAL / HO и t >= 3 за всё
время, и в плюсе при издержках x2. Для справки — держится ли VR: корреляция VR за 30 дней с VR следующих 30 дней.

    python -m research.vrgrid collect --symbols <core70>
    python -m research.vrgrid report
"""
from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .grid import grid_machine
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .tline import tf_frame

DAY0 = pd.Timestamp("2020-01-01", tz="UTC")
TFS = {"15m": "15min", "1h": "1h"}
BARS_DAY = {"15m": 96, "1h": 24}
MS = (1.0, 2.0)
FILTERS = ("нет", "VR < 0.9", "VR < 0.8", "VR < 0.8, AC")
LEVELS = 5
MIN_STEP = 0.006
VR_Q, VR_DAYS = 12, 30
COSTS = {"x1": (2.0, 5.5), "x2": (4.0, 11.0)}
VARIANTS = list(itertools.product(TFS, MS, FILTERS))


def bars(b5: pd.DataFrame, rule: str) -> pd.DataFrame:
    g = b5.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "funding": "sum"}).dropna(subset=["close"])
    return g


def mean_reversion(c1h: pd.Series) -> pd.DataFrame:
    """VR(q) и автокорреляция часовых лог-доходностей за VR_DAYS дней (известны на закрытии часа)."""
    r = np.log(c1h).diff()
    w = VR_DAYS * 24
    var1 = r.rolling(w, min_periods=w // 2).var()
    rq = r.rolling(VR_Q).sum()
    varq = rq.rolling(w, min_periods=w // 2).var()
    ac = r.rolling(w, min_periods=w // 2).corr(r.shift(1))
    return pd.DataFrame({"vr": varq / (VR_Q * var1), "ac": ac})


def active_mask(mr: pd.DataFrame, filt: str) -> np.ndarray:
    if filt == "нет":
        return np.ones(len(mr), dtype=np.bool_)
    thr = 0.9 if filt == "VR < 0.9" else 0.8
    m = (mr["vr"] < thr).to_numpy().copy()
    if filt.endswith("AC"):
        m &= (mr["ac"] < -0.05).to_numpy()
    return np.asarray(m, dtype=np.bool_)


def coin_days(b5: pd.DataFrame, n_days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(доходность по дням [вариант × издержки, день], активные дни монеты, пары VR сейчас / VR через 30 дней)."""
    h1 = bars(b5, "1h")
    mr = mean_reversion(h1["close"])
    mr.index = mr.index + pd.Timedelta("1h")                  # известно на закрытии часа
    out = np.zeros((len(VARIANTS) * len(COSTS), n_days), dtype=np.float64)
    active = np.zeros(n_days)
    for tf, rule in TFS.items():
        g = bars(b5, rule)
        c = g["close"]
        pc = c.shift(1)
        tr = pd.concat([g["high"] - g["low"], (g["high"] - pc).abs(), (g["low"] - pc).abs()], axis=1).max(axis=1)
        nd = BARS_DAY[tf]
        atr1h = (tr.ewm(alpha=1.0 / nd, adjust=False, min_periods=nd).mean() * np.sqrt(60 / (15 if tf == "15m" else 60)))
        anchor = c.ewm(span=nd, adjust=False, min_periods=nd).mean().to_numpy()
        m_tf = mr.reindex(g.index + pd.Timedelta(rule), method="ffill")
        m_tf.index = g.index
        day = ((g.index - DAY0) // pd.Timedelta("1D")).to_numpy().astype(np.int64)
        ok = (day >= 0) & (day < n_days)
        active[np.unique(day[ok])] = 1.0
        o, hi, lo, cl, f = (g[k].to_numpy(dtype=np.float64) for k in ("open", "high", "low", "close", "funding"))
        f = np.nan_to_num(f)
        for (vi, (vtf, m, filt)) in enumerate(VARIANTS):
            if vtf != tf:
                continue
            step = np.maximum(m * atr1h.to_numpy(), MIN_STEP * cl)
            act = active_mask(m_tf, filt)
            for ci, (mk, tk) in enumerate(COSTS.values()):
                ret, _, _, _, _ = grid_machine(o, hi, lo, cl, f, anchor, step, act, LEVELS, mk, tk)
                out[vi * len(COSTS) + ci] = np.bincount(day[ok], weights=ret[ok], minlength=n_days)[:n_days]
    return out, active, persistence(mr)


def persistence(mr: pd.DataFrame) -> np.ndarray:
    """Пары (VR сейчас, VR через 30 дней) раз в 30 дней — держится ли признак возврата."""
    s = mr["vr"].dropna()
    s = s[s.index.hour == 0]
    s = s.iloc[::VR_DAYS]
    return np.c_[s.to_numpy()[:-1], s.to_numpy()[1:]] if len(s) > 2 else np.zeros((0, 2))


def collect(root: Path, syms: list[str]) -> None:
    n_days = int((pd.Timestamp.now(tz="UTC").normalize() - DAY0).days) + 2
    tot = np.zeros((len(VARIANTS) * len(COSTS), n_days))
    cnt = np.zeros(n_days)
    pairs = []
    for s in mine(syms):
        try:
            b5 = tf_frame(root, s, "5m")
            if b5 is None or len(b5) < 288 * 120:
                continue
            if "funding" not in b5:
                b5["funding"] = 0.0
            p, act, pr = coin_days(b5, n_days)
            tot += p
            cnt += act
            pairs.append(pr)
            print(f"  vrgrid {s}: дней {int(act.sum())}", flush=True)
        except Exception as e:
            print(f"  vrgrid {s}: пропуск ({e})", flush=True)
    cols = {f"p{i}": tot[i] for i in range(len(tot))}
    pd.DataFrame({"day": np.arange(n_days), "n": cnt, **cols}).to_parquet(part_path("vrgrid"), index=False)
    if pairs:
        pd.DataFrame(np.vstack(pairs), columns=["vr", "vr_next"]).to_parquet(part_path("vrgrid_vr"), index=False)


def stats(r: pd.Series) -> dict:
    if len(r) < 30 or r.std() == 0:
        return {"ann": np.nan, "sh": np.nan, "dd": np.nan, "t": np.nan}
    eq = r.cumsum()
    return {"ann": r.mean() * 365, "sh": r.mean() / r.std() * np.sqrt(365), "dd": float((eq.cummax() - eq).max()),
            "t": r.mean() / r.std() * np.sqrt(len(r))}


def report() -> None:
    print("\n===== VRGRID: сетка с отбором по variance ratio и автокорреляции (core70; капитал поровну; в год / Sharpe / "
          "просадка на капитал сетки) =====")
    parts = all_parts("vrgrid")
    if not parts:
        print("  vrgrid: нет данных")
        return
    x = sum(pd.read_parquet(p).set_index("day") for p in parts)
    idx = DAY0.tz_localize(None) + pd.to_timedelta(x.index, unit="D")
    n = x["n"].replace(0, np.nan)
    vr = all_parts("vrgrid_vr")
    if vr:
        v = pd.concat([pd.read_parquet(p) for p in vr], ignore_index=True).dropna()
        low = v[v.vr < 0.8]
        print(f"  держится ли VR: корреляция VR за 30 дней со следующими 30 днями {v.vr.corr(v.vr_next):+.2f} "
              f"({len(v)} пар); у VR < 0.8 следующий VR в среднем {low.vr_next.mean():.2f} (доля < 0.8 — "
              f"{(low.vr_next < 0.8).mean():.0%}), у всех {v.vr_next.mean():.2f}")
    passed = []
    for vi, (tf, m, filt) in enumerate(VARIANTS):
        cells, ok = [], []
        res = {}
        for ci, cname in enumerate(COSTS):
            r = pd.Series((x[f"p{vi * len(COSTS) + ci}"] / n).fillna(0.0).to_numpy(), index=idx)
            r = r[(r.index >= "2020-06-01") & (r.index < "2026-10-01")]
            res[cname] = r
        r = res["x1"]
        for p, (a, b) in PER_ALL.items():
            st = stats(r[(r.index >= a) & (r.index < b)])
            if not np.isfinite(st["sh"]):
                continue
            cells.append(f"{p} {st['ann']:+.0%} / {st['sh']:.2f}")
            if p != "2020":
                ok.append(st["sh"] > 0)
        al, a2 = stats(r), stats(res["x2"])
        good = bool(ok) and all(ok) and al["t"] >= 3 and a2["ann"] > 0
        print(f"    {tf}, шаг {m:g} ATR, {filt}: {'принято' if good else 'нет'}\n      {'  '.join(cells)}\n"
              f"      всё время {al['ann']:+.0%} в год, Sharpe {al['sh']:.2f}, t {al['t']:.1f}, просадка {al['dd']:.0%}; "
              f"издержки x2: {a2['ann']:+.0%} в год")
        if good:
            passed.append(f"{tf} {m:g} ATR {filt}")
    print("\n  ИТОГ: " + ("прошли — " + "; ".join(passed) if passed else
                         "сетка не зарабатывает и с отбором по возврату цены"))


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
