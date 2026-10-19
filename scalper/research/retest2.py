"""Где ставить лимитку ретеста (по вопросу владельца): от уровня пробоя, от самой трендовой или после обновления
ближайшего свинга с закреплением над трендовой.

Сигналы — правило бота без изменений (research/noline.py: пробой линии зигзага 3 ATR закрытием, оборот от $20M,
агрессоры ≥ 55%, закрытие в своей половине свечи, тренд старшего ТФ, стоп за свингом PIV5 ± 0.1 ATR в пределах
0.3–4 ATR, цель 3R, удержание как у бота). Меняется только вход; стоп у всех вариантов — стоп бота. Ниже — для лонга
(пробой нисходящей линии вверх), шорт зеркально. line(j) — значение продолженной линии на свече j.
  level  лимитка на значении линии в свече пробоя (горизонтально), 12 свечей — бот, контроль;
  line   лимитка едет по линии: исполнение на свече j, если low[j] ≤ line(j), по line(j); 12 свечей;
  swing  S — ближайшая вершина PIV5 над закрытием пробоя (самая поздняя подтверждённая из лежащих после первой опоры
         линии и выше close[t]). Ждём закрытие выше S («обновили свинг») не дольше 24 свечей, все закрытия до этого —
         выше линии («закрепились»); затем лимитка на S на 12 свечей, отмена при закрытии ниже линии. Нет S — сигнал
         в этом варианте пропущен (0R);
  swing_mkt  вход по рынку на закрытии, обновившем S (справочно).
Все лимитки отменяются, если раньше исполнения цена дошла до 3R от цены лимитки (как у бота). На свече исполнения
касание стопа — убыток (как у бота).

Принятие (задано до запуска), 4h: R на сигнал (неисполненный = 0) у варианта выше, чем у «level», в каждом периоде
(2020 / 2021 / IS / VAL / HO) и парная разница по тем же сигналам за всё время с t ≥ 2. 15m — справочно (бот 15m
входит по рынку; там же видно «только шорты»).

    python -m research.retest2 collect --symbols <1h-qualified>      (4h; 15m — где есть 15m-свечи)
    python -m research.retest2 report
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
from .tline import (HOLD, MAKER, PIV, RETEST_BARS, STOP_BUF, TAKER, htf_trend, last_confirmed, pivots, tf_frame,
                    two_targets, zz_lines)

TFS = tuple(os.environ.get("RETEST2_TFS", "4h,15m").split(","))
CONFIRM_BARS, K = 24, 3.0
VARIANTS = {"level": "уровень пробоя (бот)", "line": "по трендовой", "swing": "свинг обновлён, ретест свинга",
            "swing_mkt": "свинг обновлён, по рынку"}


def nearest_swing(hi_or_lo: np.ndarray, piv: np.ndarray, a: int, t: int, side: int, close_t: float) -> float:
    """Ближайший свинг против сделки, ещё не пробитый закрытием пробоя: самая поздняя вершина (для лонга), подтверждённая
    к бару t, лежащая после опоры a и выше close_t. NaN — нет."""
    for p, conf in piv[::-1]:
        if conf > t:
            continue
        if p <= a:
            break
        if side * (hi_or_lo[p] - close_t) > 0:
            return float(hi_or_lo[p])
    return np.nan


def outcome(o, h, lo, c, f, fill, side, px, stop, fee, tf) -> float:
    risk = side * (px - stop)
    if not (risk > 0):
        return np.nan
    if fee == MAKER and ((side > 0 and lo[fill] <= stop) or (side < 0 and h[fill] >= stop)):
        return (side * (stop - px) - (fee + TAKER) * px) / risk
    return two_targets(o, h, lo, c, f, fill, side, px, stop, K, K, False, HOLD[tf], fee)[0]


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str) -> list[dict]:
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    atr = _atr(d).to_numpy()
    trend = htf_trend(d, tf)
    buy = (d["taker_buy_volume"] / d["volume"].replace(0, np.nan)).to_numpy()
    rng = hi - lo
    loc = (c - lo) / np.where(rng > 0, rng, np.nan)
    ph, pl = pivots(hi, PIV, True), pivots(lo, PIV, False)
    sw_lo, sw_hi = last_confirmed(pl, len(c)), last_confirmed(ph, len(c))
    rows = []
    for r in zz_lines(d):
        t, side, a = int(r["t"]), int(r["side"]), int(r["a"])
        if t <= 0 or t >= len(c) - 1 or not (atr[t] > 0) or not (liq[t] >= ADV_MIN) or trend[t] != side:
            continue
        aggr = buy[t] if side > 0 else 1 - buy[t]
        cl = loc[t] if side > 0 else 1 - loc[t]
        if not (aggr >= 0.55 and cl >= 0.5):
            continue
        sw = sw_lo[t] if side > 0 else sw_hi[t]
        if sw < 0:
            continue
        stop = (lo[sw] - STOP_BUF * atr[t]) if side > 0 else (hi[sw] + STOP_BUF * atr[t])
        line_at = (lambda j, a=a, s=r["slope"]: c[a] + s * (j - a))
        level = float(r["line_t"])
        if not (0.3 * atr[t] <= side * (level - stop) <= 4.0 * atr[t]):
            continue                                   # сигнал бота (его ретест) — общий для всех вариантов
        out = {"symbol": sym, "tf": tf, "t": d.index[t], "side": side, "risk_atr": side * (level - stop) / atr[t]}

        def take(px_fill: tuple[int, float], fee: float) -> float:
            j, px = px_fill
            if j < 0 or not (0.3 * atr[t] <= side * (px - stop) <= 4.0 * atr[t]):
                return 0.0
            v = outcome(o, hi, lo, c, f, j, side, px, stop, fee, tf)
            return float(v) if np.isfinite(v) else 0.0

        # level — горизонталь на line_t, как у бота
        lv = _fill_tp(o, hi, lo, c, side, t + 1, RETEST_BARS, lambda j: level, stop)
        out["level"], out["level_fill"] = take(lv, MAKER), lv[0] >= 0
        ln = _fill_tp(o, hi, lo, c, side, t + 1, RETEST_BARS, line_at, stop)
        out["line"], out["line_fill"] = take(ln, MAKER), ln[0] >= 0
        S = nearest_swing(hi if side > 0 else lo, ph if side > 0 else pl, a, t, side, c[t])
        out["has_swing"] = bool(np.isfinite(S))
        out["swing"], out["swing_mkt"], out["swing_fill"] = 0.0, 0.0, False
        if np.isfinite(S):
            k = -1
            for j in range(t + 1, min(t + 1 + CONFIRM_BARS, len(c) - 1)):
                if side * (c[j] - line_at(j)) <= 0:
                    break                              # закрылись обратно за линию — не закрепились
                if side * (c[j] - S) > 0:
                    k = j
                    break
            if k > 0:
                out["swing_mkt"] = take((k, c[k]), TAKER)
                sf = _fill_tp(o, hi, lo, c, side, k + 1, RETEST_BARS, lambda j: S, stop, cancel_line=line_at)
                out["swing"], out["swing_fill"] = take(sf, MAKER), sf[0] >= 0
        rows.append(out)
    return rows


def _fill_tp(o, hi, lo, c, side, start, bars, price_at, stop, cancel_line=None) -> tuple[int, float]:
    """Лимитка price_at(j) с бара start на bars свечей. Отмена до исполнения: цена дошла до 3R от лимитки или (если
    задана cancel_line) закрытие обратно за линию."""
    n = len(c)
    for j in range(start, min(start + bars, n - 1)):
        p = price_at(j)
        if (side > 0 and lo[j] <= p) or (side < 0 and hi[j] >= p):
            return j, p
        tp = p + K * (p - stop)                        # 3R от лимитки (для шорта p − stop < 0 — цель ниже)
        if (side > 0 and hi[j] >= tp) or (side < 0 and lo[j] <= tp):
            return -1, np.nan
        if cancel_line is not None and side * (c[j] - cancel_line(j)) <= 0:
            return -1, np.nan
    return -1, np.nan


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
                print(f"  retest2 {s} {tf}: пропуск ({e})", flush=True)
    print(f"  retest2: сигналов {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("retest2"), index=False)


def paired(z: pd.DataFrame, v: str) -> tuple[str, bool]:
    cells, ok = [], []
    for p, (a, b) in PER_ALL.items():
        m = z[(z.t >= a) & (z.t < b)]
        if len(m) >= 20:
            dlt = m[v].mean() - m["level"].mean()
            ok.append(dlt > 0)
            cells.append(f"{p}: {dlt:+.3f}")
    dd = (z[v] - z["level"]).to_numpy()
    t = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))) if len(dd) > 2 and dd.std(ddof=1) > 0 else np.nan
    passed = bool(ok) and all(ok) and t >= 2
    return f"{'  '.join(cells)}  всё время t {t:.1f}  → {'принято' if passed else 'нет'}", passed


def report() -> None:
    parts = all_parts("retest2")
    print("\n===== RETEST2: лимитка от уровня (бот) / по трендовой / после обновления свинга с закреплением; "
          "R на сигнал, неисполненный = 0 (ячейка: R, t по дням, доля > 0, сигналов в месяц) =====")
    if not parts:
        print("  retest2: нет данных")
        return
    x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
    for tf in TFS:
        for nm, z in (("обе стороны", x[x.tf == tf]), ("только шорты", x[(x.tf == tf) & (x.side < 0)])):
            if z.empty:
                continue
            print(f"\n  --- {tf}, {nm}: сигналов {len(z)}; исполнено: уровень {z.level_fill.mean():.0%}, линия "
                  f"{z.line_fill.mean():.0%}, свинг есть {z.has_swing.mean():.0%} / ретест свинга "
                  f"{z.swing_fill.mean():.0%} ---")
            for v, title in VARIANTS.items():
                cells = "  ".join(f"{p}: {_cell(z[(z.t >= a) & (z.t < b)].assign(R=z[v]))}"
                                  for p, (a, b) in PER_ALL.items() if ((z.t >= a) & (z.t < b)).any())
                print(f"    {v:9s} {title}: {cells}")
                fill = {"level": "level_fill", "line": "line_fill", "swing": "swing_fill"}.get(v)
                if fill:
                    q = z[z[fill]]
                    print(f"              на исполненную: {_cell(q.assign(R=q[v]))}")
                if v != "level":
                    line, _ = paired(z, v)
                    print(f"              минус уровень (R на сигнал): {line}")
            h = z[z.has_swing]
            if len(h):
                print(f"    на сигналах со свингом ({len(h)}): уровень {_cell(h.assign(R=h['level']))}; "
                      f"ретест свинга {_cell(h.assign(R=h['swing']))}")


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
