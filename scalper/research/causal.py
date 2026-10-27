"""Причинные связки из обзора 2026-10 (вопрос владельца «ищи неочевидные причинно-следственные связи»). Перпетуалы
Binance 1h с funding (1h-qualified), 2020-01 … 2026-09. Всё задано до запуска.

    F  выплата funding   толпа на дорогой стороне закрывается перед выплатой, после выплаты давление уходит.
                         Выплата с |ставкой| >= 0.03% (втрое выше обычной 0.01%): «до» — час перед выплатой против толпы
                         (сторона −sign(ставки)), «после» — час после выплаты за толпу. Ставка известна заранее лишь
                         приближённо (прогноз по премии) — берём итоговую, это мягкое заглядывание. Контроль — выплаты
                         с |ставкой| < 0.01%.
    Q  часы 21–23 UTC   лонг BTC с 21:00 до 23:00 UTC каждый день (Quantpedia, опубликовано ~2022: вне выборки —
                         2023+); справка — средний час по сессиям Азия 00–08 / Европа 08–14 / США 14–21 / вечер 21–24.
    E  экспирация       пятница 08:00 UTC (Deribit): |ход| BTC за сутки до экспирации против других дней (пиннинг) и
                         ход сутки после — справка.
    M  начало месяца    лонг BTC с закрытия предпоследнего дня месяца по закрытие 3-го дня (ребалансировки фондов).
    G  гэп CME          цена BTC в пятницу 21:00 UTC (закрытие CME) и в воскресенье 22:00 UTC (открытие); гэп >= 1%.
                         «Закрылся» — цена вернулась к пятничной до следующей пятницы 21:00. Контроль — уровень на том
                         же расстоянии по другую сторону (если и его касаются так же часто — гэп ни при чём). Сделка:
                         вход в воскресенье 22:00 к пятничной цене, цель — пятничная цена, стоп — 2 гэпа за входом, срок —
                         до пятницы 21:00.
Издержки: taker 5.5 б.п. на вход и выход (11 б.п. за круг), у Q справочно maker 2 б.п. Принятие (F «до», F «после», Q,
M, G): средний результат после издержек > 0 в каждом периоде IS / VAL / HO (сделок >= 20), t по дням >= 3 и > 0 при
издержках ×2.

    python -m research.causal collect --symbols <1h-qualified>
    python -m research.causal report
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .oos import PER_ALL
from .ppr import day_t
from .shard import all_parts, mine, part_path, shard_id
from .tline import tf_frame

HIGH, LOW = 0.0003, 0.0001
TAKER, MAKER = 5.5e-4, 2e-4
GAP_MIN = 0.01
SESSIONS = {"Азия 00–08": (0, 8), "Европа 08–14": (8, 14), "США 14–21": (14, 21), "вечер 21–24": (21, 24)}


def funding_rows(d: pd.DataFrame, sym: str) -> list[dict]:
    """Выплаты funding: ставка, ход за час до (бар перед выплатой) и час после (бар выплаты)."""
    f = d["funding"].to_numpy()
    o, c = d["open"].to_numpy(dtype="float64"), d["close"].to_numpy(dtype="float64")
    rows = []
    for i in np.flatnonzero(f != 0):
        if i < 1 or i >= len(c):
            continue
        rows.append({"symbol": sym, "t": d.index[i], "rate": float(f[i]),
                     "pre": c[i - 1] / o[i - 1] - 1, "post": c[i] / o[i] - 1})
    return rows


def collect(root: Path, syms: list[str]) -> None:
    rows = []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, "1h")
            if d is None or len(d) < 24 * 60:
                continue
            rows += funding_rows(d, s)
        except Exception as e:
            print(f"  causal {s}: пропуск ({e})", flush=True)
    print(f"  causal: выплат funding {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("causal_f"), index=False)
    if shard_id()[0] == 0:
        b = tf_frame(root, "BTCUSDT", "1h")
        if b is not None:
            b[["open", "high", "low", "close"]].reset_index(names="t").to_parquet(part_path("causal_btc"), index=False)


def verdict(z: pd.DataFrame, col: str, cost: float) -> tuple[bool, str]:
    """z: столбцы t и col (доля, до издержек); cost — за круг."""
    v = z.assign(R=z[col] - cost)
    cells, ok = [], []
    for p, (a, b) in PER_ALL.items():
        m = v[(v.t >= a) & (v.t < b)]
        if m.empty:
            continue
        cells.append(f"{p} {m.R.mean() * 1e4:+.1f} б.п. ({len(m)})")
        if len(m) >= 20 and p in ("is", "val", "ho"):
            ok.append(m.R.mean() > 0)
    t = day_t(v, "R") if len(v) > 10 else float("nan")
    x2 = (v.R - cost).mean()
    good = bool(ok) and all(ok) and np.isfinite(t) and t >= 3 and x2 > 0
    return good, (f"{'  '.join(cells)}\n        всё время {v.R.mean() * 1e4:+.1f} б.п. после издержек (до "
                  f"{z[col].mean() * 1e4:+.1f}), t {t:.1f}, в плюсе {(v.R > 0).mean():.0%}; ×2 {x2 * 1e4:+.1f} б.п.")


def report_funding(passed: list[str]) -> None:
    parts = all_parts("causal_f")
    if not parts:
        print("  causal F: нет данных")
        return
    x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
    side = -np.sign(x["rate"])
    x["pre_r"], x["post_r"] = side * x["pre"], -side * x["post"]
    hi, lo = x[x.rate.abs() >= HIGH], x[x.rate.abs() < LOW]
    print(f"\n  --- F: выплата funding — выплат {len(x)}, с |ставкой| >= {HIGH:.2%}: {len(hi)} "
          f"(плюс {(hi.rate > 0).mean():.0%}), контроль < {LOW:.2%}: {len(lo)} ---")
    for col, name in (("pre_r", "час до выплаты против толпы"), ("post_r", "час после выплаты за толпу")):
        good, txt = verdict(hi, col, 2 * TAKER)
        print(f"    {name}: {'принято' if good else 'нет'}\n        {txt}")
        _, ctl = verdict(lo, col, 2 * TAKER)
        print(f"      контроль (малая ставка):\n        {ctl}")
        if good:
            passed.append(f"F {name}")


def btc_frame() -> pd.DataFrame | None:
    parts = all_parts("causal_btc")
    if not parts:
        return None
    b = pd.concat([pd.read_parquet(p) for p in parts]).drop_duplicates("t")
    b.index = pd.to_datetime(b["t"], utc=True).dt.tz_localize(None)
    return b.drop(columns=["t"]).sort_index()


def report_hours(b: pd.DataFrame, passed: list[str]) -> None:
    r = b["close"] / b["open"] - 1
    print("\n  --- Q: часы суток BTC ---")
    rows = []
    for day, g in r.groupby(r.index.normalize()):
        w = g[(g.index.hour >= 21) & (g.index.hour < 23)]
        if len(w) == 2:
            rows.append({"t": day, "q": float((1 + w).prod() - 1)})
    z = pd.DataFrame(rows)
    good, txt = verdict(z, "q", 2 * TAKER)
    print(f"    лонг 21:00–23:00 UTC, taker: {'принято' if good else 'нет'}\n        {txt}")
    _, txt_m = verdict(z, "q", 2 * MAKER)
    print(f"    то же, maker (справка):\n        {txt_m}")
    if good:
        passed.append("Q 21–23 UTC")
    for p, (a, bb) in PER_ALL.items():
        m = r[(r.index >= a) & (r.index < bb)]
        cells = []
        for name, (h0, h1) in SESSIONS.items():
            s = m[(m.index.hour >= h0) & (m.index.hour < h1)]
            cells.append(f"{name} {s.mean() * 1e4:+.2f}")
        print(f"      {p}: средний час, б.п. — {'  '.join(cells)}")


def report_expiry(b: pd.DataFrame) -> None:
    c = b["close"]
    at8 = c[c.index.hour == 7]                                       # закрытие часа 07 — цена на 08:00
    at8.index = at8.index.normalize()
    d = pd.DataFrame({"move": at8.pct_change()})
    d["fri"] = d.index.dayofweek == 4
    d["next"] = d["move"].shift(-1)
    print("\n  --- E: экспирация Deribit (пятница 08:00 UTC), справка ---")
    for p, (a, bb) in PER_ALL.items():
        m = d[(d.index >= a) & (d.index < bb)].dropna()
        if m.empty:
            continue
        print(f"      {p}: |ход| за сутки до экспирации {m[m.fri].move.abs().mean():.2%} против {m[~m.fri].move.abs().mean():.2%}"
              f" в другие дни; сутки после экспирации {m[m.fri].next.mean():+.2%} (другие {m[~m.fri].next.mean():+.2%})")


def report_month(b: pd.DataFrame, passed: list[str]) -> None:
    c = b["close"].resample("1D").last().dropna()
    rows = []
    for m, g in c.groupby(c.index.to_period("M")):
        nxt = c[c.index.to_period("M") == m + 1]
        if len(g) < 3 or len(nxt) < 3:
            continue
        rows.append({"t": nxt.index[0], "m": float(nxt.iloc[2] / g.iloc[-2] - 1)})
    z = pd.DataFrame(rows)
    good, txt = verdict(z, "m", 2 * TAKER)
    base = c.pct_change(4).mean()
    print(f"\n  --- M: начало месяца, лонг BTC с предпоследнего дня по 3-й: {'принято' if good else 'нет'} ---\n"
          f"        {txt}\n        справка: любые 4 дня в среднем {base:+.2%}")
    if good:
        passed.append("M начало месяца")


def gap_rows(b: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fridays = b.index[(b.index.dayofweek == 4) & (b.index.hour == 20)]          # закрытие часа 20 — цена на 21:00
    for f in fridays:
        sun = f + pd.Timedelta(days=2, hours=1)                                  # воскресенье 21:00 → цена на 22:00
        end = f + pd.Timedelta(days=7)
        if sun not in b.index or end not in b.index:
            continue
        p0, p1 = float(b.at[f, "close"]), float(b.at[sun, "close"])
        gap = p1 / p0 - 1
        if abs(gap) < GAP_MIN:
            continue
        w = b.loc[sun + pd.Timedelta(hours=1): end]
        mirror = p1 * (1 + gap)                                                  # уровень по другую сторону
        side = -np.sign(gap)
        stop = p1 * (1 - side * 2 * abs(gap))
        res = np.nan
        for hi, lo in zip(w["high"].to_numpy(), w["low"].to_numpy()):
            hit_stop = (lo <= stop) if side > 0 else (hi >= stop)
            hit_tp = (hi >= p0) if side > 0 else (lo <= p0)
            if hit_stop:
                res = side * (stop / p1 - 1)
                break
            if hit_tp:
                res = side * (p0 / p1 - 1)
                break
        if not np.isfinite(res):
            res = side * (float(w["close"].iloc[-1]) / p1 - 1)
        filled = bool((w["high"] >= p0).any() if gap < 0 else (w["low"] <= p0).any())
        mir = bool((w["high"] >= mirror).any() if gap > 0 else (w["low"] <= mirror).any())
        rows.append({"t": sun, "gap": gap, "filled": filled, "mirror": mir, "g": res})
    return pd.DataFrame(rows)


def report_gap(b: pd.DataFrame, passed: list[str]) -> None:
    z = gap_rows(b)
    if z.empty:
        print("\n  --- G: гэпов нет ---")
        return
    good, txt = verdict(z, "g", 2 * TAKER)
    print(f"\n  --- G: гэп CME >= {GAP_MIN:.0%} — гэпов {len(z)}, закрылись за неделю {z.filled.mean():.0%}, уровень на том "
          f"же расстоянии с другой стороны — {z.mirror.mean():.0%} ---\n    сделка на закрытие гэпа: "
          f"{'принято' if good else 'нет'}\n        {txt}")
    if good:
        passed.append("G гэп CME")


def report() -> None:
    print("\n===== CAUSAL: выплата funding, часы суток, экспирация, начало месяца, гэп CME (перпетуалы Binance 1h) =====")
    passed: list[str] = []
    report_funding(passed)
    b = btc_frame()
    if b is None:
        print("  causal: нет BTC")
    else:
        report_hours(b, passed)
        report_expiry(b)
        report_month(b, passed)
        report_gap(b, passed)
    print("\n  ИТОГ: " + ("прошли — " + "; ".join(passed) if passed else "ни одна связка не окупает издержки"))


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
