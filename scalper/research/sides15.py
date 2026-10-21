"""Почему на 15m шорты в плюсе, а лонги в минусе (по вопросу владельца после sidegate: «почему шорты именно в плюсе
всегда?»). Правило бота на 15m без отбора по стороне: пологие линии (наклон ≤ 0.025 ATR на свечу), вход по рынку, обе
стороны; для сравнения 4h — ретест. Сигналы и вход те же, что в research/noline.py.

Что смотрим (всё в R сделки, по периодам 2020 / 2021 / IS / VAL / HO):
  R бота по сторонам и разница шорты − лонги (Welch t по сделкам);
  R без funding (тот же прогон с funding = 0) и вклад funding = R − R без funding;
  стоп в % цены (медиана) — чем он уже, тем дороже комиссии в R;
  дрейф рынка: медианная доходность монеты за месяц и средний funding монеты за месяц (в % годовых) по периодам.
Принятие «лонги на 15m отключить» (задано до запуска): R лонгов < 0 в каждом периоде, где их ≥ 20, за всё время t ≤ −2,
и шорты − лонги > 0 в каждом таком периоде. Принятие «шорты лучше из-за funding»: вклад funding в разницу шорты −
лонги больше половины разницы в R в каждом периоде. Справочно — всё остальное.

    python -m research.sides15 collect --symbols <1h-qualified>
    python -m research.sides15 report
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
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import tf_frame, zz_lines

TFS = tuple(os.environ.get("SIDES15_TFS", "15m,4h").split(","))
ENTRY = {"4h": "retest", "15m": "market"}
MAX_SLOPE_15 = 0.025
BARS_PER_YEAR = {"15m": 365 * 96, "4h": 365 * 6}


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str) -> list[dict]:
    noline.TF = tf
    x = noline.coin_trades(d, liq, sym, lines_only=True)
    if not len(x):
        return []
    nf = noline.coin_trades(d.assign(funding=0.0), liq, sym, lines_only=True)
    x = x[x.entry == ENTRY[tf]]
    nf = nf[nf.entry == ENTRY[tf]].set_index(["t", "side"]).R3
    nf = nf[~nf.index.duplicated()]
    atr = _atr(d).to_numpy()
    slope = {(int(r["t"]), int(r["side"])): abs(float(r["slope"])) for r in zz_lines(d) if r["t"] > 0}
    rows = []
    for r in x.itertuples(index=False):
        t, side = d.index.get_loc(r.t), int(r.side)
        if tf == "15m" and not (atr[t] > 0 and slope.get((t, side), np.inf) / atr[t] <= MAX_SLOPE_15):
            continue
        r_nf = nf.get((r.t, side), np.nan)
        rows.append({"symbol": sym, "tf": tf, "t": r.t, "side": side, "R": float(r.R3), "R_nofund": float(r_nf),
                     "risk_pct": float(r.risk_pct)})
    return rows


def drift_rows(d: pd.DataFrame, sym: str, tf: str) -> list[dict]:
    """По месяцам: доходность монеты и funding (сумма ставок за месяц → % годовых)."""
    if "funding" not in d:
        return []
    ret = d["close"].resample("MS").last().pct_change()
    fund = d["funding"].resample("MS").mean() * BARS_PER_YEAR[tf]
    return [{"symbol": sym, "tf": tf, "month": m, "ret": float(ret.get(m, np.nan)), "fund_apr": float(fund.get(m, np.nan))}
            for m in ret.index if np.isfinite(ret.get(m, np.nan))]


def collect(root: Path, syms: list[str]) -> None:
    trades, drift = [], []
    for s in mine(syms):
        for tf in TFS:
            try:
                if tf == "15m" and not (root / f"{s}-15m.parquet").exists():
                    continue
                d = tf_frame(root, s, tf)
                if d is None or len(d) < 500:
                    continue
                liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
                trades += coin_rows(d, liq, s, tf)
                drift += drift_rows(d, s, tf)
            except Exception as e:
                print(f"  sides15 {s} {tf}: пропуск ({e})", flush=True)
    print(f"  sides15: сделок {len(trades)}, строк дрейфа {len(drift)}", flush=True)
    if trades:
        pd.DataFrame(trades).to_parquet(part_path("sides15"), index=False)
    if drift:
        pd.DataFrame(drift).to_parquet(part_path("sides15_drift"), index=False)


def welch(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def tstat(a: np.ndarray) -> float:
    return float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))) if len(a) > 2 and a.std(ddof=1) > 0 else float("nan")


def verdict(z: pd.DataFrame) -> tuple[bool, bool, list[str]]:
    """(лонги отключить, шорты лучше из-за funding, строки по периодам)."""
    lines, neg, diff_ok, fund_ok = [], [], [], []
    for p, (a, b) in PER_ALL.items():
        m = z[(z.t >= a) & (z.t < b)]
        lg, sh = m[m.side > 0], m[m.side < 0]
        if m.empty:
            continue
        fl, fs = (lg.R - lg.R_nofund).mean(), (sh.R - sh.R_nofund).mean()
        gap, gap_f = sh.R.mean() - lg.R.mean(), fs - fl
        lines.append(f"{p}: лонги {_cell(lg) if len(lg) else '—'}; шорты {_cell(sh) if len(sh) else '—'}; шорты − лонги "
                     f"{gap:+.3f}R (t {welch(sh.R.to_numpy(), lg.R.to_numpy()):.1f}), из них funding {gap_f:+.3f}R; "
                     f"без funding лонги {lg.R_nofund.mean():+.3f} / шорты {sh.R_nofund.mean():+.3f}R; "
                     f"стоп p50 лонги {lg.risk_pct.median():.2%} / шорты {sh.risk_pct.median():.2%}")
        if len(lg) >= 20 and len(sh) >= 20:
            neg.append(lg.R.mean() < 0)
            diff_ok.append(gap > 0)
            fund_ok.append(abs(gap) > 0 and gap_f / gap > 0.5)
    lt = tstat(z[z.side > 0].R.to_numpy())
    cut = bool(neg) and all(neg) and all(diff_ok) and lt <= -2
    return cut, bool(fund_ok) and all(fund_ok), lines


def report() -> None:
    tp, dp = all_parts("sides15"), all_parts("sides15_drift")
    print("\n===== SIDES15: почему шорты на 15m в плюсе, лонги в минусе — R по сторонам, вклад funding, стоп, дрейф рынка "
          "(t по дням, прибыльных, сделок в месяц) =====")
    if not tp:
        print("  sides15: нет данных")
        return
    x = pd.concat([pd.read_parquet(p) for p in tp], ignore_index=True)
    x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
    x = x.dropna(subset=["R", "R_nofund"])
    out = []
    for tf in TFS:
        z = x[x.tf == tf]
        if len(z) < 40:
            continue
        cut, by_fund, lines = verdict(z)
        print(f"\n  --- {tf}, вход {ENTRY[tf]}: сделок {len(z)} (лонгов {(z.side > 0).sum()}, шортов {(z.side < 0).sum()}) ---")
        for ln in lines:
            print(f"    {ln}")
        lg, sh = z[z.side > 0], z[z.side < 0]
        print(f"    всё время: лонги {_cell(lg)}; шорты {_cell(sh)}; шорты − лонги t {welch(sh.R.to_numpy(), lg.R.to_numpy()):.1f}")
        print(f"    funding в среднем на сделку: лонги {(lg.R - lg.R_nofund).mean():+.3f}R, шорты {(sh.R - sh.R_nofund).mean():+.3f}R")
        out.append(f"{tf}: лонги отключить — {'да' if cut else 'нет'}; разница из-за funding — {'да' if by_fund else 'нет'}")
    if dp:
        dr = pd.concat([pd.read_parquet(p) for p in dp], ignore_index=True)
        dr["month"] = pd.to_datetime(dr["month"], utc=True).dt.tz_localize(None)
        for tf in TFS:
            q = dr[dr.tf == tf]
            if q.empty:
                continue
            print(f"\n  --- дрейф рынка, {tf}: по периодам медиана месячной доходности монеты / доля месяцев с падением / "
                  f"средний funding (годовых) ---")
            for p, (a, b) in PER_ALL.items():
                m = q[(q.month >= a) & (q.month < b)]
                if len(m):
                    print(f"    {p}: {m.ret.median():+.1%} / {(m.ret < 0).mean():.0%} / {m.fund_apr.mean():+.1%}")
    print("\n  ИТОГ: " + "; ".join(out))


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
