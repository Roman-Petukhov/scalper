"""Автовыбор стороны по «Погоде рынка» (по просьбе владельца: рынок попутный для шортов — только шорты, для лонгов —
только лонги, смешанный — обе стороны, для всех ТФ).

Режим — ровно как в панели (trader/domain/weather.py market_line), по дневкам 100 самых ликвидных монет дня (оборот за
30 дней): признаки за шорты — альты за 30 дней слабее BTC на 5%+, BTC за 7 дней в пределах ±3%, 60%+ монет ниже
дневной EMA50; за лонги — альты сильнее BTC на 5%+, BTC за 7 дней вырос больше 3%, не больше 40% монет ниже EMA50.
Попутный для шортов — за шорты ≥ 2 и за лонги 0; для лонгов — зеркально; иначе смешанный. Режим берётся по
закрытию дня до сигнала (панель считает погоду после закрытия дня) — без заглядывания вперёд.
Сделки — правило бота с обеими сторонами (research/noline.py): 4h — ретест, 15m — по рынку, пологие линии.
Принятие (задано до запуска): отсечённые правилом сделки (сторона против режима) в среднем убыточны в каждом периоде
2020 / 2021 / IS / VAL / HO, где их ≥ 20, и за всё время с t ≤ −2, а R в месяц с правилом не ниже, чем без него,
в каждом периоде. Справочно — R на сделку лонгов и шортов по режимам.

    python -m research.sidegate collect --symbols <1h-qualified>
    python -m research.sidegate report
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

TFS = tuple(os.environ.get("SIDEGATE_TFS", "4h,15m").split(","))
ENTRY = {"4h": "retest", "15m": "market"}
MAX_SLOPE_15 = 0.025
TOP, EMA, BTC = 100, 50, "BTCUSDT"
ALTS_GAP, BTC_FLAT, DOWN, UP = 0.05, 0.03, 0.6, 0.4


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str) -> list[dict]:
    noline.TF = tf
    x = noline.coin_trades(d, liq, sym, lines_only=True)
    if not len(x):
        return []
    x = x[x.entry == ENTRY[tf]]
    atr = _atr(d).to_numpy()
    slope = {(int(r["t"]), int(r["side"])): abs(float(r["slope"])) for r in zz_lines(d) if r["t"] > 0}
    rows = []
    for r in x.itertuples(index=False):
        t, side = d.index.get_loc(r.t), int(r.side)
        if tf == "15m" and not (atr[t] > 0 and slope.get((t, side), np.inf) / atr[t] <= MAX_SLOPE_15):
            continue
        rows.append({"symbol": sym, "tf": tf, "t": r.t, "side": side, "R": float(r.R3)})
    return rows


def collect(root: Path, syms: list[str]) -> None:
    trades, daily = [], []
    for s in mine(syms):
        try:
            h = tf_frame(root, s, "4h")
            if h is None or len(h) < 500:
                continue
            liq = adv30(root, s)
            day = h["close"].resample("1D").last().dropna()
            daily.append(pd.DataFrame({"symbol": s, "day": day.index, "close": day.to_numpy(),
                                       "adv": liq.reindex(day.index.floor("D")).to_numpy()}))
            for tf in TFS:
                if tf == "15m" and not (root / f"{s}-15m.parquet").exists():
                    continue
                d = h if tf == "4h" else tf_frame(root, s, tf)
                if d is None or len(d) < 500:
                    continue
                trades += coin_rows(d, liq.reindex(d.index.floor("D")).to_numpy(), s, tf)
        except Exception as e:
            print(f"  sidegate {s}: пропуск ({e})", flush=True)
    print(f"  sidegate: сделок {len(trades)}, монет с дневками {len(daily)}", flush=True)
    if trades:
        pd.DataFrame(trades).to_parquet(part_path("sidegate"), index=False)
    if daily:
        pd.concat(daily, ignore_index=True).to_parquet(part_path("sidegate_daily"), index=False)


def regimes(daily: pd.DataFrame) -> pd.Series:
    """Режим на конец каждого дня: «short» / «long» / «mixed» — как market_line панели."""
    wide = daily.pivot_table(index="day", columns="symbol", values="close").sort_index()
    adv = daily.pivot_table(index="day", columns="symbol", values="adv").reindex(wide.index)
    ema = wide.ewm(span=EMA, adjust=False, min_periods=EMA).mean()
    below = wide < ema
    r30 = wide / wide.shift(30) - 1
    btc = wide[BTC] if BTC in wide else None
    out = {}
    for day in wide.index:
        a = adv.loc[day].drop(BTC, errors="ignore").dropna().sort_values(ascending=False)
        top = [s for s in a.index[:TOP] if np.isfinite(r30.at[day, s]) and np.isfinite(ema.at[day, s])]
        if btc is None or len(top) < 20 or not np.isfinite(r30.at[day, BTC]):
            continue
        alts = float(np.median(r30.loc[day, top])) - float(r30.at[day, BTC])
        b7 = float(btc.loc[day] / btc.shift(7).loc[day] - 1) if np.isfinite(btc.shift(7).loc[day]) else np.nan
        br = float(below.loc[day, top].mean())
        good = (alts <= -ALTS_GAP) + (abs(b7) <= BTC_FLAT if np.isfinite(b7) else 0) + (br >= DOWN)
        bad = (alts >= ALTS_GAP) + (b7 > BTC_FLAT if np.isfinite(b7) else 0) + (br <= UP)
        out[day] = "short" if good >= 2 and not bad else "long" if bad >= 2 and not good else "mixed"
    return pd.Series(out)


def report() -> None:
    tp, dp = all_parts("sidegate"), all_parts("sidegate_daily")
    print("\n===== SIDEGATE: автовыбор стороны по погоде рынка (попутный для шортов — только шорты, для лонгов — только "
          "лонги, смешанный — обе); R на сделку (t по дням, прибыльных, сделок в месяц) =====")
    if not tp or not dp:
        print("  sidegate: нет данных")
        return
    x = pd.concat([pd.read_parquet(p) for p in tp], ignore_index=True)
    daily = pd.concat([pd.read_parquet(p) for p in dp], ignore_index=True)
    daily["day"] = pd.to_datetime(daily["day"], utc=True).dt.tz_localize(None)
    reg = regimes(daily)
    x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
    x["regime"] = reg.reindex(x.t.dt.floor("D") - pd.Timedelta(days=1)).to_numpy()   # погода вчерашнего закрытия
    x = x.dropna(subset=["regime", "R"])
    share = reg.value_counts(normalize=True)
    print(f"  дней: попутный шортам {share.get('short', 0):.0%}, лонгам {share.get('long', 0):.0%}, "
          f"смешанный {share.get('mixed', 0):.0%}")
    verdicts = []
    for tf in TFS:
        z = x[x.tf == tf]
        if z.empty:
            continue
        cut = ((z.regime == "short") & (z.side > 0)) | ((z.regime == "long") & (z.side < 0))
        print(f"\n  --- {tf}, вход {ENTRY[tf]}: сделок {len(z)}, правило отсекает {cut.mean():.0%} ---")
        for reg_name in ("short", "mixed", "long"):
            for sd, nm in ((-1, "шорты"), (1, "лонги")):
                m = z[(z.regime == reg_name) & (z.side == sd)]
                if len(m):
                    print(f"    режим {reg_name:5s} {nm}: всё время {_cell(m)}")
        ok_cut, ok_month, cells = [], [], []
        for p, (a, b) in PER_ALL.items():
            m = z[(z.t >= a) & (z.t < b)]
            if m.empty:
                continue
            c = cut[(z.t >= a) & (z.t < b)]
            months = max(1.0, (pd.Timestamp(b) - pd.Timestamp(a)).days / 30.4)
            base_m, gate_m = m.R.sum() / months, m.R[~c].sum() / months
            cells.append(f"{p}: отсечённые {_cell(m[c]) if c.any() else '—'}; R в месяц {base_m:+.1f} → {gate_m:+.1f}")
            if c.sum() >= 20:
                ok_cut.append(m.R[c].mean() < 0)
            ok_month.append(gate_m >= base_m)
        for line in cells:
            print(f"    {line}")
        rc = z.R[cut].to_numpy()
        t = rc.mean() / (rc.std(ddof=1) / np.sqrt(len(rc))) if len(rc) > 2 and rc.std(ddof=1) > 0 else np.nan
        passed = bool(ok_cut) and all(ok_cut) and all(ok_month) and t <= -2
        print(f"    отсечённые за всё время: {rc.mean() if len(rc) else float('nan'):+.3f}R на {len(rc)}, t {t:.1f}")
        verdicts.append(f"{tf}: {'принято' if passed else 'нет'} (отсечённые t {t:.1f}, R в месяц не ниже в "
                        f"{sum(ok_month)} из {len(ok_month)} периодов)")
    print("\n  ИТОГ: " + "; ".join(verdicts))


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
