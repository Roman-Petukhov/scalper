"""Ложный пробой как сигнал в обратную сторону (Bulkowski: неудавшийся пробой — сильный разворот).

Пробой — та же линия, что у бота (линии по значимым точкам, 4h, первое закрытие за линией). Пробой «не удался»,
если в течение K свечей цена закрылась обратно за линию дальше 0.25 ATR (ATR свечи пробоя). Тогда вход в обратную
сторону по рынку на закрытии этой свечи; стоп — за экстремумом движения с момента пробоя ∓ 0.1 ATR (риск 0.3–4 ATR);
цель — 2R или 3R, срок сделки 60 свечей, стоп проверяется раньше тейка, комиссии taker на входе и выходе, funding.
Периоды: 2020, 2021 (данные с RESEARCH_START=2020-01), IS, VAL, HO — идея новая, ни на одном не подбиралась.
Срезы: все пробои / пробои, прошедшие фильтр бота (агрессоры >= 55%, тренд старшего ТФ, закрытие в верхней половине
свечи) — их провал бьёт по позициям бота / сторона разворота / K = 3, 6, 12.
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
from .oos import PER_ALL
from .tline import HOLD, TAKER, _years, htf_trend, tf_frame, two_targets, zz_lines

KS = (3, 6, 12)
BACK_ATR = 0.25


def coin_fails(root: Path, sym: str) -> pd.DataFrame:
    d = tf_frame(root, sym, "4h")
    if d is None or len(d) < 500:
        return pd.DataFrame()
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64")
    a = _atr(d).to_numpy()
    trend = htf_trend(d, "4h")
    buy = (d["taker_buy_volume"] / d["volume"].replace(0, np.nan)).to_numpy()
    liq = adv30(root, sym).reindex(d.index.floor("D")).to_numpy()
    n = len(c)
    rows = []
    for r in zz_lines(d):
        e, side = int(r["t"]), int(r["side"])
        if e <= 0 or e >= n - 2 or not (a[e] > 0) or not (liq[e] >= ADV_MIN):
            continue
        rng = hi[e] - lo[e]
        loc = ((c[e] - lo[e]) if side > 0 else (hi[e] - c[e])) / rng if rng > 0 else 0.0
        aggr = buy[e] if side > 0 else 1 - buy[e]
        bot = bool(aggr >= 0.55 and trend[e] == side and loc >= 0.5)
        for k in KS:
            j = -1
            for i in range(e + 1, min(e + k, n - 2) + 1):
                line_i = r["line_t"] + r["slope"] * (i - e)
                if side * (c[i] - line_i) < -BACK_ATR * a[e]:
                    j = i
                    break
            if j < 0:
                continue
            rev = -side                                          # разворот: против провалившегося пробоя
            ext = hi[e:j + 1].max() if side > 0 else lo[e:j + 1].min()
            stop = ext + 0.1 * a[e] if rev < 0 else ext - 0.1 * a[e]
            risk_atr = rev * (c[j] - stop) / a[e]
            if not (0.3 <= risk_atr <= 4.0):
                continue
            out = {"symbol": sym, "t": d.index[j], "k": k, "brk_side": side, "side": rev, "bot": bot,
                   "rev_with_trend": trend[j] == rev, "risk_atr": risk_atr, "risk_pct": rev * (c[j] - stop) / c[j]}
            for tgt in (2.0, 3.0):
                out[f"R{tgt:g}"] = two_targets(o, hi, lo, c, f, j, rev, c[j], stop, tgt, tgt, False, HOLD["4h"], TAKER)[0]
            rows.append(out)
    return pd.DataFrame(rows)


def collect(root: Path, syms: list[str]) -> None:
    parts = []
    for s in mine(syms):
        try:
            x = coin_fails(root, s)
            if len(x):
                parts.append(x)
        except Exception as e:
            print(f"  failbo {s}: пропуск ({e})", flush=True)
    print(f"  failbo: монет со сделками {len(parts)}", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path("failbo"), index=False)


def report() -> None:
    parts = all_parts("failbo")
    print("\n===== FAILBO: ложный пробой линии 4h → вход в обратную сторону по рынку, стоп за экстремумом движения; "
          "ячейка — R на сделку (t по дням, прибыльных, сделок в месяц); R/мес — сумма R за месяц =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["per"] = ""
    for p, (a, b) in PER_ALL.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p

    def per_month(z: pd.DataFrame, col: str) -> str:
        return " / ".join(f"{z[z.per == p][col].sum() / ((pd.Timestamp(b) - pd.Timestamp(a)).days / 30.4):+.1f}"
                          for p, (a, b) in PER_ALL.items())
    for tgt in ("R2", "R3"):
        rows = []
        for k in KS:
            g = df[df.k == k]
            for nm, z in (("все провалы", g), ("провал сигнала бота", g[g.bot]),
                          ("провал не-сигнала", g[~g.bot]),
                          ("разворот в шорт", g[g.side == -1]), ("разворот в лонг", g[g.side == 1]),
                          ("разворот по тренду старшего ТФ", g[g.rev_with_trend]),
                          ("провал сигнала бота, разворот по тренду", g[g.bot & g.rev_with_trend])):
                z = z.assign(R=z[tgt])
                rows.append({"возврат за K свечей": k, "срез": nm, **{p: _cell(z[z.per == p]) for p in PER_ALL},
                             "R/мес " + " / ".join(PER_ALL): per_month(z, tgt)})
        print(f"\n  --- цель {tgt[1:]}R ---")
        print(pd.DataFrame(rows).to_string(index=False))
    g = df[(df.k == 6)]
    print(f"\n  по годам (K = 6, все, 3R): {_years(g, 'R3')}")
    print(f"  при издержках x2 (K = 6, все, 3R): " + ", ".join(
        f"{p}: {(z.R3 - 2 * TAKER / z.risk_pct).mean():+.3f}" for p, z in g.groupby("per") if p))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 300)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x])
    else:
        report()
