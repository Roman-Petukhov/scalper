"""
Ловля прострелов по всему рынку (research.spikes, лонг, 6σ и 4σ): реальная частота сделок, массовые обвалы и
сколько монет счёт $100 может держать под лимитками с учётом маржи Bybit.

Монеты — все USDT-перпетуалы Binance (включая делистнутые), у которых оборот за 30 прошлых дней хоть раз был в
диапазоне $5–150 млн за 2025-07 … 2026-09; сделка учитывается, только если в её день монета была в диапазоне.
Свечи 1m Binance. Настройки (зафиксированы ранее, research.spikes.EXAM): m = 6 и 4, f = 0.5, T = 60 мин,
стоп нет / 5D, только лонг; maker 2 б.п. (вход, тейк), taker 5.5 б.п.

Счёт $100 (report): каждую неделю выбираем N монет под лимитки — с наибольшим числом прострелов за прошлые 90 дней
(только прошлое); N = резерв x капитал x плечо / ордер (Bybit блокирует маржу под стоящие ордера: ордер / плечо).
Все исполнения в наблюдаемых монетах засчитываются — даже десятки одновременно в обвал (их нельзя отменить
заранее), это и есть главный риск. Сетка: ордер 5 / 10 / 20% капитала (не меньше $5), плечо 5 / 10, резерв 0.5 / 1.

    python -m research.spikes_full collect --symbols <все перпетуалы>   (по частям)
    python -m research.spikes_full report
"""
from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .shard import all_parts, mine, part_path
from .spikes import MAKER, TAKER, load, spike_machine

ADV_LO, ADV_HI = 5e6, 150e6
START, END = "2025-07-01", "2026-10-01"
CONFIGS = [{"name": "6σ", "m": 6.0, "stop": np.inf}, {"name": "6σ стоп 5D", "m": 6.0, "stop": 5.0},
           {"name": "4σ", "m": 4.0, "stop": np.inf}]
M1_ROOT = Path.home() / "m1"


def adv_daily(root: Path, sym: str) -> pd.Series | None:
    p = root / f"{sym}-1h.parquet"
    if not p.exists():
        return None
    k = pd.read_parquet(p, columns=["open_time", "quote_volume"])
    d = pd.Series(k["quote_volume"].to_numpy(), index=pd.to_datetime(k["open_time"], unit="ms", utc=True)).resample("1D").sum()
    return d.rolling(30, min_periods=20).mean().shift(1)


def collect(root: Path, syms: list[str]) -> None:
    cand = []
    for s in mine(syms):
        a = adv_daily(root, s)
        if a is None:
            continue
        a = a[(a.index >= START) & (a.index < END)]
        if ((a >= ADV_LO) & (a <= ADV_HI)).any():
            cand.append((s, a))
    print(f"монет в диапазоне оборота: {len(cand)}", flush=True)
    D.set_host("cdn")
    D.build([s for s, _ in cand], "1m", "2025-06", "2026-09", M1_ROOT, 32)
    out = []
    for s, a in cand:
        try:
            k = load(M1_ROOT, s)
            if k is None or len(k) < 5000:
                continue
            ok_day = ((a >= ADV_LO) & (a <= ADV_HI))
            for cfg in CONFIGS:
                i, r, _ = spike_machine(k["open"].to_numpy(), k["high"].to_numpy(), k["low"].to_numpy(),
                                        k["close"].to_numpy(), k["sig"].to_numpy(), cfg["m"], 0.5, 60, cfg["stop"],
                                        MAKER, TAKER, True)
                if not len(i):
                    continue
                t = k.index[i]
                keep = (t >= START) & (t < END) & ok_day.reindex(t.floor("D")).fillna(False).to_numpy()
                out.append(pd.DataFrame({"cfg": cfg["name"], "symbol": s, "t": t[keep], "r": r[keep]}))
        except Exception as e:
            print(f"  {s}: пропуск ({e})", flush=True)
        finally:
            for f in M1_ROOT.glob(f"{s}-1m.parquet"):
                f.unlink()                                      # место на диске раннера
    if out:
        pd.concat(out, ignore_index=True).to_parquet(part_path("spikes_full"), index=False)


def account(tr: pd.DataFrame, order_frac: float, lev: float, reserve: float) -> dict:
    """Счёт $100: еженедельный выбор монет по прострелам за прошлые 90 дней; позиции живут до 60 мин."""
    eq, start = 100.0, 100.0
    tr = tr.sort_values("t").reset_index(drop=True)
    weeks = pd.date_range(pd.Timestamp(START, tz="UTC"), pd.Timestamp(END, tz="UTC"), freq="7D")
    hist = tr[["t", "symbol"]]
    month_eq, peak, mdd, fills, max_conc = {}, eq, 0.0, 0, 0
    for w0, w1 in zip(weeks[:-1], weeks[1:]):
        past = hist[(hist.t < w0) & (hist.t >= w0 - pd.Timedelta(days=90))]["symbol"].value_counts()
        order = max(5.0, order_frac * eq)
        n = int(reserve * eq * lev // order)
        watch = set(past.index[:n])
        wk = tr[(tr.t >= w0) & (tr.t < w1) & tr.symbol.isin(watch)]
        if len(wk):
            conc = wk.groupby(wk.t.dt.floor("h"))["r"].size().max()
            max_conc = max(max_conc, int(conc))
        for _, g in wk.groupby(wk.t.dt.floor("h")):          # исполнения в один час — на капитал начала часа
            order = max(5.0, order_frac * eq)
            eq += (order * g["r"]).sum()
            fills += len(g)
            peak = max(peak, eq)
            mdd = min(mdd, eq / peak - 1)
            if eq < 10:
                break
        month_eq[w1.strftime("%Y-%m")] = eq
        if eq < 10:
            break
    s = pd.concat([pd.Series({"start": start}), pd.Series(month_eq)])
    s = s[~s.index.duplicated(keep="last")]
    m = s.pct_change().dropna()
    return {"сделок/мес": round(fills / max(len(m), 1), 1), "итог $": round(eq, 0),
            "в мес. (CAGR)": f"{(eq / start) ** (1 / max(len(m), 1)) - 1:+.1%}", "худший мес.": f"{m.min():+.0%}",
            "макс. просадка": f"{mdd:.0%}", "макс. исполнений за час": max_conc, "разорён": eq < 10}


def report() -> None:
    parts = all_parts("spikes_full")
    if not parts:
        print("частей нет")
        return
    tr = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    tr["t"] = pd.to_datetime(tr["t"], utc=True)
    print(f"===== SPIKES FULL: 2025-07 … 2026-09, монет со сделками {tr.symbol.nunique()}, частей {len(parts)} =====")
    for name, g in tr.groupby("cfg"):
        r = g["r"]
        print(f"\n--- {name}: сделок {len(g)} ({len(g) / 15:.0f} в месяц), монет {g.symbol.nunique()}, "
              f"средняя {r.mean():+.2%}, медиана {r.median():+.2%}, прибыльных {(r > 0).mean():.0%}, "
              f"1% худших {r.quantile(0.01):+.1%}, худшая {r.min():+.1%}")
        mo = g.groupby(g.t.dt.strftime("%Y-%m"))["r"].agg(["size", "mean"])
        print("  по месяцам (n, средняя %): " + ", ".join(f"{k}: {int(v['size'])} {v['mean'] * 100:+.1f}" for k, v in mo.iterrows()))
        hours = g.groupby(g.t.dt.floor("h"))["r"].agg(["size", "sum", "mean"]).sort_values("size", ascending=False)
        print("  часы с наибольшим числом исполнений (n, средняя %): " +
              "; ".join(f"{k:%Y-%m-%d %H:00} n={int(v['size'])} {v['mean'] * 100:+.1f}" for k, v in hours.head(6).iterrows()))
        top = g.symbol.value_counts()
        print(f"  сделок на монету: медиана {top.median():.0f}, топ-20 монет дают {top.head(20).sum() / len(g):.0%} сделок")
        rows = []
        for of, lev, res in itertools.product((0.05, 0.10, 0.20), (5.0, 10.0), (0.5, 1.0)):
            rows.append({"ордер, доля": of, "плечо": lev, "резерв": res, **account(g, of, lev, res)})
        print("  счёт $100:")
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 320)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x])
    else:
        report()
