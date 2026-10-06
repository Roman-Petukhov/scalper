"""Размер «как на Bybit» (маржа 10% капитала × плечо 10× ≈ номинал в капитал на сделку) с хеджем BTC на реальных
сделках бота 4h (линия, ретест, 3R — research/noline.py): сколько это в процентах капитала.

В режиме маржи убыток по стопу = стоп в % цены × номинал, то есть расстояние до стопа решает риск (в режиме «по
риску» он постоянный). Здесь по сделкам бэктеста 2020–2026 проигрывается счёт с теми же правилами, что в боте
(trader/domain/execution.build_order): размер от капитала на момент входа, хедж BTC на бету × номинал одной позицией
(неттинг, как в domain/hedge.py), «не больше N позиций», дневной стоп, свободная маржа с запасом 5% и урезанием
размера, комиссии хеджа. Не моделируется: нереализованный результат внутри сделки (просадка по закрытым сделкам
занижена), потолок плеча отдельных монет (бот в таких случаях берёт меньшее плечо), ликвидация при стопе дальше
границы плеча (считаем долю таких сделок), funding хеджа.
"""
from __future__ import annotations

import argparse
import heapq
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import adv30
from .noline import TF, coin_trades
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .tline import TAKER, tf_frame

BAR = pd.Timedelta(hours=4)
E0 = 1000.0                 # стартовый капитал, $ (только масштаб: минимальный ордер $5 важен для мелких счетов)
RESERVE = 0.95              # 5% свободной маржи — запас на комиссии и проскальзывание (как в боте)
MIN_NOTIONAL = 5.0
MAX_POS = 12                # Settings.max_positions по умолчанию
DAILY_LOSS = 4.0            # Settings.daily_loss_pct по умолчанию
BETA_BARS, BETA_MIN_BARS, BETA_RANGE = 360, 120, (0.0, 3.0)      # как в domain/hedge.py
LIQ_STOP = 0.09             # стоп дальше ~9% цены при плече 10× — до стопа сработает ликвидация изолированной маржи


@dataclass(frozen=True)
class Cfg:
    name: str
    by_margin: bool         # True — маржа size_pct% капитала × плечо; False — убыток по стопу size_pct% капитала
    size_pct: float
    lev: int
    hedge: bool
    hedge_ratio: float = 1.0    # доля беты в хедже (1 — полная, как в боте)
    hedge_lev: int = 0          # плечо BTC; 0 — то же, что у сделок (в боте сейчас так)
    max_pos: int = MAX_POS


CONFIGS = (
    Cfg("маржа 10% x10 + хедж", True, 10.0, 10, True),
    Cfg("маржа 2.5% x10 + хедж", True, 2.5, 10, True),
    Cfg("риск 1% x5 (сейчас)", False, 1.0, 5, False),
    Cfg("риск 1% x10 + хедж", False, 1.0, 10, True),
    Cfg("риск 2% x10 без хеджа", False, 2.0, 10, False),
    Cfg("риск 2% x10 + хедж", False, 2.0, 10, True),
    Cfg("  то же, хедж 50% беты", False, 2.0, 10, True, hedge_ratio=0.5),
    Cfg("  то же, не больше 6 поз.", False, 2.0, 10, True, max_pos=6),
    Cfg("  то же, плечо BTC x25", False, 2.0, 10, True, hedge_lev=25),
    Cfg("риск 2.5% x10 + хедж", False, 2.5, 10, True),
    Cfg("риск 5% x10 + хедж", False, 5.0, 10, True),
)


def collect(root: Path, syms: list[str]) -> None:
    btc_d = tf_frame(root, "BTCUSDT", TF)
    btc = btc_d["close"] if btc_d is not None else None
    parts = []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, TF)
            if d is None or len(d) < 500:
                continue
            liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
            x = coin_trades(d, liq, s)
            if not len(x):
                continue
            x = x[(x.trig == "line") & (x.entry == "retest")]
            if not len(x):
                continue
            b = btc.reindex(d.index).ffill().to_numpy() if btc is not None and s != "BTCUSDT" else None
            rc = np.diff(np.log(d["close"].to_numpy()), prepend=np.nan)
            rb = np.diff(np.log(b), prepend=np.nan) if b is not None else None
            betas, rets = [], []
            for t_ts, fill, ex in zip(x["t"], x["fill_i"], x["exit_i"]):
                if b is None:
                    betas.append(np.nan)
                    rets.append(np.nan)
                    continue
                t = d.index.get_loc(t_ts)
                w = slice(max(1, t - BETA_BARS), t + 1)
                a_, b_ = rc[w], rb[w]
                ok = np.isfinite(a_) & np.isfinite(b_)
                beta = float(np.cov(a_[ok], b_[ok])[0, 1] / np.var(b_[ok], ddof=1)) if ok.sum() >= BETA_MIN_BARS else np.nan
                betas.append(min(max(beta, BETA_RANGE[0]), BETA_RANGE[1]) if np.isfinite(beta) else np.nan)
                rets.append(b[ex] / b[fill] - 1 if b[fill] > 0 else np.nan)
            parts.append(pd.DataFrame({"symbol": s, "t": x["t"].to_numpy(), "side": x["side"].to_numpy(),
                                       "R3": x["R3"].to_numpy(), "risk_pct": x["risk_pct"].to_numpy(),
                                       "t_in": d.index[x["fill_i"].to_numpy()] + BAR,
                                       "t_out": d.index[x["exit_i"].to_numpy()] + BAR,
                                       "beta": betas, "btc_ret": rets}))
        except Exception as e:
            print(f"  margsim {s}: пропуск ({e})", flush=True)
    print(f"  margsim: монет со сделками {len(parts)}", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path("margsim"), index=False)


@dataclass
class Result:
    eq: pd.Series                       # капитал по закрытым сделкам (момент выхода → $)
    opened: int = 0
    rej_pos: int = 0
    rej_day: int = 0
    rej_margin: int = 0
    cut: int = 0
    peak_open: int = 0
    peak_margin: float = 0.0            # доля капитала под маржой сделок и хеджа
    loss_at_stop: np.ndarray | None = None   # убыток по стопу каждой открытой сделки, % капитала на момент входа
    util_t: list | None = None          # моменты изменения загрузки маржи
    util_v: list | None = None          # загрузка после изменения: (маржа сделок + маржа хеджа) / капитал


def simulate(df: pd.DataFrame, cfg: Cfg, e0: float = E0) -> Result:
    """Счёт по сделкам в порядке входа: выходы, наступившие к моменту входа, закрываются первыми."""
    d = df.sort_values("t_in", kind="stable")
    rows = zip(d["t_in"], d["t_out"], d["side"].astype(int), d["R3"], d["risk_pct"], d["beta"], d["btc_ret"])
    hl = cfg.hedge_lev or cfg.lev
    eq, open_, seq = e0, [], 0
    marg, hedge_net = 0.0, 0.0          # маржа открытых сделок; нетто-номинал хеджа BTC (плюс — лонг)
    times, vals = [], []
    res = Result(pd.Series(dtype="float64"), util_t=[], util_v=[])
    losses: list[float] = []
    day, day_start = None, e0

    def util() -> float:
        return (marg + abs(hedge_net) / hl) / eq if eq > 0 else 1.0

    def close_until(t: pd.Timestamp | None) -> None:
        """Закрыть выходы не позже t (None — все оставшиеся)."""
        nonlocal eq, marg, hedge_net
        while open_ and (t is None or open_[0][0] <= t):
            t_out, _, pnl, m, hs = heapq.heappop(open_)
            eq += pnl
            marg -= m
            hedge_net -= hs
            times.append(t_out)
            vals.append(eq)
            res.util_t.append(t_out)
            res.util_v.append(util())

    for t_in, t_out, side, r3, rp, beta, btc_ret in rows:
        close_until(t_in)
        if t_in.floor("D") != day:
            day, day_start = t_in.floor("D"), eq
        if len(open_) >= cfg.max_pos:
            res.rej_pos += 1
            continue
        if eq <= day_start * (1 - DAILY_LOSS / 100):
            res.rej_day += 1
            continue
        b = float(beta) * cfg.hedge_ratio if cfg.hedge and np.isfinite(beta) and np.isfinite(btc_ret) else 0.0
        want = eq * cfg.size_pct / 100 * cfg.lev if cfg.by_margin else min(eq * cfg.size_pct / 100 / rp, eq * cfg.lev)
        avail = eq - marg - abs(hedge_net) / hl
        cap = max(avail, 0.0) * RESERVE / (1 / cfg.lev + b / hl)         # маржа сделки + маржа её хеджа ≤ свободной
        n = min(want, cap)
        if n < MIN_NOTIONAL:
            res.rej_margin += 1
            continue
        if n < 0.9 * want:
            res.cut += 1
        hedge_pnl = b * n * (-side * btc_ret) - 2 * TAKER * b * n if b > 0 else 0.0
        pnl = n * rp * r3 + hedge_pnl
        hs = -side * b * n
        m = n / cfg.lev
        seq += 1
        heapq.heappush(open_, (t_out, seq, pnl, m, hs))
        marg += m
        hedge_net += hs
        res.opened += 1
        res.peak_open = max(res.peak_open, len(open_))
        res.peak_margin = max(res.peak_margin, util())
        res.util_t.append(t_in)
        res.util_v.append(util())
        losses.append(n * rp / eq * 100)
    close_until(None)
    res.eq = pd.Series(vals, index=pd.DatetimeIndex(times), dtype="float64")
    res.loss_at_stop = np.asarray(losses)
    return res


def util_stats(res: Result) -> tuple[float, float, float]:
    """Загрузка маржи по времени (от первого входа до последнего выхода): среднее, доля времени выше 50% и выше 80%."""
    if not res.util_t or len(res.util_t) < 2:
        return 0.0, 0.0, 0.0
    t = pd.DatetimeIndex(res.util_t)
    order = np.argsort(t.asi8, kind="stable")
    ts, v = t.asi8[order].astype("float64"), np.asarray(res.util_v)[order]
    w = np.diff(ts)
    v = v[:-1]
    if w.sum() <= 0:
        return 0.0, 0.0, 0.0
    return float((v * w).sum() / w.sum()), float(w[v > 0.5].sum() / w.sum()), float(w[v > 0.8].sum() / w.sum())


def monthly(eq: pd.Series, e0: float = E0) -> pd.Series:
    """Месячная доходность по капиталу на конец месяца."""
    if eq.empty:
        return pd.Series(dtype="float64")
    m = eq.resample("ME").last().ffill()
    prev = m.shift(1)
    prev.iloc[0] = e0
    return (m / prev - 1) * 100


def max_dd(eq: pd.Series, e0: float = E0) -> float:
    if eq.empty:
        return 0.0
    v = np.r_[e0, eq.to_numpy()]
    return float(np.max((np.maximum.accumulate(v) - v) / np.maximum.accumulate(v)) * 100)


def summarize(res: Result, e0: float = E0) -> dict[str, object]:
    mo = monthly(res.eq, e0)
    row: dict[str, object] = {
        "итог ×": f"{res.eq.iloc[-1] / e0:.2f}" if len(res.eq) else "-",
        "просадка %": f"{max_dd(res.eq, e0):.0f}",
        "худший мес. %": f"{mo.min():.0f}" if len(mo) else "-",
        "ср. мес. %": f"{mo.mean():+.1f}" if len(mo) else "-",
        "σ мес. %": f"{mo.std():.1f}" if len(mo) else "-",
    }
    for p, (a, b) in PER_ALL.items():
        sub = mo[(mo.index >= a) & (mo.index < b)]
        row[p] = f"{sub.mean():+.1f}" if len(sub) else "-"
    ls = res.loss_at_stop if res.loss_at_stop is not None and len(res.loss_at_stop) else np.array([np.nan])
    um, u50, u80 = util_stats(res)
    row |= {"сделок": res.opened, "лимит поз.": res.rej_pos, "дн. стоп": res.rej_day, "нет маржи": res.rej_margin,
            "урезано": res.cut, "пик поз.": res.peak_open,
            "маржа %: ср. / пик": f"{um * 100:.0f} / {res.peak_margin * 100:.0f}",
            "время >50% / >80%": f"{u50:.0%} / {u80:.0%}",
            "стоп = % капитала (мед / p90 / макс)": f"{np.nanmedian(ls):.1f} / {np.nanpercentile(ls, 90):.1f} / {np.nanmax(ls):.1f}"}
    return row


def report() -> None:
    parts = all_parts("margsim")
    print("\n===== MARGSIM: размер «как на Bybit» (маржа x плечо) и хедж BTC на сделках бота 4h (линия, ретест, 3R); "
          "капитал считается по закрытым сделкам, лимит 12 позиций, дневной стоп 4% =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    for c in ("t", "t_in", "t_out"):
        df[c] = pd.to_datetime(df[c], utc=True)
    df = df.sort_values("t_in", kind="stable").reset_index(drop=True)
    rp = df["risk_pct"].to_numpy(dtype="float64") * 100
    q = np.percentile(rp, [10, 50, 90, 99])
    print(f"  сделок {len(df)} с {df.t.min():%Y-%m} по {df.t.max():%Y-%m}, шортов {(df.side < 0).mean():.0%}, "
          f"с хеджем (бета известна) {df.beta.notna().mean():.0%}, бета медиана {df.beta.median():.2f}")
    print(f"  стоп, % цены: p10 {q[0]:.1f} / медиана {q[1]:.1f} / p90 {q[2]:.1f} / p99 {q[3]:.1f} / макс {rp.max():.1f}; "
          f"дальше {LIQ_STOP * 100:.0f}% (при x10 раньше стопа ликвидация изолированной маржи): {(rp >= LIQ_STOP * 100).mean():.1%} сделок")
    rows = []
    for cfg in CONFIGS:
        rows.append({"вариант": cfg.name, **summarize(simulate(df, cfg))})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 400)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x])
    else:
        report()
