"""Хеджированные стратегии (лонг одного + шорт другого), 2020–2026, комиссии Bybit.

1. Пары по расстоянию (Gatev, Goetzmann, Rouwenhorst 2006) на 1h и 4h: каждые 30 дней — топ-50 монет по обороту,
   отбор FORM_D дней, 20 пар с наименьшей суммой квадратов расхождения нормированных цен; торговля TRADE_D дней:
   расхождение > 2σ (σ — на отборе) — лонг дешёвой, шорт дорогой на равные суммы; выход — схождение (спред через 0)
   или конец периода. Итог в % от суммы одной ноги: разница ног, фандинг, 4 сделки по тейкеру (или по мейкеру).
2. Кросс-секционный фандинг: раз в неделю среди топ-50 по обороту — лонг K монет с самым низким фандингом за 7 дней,
   шорт K с самым высоким, равные веса, нейтрально по деньгам. Неделя: цена + фандинг − комиссии с оборота.
3. Бот 4h с хеджем BTC: к каждой сделке бота (правило линии, ретест, 3R — research/noline.py) — противоположная
   позиция BTC на бету монеты к BTC (4h, 60 дней до сигнала) от исполнения до выхода; результат в R сделки.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30
from .noline import coin_trades as line_trades
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _cell
from .tline import MAKER, TAKER, _years, tf_frame

TOP = 50
FORM_D, TRADE_D = 90, 30
N_PAIRS = 20
OPEN_SD = 2.0
CARRY_K = (5, 10)
BETA_BARS = 360                     # 60 дней 4h-свечей


# ---------------- сбор ----------------
def _bot_hedge(sym: str, d: pd.DataFrame, liq: np.ndarray, btc: pd.Series) -> pd.DataFrame:
    x = line_trades(d, liq, sym)
    if not len(x):
        return x
    x = x[(x.trig == "line") & (x.entry == "retest")].copy()
    if not len(x):
        return x
    b = btc.reindex(d.index).ffill().to_numpy()
    rc, rb = np.diff(np.log(d["close"].to_numpy()), prepend=np.nan), np.diff(np.log(b), prepend=np.nan)
    betas, hedge = [], []
    for t_ts, fill, ex, side, rp in zip(x.t, x.fill_i, x.exit_i, x.side, x.risk_pct):
        t = d.index.get_loc(t_ts)
        w = slice(max(1, t - BETA_BARS), t + 1)
        a_, b_ = rc[w], rb[w]
        ok = np.isfinite(a_) & np.isfinite(b_)
        beta = float(np.cov(a_[ok], b_[ok])[0, 1] / np.var(b_[ok], ddof=1)) if ok.sum() > 60 else np.nan
        btc_ret = b[ex] / b[fill] - 1 if b[fill] > 0 else np.nan
        betas.append(beta)
        # BTC в противоположную сторону на beta · сумму позиции; позиция = риск / risk_pct → в R делим на risk_pct
        hedge.append((-side * beta * btc_ret - 2 * TAKER * abs(beta)) / rp)
    x["beta"], x["hedge_R"] = betas, hedge
    x["btc_ret"] = [b[ex] / b[f] - 1 for f, ex in zip(x.fill_i, x.exit_i)]
    return x[["symbol", "t", "side", "R3", "risk_pct", "beta", "hedge_R", "btc_ret"]]


def collect(root: Path, syms: list[str]) -> None:
    btc_d = tf_frame(root, "BTCUSDT", "4h")
    btc = btc_d["close"] if btc_d is not None else None
    panel, bot = [], []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, "1h")
            if d is None or len(d) < 2000:
                continue
            al = adv30(root, s)
            panel.append(pd.DataFrame({"symbol": s, "t": d.index, "close": d["close"].to_numpy(),
                                       "funding": d["funding"].to_numpy(dtype="float32"),
                                       "adv": al.reindex(d.index.floor("D")).to_numpy(dtype="float32")}))
            d4 = tf_frame(root, s, "4h")
            if btc is not None and d4 is not None and len(d4) >= 500 and s != "BTCUSDT":
                x = _bot_hedge(s, d4, al.reindex(d4.index.floor("D")).to_numpy(), btc)
                if len(x):
                    bot.append(x)
        except Exception as e:
            print(f"  hedged {s}: пропуск ({e})", flush=True)
    print(f"  hedged: монет {len(panel)}, со сделками бота {len(bot)}", flush=True)
    if panel:
        pd.concat(panel, ignore_index=True).to_parquet(part_path("hedged_panel"), index=False)
    if bot:
        pd.concat(bot, ignore_index=True).to_parquet(part_path("hedged_bot"), index=False)


# ---------------- общее ----------------
def _per_of(t: pd.Series) -> pd.Series:
    out = pd.Series("", index=t.index)
    for p, (a, b) in PER_ALL.items():
        out[(t >= pd.Timestamp(a, tz="UTC")) & (t < pd.Timestamp(b, tz="UTC"))] = p
    return out


def _stat(x: pd.Series, periods_per_year: float) -> str:
    x = x.dropna()
    if len(x) < 5:
        return f"n={len(x)}"
    sd = x.std()
    sh = x.mean() / sd * np.sqrt(periods_per_year) if sd > 0 else np.nan
    return f"{x.mean() * 100:+.2f}% (Sharpe {sh:+.1f}, {len(x)})"


# ---------------- 1. пары ----------------
def _pairs(close: pd.DataFrame, fund: pd.DataFrame, adv: pd.DataFrame, bars_d: int) -> pd.DataFrame:
    trades = []
    starts = pd.date_range(close.index[0].normalize() + pd.Timedelta(days=FORM_D), close.index[-1], freq=f"{TRADE_D}D")
    for t0 in starts:
        f0 = t0 - pd.Timedelta(days=FORM_D)
        t1 = t0 + pd.Timedelta(days=TRADE_D)
        a = adv.loc[:t0].iloc[-1] if len(adv.loc[:t0]) else None
        if a is None:
            continue
        form = close.loc[f0:t0 - pd.Timedelta(seconds=1)]
        good = form.columns[form.notna().all() & (form.iloc[:1] > 0).all()] if len(form) else []
        cand = a.reindex(good)
        cand = cand[cand >= ADV_MIN].sort_values(ascending=False).index[:TOP]
        if len(cand) < 10 or len(form) < FORM_D * bars_d * 0.9:
            continue
        x = (form[cand] / form[cand].iloc[0]).to_numpy()
        sq = (x ** 2).sum(0)
        ssd = sq[:, None] + sq[None, :] - 2 * x.T @ x
        iu = np.triu_indices(len(cand), 1)
        order = np.argsort(ssd[iu])[:N_PAIRS]
        trade = close.loc[t0:t1 - pd.Timedelta(seconds=1), cand]
        ftr = fund.loc[t0:t1 - pd.Timedelta(seconds=1), cand]
        if len(trade) < 10:
            continue
        for k in order:
            i, j = iu[0][k], iu[1][k]
            si, sj = cand[i], cand[j]
            sd = np.std(x[:, i] - x[:, j])
            if not sd > 0:
                continue
            pi, pj = trade[si].to_numpy(), trade[sj].to_numpy()
            if not (np.isfinite(pi).all() and np.isfinite(pj).all()):
                continue
            ni, nj = pi / form[si].iloc[0], pj / form[sj].iloc[0]
            spread = (ni - nj) / sd
            fi, fj = ftr[si].to_numpy(), ftr[sj].to_numpy()
            pos, e = 0, 0
            for b in range(len(spread)):
                if pos == 0 and abs(spread[b]) > OPEN_SD and b < len(spread) - 1:
                    pos, e = (-1 if spread[b] > 0 else 1), b       # pos=+1: лонг i, шорт j
                elif pos != 0 and (np.sign(spread[b]) == pos or spread[b] == 0 or b == len(spread) - 1):
                    gross = pos * ((pi[b] / pi[e] - 1) - (pj[b] / pj[e] - 1))
                    fsum = -pos * fi[e + 1: b + 1].sum() + pos * fj[e + 1: b + 1].sum()
                    trades.append({"t": trade.index[e], "pair": f"{si}/{sj}", "bars": b - e,
                                   "gross": gross, "funding": fsum, "conv": b < len(spread) - 1})
                    pos = 0
    return pd.DataFrame(trades)


def pairs_report(close1: pd.DataFrame, fund1: pd.DataFrame, adv: pd.DataFrame) -> None:
    print("\n  === 1. ПАРЫ ПО РАССТОЯНИЮ: на сделку в % от суммы одной ноги (Sharpe — по месяцам портфеля из 20 пар) ===")
    for tf, rule, bars_d in (("1h", None, 24), ("4h", "4h", 6)):
        close = close1 if rule is None else close1.resample(rule, label="left", closed="left").last()
        fund = fund1 if rule is None else fund1.resample(rule, label="left", closed="left").sum()
        tr = _pairs(close, fund, adv, bars_d)
        if not len(tr):
            print(f"  {tf}: сделок нет")
            continue
        tr["per"] = _per_of(tr.t)
        rows = []
        for nm, cost in (("без комиссий", 0.0), ("тейкер 4×0.055%", 4 * TAKER), ("мейкер 4×0.02%", 4 * MAKER)):
            net = tr.gross + tr.funding - cost
            row = {"издержки": nm}
            for p in PER_ALL:
                z = net[tr.per == p]
                mon = z.groupby(tr.t[tr.per == p].dt.to_period("M")).sum() / N_PAIRS
                row[p] = (f"{z.mean() * 100:+.2f}% ({(z > 0).mean():.0%}, {len(z)}) · мес {mon.mean() * 100:+.2f}% "
                          f"Sh {mon.mean() / mon.std() * np.sqrt(12):+.1f}" if len(z) > 5 and mon.std() > 0 else f"n={len(z)}")
            rows.append(row)
        print(f"\n  --- {tf}: сделок {len(tr)}, сошлись {tr.conv.mean():.0%}, держали в среднем {tr.bars.mean():.0f} свечей; "
              f"фандинг на сделку {tr.funding.mean() * 100:+.3f}% ---")
        print(pd.DataFrame(rows).to_string(index=False))


# ---------------- 2. фандинг ----------------
def carry_report(close1: pd.DataFrame, fund1: pd.DataFrame, adv: pd.DataFrame) -> None:
    print("\n  === 2. КРОСС-СЕКЦИОННЫЙ ФАНДИНГ: лонг низкого фандинга, шорт высокого, неделя; % на капитал "
          "(сумма длинной ноги = короткой = 1) ===")
    wk = close1.resample("7D", origin="start").last()
    fw = fund1.resample("7D", origin="start").sum()
    advw = adv.reindex(wk.index, method="ffill")
    ret = wk.shift(-1) / wk - 1                         # цена: следующая неделя
    fnext = fw.shift(-1)                                # фандинг, начисленный за следующую неделю
    btc = ret["BTCUSDT"] if "BTCUSDT" in ret else None
    for k in CARRY_K:
        rows, prev_w = [], None
        res = []
        for t in wk.index[:-1]:
            a = advw.loc[t]
            sig = fw.loc[t]
            u = sig.index[(a >= ADV_MIN) & sig.notna() & ret.loc[t].notna() & (sig != 0)]
            u = a[u].sort_values(ascending=False).index[:TOP]
            if len(u) < 4 * k:
                continue
            srt = sig[u].sort_values()
            w = pd.Series(0.0, index=close1.columns)
            w[srt.index[:k]] = 1.0 / k                     # лонг: низкий фандинг (нам платят или почти)
            w[srt.index[-k:]] = -1.0 / k                   # шорт: высокий фандинг (нам платят)
            price = float((w * ret.loc[t]).sum())
            fund = float(-(w * fnext.loc[t].fillna(0)).sum())
            turn = float((w - prev_w).abs().sum()) if prev_w is not None else 2.0
            prev_w = w
            res.append({"t": t, "price": price, "fund": fund, "cost": turn * TAKER,
                        "btc": float(btc.loc[t]) if btc is not None else np.nan})
        r = pd.DataFrame(res)
        if not len(r):
            print(f"  K={k}: недель нет")
            continue
        r["net"] = r.price + r.fund - r.cost
        r["per"] = _per_of(r.t)
        for col, nm in (("net", "итог"), ("price", "  цена"), ("fund", "  фандинг"), ("cost", "  комиссии")):
            rows.append({"K": k, "часть": nm, **{p: _stat(r[col][r.per == p], 52) for p in PER_ALL}})
        print(pd.DataFrame(rows).to_string(index=False))
        print(f"  K={k}: корреляция недель с BTC {r.net.corr(r.btc):+.2f}; по годам: " + ", ".join(
            f"{y}: {g.net.sum() * 100:+.1f}%" for y, g in r.groupby(r.t.dt.year)))


# ---------------- 3. бот с хеджем ----------------
def _dd(r: pd.Series) -> float:
    eq = r.cumsum()
    return float((eq.cummax() - eq).max()) if len(eq) else np.nan


def bot_report(bot: pd.DataFrame) -> None:
    print("\n  === 3. БОТ 4h С ХЕДЖЕМ BTC: R на сделку без хеджа и с хеджем (BTC в обратную сторону на бету, 60 дней) ===")
    bot = bot.dropna(subset=["R3", "hedge_R"]).sort_values("t").copy()
    bot["H"] = bot.R3 + bot.hedge_R
    bot["per"] = _per_of(bot.t)
    rows = []
    for col, nm in (("R3", "без хеджа"), ("H", "с хеджем BTC"), ("hedge_R", "  сама нога BTC")):
        rows.append({"вариант": nm, **{p: _cell(bot.assign(R=bot[col])[bot.per == p]) for p in PER_ALL}})
    print(pd.DataFrame(rows).to_string(index=False))
    rows = []
    for col, nm in (("R3", "без хеджа"), ("H", "с хеджем BTC")):
        m = bot.groupby(bot.t.dt.to_period("M"))[col].sum()
        rows.append({"вариант": nm, "R/мес": f"{m.mean():+.2f}", "σ месяца": f"{m.std():.2f}",
                     "Sharpe мес.": f"{m.mean() / m.std() * np.sqrt(12):+.2f}", "худший месяц": f"{m.min():+.1f}",
                     "просадка R": f"{_dd(bot[col]):.1f}", **{p: _dd(bot[col][bot.per == p]) for p in PER_ALL}})
    print(pd.DataFrame(rows).round(1).to_string(index=False))
    print(f"  бета монет к BTC: медиана {bot.beta.median():.2f}, доля шортов {(bot.side < 0).mean():.0%}; "
          f"корреляция R сделки с ходом BTC × сторона: {bot.R3.corr(bot.side * bot.btc_ret / bot.risk_pct):+.2f}")
    print(f"  без хеджа по годам: {_years(bot)}")
    print(f"  с хеджем по годам:  {_years(bot, 'H')}")


def report() -> None:
    print("\n===== HEDGED: хеджированные стратегии — пары, фандинг, бот с хеджем BTC (2020 / 2021 / IS / VAL / HO) =====")
    pp, bp = all_parts("hedged_panel"), all_parts("hedged_bot")
    if bp:
        bot_report(pd.concat([pd.read_parquet(p) for p in bp], ignore_index=True).assign(
            t=lambda z: pd.to_datetime(z.t, utc=True)))
    if not pp:
        print("частей панели нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in pp], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df.drop_duplicates(["symbol", "t"])
    close = df.pivot(index="t", columns="symbol", values="close").sort_index()
    fund = df.pivot(index="t", columns="symbol", values="funding").reindex(close.index).fillna(0.0)
    adv = df.pivot(index="t", columns="symbol", values="adv").reindex(close.index).resample("1D").last()
    del df
    print(f"  панель: {close.shape[1]} монет, {close.index[0]:%Y-%m} — {close.index[-1]:%Y-%m}")
    carry_report(close, fund, adv)
    pairs_report(close, fund, adv)


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
