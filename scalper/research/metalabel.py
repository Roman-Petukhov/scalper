"""Мета-разметка (López de Prado, «Advances in Financial Machine Learning», гл. 3): первичная модель — правило бота
(пробой линии зигзага 3 ATR задаёт сторону), вторичная — LightGBM решает, брать ли сигнал и каким объёмом.

Кандидаты — все пробои линий бота (research/noline.py) с тем же стопом, входом и целью 3R, но без фильтров
агрессоров, закрытия и тренда: эти величины идут в модель признаками, так у неё в разы больше примеров. Оценка —
только на сделках бота (агрессоры ≥ 55%, закрытие в своей половине, по тренду; 15m — ещё пологие шорты).
4h — вход на ретесте (основной бот), 15m — по рынку; монеты 1h-qualified с 2020 (как research/tlcanon.py).
Признаки (все считаются ботом в момент сигнала по свечам монеты, BTC и funding): сторона, агрессоры, место закрытия,
по тренду, глубина пробоя в ATR, риск в ATR, наклон линии, длина линии, касания, объём к среднему за 20 свечей, тело
свечи, ATR к среднему за 100, ширина 10 свечей в ATR, ход за 30 свечей в ATR в сторону сделки, funding за сутки (со
знаком сделки), BTC за 30 свечей и BTC против EMA50 (со знаком сделки), круглый уровень, час, день недели.
Цель — R сделки (обрезан −1.5…3.5), регрессия; параметры — как в research/tline.py.
Обучение «вперёд» (без заглядывания): модель A — на кандидатах 2020 / 2021 / IS, проверка на VAL и HO; модель B — до
конца VAL, проверка на HO. Пороги — по прогнозам сделок бота в периоде обучения. Политики (заданы до запуска):
  filter  брать только сделки бота выше 40-го перцентиля прогноза (отсечь нижние 40%)
  size    риск ×0.5 / ×1 / ×1.5 по третям прогноза (в среднем ×1)
Принятие: filter — R на сделку у оставшихся выше бота в VAL и HO (модель A) и в HO (модель B), разница «оставшиеся
минус отсечённые» на VAL + HO с t ≥ 2; size — R на сделку с весами выше бота там же и t ≥ 2 у (вес − 1)·R на VAL + HO.
При десятках уже проверенных идей уверенный результат — t ≥ 3 (Harvey, Liu, Zhu 2016).

    python -m research.metalabel collect --symbols <1h-qualified>
    python -m research.metalabel report
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
from .tline import (HOLD, MAKER, PIV, RETEST_BARS, STOP_BUF, TAKER, crossed_round, htf_trend, last_confirmed,
                    line_touches, pivots, retest_fill, tf_frame, two_targets, zz_lines)

TFS = tuple(os.environ.get("METALABEL_TFS", "4h,15m").split(","))
ENTRY = {"4h": "retest", "15m": "market"}
MAX_SLOPE = 0.025
FEATS = ["side", "aggr", "close_loc", "with_trend", "brk_atr", "risk_atr", "slope_atr", "line_len", "touches",
         "vol_ratio", "body", "atr_ratio", "range10", "ret30", "fund_day", "btc_ret30", "btc_trend", "round", "hour",
         "dow"]
PARAMS = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 200,
          "bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.8, "lambda_l2": 5.0, "verbose": -1, "seed": 7}
ROUNDS, FILTER_Q, SIZES = 300, 0.4, (0.5, 1.0, 1.5)
BARS_DAY = {"4h": 6, "15m": 96}
_BTC: dict[str, pd.DataFrame | None] = {}


def _btc(root: Path, tf: str) -> pd.DataFrame | None:
    if tf not in _BTC:
        b = tf_frame(root, "BTCUSDT", tf)
        if b is not None:
            b = pd.DataFrame({"close": b["close"], "ema": b["close"].ewm(span=50, adjust=False).mean()})
        _BTC[tf] = b
    return _BTC[tf]


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str, btc: pd.DataFrame | None) -> list[dict]:
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    vol = d["volume"].to_numpy(dtype="float64")
    a = _atr(d).to_numpy()
    trend = htf_trend(d, tf)
    buy = (d["taker_buy_volume"] / d["volume"].replace(0, np.nan)).to_numpy()
    rng = hi - lo
    loc = (c - lo) / np.where(rng > 0, rng, np.nan)
    sw_lo, sw_hi = last_confirmed(pivots(lo, PIV, False), len(c)), last_confirmed(pivots(hi, PIV, True), len(c))
    if btc is not None:
        bb = btc.reindex(d.index)
        bc, be = bb["close"].to_numpy(), bb["ema"].to_numpy()
    else:
        bc = be = np.full(len(c), np.nan)
    hours, dows = d.index.hour.to_numpy(), d.index.dayofweek.to_numpy()
    day = BARS_DAY.get(tf, 6)
    entry = ENTRY[tf]
    rows = []
    for ln in zz_lines(d):
        t, side = int(ln["t"]), int(ln["side"])
        if t <= 100 or t >= len(c) - 1 or not (a[t] > 0) or not (liq[t] >= ADV_MIN):
            continue
        sw = sw_lo[t] if side > 0 else sw_hi[t]
        if sw < 0:
            continue
        level = float(ln["line_t"])
        stop = (lo[sw] - STOP_BUF * a[t]) if side > 0 else (hi[sw] + STOP_BUF * a[t])
        if entry == "market":
            fill, px, fee = t, c[t], TAKER
        else:
            if not (side * (level - stop) > 0):
                continue
            fill = retest_fill(hi, lo, t, side, level, 3 * side * (level - stop), RETEST_BARS)
            if fill < 0:
                continue
            px, fee = level, MAKER
        risk = side * (px - stop)
        if not (0.3 * a[t] <= risk <= 4.0 * a[t]):
            continue
        if entry == "retest" and ((side > 0 and lo[fill] <= stop) or (side < 0 and hi[fill] >= stop)):
            r3 = (side * (stop - px) - (fee + TAKER) * px) / risk
        else:
            r3, _ = two_targets(o, hi, lo, c, f, fill, side, px, stop, 3.0, 3.0, False, HOLD[tf], fee)
        aggr = buy[t] if side > 0 else 1 - buy[t]
        cl = loc[t] if side > 0 else 1 - loc[t]
        slope, la = float(ln["slope"]), int(ln["a"])
        slope_atr = abs(slope) / a[t]
        bot = bool(aggr >= 0.55 and cl >= 0.5 and trend[t] == side)
        if tf == "15m":
            bot = bot and side < 0 and slope_atr <= MAX_SLOPE
        rows.append({
            "symbol": sym, "tf": tf, "t": d.index[t], "R": float(r3), "bot": bot,
            "side": side, "aggr": aggr, "close_loc": cl, "with_trend": float(trend[t] == side),
            "brk_atr": side * (c[t] - level) / a[t], "risk_atr": risk / a[t], "slope_atr": slope_atr,
            "line_len": t - la, "touches": line_touches(c, a, la, t, side, float(c[la]), slope),
            "vol_ratio": vol[t] / vol[t - 20:t].mean() if vol[t - 20:t].mean() > 0 else np.nan,
            "body": abs(c[t] - o[t]) / rng[t] if rng[t] > 0 else np.nan,
            "atr_ratio": a[t] / np.nanmean(a[t - 100:t]), "range10": (hi[t - 9:t + 1].max() - lo[t - 9:t + 1].min()) / a[t],
            "ret30": side * (c[t] - c[t - 30]) / a[t], "fund_day": side * f[max(0, t - day + 1):t + 1].sum(),
            "btc_ret30": side * (bc[t] / bc[t - 30] - 1) if bc[t - 30] > 0 else np.nan,
            "btc_trend": side * np.sign(bc[t] - be[t]) if np.isfinite(bc[t]) else np.nan,
            "round": float(c[t - 1] > 0 and crossed_round(c[t - 1], c[t])), "hour": int(hours[t]), "dow": int(dows[t])})
    return rows


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
                rows += coin_rows(d, liq, s, tf, _btc(root, tf))
            except Exception as e:
                print(f"  metalabel {s} {tf}: пропуск ({e})", flush=True)
    print(f"  metalabel: кандидатов {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("metalabel"), index=False)


def _in(z: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    return z[(z.t >= a) & (z.t < b)]


def _tstat(x: np.ndarray) -> float:
    sd = x.std(ddof=1) if len(x) > 2 else 0.0
    return float(x.mean() / (sd / np.sqrt(len(x)))) if sd > 0 else np.nan


def _welch(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or len(b) < 3:
        return np.nan
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else np.nan


def evaluate(bot: pd.DataFrame, score: np.ndarray, train: np.ndarray) -> pd.DataFrame:
    """Сделки бота с прогнозом: keep — выше 40-го перцентиля прогноза по сделкам бота периода обучения; w — вес 0.5 / 1 /
    1.5 по третям того же распределения."""
    z = bot.assign(score=score)
    ref = z.score[train]
    q = ref.quantile(FILTER_Q)
    t1, t2 = ref.quantile([1 / 3, 2 / 3]).to_numpy()
    z["keep"] = z.score > q
    z["w"] = np.where(z.score <= t1, SIZES[0], np.where(z.score <= t2, SIZES[1], SIZES[2]))
    return z


def report() -> None:
    parts = all_parts("metalabel")
    print("\n===== METALABEL: мета-разметка — LightGBM поверх правила бота решает, брать ли сигнал (filter) и каким объёмом "
          "(size); обучение вперёд, проверка на VAL / HO; R на сделку (t по дням, прибыльных, сделок в месяц) =====")
    if not parts:
        print("  metalabel: нет данных")
        return
    try:
        import lightgbm as lgb
    except ImportError:
        print("  lightgbm не установлен — модель пропущена")
        return
    x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
    x = x.dropna(subset=["R"]).sort_values("t").reset_index(drop=True)
    val_a, val_b = PER_ALL["val"]
    ho_a, ho_b = PER_ALL["ho"]
    start = min(a for a, _ in PER_ALL.values())
    verdicts = []
    for tf in TFS:
        z = x[x.tf == tf].reset_index(drop=True)
        if z.empty:
            continue
        X = z[FEATS].astype("float64")
        y = z.R.clip(-1.5, 3.5)
        tr_a = (z.t >= start) & (z.t < val_a)
        tr_b = (z.t >= start) & (z.t < ho_a)
        print(f"\n  --- {tf}, вход {ENTRY[tf]}: кандидатов {len(z)} (у бота {int(z.bot.sum())}); обучение A {int(tr_a.sum())}, "
              f"B {int(tr_b.sum())} ---")
        if tr_a.sum() < 1000:
            print("    мало кандидатов для обучения — пропуск")
            continue
        ma = lgb.train(PARAMS, lgb.Dataset(X[tr_a], y[tr_a]), num_boost_round=ROUNDS)
        mb = lgb.train(PARAMS, lgb.Dataset(X[tr_b], y[tr_b]), num_boost_round=ROUNDS)
        imp = pd.Series(ma.feature_importance("gain"), index=FEATS).sort_values(ascending=False)
        print("    важность A (gain): " + ", ".join(f"{k} {v / imp.sum():.0%}" for k, v in imp.head(10).items()))
        bm = z.bot.to_numpy()
        bot = z[bm].reset_index(drop=True)
        ea = evaluate(bot, ma.predict(X[bm]), tr_a[bm].to_numpy())
        eb = evaluate(bot, mb.predict(X[bm]), tr_b[bm].to_numpy())
        ic = [f"{p} {(_in(ea, a, b).score.corr(_in(ea, a, b).R, method='spearman')):+.3f}"
              for p, (a, b) in PER_ALL.items() if p in ("val", "ho")]
        print("    ранговая корреляция прогноза A с R сделок бота: " + ", ".join(ic))
        for p, (a, b) in PER_ALL.items():
            g = _in(ea, a, b)
            if g.empty:
                continue
            tag = " (обучение)" if b <= val_a else ""
            kept, cut = g[g.keep], g[~g.keep]
            sized = (g.w * g.R).mean()
            print(f"    {p}{tag}: бот {_cell(g)};  filter {_cell(kept)}, отсечено {_cell(cut)};  size {sized:+.3f}R "
                  f"(к боту {sized - g.R.mean():+.3f})")
        gb = _in(eb, ho_a, ho_b)
        if len(gb):
            print(f"    модель B на HO: filter {_cell(gb[gb.keep])}, отсечено {_cell(gb[~gb.keep])};  size "
                  f"{(gb.w * gb.R).mean():+.3f}R (к боту {(gb.w * gb.R).mean() - gb.R.mean():+.3f})")
        oos = pd.concat([_in(ea, val_a, val_b), _in(ea, ho_a, ho_b)])
        per = [_in(ea, val_a, val_b), _in(ea, ho_a, ho_b)]
        t_f = _welch(oos.R[oos.keep].to_numpy(), oos.R[~oos.keep].to_numpy())
        ok_f = all(len(g) >= 20 and g.R[g.keep].mean() > g.R.mean() for g in per) and \
            len(gb) >= 20 and gb.R[gb.keep].mean() > gb.R.mean() and t_f >= 2
        d_s = ((oos.w - 1) * oos.R).to_numpy()
        t_s = _tstat(d_s)
        ok_s = all(len(g) >= 20 and (g.w * g.R).mean() > g.R.mean() for g in per) and \
            len(gb) >= 20 and (gb.w * gb.R).mean() > gb.R.mean() and t_s >= 2
        mark = lambda ok, t: ("принято" + (" (t ≥ 3)" if t >= 3 else "")) if ok else "нет"
        verdicts.append(f"{tf}: filter {mark(ok_f, t_f)} (оставшиеся минус отсечённые на VAL + HO t {t_f:.1f}); "
                        f"size {mark(ok_s, t_s)} (t {t_s:.1f})")
    print("\n  ИТОГ: " + ("; ".join(verdicts) if verdicts else "нет данных"))


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
