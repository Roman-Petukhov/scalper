"""Модель сопровождения: «следить за сделкой как человек» (по вопросу владельца после research/exit5.py и research/at2r.py).

Сделки — правило бота (research/noline.py), все ~725 монет с 2022 (как research/holdlen.py); 5m и 15m — по рынку
(срок 800 / 200 свечей), 4h — ретест (срок 60). Внутри открытой сделки — точки решения на закрытии каждой STRIDE-й
свечи (5m — каждые 30 минут, 15m — каждый час, 4h — каждая свеча), пока позиция открыта после этой свечи.
В точке решения — картина, которую видит трейдер (только данные до закрытия этой свечи):
  где цена в R, лучший и худший ход, сколько отдали от лучшего, свечей в сделке; агрессоры в сторону сделки от входа,
  за 3 и 12 свечей; объём последних 3 свечей к свече пробоя и последних 12 к обычному до входа; где закрылась свеча
  (у экстремума в нашу сторону или откатилась); расстояние до пробитой линии в ATR; отрыв от EMA50; импульс за 3 и 12
  свечей; ATR сейчас к ATR входа; риск входа в ATR; наклон линии; сторона; час; круглый уровень и экстремум 500 свечей
  до входа между ценой и целью.
Цель обучения — выгода удержания: R бота (стоп / 3R / срок) минус R выхода по закрытию этой свечи. LightGBM
(регрессия, параметры как в research/tline.py), обучение на всех сделках ТФ (обе стороны, любой наклон):
  модель A — на точках сделок, открытых в IS; проверка на VAL и HO (модель их не видела);
  модель B — на IS + VAL; проверка на HO (обучение «на всём, что было», как в боте).
Политика: выйти в первой точке, где прогноз выгоды удержания < −порог; иначе — как бот. Порог 0.25R задан до запуска
(0 и 0.5R — справочно). Оценка — на сделках среза бота (5m / 15m — пологие шорты, 4h — все).
Принятие: политика A лучше бота по R на сделку в VAL и в HO (≥ 20 сделок), политика B лучше в HO, и парная разница A
на VAL + HO с t ≥ 2. Дополнительно — ранговая корреляция прогноза с выгодой на точках VAL / HO: есть ли у модели вообще
информация.

    python -m research.trademl collect --symbols <hot-wide>
    python -m research.trademl report
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from . import noline
from .broad import adv30
from .oos import PER
from .roundnum import level_ahead
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import HOLD, MAKER, TAKER, tf_frame, zz_lines

TFS = tuple(os.environ.get("TRADEML_TFS", "5m,15m,4h").split(","))
#      вход, срез бота — пологие шорты, срок, шаг точек решения
SLICE = {"5m": ("market", True, 800, 6), "15m": ("market", True, HOLD["15m"], 4), "4h": ("retest", False, HOLD["4h"], 1)}
MAX_SLOPE, SWING_BACK, PRE_VOL, CLIP = 0.025, 500, 100, 4.0
THRESHOLDS, MAIN_THR = (0.0, 0.25, 0.5), 0.25
FEATS = ["side", "slope_atr", "risk_atr", "bars_in", "r_now", "mfe", "mae", "giveback", "aggr_in", "aggr3", "aggr12",
         "vol3_bo", "vol12_pre", "loc", "line_atr", "stretch", "mom3", "mom12", "atr_ratio", "hour", "round_ahead",
         "swing_ahead"]
PARAMS = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 200,
          "bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.8, "lambda_l2": 5.0, "verbose": -1, "seed": 7}
ROUNDS = 300


@njit(cache=True)
def trade(o, h, lo, c, fund, e, side, entry, stop, k, max_hold, entry_fee):
    """Бот: стоп / k·R / срок. Возвращает (R, свеча выхода). Стоп раньше тейка."""
    n = len(c)
    risk = side * (entry - stop)
    if not (risk > 0):
        return np.nan, -1
    tp = entry + side * k * risk
    paid = 0.0
    for j in range(e + 1, n):
        paid += fund[j]
        if (side > 0 and lo[j] <= stop) or (side < 0 and h[j] >= stop):
            ex = min(stop, o[j]) if side > 0 else max(stop, o[j])
            return (side * (ex - entry) - (entry_fee + TAKER) * entry) / risk - side * paid * entry / risk, j
        if (side > 0 and h[j] >= tp) or (side < 0 and lo[j] <= tp):
            return (k * risk - (entry_fee + MAKER) * entry) / risk - side * paid * entry / risk, j
        if j - e >= max_hold or j == n - 1:
            return (side * (c[j] - entry) - (entry_fee + TAKER) * entry) / risk - side * paid * entry / risk, j
    return np.nan, -1


def _share(buy: np.ndarray, vol: np.ndarray, a: int, b: int, side: int) -> float:
    v = vol[a:b].sum()
    if not v > 0:
        return np.nan
    s = buy[a:b].sum() / v
    return s if side > 0 else 1 - s


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str) -> list[dict]:
    entry_kind, bot_slice, hold, stride = SLICE[tf]
    noline.TF = tf
    x = noline.coin_trades(d, liq, sym, lines_only=True)
    if not len(x):
        return []
    x = x[x.entry == entry_kind]
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    vol = d["volume"].to_numpy(dtype="float64")
    buy = d["taker_buy_volume"].to_numpy(dtype="float64")
    atr = _atr(d).to_numpy()
    ema = d["close"].ewm(span=50, adjust=False).mean().to_numpy()
    hours = d.index.hour.to_numpy()
    lines = {(int(r["t"]), int(r["side"])): r for r in zz_lines(d) if r["t"] > 0}
    fee = MAKER if entry_kind == "retest" else TAKER
    rows = []
    for r in x.itertuples(index=False):
        t, side, e, px, stop = d.index.get_loc(r.t), int(r.side), int(r.fill_i), float(r.px), float(r.stop)
        ln = lines.get((t, side))
        if ln is None or not (atr[t] > 0) or t < PRE_VOL:
            continue
        if entry_kind == "retest" and ((side > 0 and lo[e] <= stop) or (side < 0 and hi[e] >= stop)):
            continue                                                   # стоп в свече исполнения: решать нечего
        hold_r, ex = trade(o, hi, lo, c, f, e, side, px, stop, 3.0, hold, fee)
        if ex < 0 or not np.isfinite(hold_r):
            continue
        risk = side * (px - stop)
        slope = float(ln["slope"])
        line_t = float(ln["line_t"])
        slope_atr = abs(slope) / atr[t]
        tp = px + side * 3.0 * risk
        ext = lo[max(0, e - SWING_BACK):e].min() if side < 0 else hi[max(0, e - SWING_BACK):e].max()
        pre_vol = vol[t - PRE_VOL:t].mean()
        bot = (not bot_slice) or (side < 0 and slope_atr <= MAX_SLOPE)
        paid, mfe, mae, n0 = 0.0, 0.0, 0.0, len(rows)
        for j in range(e + 1, ex):                                     # позиция открыта после закрытия свечи j
            paid += f[j]
            fav = (hi[j] - px) / risk if side > 0 else (px - lo[j]) / risk
            adv = (px - lo[j]) / risk if side > 0 else (hi[j] - px) / risk
            mfe, mae = max(mfe, fav), max(mae, adv)
            if (j - e) % stride:
                continue
            exit_r = (side * (c[j] - px) - (fee + TAKER) * px) / risk - side * paid * px / risk
            rng = hi[j] - lo[j]
            r_now = side * (c[j] - px) / risk
            rows.append({
                "symbol": sym, "tf": tf, "t": r.t, "j": j - e, "bot": bot, "hold": float(hold_r), "exit": float(exit_r),
                "side": side, "slope_atr": slope_atr, "risk_atr": risk / atr[t], "bars_in": j - e, "r_now": r_now,
                "mfe": mfe, "mae": mae, "giveback": mfe - r_now,
                "aggr_in": _share(buy, vol, e + 1, j + 1, side), "aggr3": _share(buy, vol, max(e + 1, j - 2), j + 1, side),
                "aggr12": _share(buy, vol, max(e + 1, j - 11), j + 1, side),
                "vol3_bo": vol[j - 2:j + 1].mean() / vol[t] if vol[t] > 0 else np.nan,
                "vol12_pre": vol[j - 11:j + 1].mean() / pre_vol if pre_vol > 0 else np.nan,
                "loc": ((c[j] - lo[j]) / rng if side > 0 else (hi[j] - c[j]) / rng) if rng > 0 else 0.5,
                "line_atr": side * (c[j] - (line_t + slope * (j - t))) / atr[t],
                "stretch": side * (c[j] - ema[j]) / atr[j] if atr[j] > 0 else np.nan,
                "mom3": side * (c[j] - c[j - 3]) / atr[j] if atr[j] > 0 else np.nan,
                "mom12": side * (c[j] - c[j - 12]) / atr[j] if atr[j] > 0 else np.nan,
                "atr_ratio": atr[j] / atr[t], "hour": int(hours[j]),
                "round_ahead": float(level_ahead(c[j], side, abs(tp - c[j]), True)) if side * (tp - c[j]) > 0 else 0.0,
                "swing_ahead": float(min(c[j], tp) < ext < max(c[j], tp)), "pt": True})
        if len(rows) == n0:                                            # вышла до первой точки: решения не было
            rows.append({"symbol": sym, "tf": tf, "t": r.t, "j": 0, "bot": bot, "hold": float(hold_r),
                         "exit": float(hold_r), "pt": False, **{k: np.nan for k in FEATS}})
    return rows


def collect(root: Path, syms: list[str]) -> None:
    rows = []
    for s in mine(syms):
        for tf in TFS:
            try:
                d = tf_frame(root, s, tf)
                if d is None or len(d) < 2000:
                    continue
                liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
                rows += coin_rows(d, liq, s, tf)
            except Exception as e:
                print(f"  trademl {s} {tf}: пропуск ({e})", flush=True)
    print(f"  trademl: точек решения {len(rows)}", flush=True)
    if rows:
        df = pd.DataFrame(rows)
        num = df.select_dtypes("float64").columns
        df[num] = df[num].astype("float32")
        df.to_parquet(part_path("trademl"), index=False)


def policy(pts: pd.DataFrame, pred: np.ndarray, thr: float) -> pd.DataFrame:
    """R на сделку: выход в первой точке, где прогноз < −thr, иначе R бота. pts — точки, отсортированные по сделке и j;
    сделки без точек решения (вышли раньше первой) остаются как у бота."""
    p = pts.assign(pred=pred, go=(pred < -thr) & pts["pt"].to_numpy())
    key = ["symbol", "t"]
    first = p[p.go].groupby(key, sort=False)["exit"].first()
    out = p.groupby(key, sort=False)["hold"].first().rename("bot").to_frame()
    out["R"] = first.reindex(out.index).fillna(out["bot"]).astype("float64")
    out["bot"] = out["bot"].astype("float64")
    out["exited"] = first.reindex(out.index).notna()
    return out.reset_index()


def _per(z: pd.DataFrame, p: str) -> pd.DataFrame:
    a, b = PER[p]
    return z[(z.t >= a) & (z.t < b)]


def _paired(z: pd.DataFrame) -> tuple[float, float]:
    dd = (z.R - z.bot).to_numpy()
    sd = dd.std(ddof=1) if len(dd) > 2 else 0.0
    return float(dd.mean()) if len(dd) else np.nan, float(dd.mean() / (sd / np.sqrt(len(dd)))) if sd > 0 else np.nan


def report() -> None:
    parts = all_parts("trademl")
    print("\n===== TRADEML: модель сопровождения (LightGBM по картине открытой сделки) — выход, когда модель ждёт, что "
          "держать хуже; R на сделку среза бота (t по дням, прибыльных, сделок в месяц) =====")
    if not parts:
        print("  trademl: нет данных")
        return
    try:
        import lightgbm as lgb
    except ImportError:
        print("  lightgbm не установлен — модель пропущена")
        return
    x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
    x = x.dropna(subset=["hold", "exit"]).sort_values(["tf", "symbol", "t", "j"]).reset_index(drop=True)
    x["y"] = (x["hold"] - x["exit"]).clip(-CLIP, CLIP)
    verdicts = []
    for tf in TFS:
        z = x[x.tf == tf].reset_index(drop=True)
        if z.empty:
            continue
        per = pd.Series("", index=z.index)
        for p in PER:
            a, b = PER[p]
            per[(z.t >= a) & (z.t < b)] = p
        X = z[FEATS].astype("float64")
        pt = z["pt"].to_numpy(dtype=bool)
        tr_a, tr_b = (per == "is").to_numpy() & pt, per.isin(["is", "val"]).to_numpy() & pt
        n_tr = z.loc[tr_a, ["symbol", "t"]].drop_duplicates().shape[0]
        print(f"\n  --- {tf}: точек {pt.sum()}, сделок {z[['symbol', 't']].drop_duplicates().shape[0]} "
              f"(в обучении A — {n_tr}); у среза бота {z[z.bot][['symbol', 't']].drop_duplicates().shape[0]} ---")
        if tr_a.sum() < 2000:
            print("    мало точек для обучения — пропуск")
            continue
        ma = lgb.train(PARAMS, lgb.Dataset(X[tr_a], z.y[tr_a]), num_boost_round=ROUNDS)
        mb = lgb.train(PARAMS, lgb.Dataset(X[tr_b], z.y[tr_b]), num_boost_round=ROUNDS)
        pa, pb = ma.predict(X), mb.predict(X)
        imp = pd.Series(ma.feature_importance("gain"), index=FEATS).sort_values(ascending=False)
        print("    важность A (gain): " + ", ".join(f"{k} {v / imp.sum():.0%}" for k, v in imp.head(8).items()))
        ic = []
        for p in ("val", "ho"):
            m = (per == p).to_numpy() & pt
            if m.sum() > 100:
                ic.append(f"{p} {pd.Series(pa[m]).corr(z.y[m].reset_index(drop=True), method='spearman'):+.3f}")
        print("    ранговая корреляция прогноза A с выгодой удержания на точках: " + ", ".join(ic))
        bm = z.bot.to_numpy()
        zb = z[bm]
        for thr in THRESHOLDS:
            ra = policy(zb, pa[bm], thr)
            rb = policy(zb, pb[bm], thr)
            cells = []
            for p in PER:
                g = _per(ra, p)
                tag = " (обучение)" if p == "is" else ""
                cells.append(f"{p}{tag}: {_cell(g)} вышли {g.exited.mean():.0%}, к боту {(g.R - g.bot).mean():+.3f}")
            gb = _per(rb, "ho")
            print(f"    порог {thr:.2f}{' (основной)' if thr == MAIN_THR else ''}: A — " + "  ".join(cells) +
                  f";  B на HO: {_cell(gb)} к боту {(gb.R - gb.bot).mean():+.3f}")
            if thr == MAIN_THR:
                oos = pd.concat([_per(ra, "val"), _per(ra, "ho")])
                dm, tt = _paired(oos)
                ok_a = all(len(_per(ra, p)) >= 20 and (_per(ra, p).R - _per(ra, p).bot).mean() > 0 for p in ("val", "ho"))
                ok_b = len(gb) >= 20 and (gb.R - gb.bot).mean() > 0
                passed = ok_a and ok_b and tt >= 2
                verdicts.append(f"{tf}: {'принято' if passed else 'нет'} (A на VAL + HO {dm:+.3f}R, t {tt:.1f}; "
                                f"B на HO {(gb.R - gb.bot).mean():+.3f}R)")
        print("    бот как есть: " + "  ".join(f"{p}: {_cell(_per(policy(zb, np.zeros(bm.sum()), 1.0), p))}" for p in PER))
    print("\n  ИТОГ (порог 0.25R): " + ("; ".join(verdicts) if verdicts else "нет данных"))


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
