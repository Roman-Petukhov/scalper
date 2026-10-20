"""Мета-разметка со скользящим переобучением (по вопросу владельца: «обучить на свежих данных и признаках нынешнего
рынка»). Модель переобучается каждый квартал на последних W месяцах и торгует следующий квартал, не заглядывая
вперёд, — так её применял бы бот. Кандидаты и признаки — research/metalabel.py (части metalabel) плюс признаки режима:
  btc_vol      волатильность BTC за 30 свечей к её уровню за 500 (из сбора)
  breadth      сколько монет в ту же свечу пробили линию в ту же сторону минус в обратную (без самой монеты)
  mkt_funding  средний funding кандидатов рынка за 7 дней до сигнала (без знака сделки)
Доминации BTC в данных нет — не используется.
Окна W = 6 / 12 / 24 месяца; проверочные кварталы — с 2021-01 по последний полный. Чистка: из обучения убираются
сделки, открытые ближе PURGE к началу квартала (их исход ещё не известен). Параметры LightGBM — под малую выборку
(min_data_in_leaf ≥ 30). Пороги — по прогнозам сделок бота окна обучения (отсечь нижние 40%; размер ×0.5 / 1 / 1.5 по
третям); квартал без 30 сделок бота в окне — без модели.
Принятие (задано до запуска): на всех проверочных кварталах вместе оставшиеся лучше отсечённых с t ≥ 2, оставшиеся
лучше бота и «оставшиеся > отсечённые» не меньше чем в 4 из 6 лет 2021–2026; для размера — t ≥ 2 у (вес − 1)·R.

    python -m research.metalabel collect --symbols <1h-qualified>     (данные — части metalabel)
    python -m research.metaroll report
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from .metalabel import ENTRY, FEATS, FILTER_Q, SIZES, _tstat, _welch
from .shard import all_parts
from .smc import _cell

TFS = ("4h", "15m")
WINDOWS = (6, 12, 24)
REGIME = ["btc_vol", "breadth", "mkt_funding"]
ALL_FEATS = FEATS + REGIME
PURGE = {"4h": pd.Timedelta(days=14), "15m": pd.Timedelta(days=3)}
FIRST, MIN_BOT, ROUNDS = "2021-01-01", 30, 200


def add_regime(z: pd.DataFrame) -> pd.DataFrame:
    """breadth и mkt_funding по всем кандидатам ТФ (все монеты собраны вместе, поэтому только в отчёте)."""
    z = z.copy()
    cnt = z.groupby(["t", "side"]).size()
    up = cnt.xs(1, level="side").reindex(z.t).to_numpy(dtype="float64") if 1 in cnt.index.get_level_values(1) else 0.0
    dn = cnt.xs(-1, level="side").reindex(z.t).to_numpy(dtype="float64") if -1 in cnt.index.get_level_values(1) else 0.0
    up, dn = np.nan_to_num(up), np.nan_to_num(dn)
    same = np.where(z.side > 0, up, dn) - 1
    other = np.where(z.side > 0, dn, up)
    z["breadth"] = same - other
    raw = (z.fund_day * z.side).rename("f")
    daily = raw.groupby(z.t.dt.floor("D")).mean()
    roll = daily.rolling(7, min_periods=1).mean().shift(1)
    z["mkt_funding"] = roll.reindex(z.t.dt.floor("D")).to_numpy()
    return z


def params(n: int) -> dict:
    return {"objective": "regression", "learning_rate": 0.03, "num_leaves": 7, "min_data_in_leaf": max(30, n // 100),
            "bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.8, "lambda_l2": 5.0, "verbose": -1,
            "seed": 7}


def roll(z: pd.DataFrame, tf: str, months: int, lgb) -> pd.DataFrame:
    """Сделки бота проверочных кварталов с keep / w; без модели — keep = True, w = 1."""
    end = z.t.max()
    starts = pd.date_range(FIRST, end, freq="QS")
    out = []
    for q0 in starts:
        q1 = q0 + pd.DateOffset(months=3)
        test = z[(z.t >= q0) & (z.t < q1) & z.bot]
        if test.empty:
            continue
        tr = z[(z.t >= q0 - pd.DateOffset(months=months)) & (z.t < q0 - PURGE[tf])]
        res = test.assign(q=q0, keep=True, w=1.0, modeled=False)
        if len(tr) >= 300 and tr.bot.sum() >= MIN_BOT:
            m = lgb.train(params(len(tr)), lgb.Dataset(tr[ALL_FEATS].astype("float64"), tr.R.clip(-1.5, 3.5)),
                          num_boost_round=ROUNDS)
            ref = m.predict(tr.loc[tr.bot, ALL_FEATS].astype("float64"))
            sc = m.predict(test[ALL_FEATS].astype("float64"))
            q = np.quantile(ref, FILTER_Q)
            t1, t2 = np.quantile(ref, [1 / 3, 2 / 3])
            res = res.assign(keep=sc > q, w=np.where(sc <= t1, SIZES[0], np.where(sc <= t2, SIZES[1], SIZES[2])),
                             modeled=True, score=sc)
        out.append(res)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def report() -> None:
    parts = all_parts("metalabel")
    print("\n===== METAROLL: мета-разметка со скользящим переобучением каждый квартал на последних 6 / 12 / 24 месяцах "
          "(+ признаки режима); проверочные кварталы 2021–2026 вместе; R на сделку (t по дням, прибыльных, сделок в месяц) =====")
    if not parts:
        print("  metaroll: нет данных")
        return
    try:
        import lightgbm as lgb
    except ImportError:
        print("  lightgbm не установлен — модель пропущена")
        return
    x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
    x = x.dropna(subset=["R"]).sort_values("t").reset_index(drop=True)
    if "btc_vol" not in x:
        x["btc_vol"] = np.nan
    verdicts = []
    for tf in TFS:
        z = x[x.tf == tf].reset_index(drop=True)
        if z.empty:
            continue
        z = add_regime(z)
        print(f"\n  --- {tf}, вход {ENTRY[tf]}: кандидатов {len(z)}, у бота {int(z.bot.sum())} ---")
        for months in WINDOWS:
            r = roll(z, tf, months, lgb)
            if r.empty:
                continue
            m = r[r.modeled]
            if m.empty:
                print(f"    окно {months} мес.: мало данных в окне — модель ни разу не обучалась")
                verdicts.append(f"{tf} {months} мес.: мало данных")
                continue
            kept, cut = m[m.keep], m[~m.keep]
            t_f = _welch(kept.R.to_numpy(), cut.R.to_numpy())
            t_s = _tstat(((m.w - 1) * m.R).to_numpy())
            years = sorted(m.t.dt.year.unique())
            wins = 0
            cells = []
            for y in years:
                g = m[m.t.dt.year == y]
                k, c = g[g.keep].R.mean(), g[~g.keep].R.mean()
                wins += int(k > c)
                cells.append(f"{y}: {k:+.2f} / {c:+.2f} ({len(g)})")
            ok_f = len(kept) >= 20 and len(cut) >= 20 and t_f >= 2 and kept.R.mean() > m.R.mean() and wins >= 4
            ok_s = t_s >= 2 and (m.w * m.R).mean() > m.R.mean()
            print(f"    окно {months} мес.: кварталов с моделью {m.q.nunique()} из {r.q.nunique()}; бот {_cell(m)}")
            print(f"       filter: оставшиеся {_cell(kept)}  отсечённые {_cell(cut)}  разница t {t_f:.1f}")
            print(f"       по годам оставшиеся / отсечённые, R: " + "  ".join(cells) + f"  (лучше в {wins} из {len(years)})")
            print(f"       size ×0.5 / 1 / 1.5: {(m.w * m.R).mean():+.3f}R против {m.R.mean():+.3f}R, t {t_s:.1f}")
            verdicts.append(f"{tf} {months} мес.: filter {'принято' if ok_f else 'нет'} (t {t_f:.1f}, лет {wins}/{len(years)}), "
                            f"size {'принято' if ok_s else 'нет'} (t {t_s:.1f})")
    print("\n  ИТОГ: " + ("; ".join(verdicts) if verdicts else "нет данных"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 400)
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
