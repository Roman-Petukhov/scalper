"""
Ловля иксов «вилкой»: модель research.moonshots знает, что монету скоро понесёт, но не знает куда — пусть направление
покажет рынок. На срезе (каждые 4 ч, закрытие часа) ставим стоп-ордер на покупку на +b и на продажу на −b от цены,
живут 24 ч; сработал один — второй снят. Стоп s от входа, трейлинг: стоп подтягивается на trail от лучшей цены
(для шорта — от минимума), без тейка — прибыль не ограничена; не дольше 30 дней. Если оба уровня пробиты в одном
часе — считаем худшее (вошли и выбило стопом). Издержки: 12 б.п. комиссий + проскальзывание 0.25% на входе и на выходе
по стопу (стоп-ордера в резком движении), funding по часам. Монета делистнута, пока сделка открыта, — выход по
последней цене.

Вселенная и признаки — как в research.moonshots (все перпетуалы Binance, включая делистнутые, оборот >= $2M/день,
признаки A). Модель up50 обучается на IS (внефолдовые прогнозы для IS), порог — верхние 0.5 / 1 / 2% прогнозов IS;
на VAL / HOLDOUT применяется как есть. Для сравнения: правило по одному признаку (vol_ratio в верхнем 1% IS) и все
срезы подряд (без отбора). Одна «вилка» на монету: следующий срез монеты — только после выхода из сделки или
истечения неисполненной вилки.

    python -m research.moonbracket collect --symbols <все перпетуалы>   (по частям)
    python -m research.moonbracket report
"""
from __future__ import annotations

import argparse
import heapq
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from .broad import adv30
from .moonshots import FEATS_A, PER, _fit, coin_rows
from .shard import all_parts, mine, part_path
from .wave2 import Data2

BANDS = (0.03, 0.05, 0.10)
EXITS = ((0.05, 0.15), (0.10, 0.30), (0.10, 0.50))
WINDOW = 24
MAX_HOLD = 24 * 30
COST = 12e-4
SLIP = 0.0025
TOPS = (0.005, 0.01, 0.02)
CONFIGS = [(b, s, tr) for b in BANDS for s, tr in EXITS]
MAX_OPEN = 10


def cfg_name(b: float, s: float, tr: float) -> str:
    return f"b{int(b * 100)}_s{int(s * 100)}_t{int(tr * 100)}"


@njit(cache=True)
def bracket(o, h, lo, c, fund, starts, band, stop, trail, window, max_hold, cost, slip):
    """Для каждого среза: результат сделки (доля, NaN — вилка не сработала), сторона (+1/−1/0), бар выхода
    (или конец окна вилки) и лучший ход в пользу позиции (доля от входа)."""
    n, m = len(c), len(starts)
    res = np.full(m, np.nan)
    side_out = np.zeros(m, np.int8)
    end_out = np.zeros(m, np.int64)
    mfe = np.full(m, np.nan)
    for k in range(m):
        i = starts[k]
        ref = c[i]
        bu, sd = ref * (1 + band), ref * (1 - band)
        j = i + 1
        last_wait = min(i + window, n - 1)
        side = 0
        entry = 0.0
        while j <= last_wait:
            up, dn = h[j] >= bu, lo[j] <= sd
            if up and dn:
                side = 2                                   # оба уровня в одном часе — худший исход
                break
            if up:
                side, entry = 1, max(bu, o[j])
                break
            if dn:
                side, entry = -1, min(sd, o[j])
                break
            j += 1
        if side == 0:
            end_out[k] = last_wait
            continue
        if side == 2:
            res[k] = -stop - cost - 2 * slip
            end_out[k] = j
            mfe[k] = 0.0
            continue
        entry = entry * (1 + side * slip)
        sp = entry * (1 - side * stop)
        if (side > 0 and lo[j] <= sp) or (side < 0 and h[j] >= sp):          # выбило в часе входа
            res[k] = side * (sp / entry - 1) - cost - slip - side * fund[j]
            side_out[k], end_out[k], mfe[k] = side, j, 0.0
            continue
        ext = h[j] if side > 0 else lo[j]
        if side > 0:
            sp = max(sp, ext * (1 - trail))
        else:
            sp = min(sp, ext * (1 + trail))
        paid = fund[j]
        ex = np.nan
        q = j + 1
        last = min(j + max_hold, n - 1)
        while q <= last:
            paid += fund[q]
            if side > 0:
                if o[q] <= sp:
                    ex = o[q]
                elif lo[q] <= sp:
                    ex = sp
                else:
                    ext = max(ext, h[q])
                    sp = max(sp, ext * (1 - trail))
            else:
                if o[q] >= sp:
                    ex = o[q]
                elif h[q] >= sp:
                    ex = sp
                else:
                    ext = min(ext, lo[q])
                    sp = min(sp, ext * (1 + trail))
            if not np.isnan(ex):
                break
            q += 1
        if np.isnan(ex):
            q = last
            ex = c[last]                                   # таймаут или конец истории (делистинг)
        res[k] = side * (ex / entry - 1) - cost - slip - side * paid
        side_out[k], end_out[k] = side, q
        mfe[k] = side * (ext / entry - 1)
    return res, side_out, end_out, mfe


def collect(root: Path, syms: list[str]) -> None:
    data = Data2(root, ["BTCUSDT", *syms])
    btc = data.get("BTCUSDT", "1h", "full")["close"]
    parts = []
    for k, s in enumerate(mine(syms)):
        try:
            h = data.get(s, "1h", "full")
            if len(h) < 24 * 3:
                continue
            rows = coin_rows(h, adv30(root, s), btc, s, False)
            if not len(rows):
                continue
            idx = h.index.get_indexer(pd.to_datetime(rows["t"], utc=True))
            arr = {c_: h[c_].to_numpy(dtype="float64") for c_ in ("open", "high", "low", "close")}
            f = h["funding"].fillna(0.0).to_numpy(dtype="float64")
            dn = h["low"][::-1].rolling(24, min_periods=24).min()[::-1].shift(-1)
            rows["dn24"] = (dn.to_numpy()[idx] / arr["close"][idx] - 1).astype("float32")
            for b, s_, tr in CONFIGS:
                r, sd, end, mf = bracket(arr["open"], arr["high"], arr["low"], arr["close"], f, idx.astype(np.int64),
                                         b, s_, tr, WINDOW, MAX_HOLD, COST, SLIP)
                nm = cfg_name(b, s_, tr)
                rows[f"r_{nm}"] = r.astype("float32")
                rows[f"side_{nm}"] = sd
                rows[f"end_{nm}"] = h.index[end].to_numpy()
                rows[f"mfe_{nm}"] = mf.astype("float32")
            parts.append(rows.drop(columns=[c_ for c_ in rows.columns if c_.startswith("tr_")]))
        except Exception as e:
            print(f"  {s}: пропуск ({e})", flush=True)
        finally:
            data.cache.pop((s, "1h", "m"), None)
            data.cache.pop((s, "1h"), None)
    print(f"  монет со срезами: {len(parts)}", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path("moonbracket"), index=False)


# ---------- отчёт ----------

def dedupe(sel: pd.DataFrame, nm: str) -> pd.DataFrame:
    """Одна вилка на монету: следующий срез — только после конца предыдущей (выход или истечение окна)."""
    sel = sel.sort_values(["symbol", "t"])
    keep = np.zeros(len(sel), bool)
    t, end, sym = sel["t"].to_numpy(), sel[f"end_{nm}"].to_numpy(), sel["symbol"].to_numpy()
    busy_until, cur = None, None
    for k in range(len(sel)):
        if sym[k] != cur:
            cur, busy_until = sym[k], None
        if busy_until is None or t[k] > busy_until:
            keep[k] = True
            busy_until = end[k]
    return sel[keep]


def trade_stats(x: pd.DataFrame, nm: str) -> dict:
    r = x[f"r_{nm}"].dropna()
    if len(r) < 5:
        return {"вилок": len(x), "сделок": len(r)}
    tt = x.loc[r.index, "t"]
    day = tt.dt.floor("D").to_numpy()
    rv = r.to_numpy(dtype="float64")
    g = pd.Series(rv - rv.mean()).groupby(day).sum()
    t_day = rv.sum() / np.sqrt((g ** 2).sum()) if (g ** 2).sum() > 0 else np.nan
    srt = np.sort(rv)[::-1]
    top = max(1, int(np.ceil(0.01 * len(rv))))
    losing = (r.loc[tt.sort_values().index] <= 0).astype(int).to_numpy()
    streak = max((len(list(gr)) for v, gr in itertools.groupby(losing) if v == 1), default=0)
    sd = x.loc[r.index, f"side_{nm}"]
    return {"вилок": len(x), "сделок": len(rv), "средняя": f"{rv.mean():+.2%}", "медиана": f"{np.median(rv):+.1%}",
            "t_day": round(t_day, 1), "прибыльных": f"{(rv > 0).mean():.0%}",
            "доля 1% лучших": f"{srt[:top].sum() / rv.sum():.0%}" if rv.sum() > 0 else "—",
            "лучшая": f"{srt[0]:+.0%}", ">=+100%": int((rv >= 1).sum()), ">=+300%": int((rv >= 3).sum()),
            ">=+1000%": int((rv >= 10).sum()), "серия убытков": streak,
            "лонг": f"{r[sd == 1].mean():+.1%} n={int((sd == 1).sum())}",
            "шорт": f"{r[sd == -1].mean():+.1%} n={int((sd == -1).sum())}"}


def account(x: pd.DataFrame, nm: str, stop: float, risk: float) -> dict:
    """Риск `risk` капитала на сделку (номинал = риск / стоп), не больше MAX_OPEN позиций; прибыль зачисляется
    на выходе, размер — от капитала на момент входа."""
    tr = x[x[f"r_{nm}"].notna()].sort_values("t")
    eq, open_, curve = 1.0, [], []
    for t, end, r in zip(tr["t"], tr[f"end_{nm}"], tr[f"r_{nm}"]):
        while open_ and open_[0][0] <= t:
            e_, pnl = heapq.heappop(open_)
            eq += pnl
            curve.append((e_, eq))
        if len(open_) >= MAX_OPEN or eq <= 0.05:
            continue
        heapq.heappush(open_, (pd.Timestamp(end), eq * risk / stop * float(r)))
    while open_:
        e_, pnl = heapq.heappop(open_)
        eq += pnl
        curve.append((e_, eq))
    if not curve:
        return {}
    s = pd.Series([v for _, v in curve], index=pd.DatetimeIndex([t for t, _ in curve]))
    months = max(1, (s.index.max() - tr["t"].min()).days / 30.4)
    m = pd.concat([pd.Series([1.0], index=[tr["t"].min()]), s]).resample("ME").last().ffill()
    mret = m.pct_change().dropna()
    return {"итог x": round(eq, 2), "в мес.": f"{max(eq, 1e-9) ** (1 / months) - 1:+.1%}",
            "худший мес.": f"{mret.min():+.0%}" if len(mret) else "—",
            "макс. просадка": f"{(s / s.cummax().clip(lower=1.0) - 1).min():.0%}"}


def report() -> None:
    parts = all_parts("moonbracket")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    for c_ in [c_ for c_ in df.columns if c_.startswith("end_")]:
        df[c_] = pd.to_datetime(df[c_], utc=True)
    hot = (df["full24"] > np.log(1.3)).groupby(df["t"]).sum()
    df["heat"] = df["t"].map(hot.rolling("7D").sum()).astype("float32")
    df["up50"] = (df["up24"] >= 0.5).astype("int8")
    df["up100"] = (df["up24"] >= 1.0).astype("int8")
    print(f"===== MOONBRACKET: срезов {len(df):,}, монет {df['symbol'].nunique()}, частей {len(parts)} =====")
    print(f"вилка: ±b от цены на {WINDOW} ч; стоп s, трейлинг t от лучшей цены, до {MAX_HOLD // 24} дней; "
          f"издержки {COST * 1e4:.0f} б.п. + проскальзывание {SLIP:.2%} x2")

    df["symbol"] = df["symbol"].astype("category")
    is_ = df[(df.t >= PER["is"][0]) & (df.t < PER["is"][1])].sort_values("t")
    folds = np.array_split(np.arange(len(is_)), 4)
    oof = np.full(len(is_), np.nan)
    for k in range(1, 4):
        tr_ = np.concatenate(folds[:k])
        oof[folds[k]] = _fit(is_[FEATS_A].iloc[tr_], is_["up50"].iloc[tr_]).predict(is_[FEATS_A].iloc[folds[k]])
    m = _fit(is_[FEATS_A], is_["up50"])
    df["p"] = m.predict(df[FEATS_A])
    df.loc[df.t < PER["is"][1], "p"] = np.nan
    df.loc[is_.index, "p"] = oof                           # IS — только внефолдовые прогнозы
    thr = {q: np.nanquantile(oof, 1 - q) for q in TOPS}
    vr_thr = is_["vol_ratio"].quantile(0.99)
    selectors = {f"модель, верх {q:.1%}": df["p"] >= thr[q] for q in TOPS}
    selectors["vol_ratio, верх 1%"] = df["vol_ratio"] >= vr_thr
    rng = np.random.default_rng(0)
    selectors["все срезы (5% случайных)"] = pd.Series(rng.random(len(df)) < 0.05, index=df.index)

    for b, s_, tr in CONFIGS:
        nm = cfg_name(b, s_, tr)
        print(f"\n=== вилка ±{b:.0%}, стоп {s_:.0%}, трейлинг {tr:.0%} ===")
        rows = []
        for sname, mask in selectors.items():
            for p, (a, z) in PER.items():
                x = dedupe(df[mask & (df.t >= a) & (df.t < z)], nm)
                rows.append({"отбор": sname, "период": p, **trade_stats(x, nm)})
        print(pd.DataFrame(rows).to_string(index=False))

    # иксы: что мы взяли из настоящих +100% за сутки
    nm_main = cfg_name(0.05, 0.10, 0.30)
    print(f"\n=== Настоящие +100% за сутки среди отобранных срезов (модель, верх 1%, {nm_main}) ===")
    for p, (a, z) in PER.items():
        x = dedupe(df[selectors["модель, верх 1.0%"] & (df.t >= a) & (df.t < z)], nm_main)
        u = x[x["up100"] == 1]
        r = u[f"r_{nm_main}"]
        print(f"  {p}: срезов перед +100% {len(u)}, вилка сработала {int(r.notna().sum())}, из них в лонг "
              f"{int((u[f'side_{nm_main}'] == 1).sum())}; средний результат {r.mean():+.0%}, медиана {r.median():+.0%}; "
              f"лучший ход в нашу сторону (медиана) {u[f'mfe_{nm_main}'].median():+.0%}")

    print("\n=== Счёт: риск 1% / 2% капитала на сделку, до 10 позиций (VAL + HOLDOUT подряд и только HOLDOUT) ===")
    rows = []
    for sname in ("модель, верх 0.5%", "модель, верх 1.0%", "модель, верх 2.0%"):
        for b, s_, tr in CONFIGS:
            nm = cfg_name(b, s_, tr)
            for lab, a in (("VAL+HO", PER["val"][0]), ("HO", PER["ho"][0])):
                x = dedupe(df[selectors[sname] & (df.t >= a) & (df.t < PER["ho"][1])], nm)
                for risk in (0.01, 0.02):
                    rows.append({"отбор": sname, "вилка": nm, "период": lab, "риск": f"{risk:.0%}",
                                 **account(x, nm, s_, risk)})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 340)
    pd.set_option("display.max_columns", 30)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x])
    else:
        report()
