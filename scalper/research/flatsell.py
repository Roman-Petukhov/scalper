"""Цена стоит, а продавцы давят — куда потом? Проверка «поглощения в боковике» (накопление по Вайкоффу) на 4h.

События (заданы до запуска), окно N = 30 / 60 свечей (5 / 10 дней), монета с оборотом от $20M:
    боковик     высота окна (max high − min low) ≤ 6 ATR и закрытие сдвинулось за окно не больше чем на 1 ATR
    продавцы    дельта агрессоров за окно (покупки − продажи) / объём ≤ −2% и ниже своей нормы (медиана той же
                величины за 180 прошлых свечей) на 2 п.п. и больше
    покупатели  зеркально: ≥ +2% и выше нормы на 2 п.п. (контроль «распределения»)
    нейтрально  остальные боковики (контроль)
На монете после события следующее — не раньше чем через N свечей.
Факторы в поддержку роста (метки события, заданы до запуска):
    тренд       дневное закрытие выше EMA50
    btc         BTC за окно не упал (≥ 0)
    фандинг     сумма funding за окно < 0 — шорты платят, толпа в шортах (топливо для выноса)
    минимумы    минимум второй половины окна выше минимума первой — продавцы не могут продавить ниже
    балл        число факторов из четырёх; «сильный» — 3 и больше
Исход: какую границу окна закрытие пробьёт первым (до 180 свечей, 30 дней) — доля «вверх»; доходность через 1 / 3 / 7
дней — сырая и сверх BTC за тот же отрезок. Сделок нет — это вероятности и средние ходы.
Вывод в пользу гипотезы: у «продавцов» доля пробоев вверх и доходность сверх BTC выше, чем у «нейтрально», во всех
периодах 2021 / IS / VAL / HO / 2026, где событий ≥ 30. То же для каждого фактора и «сильного»
балла: «продавцы + фактор» против «нейтрально + тот же фактор» — чтобы отделить эффект продавцов от эффекта фактора.

    python -m research.flatsell collect --symbols <1h-qualified>
    python -m research.flatsell report
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .smc import _atr
from .tline import htf_trend, tf_frame

TF = "4h"
WINDOWS = (30, 60)
BOX_ATR, DRIFT_ATR = 6.0, 1.0
DELTA_MIN, DELTA_REL = 0.02, 0.02
NORM_BARS, BREAK_BARS = 180, 180
HORIZONS = {"1д": 6, "3д": 18, "7д": 42}
GROUPS = ("продавцы", "нейтрально", "покупатели")
FACTORS = ("тренд", "btc", "фандинг", "минимумы")


def first_break(c: np.ndarray, t: int, top: float, bot: float, limit: int = BREAK_BARS) -> int:
    """+1 — закрытие выше top раньше, чем ниже bot; −1 — наоборот; 0 — ни то ни другое за limit свечей."""
    for j in range(t + 1, min(len(c), t + 1 + limit)):
        if c[j] > top:
            return 1
        if c[j] < bot:
            return -1
    return 0


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, btc: pd.Series | None) -> list[dict]:
    hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("high", "low", "close"))
    v = d["volume"]
    a = _atr(d).to_numpy()
    flow = 2 * d["taker_buy_volume"] - v
    trend = htf_trend(d, TF)
    fund = d["funding"].fillna(0.0) if "funding" in d else pd.Series(0.0, index=d.index)
    bc = (btc.reindex(d.index, method="ffill").to_numpy(dtype="float64") if btc is not None
          else np.full(len(c), np.nan))
    rows = []
    for n in WINDOWS:
        delta = (flow.rolling(n).sum() / v.rolling(n).sum().replace(0, np.nan))
        norm = delta.shift(n).rolling(NORM_BARS, min_periods=NORM_BARS // 2).median().to_numpy()
        delta = delta.to_numpy()
        top = d["high"].rolling(n).max().to_numpy()
        bot = d["low"].rolling(n).min().to_numpy()
        fsum = fund.rolling(n).sum().to_numpy()
        half_old = d["low"].shift(n // 2).rolling(n - n // 2).min().to_numpy()
        half_new = d["low"].rolling(n // 2).min().to_numpy()
        nxt = 0
        for t in range(n, len(c) - 1):
            if t < nxt or not (a[t] > 0) or not (liq[t] >= ADV_MIN) or not np.isfinite(delta[t]):
                continue
            if top[t] - bot[t] > BOX_ATR * a[t] or abs(c[t] - c[t - n]) > DRIFT_ATR * a[t]:
                continue
            rel = delta[t] - norm[t] if np.isfinite(norm[t]) else np.nan
            if delta[t] <= -DELTA_MIN and rel <= -DELTA_REL:
                g = "продавцы"
            elif delta[t] >= DELTA_MIN and rel >= DELTA_REL:
                g = "покупатели"
            else:
                g = "нейтрально"
            row = {"symbol": sym, "t": d.index[t], "n": n, "group": g, "delta": delta[t], "rel": rel,
                   "brk": first_break(c, t, top[t], bot[t]), "тренд": bool(trend[t] > 0),
                   "btc": bool(np.isfinite(bc[t]) and np.isfinite(bc[t - n]) and bc[t] >= bc[t - n]),
                   "фандинг": bool(fsum[t] < 0), "минимумы": bool(half_new[t] > half_old[t])}
            row["балл"] = sum(row[k] for k in FACTORS)
            for h, k in HORIZONS.items():
                if t + k < len(c):
                    r = c[t + k] / c[t] - 1
                    rb = bc[t + k] / bc[t] - 1 if np.isfinite(bc[t]) and np.isfinite(bc[t + k]) else np.nan
                    row[f"r{h}"], row[f"x{h}"] = r, r - rb
                else:
                    row[f"r{h}"] = row[f"x{h}"] = np.nan
            rows.append(row)
            nxt = t + n
    return rows


def collect(root: Path, syms: list[str]) -> None:
    b = tf_frame(root, "BTCUSDT", TF)
    btc = b["close"] if b is not None else None
    rows = []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, TF)
            if d is None or len(d) < 500:
                continue
            liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
            rows += coin_rows(d, liq, s, btc)
        except Exception as e:
            print(f"  flatsell {s}: пропуск ({e})", flush=True)
    print(f"  flatsell: событий {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("flatsell"), index=False)


def _up(z: pd.DataFrame) -> float:
    """Доля пробоев вверх среди событий, где окно пробито."""
    return (z.brk > 0).sum() / max(1, (z.brk != 0).sum())


def _line(z: pd.DataFrame) -> str:
    if len(z) < 30:
        return f"n={len(z)}"
    b = z.brk
    up = (b > 0).sum() / max(1, (b != 0).sum())
    x = z["x7д"].dropna()
    t = x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else np.nan
    return (f"n={len(z)} вверх первым {up:.0%}; 1д {z['r1д'].mean():+.1%}, 3д {z['r3д'].mean():+.1%}, "
            f"7д {z['r7д'].mean():+.1%} (сверх BTC {x.mean():+.1%}, t {t:.1f})")


def report() -> None:
    parts = all_parts("flatsell")
    print("\n===== FLATSELL: цена в боковике, а продавцы давят — куда пойдёт? 4h, монеты от $20M; доля пробоев окна "
          "вверх первым и доходность через 1 / 3 / 7 дней (сверх BTC — за 7 дней) =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    per = {**PER_ALL, "2026": ("2026-01-01", "2026-10-01")}
    judged = [k for k in per if k != "2020"]
    for n in WINDOWS:
        z0 = df[df.n == n]
        print(f"\n  окно {n} свечей ({n // 6} дней): событий {len(z0)}; доли — " + ", ".join(
            f"{g} {(z0.group == g).mean():.0%}" for g in GROUPS))
        stats = {}
        for g in GROUPS:
            z = z0[z0.group == g]
            print(f"    {g}:")
            for p, (a, b) in per.items():
                m = z[(z.t >= a) & (z.t < b)]
                print(f"      {p:>4}: {_line(m)}")
                br = m.brk
                stats[(g, p)] = (len(m), (br > 0).sum() / max(1, (br != 0).sum()), m["x7д"].mean())
        ok = [stats[("продавцы", p)][1] > stats[("нейтрально", p)][1] and
              stats[("продавцы", p)][2] > stats[("нейтрально", p)][2]
              for p in judged if stats[("продавцы", p)][0] >= 30]
        print(f"    → продавцы в боковике к росту: {'да, во всех периодах' if ok and all(ok) else 'нет'} "
              f"({sum(ok)} из {len(ok)} периодов)")
        z0 = z0.assign(сильный=z0["балл"] >= 3)
        print("    факторы в поддержку роста: продавцы + фактор против нейтрально + фактор")
        for fc in (*FACTORS, "сильный"):
            res = []
            for p in judged:
                a, b = per[p]
                m = (z0.t >= a) & (z0.t < b) & z0[fc]
                s_, n_ = z0[m & (z0.group == "продавцы")], z0[m & (z0.group == "нейтрально")]
                if len(s_) < 30 or len(n_) < 30:
                    continue
                res.append(_up(s_) > _up(n_) and s_["x7д"].mean() > n_["x7д"].mean())
            sel = z0[z0[fc] & (z0.group == "продавцы")]
            print(f"      {fc} (у продавцов {z0[z0.group == 'продавцы'][fc].mean():.0%}): "
                  f"{'лучше во всех' if res and all(res) else 'нет'} ({sum(res)} из {len(res)})")
            for p, (a, b) in per.items():
                print(f"        {p:>4}: {_line(sel[(sel.t >= a) & (sel.t < b)])}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x])
    else:
        report()
