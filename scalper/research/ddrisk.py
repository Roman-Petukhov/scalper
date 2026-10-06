"""Снижать ли риск в просадке: правило бота 4h (линия, ретест, 3R — research/noline.py) и управление размером
по состоянию собственной кривой результата.

Идея (Howden, Andreev 2026, SSRN 7115459: тренд по крупным монетам, «при просадке > 15% от пика — меньше позиция»)
работает, только если плохие сделки идут сериями. Проверяем на сделках бота:
  base     — риск постоянный (1 единица на сделку), как сейчас;
  dd5/dd8/dd12 — если к моменту входа кривая ниже пика больше чем на 5/8/12 R, риск сделки × 0.5;
  dd8_off  — то же при 8 R, но риск 0 (бот на паузе до восстановления — по бумажному результату пропущенных сделок);
  ma50     — кривая ниже своей средней за 50 сделок → риск × 0.5 (классический фильтр кривой).
Кривая — по закрытым сделкам (результат известен в момент выхода), решение — в момент входа. Пропущенные и
урезанные сделки всё равно двигают «бумажную» кривую базового правила: иначе пауза никогда не кончится.

Мерило — не сумма R (урезанный риск механически её уменьшает), а отношение суммы к максимальной просадке и
перестановочный тест: результаты сделок случайно переставляются по тем же моментам входа/выхода; если правило
выигрывает и на перестановках, дело не в сериях, а в арифметике.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import adv30
from .noline import TF, coin_trades
from .oos import PER_ALL
from .shard import all_parts, mine, part_path
from .tline import tf_frame

BAR = pd.Timedelta(hours=4)
RULES = ("base", "dd5", "dd8", "dd12", "dd8_off", "ma50")
SHUFFLES = 300


def collect(root: Path, syms: list[str]) -> None:
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
            if len(x):
                parts.append(pd.DataFrame({"symbol": s, "t": x["t"].to_numpy(), "R": x["R3"].to_numpy(),
                                           "t_in": d.index[x["fill_i"].to_numpy()] + BAR,
                                           "t_out": d.index[x["exit_i"].to_numpy()] + BAR}))
        except Exception as e:
            print(f"  ddrisk {s}: пропуск ({e})", flush=True)
    print(f"  ddrisk: монет со сделками {len(parts)}", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path("ddrisk"), index=False)


def weights(t_in: np.ndarray, t_out: np.ndarray, r: np.ndarray, rule: str) -> np.ndarray:
    """Множитель риска каждой сделки по состоянию базовой кривой (сумма R закрытых к моменту входа сделок)."""
    if rule == "base":
        return np.ones(len(r))
    order_out = np.argsort(t_out, kind="stable")
    curve = np.cumsum(r[order_out])                         # базовая кривая по моментам выхода
    peak = np.maximum.accumulate(curve)
    ma = pd.Series(curve).rolling(50, min_periods=50).mean().to_numpy()
    k = np.searchsorted(t_out[order_out], t_in, side="right") - 1      # последняя сделка, закрытая до входа
    have = k >= 0
    dd = np.where(have, peak[np.maximum(k, 0)] - curve[np.maximum(k, 0)], 0.0)
    if rule == "ma50":
        below = have & ~np.isnan(ma[np.maximum(k, 0)]) & (curve[np.maximum(k, 0)] < ma[np.maximum(k, 0)])
        return np.where(below, 0.5, 1.0)
    lim = {"dd5": 5.0, "dd8": 8.0, "dd12": 12.0, "dd8_off": 8.0}[rule]
    return np.where(dd > lim, 0.0 if rule.endswith("_off") else 0.5, 1.0)


def stats(t_out: np.ndarray, pnl: np.ndarray) -> tuple[float, float]:
    """Сумма R и максимальная просадка кривой по моментам выхода."""
    if not len(pnl):
        return 0.0, 0.0
    c = np.cumsum(pnl[np.argsort(t_out, kind="stable")])
    return float(c[-1]), float(np.max(np.maximum.accumulate(np.r_[0.0, c])[1:] - c))


def evaluate(df: pd.DataFrame, r: np.ndarray) -> dict[str, dict[str, tuple[float, float, float]]]:
    """rule → period → (сумма R, макс. просадка R, средний множитель риска)."""
    t_in, t_out = df["t_in"].to_numpy(), df["t_out"].to_numpy()
    out = {}
    for rule in RULES:
        w = weights(t_in, t_out, r, rule)
        row = {}
        for p, (a, b) in {**PER_ALL, "всё": ("2000-01-01", "2100-01-01")}.items():
            m = (df["t"] >= a).to_numpy() & (df["t"] < b).to_numpy()
            s, dd = stats(t_out[m], (w * r)[m])
            row[p] = (s, dd, float(w[m].mean()) if m.any() else np.nan)
        out[rule] = row
    return out


def _ratio(s: float, dd: float) -> float:
    return s / dd if dd > 0 else np.nan


def report() -> None:
    parts = all_parts("ddrisk")
    print("\n===== DDRISK: снижать ли риск в просадке — сделки бота 4h (линия, ретест, 3R); сумма R / макс. просадка R "
          "/ их отношение; правило решает по кривой закрытых сделок к моменту входа =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    for c in ("t", "t_in", "t_out"):
        df[c] = pd.to_datetime(df[c], utc=True)
    df = df.sort_values("t_in", kind="stable").reset_index(drop=True)
    r = df["R"].to_numpy(dtype="float64")
    print(f"  сделок {len(df)}, R на сделку {r.mean():+.3f}, с {df.t.min():%Y-%m} по {df.t.max():%Y-%m}")
    res = evaluate(df, r)
    pers = [*PER_ALL, "всё"]
    rows = []
    for rule in RULES:
        rows.append({"правило": rule, **{p: f"{res[rule][p][0]:+.0f} / {res[rule][p][1]:.1f} / "
                                            f"{_ratio(*res[rule][p][:2]):.2f}" for p in pers},
                     "риск ×": f"{res[rule]['всё'][2]:.2f}"})
    print(pd.DataFrame(rows).to_string(index=False))

    # серийность: автокорреляция результатов соседних по выходу сделок и месячных сумм
    ro = r[np.argsort(df["t_out"].to_numpy(), kind="stable")]
    ac1 = np.corrcoef(ro[:-1], ro[1:])[0, 1]
    mon = pd.Series(r, index=df["t_out"]).resample("ME").sum()
    print(f"  автокорреляция: соседние сделки {ac1:+.3f}, месяцы {mon.autocorr(1):+.3f} ({len(mon)} мес.)")

    # перестановки: те же моменты входа/выхода, результаты перемешаны — выигрыш правила без серий
    rng = np.random.default_rng(7)
    base_all = _ratio(*res["base"]["всё"][:2])
    gains = {rule: _ratio(*res[rule]["всё"][:2]) - base_all for rule in RULES[1:]}
    null = {rule: [] for rule in RULES[1:]}
    for _ in range(SHUFFLES):
        rs = rng.permutation(r)
        sh = evaluate(df, rs)
        b = _ratio(*sh["base"]["всё"][:2])
        for rule in RULES[1:]:
            null[rule].append(_ratio(*sh[rule]["всё"][:2]) - b)
    print(f"  перестановки ({SHUFFLES}): прирост отношения сумма/просадка за всё время против случайного порядка")
    for rule in RULES[1:]:
        z = np.asarray(null[rule])
        print(f"    {rule:8s} факт {gains[rule]:+.3f}; случайно: среднее {np.nanmean(z):+.3f}, 95% {np.nanpercentile(z, 95):+.3f}; "
              f"p = {np.mean(z >= gains[rule]):.2f}")


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
