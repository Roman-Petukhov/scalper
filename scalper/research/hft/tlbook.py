"""
Стакан Bybit (дельты L2, 500 уровней) и тиковые сделки в момент входа по пробою трендовой (research.tline, 4h):
отличает ли то, что видно в стакане, пробои, дошедшие до целей, от выбитых по стопу.

Момент входа T — закрытие свечи закрепления (снимок стакана за 1 с до T, внутри того же дня файла).
Окно — свеча закрепления [T − 4 ч, T], обрезанное началом суток UTC (файлы стакана — по дням).
d = +1 — пробой вверх (лонг), −1 — вниз (шорт); «путь» — сторона стакана, через которую идёт цена (ask для d = +1).
    flow_d        перевес агрессоров в окне по объёму: (купили − продали) / всего, умноженный на d
    imb{b}_d      перекос лимитных заявок в b б.п. от mid: (bid − ask) / (bid + ask), умноженный на d
                  (плюс — за нами плотнее, чем впереди)
    thin_{b}      заявки на пути в b б.п. / средний объём сделок за час окна (меньше — пустота впереди)
    consumed_path стены на пути, съеденные агрессорами, в час окна (настоящий участник пробивает уровень)
    absorbed_path стены на пути, которые поглотили >= 2 своих объёмов и стоят (айсберг против пробоя)
    pulled_path   стены на пути, снятые у цены (фальшивое сопротивление убрали)
    pulled_back   стены за нами, снятые у цены (фальшивая поддержка пробоя)
    consumed_back стены за нами, съеденные (встречный агрессор пробил опору)
Стены и их исходы — определения research.hft.walls (>= 10 медиан уровня, в 30 б.п., простояла >= 5 с,
одно событие типа / стороны в минуту).
Классы зафиксированы до запуска:
    confirmed  flow_d > 0, imb100_d > 0, absorbed_path == 0
    fake       absorbed_path > 0 или (flow_d < 0 и imb100_d < 0)
    other      остальное
Результат — R сделки из research.tline (тейки 3R / 5R, стоп за свингом).

    python -m research.hft.tlbook --sample <tline_breakouts.parquet> --out research/hft      (выборка и задачи)
    python -m research.hft.tlbook --events research/hft/tlbook_events.csv --chunk 0/20 --out ../out
    python -m research.hft.tlbook --report ../out
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import walls
from .build import BYBIT_TRADES, OB_URL, _get
from .luxbook import _window_sum

BANDS = (50, 100)
WIN_MS = 4 * 3_600_000
PROBE_LAG_MS = 1_000
DAY_MS = 86_400_000
FROM = "2024-01-01"
N_SAMPLE = 700
N_CHUNKS = 20
FEATS = ("flow_d", "imb50_d", "imb100_d", "thin_50", "thin_100", "consumed_path", "absorbed_path", "pulled_path",
         "pulled_back", "consumed_back")


def period_of(day: str) -> str:
    return "is" if day < "2024-07-01" else "val" if day < "2025-07-01" else "ho"


def sample(src: Path, out: Path, n: int = N_SAMPLE, seed: int = 7, focus: str = "", exclude: Path | None = None) -> None:
    """Пробои 4h с FROM: одна сделка на (монета, время, сторона) — линии «чистая» и «по значимым точкам» дают одни
    и те же входы; затем случайные N монето-дней (все входы этих дней). focus="short55" — только шорты с долей
    продавцов >= 55% (проверка гипотезы); exclude — события прошлой выборки: их монето-дни не берутся."""
    df = pd.read_parquet(src)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df[(df.tf == "4h") & (df.t >= FROM)]
    if focus == "short55":
        df = df[(df.side < 0) & (df.aggr >= 0.55)]
    df = df.sort_values("line").drop_duplicates(["symbol", "t", "side"])
    df["t_close"] = df["t"] + pd.Timedelta(hours=4)
    df["probe_ms"] = (df["t_close"] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1) - PROBE_LAG_MS
    df["day"] = pd.to_datetime(df["probe_ms"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    days = df[["symbol", "day"]].drop_duplicates()
    if exclude is not None:
        old_days = pd.read_csv(exclude)[["symbol", "day"]].drop_duplicates()
        days = days.merge(old_days, how="left", indicator=True).query("_merge == 'left_only'").drop(columns="_merge")
    days = days.sample(frac=1.0, random_state=seed).head(n)
    ev = df.merge(days, on=["symbol", "day"])
    cols = ["symbol", "day", "probe_ms", "side", "line", "R", "aggr", "with_trend", "per"]
    ev[cols].sort_values(["symbol", "day", "probe_ms"]).to_csv(out / "tlbook_events.csv", index=False)
    jobs = [{"chunk": f"{k}/{N_CHUNKS}"} for k in range(N_CHUNKS)]
    (out / "tlbook_jobs.json").write_text(json.dumps(jobs, indent=1))
    print(f"пробоев {len(ev)}, монето-дней {len(days)}, шортов {int((ev.side < 0).sum())}")


def features(evd: pd.DataFrame, snaps: pd.DataFrame, wall_ev: pd.DataFrame, trades: pd.DataFrame,
             day: str) -> pd.DataFrame:
    out = evd.merge(snaps.rename(columns={"t_ms": "probe_ms"}), on="probe_ms", how="left")
    t = out["probe_ms"].to_numpy(dtype="float64")
    d = out["side"].to_numpy(dtype="float64")
    day0 = pd.Timestamp(day, tz="UTC").value // 10**6
    win = np.minimum(WIN_MS, t - day0)                                   # окно, обрезанное началом суток
    hours = np.maximum(win, 1.0) / 3_600_000

    tts = trades["timestamp"].to_numpy(dtype="float64") * 1000.0
    o = np.argsort(tts, kind="stable")
    tts = tts[o]
    sz = trades["size"].to_numpy(dtype="float64")[o]
    sgn = np.where(trades["side"].to_numpy()[o] == "Buy", 1.0, -1.0)
    net = np.array([_window_sum(tts, np.cumsum(sz * sgn), t[i:i + 1], int(win[i]))[0] for i in range(len(t))])
    tot = np.array([_window_sum(tts, np.cumsum(sz), t[i:i + 1], int(win[i]))[0] for i in range(len(t))])
    out["flow_d"] = np.where(tot > 0, net / np.where(tot > 0, tot, 1.0), np.nan) * d
    per_hour = tot / hours

    for b in BANDS:
        bid, ask = out[f"bid{b}"].to_numpy(dtype="float64"), out[f"ask{b}"].to_numpy(dtype="float64")
        s = bid + ask
        out[f"imb{b}_d"] = np.where(s > 0, (bid - ask) / np.where(s > 0, s, 1.0), np.nan) * d
        path = np.where(d > 0, ask, bid)
        out[f"thin_{b}"] = np.where(per_hour > 0, path / np.where(per_hour > 0, per_hour, 1.0), np.nan)

    e = walls.clean(wall_ev.assign(symbol=evd["symbol"].iloc[0], day=day)) if len(wall_ev) else wall_ev

    def rate(kind: str, side_of: np.ndarray) -> np.ndarray:
        """Событий kind в час окна на стороне side_of[i] (1 — bid, −1 — ask)."""
        res = np.zeros(len(t))
        if not len(e):
            return res
        for s in (1, -1):
            ts = np.sort(e[(e["kind"] == kind) & (e["side"] == s)]["ts"].to_numpy(dtype="float64"))
            cs = np.cumsum(np.ones(len(ts)))
            c = np.array([_window_sum(ts, cs, t[i:i + 1], int(win[i]))[0] for i in range(len(t))]) if len(ts) else res
            res = np.where(side_of == s, c, res)
        return res / hours

    out["consumed_path"] = rate("consumed", -d)
    out["absorbed_path"] = rate("absorbed", -d)
    out["pulled_path"] = rate("pulled_near", -d)
    out["pulled_back"] = rate("pulled_near", d)
    out["consumed_back"] = rate("consumed", d)
    out["win_h"] = hours
    return out


def run_symbol_day(args: tuple[str, str, pd.DataFrame]) -> pd.DataFrame:
    sym, day, evd = args
    ob = None
    for n in (500, 200):
        ob = walls._fetch_to_file(OB_URL.format(s=sym, d=day, n=n))
        if ob is not None:
            break
    tr = _get(BYBIT_TRADES.format(s=sym, d=day))
    if ob is None or tr is None:
        if ob is not None:
            os.unlink(ob)
        print(f"  {sym} {day}: нет данных Bybit", flush=True)
        return pd.DataFrame()
    trades = pd.read_csv(io.BytesIO(gzip.decompress(tr)), usecols=["timestamp", "side", "size", "price"])
    del tr
    try:
        wall_ev, snaps = walls.replay(ob, trades, np.sort(evd["probe_ms"].to_numpy(dtype="int64")))
    finally:
        os.unlink(ob)
    res = features(evd, snaps, wall_ev, trades, day)
    print(f"  {sym} {day}: пробоев {len(res)}, событий стен {len(wall_ev)}", flush=True)
    return res


def classify(df: pd.DataFrame) -> pd.Series:
    conf = (df["flow_d"] > 0) & (df["imb100_d"] > 0) & (df["absorbed_path"] == 0)
    fake = (df["absorbed_path"] > 0) | ((df["flow_d"] < 0) & (df["imb100_d"] < 0))
    return pd.Series(np.select([fake, conf], ["fake", "confirmed"], "other"), index=df.index)


def _cell(g: pd.DataFrame) -> str:
    r = g["R"].dropna()
    if len(r) < 10:
        return f"n={len(r)}"
    by_day = (r - r.mean()).groupby(g.loc[r.index, "day"]).sum()
    t = r.sum() / np.sqrt((by_day ** 2).sum()) if (by_day ** 2).sum() > 0 else np.nan
    return f"{r.mean():+.2f}R (t {t:.1f}, n {len(r)})"


def hypothesis(df: pd.DataFrame) -> None:
    """Гипотезы, заявленные по первой выборке (шорты 4h, продавцов >= 55%, ρ ≈ −0.33 для thin_100) и проверяемые
    на новых монето-днях: H1 — пустой стакан впереди (thin_100 ниже медианы) лучше; H2 — перекос заявок в 1%
    за нами (imb100_d выше медианы) лучше. Медианы — по проверочной выборке."""
    g = df[(df.side < 0) & (df.aggr >= 0.55)]
    if len(g) < 100:
        return
    print(f"\n  ПРОВЕРКА ГИПОТЕЗ (шорты, агрессоры >= 55%, n {len(g)}):")
    for f, better_low, name in (("thin_100", True, "H1: заявок впереди в 1% мало"), ("imb100_d", False, "H2: за нами плотнее")):
        med = g[f].median()
        lo, hi = g[g[f] <= med], g[g[f] > med]
        good, bad = (lo, hi) if better_low else (hi, lo)
        print(f"    {name}: да {_cell(good)}, нет {_cell(bad)}; по периодам: " +
              "; ".join(f"{p}: да {_cell(good[good.period == p])}, нет {_cell(bad[bad.period == p])}" for p in ("is", "val", "ho")))


def report(df: pd.DataFrame) -> None:
    df = df[df["mid"].notna()].copy()
    df["period"] = df["day"].map(period_of)
    print(f"\n===== Стакан Bybit в момент входа по пробою 4h: пробоев {len(df)}, монет {df.symbol.nunique()}, "
          f"монето-дней {df.groupby(['symbol', 'day']).ngroups} =====")
    print("ячейка: средний R (t по дням, n); квинтили по всей выборке (1 — меньше значение)")
    for name, g in (("все", df), ("шорты", df[df.side < 0]), ("шорты, агрессоры >= 55%", df[(df.side < 0) & (df.aggr >= 0.55)])):
        if len(g) < 50:
            continue
        print(f"\n  {name}: {_cell(g)}")
        rows = []
        for f in FEATS:
            x = g[f]
            rho = g[[f, "R"]].dropna().corr(method="spearman").iloc[0, 1]
            if x.nunique() > 5:
                q = pd.qcut(x.rank(method="first"), 5, labels=False) + 1
                cells = {f"q{k}": _cell(g[q == k]) for k in range(1, 6)}
            else:
                cells = {"=0": _cell(g[x == 0]), ">0": _cell(g[x > 0])}
            rows.append({"признак": f, "ρ": f"{rho:+.3f}", **cells})
        print(pd.DataFrame(rows).fillna("").to_string(index=False))
        cl = classify(g)
        print("  классы (заданы до запуска): " + ", ".join(f"{k}: {_cell(g[cl == k])}" for k in ("confirmed", "other", "fake")))
        print("  по периодам: " + "; ".join(f"{p}: " + ", ".join(f"{k} {_cell(gp[classify(gp) == k])}"
                                                                  for k in ("confirmed", "fake"))
                                           for p, gp in g.groupby("period")))
    hypothesis(df)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 300)
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=None, help="tline_breakouts.parquet: выбрать монето-дни и записать задачи")
    ap.add_argument("--events", default=None)
    ap.add_argument("--chunk", default="0/1")
    ap.add_argument("--report", default=None, help="папка с tlbook_*.parquet")
    ap.add_argument("--out", default=".")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--focus", default="")
    ap.add_argument("--exclude", default=None)
    ap.add_argument("--n", type=int, default=N_SAMPLE)
    a = ap.parse_args()
    walls.BANDS_BPS = BANDS                                  # снимок глубины в 50 / 100 б.п. вместо 10 / 25
    if a.sample:
        sample(Path(a.sample), Path(a.out), a.n, focus=a.focus, exclude=Path(a.exclude) if a.exclude else None)
        sys.exit(0)
    if a.report:
        files = sorted(Path(a.report).rglob("tlbook_*.parquet"))
        if files:
            report(pd.concat([pd.read_parquet(p) for p in files], ignore_index=True))
        sys.exit(0)
    k, n = (int(x) for x in a.chunk.split("/"))
    ev = pd.read_csv(a.events)
    keys = ev[["symbol", "day"]].drop_duplicates().reset_index(drop=True)
    keys = keys[keys.index % n == k]
    jobs = [(s, d, ev[(ev.symbol == s) & (ev.day == d)].reset_index(drop=True)) for s, d in keys.itertuples(index=False)]
    with ProcessPoolExecutor(a.workers) as ex:
        res = [r for r in ex.map(run_symbol_day, jobs) if len(r)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if res:
        pd.concat(res, ignore_index=True).to_parquet(out / f"tlbook_{k}.parquet")
        print(f"часть {k}: пробоев со стаканом {sum(len(r) for r in res)}")
