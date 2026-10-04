"""
Что было в стакане и в крупных сделках перед +50% / +100% за сутки (событийный тест, 2023-01 … 2026-09).

Данные (архив Binance, перпетуалы USDT): bookDepth — снимки суммарного объёма заявок в пределах ±1…5% от цены
(усредняем по часам); aggTrades — все сделки (агрегируем по часам: рыночные покупки и продажи, крупные сделки
от $10k, от $50k и от 0.01% среднего дневного оборота монеты). Решение принимается на закрытии часа T, признаки —
только из окна до T.

Выборки (оборот монеты за 30 прошлых дней >= $2M):
    UP      первый час, после которого максимум за 24 ч выше на 50% (кластер 72 ч), подмножество UP100 — на 100%
    DOWN    первый час, после которого минимум за 24 ч ниже на 35%
    HOT     «горячая» монета по прошлому: объём за 24 ч в верхних 2% её прошлых 90 дней (без взгляда в будущее)
    RAND    случайные часы
Признаки: перекос стакана (bid − ask) / (bid + ask) в пределах 1 / 2 / 5% за 4 ч и 24 ч; «истончение» стакана —
объём заявок за последние 4 ч против T-72…T-24 ч (по каждой стороне); перекос рыночных покупок (CVD) и крупных
сделок за 4 / 24 ч, доля крупных сделок, рост активности (число сделок за 4 ч против предыдущих 20 ч).

Отчёт: медианы признаков по выборкам; AUC «UP против DOWN» по периодам (подсказывает ли признак направление);
в HOT (отбор только по прошлому — это торгуемая выборка): лонг на 24 ч и частоты +30% / −20% по квинтилям признака.

    python -m research.precursors collect --symbols <все перпетуалы>   (по частям)
    python -m research.precursors report
"""
from __future__ import annotations

import argparse
import io
import sys
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .shard import all_parts, mine, part_path, smoke

START, END = pd.Timestamp("2023-01-05", tz="UTC"), pd.Timestamp("2026-09-28", tz="UTC")
PER = {"is": ("2023-01-01", "2024-07-01"), "val": ("2024-07-01", "2025-07-01"), "ho": ("2025-07-01", "2026-10-01")}
ADV_LO = 2e6
UP, UP100, DOWN = 0.5, 1.0, -0.35
CAPS = {"UP": 300, "DOWN": 150, "HOT": 400, "RAND": 200}
BIG = (10_000.0, 50_000.0)
REL_BIG = 1e-4
COST = 12e-4
BOOK_FEATS = ["imb1_4", "imb2_4", "imb5_4", "imb2_24", "ask_thin2", "bid_thin2", "ask_thin5", "bid_thin5"]
FLOW_FEATS = ["cvd4", "cvd24", "big10_net4", "big10_net24", "big50_net24", "bigr_net4", "bigr_net24", "big_share24",
              "act4"]
FEATS = BOOK_FEATS + FLOW_FEATS


# ---------- данные одного дня ----------

def _zip_csv(url: str) -> pd.DataFrame | None:
    try:
        blob = D._get(url, strict=True)
    except D.NotFound:
        return None
    if blob is None:
        return None
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0])
    header = not raw[:1].isdigit()
    return pd.read_csv(io.BytesIO(raw), header=0 if header else None)


def book_day(sym: str, day: str, cache: Path) -> pd.DataFrame | None:
    """Часовые средние суммарного объёма заявок ($) по уровням ±1…5%: колонки p-5 … p5."""
    out = cache / f"{sym}-book-{day}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    if out.with_suffix(".missing").exists():
        return None
    df = _zip_csv(f"{D.DAILY}/bookDepth/{sym}/{sym}-bookDepth-{day}.zip")
    if df is None or df.shape[1] < 4:
        out.with_suffix(".missing").touch()
        return None
    df = df.iloc[:, :4]
    df.columns = ["timestamp", "percentage", "depth", "notional"]
    ts = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
    pct = pd.to_numeric(df["percentage"], errors="coerce")
    g = pd.DataFrame({"h": ts.dt.floor("h"), "p": pct, "n": pd.to_numeric(df["notional"], errors="coerce")}).dropna()
    g = g[g["p"].isin([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])]
    wide = g.pivot_table(index="h", columns="p", values="n", aggfunc="mean")
    wide.columns = [f"p{int(c)}" for c in wide.columns]
    wide.to_parquet(out)
    return wide


def flow_day(sym: str, day: str, rel_big: float, cache: Path) -> pd.DataFrame | None:
    """Часовые суммы: рыночные покупки / продажи ($), число сделок, крупные покупки / продажи по порогам."""
    out = cache / f"{sym}-flow-{day}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    if out.with_suffix(".missing").exists():
        return None
    df = _zip_csv(f"{D.DAILY}/aggTrades/{sym}/{sym}-aggTrades-{day}.zip")
    if df is None or df.shape[1] < 7:
        out.with_suffix(".missing").touch()
        return None
    px = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
    qty = pd.to_numeric(df.iloc[:, 2], errors="coerce").to_numpy()
    ts = pd.to_numeric(df.iloc[:, 5], errors="coerce").to_numpy(dtype="float64")
    ts = np.where(ts > 1e14, ts / 1000, ts)                    # микросекунды в новых архивах
    maker = df.iloc[:, 6].astype(str).str.lower().isin(["true", "1"]).to_numpy()
    notional = px * qty
    hour = pd.to_datetime(ts, unit="ms", utc=True).floor("h")
    buy = np.where(~maker, notional, 0.0)
    sell = np.where(maker, notional, 0.0)
    cols = {"buy": buy, "sell": sell, "cnt": np.ones(len(notional))}
    for lim, nm in ((BIG[0], "10"), (BIG[1], "50"), (rel_big, "r")):
        big = notional >= lim
        cols[f"bb{nm}"] = np.where(big, buy, 0.0)
        cols[f"bs{nm}"] = np.where(big, sell, 0.0)
    agg = pd.DataFrame(cols, index=hour).groupby(level=0).sum()
    agg.to_parquet(out)
    return agg


# ---------- выборки и признаки ----------

def hourly(root: Path, sym: str) -> pd.DataFrame | None:
    p = root / f"{sym}-1h.parquet"
    if not p.exists():
        return None
    k = pd.read_parquet(p, columns=["open_time", "high", "low", "close", "quote_volume"])
    k.index = pd.to_datetime(k["open_time"], unit="ms", utc=True) + pd.Timedelta(hours=1)   # T = закрытие часа
    k = k[~k.index.duplicated()].sort_index()
    d = k["quote_volume"].resample("1D").sum()
    adv = d.rolling(30, min_periods=20).mean().shift(1)
    k["adv"] = adv.reindex(k.index.floor("D")).to_numpy()
    hi_f = k["high"][::-1].rolling(24, min_periods=24).max()[::-1].shift(-1)
    lo_f = k["low"][::-1].rolling(24, min_periods=24).min()[::-1].shift(-1)
    k["up24"] = hi_f / k["close"] - 1
    k["dn24"] = lo_f / k["close"] - 1
    k["ret24"] = k["close"].shift(-24) / k["close"] - 1
    v = k["quote_volume"]
    k["vol_ratio"] = np.log((v.rolling(24).sum() + 1) / (v.rolling(720, min_periods=240).sum() / 30 + 1))
    k["hot_thr"] = k["vol_ratio"].rolling(2160, min_periods=720).quantile(0.98).shift(1)
    return k


def _first_in_clusters(t: pd.DatetimeIndex, gap: pd.Timedelta) -> pd.DatetimeIndex:
    keep, last = [], None
    for x in t:
        if last is None or x - last > gap:
            keep.append(x)
            last = x
    return pd.DatetimeIndex(keep)


def candidates(k: pd.DataFrame, sym: str, rng: np.random.Generator) -> pd.DataFrame:
    ok = (k.index >= START) & (k.index < END) & (k["adv"] >= ADV_LO) & k["up24"].notna()
    e = k[ok]
    rows = []
    for kind, mask, gap in (("UP", e["up24"] >= UP, 72), ("DOWN", e["dn24"] <= DOWN, 72),
                            ("HOT", e["vol_ratio"] >= e["hot_thr"], 24)):
        for t in _first_in_clusters(e.index[mask.to_numpy()], pd.Timedelta(hours=gap)):
            rows.append((sym, t, kind))
    if len(e):
        for i in sorted(rng.choice(len(e), size=min(len(e), 40), replace=False)):
            rows.append((sym, e.index[i], "RAND"))
    return pd.DataFrame(rows, columns=["symbol", "t", "kind"])


def _days(a: pd.Timestamp, b: pd.Timestamp) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(a.floor("D"), (b - pd.Timedelta(seconds=1)).floor("D"))]


def _win(df: pd.DataFrame | None, a: pd.Timestamp, b: pd.Timestamp) -> pd.DataFrame:
    if df is None or not len(df):
        return pd.DataFrame()
    return df[(df.index >= a) & (df.index < b)]


def _imb(w: pd.DataFrame, lv: int) -> float:
    if not len(w) or f"p{lv}" not in w or f"p-{lv}" not in w:
        return np.nan
    b, a = w[f"p-{lv}"].mean(), w[f"p{lv}"].mean()
    return (b - a) / (b + a) if (b + a) > 0 else np.nan


def _thin(w4: pd.DataFrame, base: pd.DataFrame, col: str) -> float:
    if not len(w4) or not len(base) or col not in w4 or col not in base:
        return np.nan
    x, y = w4[col].mean(), base[col].mean()
    return float(np.log(x / y)) if x > 0 and y > 0 else np.nan


def features(book: pd.DataFrame | None, flow: pd.DataFrame | None, t: pd.Timestamp) -> dict:
    h = pd.Timedelta(hours=1)
    # метки часов — начало часа, поэтому окно [t-4h, t) не заглядывает за t
    b4, b24, base = _win(book, t - 4 * h, t), _win(book, t - 24 * h, t), _win(book, t - 72 * h, t - 24 * h)
    f = {"imb1_4": _imb(b4, 1), "imb2_4": _imb(b4, 2), "imb5_4": _imb(b4, 5), "imb2_24": _imb(b24, 2),
         "ask_thin2": _thin(b4, base, "p2"), "bid_thin2": _thin(b4, base, "p-2"),
         "ask_thin5": _thin(b4, base, "p5"), "bid_thin5": _thin(b4, base, "p-5")}
    w4, w24, prev = _win(flow, t - 4 * h, t), _win(flow, t - 24 * h, t), _win(flow, t - 24 * h, t - 4 * h)

    def net(w: pd.DataFrame, a: str, b: str) -> float:
        tot = w["buy"].sum() + w["sell"].sum() if len(w) else 0.0
        return float((w[a].sum() - w[b].sum()) / tot) if tot > 0 else np.nan

    f.update({"cvd4": net(w4, "buy", "sell"), "cvd24": net(w24, "buy", "sell"),
              "big10_net4": net(w4, "bb10", "bs10"), "big10_net24": net(w24, "bb10", "bs10"),
              "big50_net24": net(w24, "bb50", "bs50"), "bigr_net4": net(w4, "bbr", "bsr"),
              "bigr_net24": net(w24, "bbr", "bsr")})
    tot24 = w24["buy"].sum() + w24["sell"].sum() if len(w24) else 0.0
    f["big_share24"] = float((w24["bb10"].sum() + w24["bs10"].sum()) / tot24) if tot24 > 0 else np.nan
    c4, cp = (w4["cnt"].sum() if len(w4) else 0.0), (prev["cnt"].sum() if len(prev) else 0.0)
    f["act4"] = float(np.log((c4 + 1) / (cp / 5 + 1))) if len(w4) and len(prev) else np.nan
    return f


def collect(root: Path, syms: list[str]) -> None:
    D.set_host("cdn")
    cache = root / "cache" / "prec"
    cache.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    hours, cands = {}, []
    for s in mine(syms):
        k = hourly(root, s)
        if k is None:
            continue
        hours[s] = k
        cands.append(candidates(k, s, rng))
    if not cands:
        return
    c = pd.concat(cands, ignore_index=True)
    caps = {k: (5 if smoke() else v) for k, v in CAPS.items()}
    c = pd.concat([g.sample(n=min(len(g), caps[kind]), random_state=1) for kind, g in c.groupby("kind")],
                  ignore_index=True)
    print(f"  событий в части: " + ", ".join(f"{k} {v}" for k, v in c["kind"].value_counts().items()), flush=True)

    jobs = set()
    for r in c.itertuples():
        for d in _days(r.t - pd.Timedelta(hours=72), r.t):
            jobs.add(("book", r.symbol, d))
        for d in _days(r.t - pd.Timedelta(hours=24), r.t):
            jobs.add(("flow", r.symbol, d))

    def run(job):
        kind, s, d = job
        try:
            if kind == "book":
                return job, book_day(s, d, cache)
            adv = hours[s]["adv"].asof(pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=1))
            return job, flow_day(s, d, max(BIG[0], REL_BIG * float(adv) if np.isfinite(adv) else BIG[0]), cache)
        except Exception as e:
            print(f"  {kind} {s} {d}: {e}", flush=True)
            return job, None

    got = {}
    with ThreadPoolExecutor(16) as ex:
        for job, df in ex.map(run, sorted(jobs)):
            got[job] = df
    print(f"  файлов: стакан {sum(1 for j in got if j[0] == 'book' and got[j] is not None)} из "
          f"{sum(1 for j in got if j[0] == 'book')}, сделки {sum(1 for j in got if j[0] == 'flow' and got[j] is not None)} "
          f"из {sum(1 for j in got if j[0] == 'flow')}", flush=True)

    rows = []
    for r in c.itertuples():
        def cat(kind: str, days: list[str]) -> pd.DataFrame | None:
            parts = [got.get((kind, r.symbol, d)) for d in days]
            parts = [p for p in parts if p is not None and len(p)]
            return pd.concat(parts).sort_index() if parts else None
        book = cat("book", _days(r.t - pd.Timedelta(hours=72), r.t))
        flow = cat("flow", _days(r.t - pd.Timedelta(hours=24), r.t))
        k = hours[r.symbol]
        rows.append({"symbol": r.symbol, "t": r.t, "kind": r.kind, "up24": k.at[r.t, "up24"], "dn24": k.at[r.t, "dn24"],
                     "ret24": k.at[r.t, "ret24"], "vol_ratio": k.at[r.t, "vol_ratio"], **features(book, flow, r.t)})
    pd.DataFrame(rows).to_parquet(part_path("precursors"), index=False)


# ---------- отчёт ----------

def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) < 10 or len(neg) < 10:
        return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _period(t: pd.Series) -> pd.Series:
    out = pd.Series("", index=t.index)
    for p, (a, b) in PER.items():
        out[(t >= a) & (t < b)] = p
    return out


def report() -> None:
    parts = all_parts("precursors")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["per"] = _period(df["t"])
    up100 = df[(df.kind == "UP") & (df.up24 >= UP100)].assign(kind="UP100")
    full = pd.concat([df, up100], ignore_index=True)
    print(f"===== PRECURSORS: событий {len(df)}, монет {df.symbol.nunique()}, частей {len(parts)} =====")
    print(pd.crosstab(full["kind"], full["per"]).to_string())
    have = full.groupby("kind")[["imb2_4", "cvd24"]].apply(lambda g: g.notna().mean()).round(2)
    print("доля событий с данными стакана / сделок:\n" + have.to_string())

    print("\n=== 1. Медианы признаков по выборкам (все периоды) ===")
    print(full.groupby("kind")[FEATS].median().T.round(3).to_string())

    print("\n=== 2. Подсказывает ли признак направление: AUC «UP против DOWN» (0.5 — нет; >0.55 или <0.45 — есть) ===")
    rows = []
    for f in FEATS:
        row = {"признак": f}
        for p in [*PER, "все"]:
            x = df if p == "все" else df[df.per == p]
            row[p] = auc(x.loc[x.kind == "UP", f].to_numpy(dtype=float), x.loc[x.kind == "DOWN", f].to_numpy(dtype=float))
        row["UP100 vs HOT"] = auc(up100[f].to_numpy(dtype=float), df.loc[df.kind == "HOT", f].to_numpy(dtype=float))
        row["UP vs RAND"] = auc(df.loc[df.kind == "UP", f].to_numpy(dtype=float),
                                df.loc[df.kind == "RAND", f].to_numpy(dtype=float))
        rows.append(row)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n=== 3. Торгуемая выборка HOT (отбор по прошлому): лонг на 24 ч (с издержками) и частоты по квинтилям ===")
    hot = df[df.kind == "HOT"].copy()
    print(f"HOT: {len(hot)} событий; все: лонг 24 ч {hot['ret24'].mean() - COST:+.2%}, P(+30%) {(hot.up24 >= 0.3).mean():.1%}, "
          f"P(−20%) {(hot.dn24 <= -0.2).mean():.1%}")
    rows = []
    is_hot = hot[hot.per == "is"]
    for f in FEATS:
        x = is_hot[f].dropna()
        if x.nunique() < 5:
            continue
        edges = np.unique(np.quantile(x, [0, 0.2, 0.4, 0.6, 0.8, 1]))
        if len(edges) < 4:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        for p in PER:
            y = hot[hot.per == p]
            q = pd.cut(y[f], edges, labels=False)
            g = y.groupby(q, observed=True)
            rows.append({"признак": f, "период": p,
                         **{f"q{int(i) + 1} лонг": f"{v:+.1%}" for i, v in (g["ret24"].mean() - COST).items()},
                         "q5 P(+30%)": f"{(g['up24'].apply(lambda s: (s >= 0.3).mean())).get(len(edges) - 2, np.nan):.1%}",
                         "q1 P(+30%)": f"{(g['up24'].apply(lambda s: (s >= 0.3).mean())).get(0, np.nan):.1%}",
                         "n": len(y[f].dropna())})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 320)
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
