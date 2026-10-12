"""Крупные и мелкие рыночные ордера в свече пробоя 4h (сигналы бота).

Фильтр бота смотрит на долю агрессоров по всему объёму свечи (taker buy / volume). Здесь тот же поток делится
по размеру рыночного ордера: делают ли пробой крупные участники или толпа мелких.

Данные: архив Binance aggTrades (перпетуалы USDT) за свечу пробоя [open, open + 4 ч). Рыночный ордер — подряд
идущие aggTrades с одним временем и одной стороной (крупная заявка, прошедшая несколько уровней, в архиве разбита
по ценам). Размер ордера — относительно оборота монеты (adv30, $ в день, только прошлое):
    small   < 1e-5 · adv   ($200 при $20M в день)
    big     >= 1e-4 · adv  ($2k при $20M, $50k при $500M)
    whale   >= 1e-3 · adv  ($20k при $20M, $500k при $500M)
Признаки (по стороне пробоя: для шорта — продажи): aggr_all, aggr_small, aggr_big, aggr_whale — доля агрессоров
по стороне в объёме группы; share_big, share_whale — доля группы в обороте свечи; div = aggr_big − aggr_small.

События — первые пробои бота 4h (research.rebreak: линии zz 3 ATR, тренд 1D, оборот от $20M, закрытие в своей
половине свечи), агрессоры по свече >= 50%, входы по рынку и ретест, цель 3R. Квинтили — по границам IS.
Правила зафиксированы до запуска (принимаются, только если лучше дополнения в каждом из шести периодов):
    H1 whale с нами     aggr_whale >= 0.5 против < 0.5 (сигналы бота, агрессоры >= 55%)
    H2 big с нами       aggr_big >= 0.55 против < 0.55
    H3 расхождение      div > 0 против <= 0
    H4 спасение         агрессоры 50–55% (бот отсекает) и aggr_big >= 0.55 — не хуже ли сигналов бота

    python -m research.bigflow collect --symbols <монеты 1h-qualified>
    python -m research.bigflow report
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
from .broad import ADV_MIN, adv30
from .oos import PER_ALL
from .rebreak import coin_rows
from .shard import all_parts, mine, part_path
from .smc import _cell
from .tline import tf_frame

TF = "4h"
BAR = pd.Timedelta(hours=4)
SMALL, BIG, WHALE = 1e-5, 1e-4, 1e-3
MIN_AGGR = 0.5
FEATS = ("aggr_all", "aggr_small", "aggr_big", "aggr_whale", "share_big", "share_whale", "div")


def order_features(price: np.ndarray, qty: np.ndarray, ts: np.ndarray, maker: np.ndarray, side: int,
                   adv: float) -> dict[str, float]:
    """Склейка aggTrades в рыночные ордера (одно время + одна сторона подряд) и признаки по размеру ордера."""
    out = {k: np.nan for k in FEATS} | {"orders": 0}
    if len(price) == 0 or not adv > 0:
        return out
    notional = price * qty
    new = np.ones(len(ts), dtype=bool)
    new[1:] = (ts[1:] != ts[:-1]) | (maker[1:] != maker[:-1])
    gid = np.cumsum(new) - 1
    size = np.bincount(gid, weights=notional)
    sell = maker[new]                                    # покупатель — мейкер: агрессор продаёт
    with_us = ~sell if side > 0 else sell
    total = size.sum()

    def aggr(mask: np.ndarray) -> float:
        v = size[mask].sum()
        return float(size[mask & with_us].sum() / v) if v > 0 else np.nan

    out["orders"] = int(len(size))
    out["aggr_all"] = aggr(np.ones(len(size), dtype=bool))
    out["aggr_small"] = aggr(size < SMALL * adv)
    out["aggr_big"] = aggr(size >= BIG * adv)
    out["aggr_whale"] = aggr(size >= WHALE * adv)
    out["share_big"] = float(size[size >= BIG * adv].sum() / total) if total > 0 else np.nan
    out["share_whale"] = float(size[size >= WHALE * adv].sum() / total) if total > 0 else np.nan
    out["div"] = out["aggr_big"] - out["aggr_small"]
    return out


def window_trades(blob: bytes, a_ms: float, b_ms: float) -> tuple[np.ndarray, ...]:
    """Сделки дня в окне [a_ms, b_ms): цена, количество, время (мс), покупатель — мейкер. Читается кусками."""
    keep = []
    with zipfile.ZipFile(io.BytesIO(blob)) as z, z.open(z.namelist()[0]) as fh:
        header = not fh.peek(1)[:1].isdigit()
        for ch in pd.read_csv(fh, header=0 if header else None, usecols=[1, 2, 5, 6], chunksize=1_000_000):
            ch.columns = ["price", "qty", "ts", "maker"]
            ts = pd.to_numeric(ch["ts"], errors="coerce").to_numpy(dtype="float64")
            ts = np.where(ts > 1e14, ts / 1000, ts)                    # микросекунды в новых архивах
            m = (ts >= a_ms) & (ts < b_ms)
            if m.any():
                keep.append((pd.to_numeric(ch["price"], errors="coerce").to_numpy()[m],
                             pd.to_numeric(ch["qty"], errors="coerce").to_numpy()[m], ts[m],
                             ch["maker"].astype(str).str.lower().isin(["true", "1"]).to_numpy()[m]))
    if not keep:
        return tuple(np.array([]) for _ in range(4))
    return tuple(np.concatenate([k[i] for k in keep]) for i in range(4))


def event_features(sym: str, t: pd.Timestamp, side: int, adv: float, cache: Path) -> dict[str, float] | None:
    out = cache / f"{sym}-bigflow-{t:%Y%m%d%H}-{side:+d}.parquet"
    if out.exists():
        return pd.read_parquet(out).iloc[0].to_dict()
    if out.with_suffix(".missing").exists():
        return None
    try:
        blob = D._get(f"{D.DAILY}/aggTrades/{sym}/{sym}-aggTrades-{t:%Y-%m-%d}.zip", strict=True)
    except D.NotFound:
        out.with_suffix(".missing").touch()
        return None
    if blob is None:
        return None
    a_ms = t.value / 1e6
    p, q, ts, mk = window_trades(blob, a_ms, a_ms + BAR.value / 1e6)
    del blob
    f = order_features(p, q, ts, mk.astype(bool), side, adv)
    pd.DataFrame([f]).to_parquet(out, index=False)
    return f


def collect(root: Path, syms: list[str]) -> None:
    D.set_host("cdn")
    cache = root / "cache" / "bigflow"
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, TF)
            if d is None or len(d) < 500:
                continue
            x = coin_rows(d, adv30(root, s).reindex(d.index.floor("D")).to_numpy(), s, TF, ("market", "retest"))
            if len(x):
                x = x[(x.kind == "first") & x.with_trend & (x.adv >= ADV_MIN) & (x.close_loc >= 0.5)
                      & (x.aggr >= MIN_AGGR)]
                rows.append(x)
        except Exception as e:
            print(f"  bigflow {s}: пропуск ({e})", flush=True)
    if not rows:
        return
    df = pd.concat(rows, ignore_index=True)
    ev = df.drop_duplicates(["symbol", "t", "side"])[["symbol", "t", "side", "adv"]]
    print(f"  bigflow: событий {len(ev)}, монет {ev.symbol.nunique()}", flush=True)

    def run(r):
        try:
            return (r.symbol, r.t, r.side), event_features(r.symbol, r.t, int(r.side), float(r.adv), cache)
        except Exception as e:
            print(f"  bigflow {r.symbol} {r.t}: {e}", flush=True)
            return (r.symbol, r.t, r.side), None

    with ThreadPoolExecutor(6) as ex:
        got = dict(ex.map(run, list(ev.itertuples())))
    print(f"  bigflow: со сделками {sum(v is not None for v in got.values())} из {len(got)}", flush=True)
    feats = pd.DataFrame([{"symbol": k[0], "t": k[1], "side": k[2], **v} for k, v in got.items() if v is not None])
    if len(feats):
        df.drop(columns=["first_t", "exit_t", "kind", "fail_lag"]).merge(feats, on=["symbol", "t", "side"]).to_parquet(
            part_path("bigflow"), index=False)


# ---------- отчёт ----------

def _per(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["per"] = ""
    for p, (a, b) in PER_ALL.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    return df


COLS = list(PER_ALL) + ["2026"]


def _row(name: str, g: pd.DataFrame) -> dict:
    g = g.assign(R=g.R3)
    return {"группа": name, **{p: _cell(g[g.per == p]) for p in PER_ALL}, "2026": _cell(g[g.t.dt.year == 2026])}


def _means(g: pd.DataFrame) -> list[float]:
    return [g.loc[g.per == p, "R3"].mean() for p in PER_ALL] + [g.loc[g.t.dt.year == 2026, "R3"].mean()]


def _rule(name: str, z: pd.DataFrame, feat: str, cond: pd.Series) -> None:
    valid = z[feat].notna()
    a, b = z[valid & cond], z[valid & ~cond]
    rows = [_row(f"{name}: да", a), _row(f"{name}: нет", b)]
    print(pd.DataFrame(rows)[["группа"] + COLS].to_string(index=False))
    da, db = _means(a), _means(b)
    diff = [x - y for x, y in zip(da, db)]
    ok = all(np.isfinite(x) and x > 0 for x in diff)
    print("  разница «да − нет»: " + ", ".join(f"{p} {x:+.2f}" for p, x in zip(COLS, diff))
          + f" → {'лучше во всех периодах' if ok else 'не во всех периодах'}")


def report() -> None:
    parts = all_parts("bigflow")
    if not parts:
        print("bigflow: частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = _per(df)
    ev = df.drop_duplicates(["symbol", "t", "side"])
    print(f"\n===== BIGFLOW 4h: крупные и мелкие рыночные ордера в свече пробоя (aggTrades Binance); "
          f"событий {len(ev)}, R на сделку, 3R =====")
    ok = ev[["aggr", "aggr_all"]].dropna()
    print(f"  сверка со свечой: корреляция aggr (свеча) и aggr_all (сделки) {ok.aggr.corr(ok.aggr_all):.3f}, "
          f"медиана |разницы| {(ok.aggr - ok.aggr_all).abs().median():.4f}")
    print("  медианы признаков (сигналы бота): " + ", ".join(
        f"{k} {ev.loc[ev.aggr >= 0.55, k].median():.3f}" for k in FEATS + ("orders",)))
    for entry in ("retest", "market"):
        z = df[(df.entry == entry)]
        bot = z[z.aggr >= 0.55]
        print(f"\n  --- вход: {'ретест' if entry == 'retest' else 'по рынку'} ---")
        print(pd.DataFrame([_row("сигналы бота (агрессоры >= 55%)", bot)])[["группа"] + COLS].to_string(index=False))
        isb = bot[bot.per == "is"]
        for f in FEATS:
            x = isb[f].dropna()
            if len(x) < 50:
                print(f"  {f}: мало данных в IS ({len(x)})")
                continue
            edges = np.unique(np.quantile(x, [0.2, 0.4, 0.6, 0.8]))
            q = np.digitize(bot[f].to_numpy(), edges)
            rows = [_row(f"{f} Q{k + 1}" + (f" (< {edges[k]:.3f})" if k < len(edges) else ""), bot[(q == k) & bot[f].notna()])
                    for k in range(len(edges) + 1)]
            print(pd.DataFrame(rows)[["группа"] + COLS].to_string(index=False))
        print("\n  правила, зафиксированные до запуска:")
        _rule("H1 whale с нами (aggr_whale >= 0.5)", bot, "aggr_whale", bot["aggr_whale"] >= 0.5)
        _rule("H2 big с нами (aggr_big >= 0.55)", bot, "aggr_big", bot["aggr_big"] >= 0.55)
        _rule("H3 крупные сильнее мелких (div > 0)", bot, "div", bot["div"] > 0)
        low = z[(z.aggr >= 0.5) & (z.aggr < 0.55)]
        rows = [_row("сигналы бота", bot), _row("агрессоры 50–55%, все", low),
                _row("H4 агрессоры 50–55% и aggr_big >= 0.55", low[low.aggr_big >= 0.55]),
                _row("   агрессоры 50–55% и aggr_big < 0.55", low[low.aggr_big < 0.55])]
        print(pd.DataFrame(rows)[["группа"] + COLS].to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 400)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    syms = [x for x in a.symbols.split(",") if x]
    {"collect": lambda: collect(Path(a.root).expanduser(), syms), "report": report}[a.mode]()
