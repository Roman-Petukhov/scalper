"""Сверка линий с ручной разметкой: свежие свечи Binance USDⓈ-M, линии по закрытиям с разными правилами вершин
(зигзаг 3 ATR и «видимые» экстремумы ±N свечей), картинки в out/linecheck. Плюс разметка монет в exchangeInfo
(underlyingType / underlyingSubType) — чтобы отсеять золото, акции и прочее не-крипто.

    python -m tools.linecheck DOTUSDT:15m XAUTUSDT:1h

Вариант «видимых» вершин здесь — черновик для показа; после согласования переносится в research.tline.zz_lines.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from research.smc import _atr  # noqa: E402
from research.tline import ZZ_ANCHOR, ZZ_LIFE, ZZ_SPAN, pivots, zz_lines  # noqa: E402

VIS_TOL = 0.05              # закрытия между точками касания не заходят за линию дальше 0.05 ATR


def vis_lines(d: pd.DataFrame, n: int) -> list[dict]:
    """Как zz_lines, но вершины — «видимые на этом ТФ»: закрытие — экстремум среди n свечей с каждой стороны
    (известно через n свечей), и между двумя точками линии ни одно закрытие не заходит за линию."""
    c = d["close"].to_numpy(dtype="float64")
    atr = _atr(d).to_numpy()
    m = len(c)
    out = []
    for side, piv in ((1, pivots(c, n, True)), (-1, pivots(c, n, False))):
        seen = set()
        for a, conf_a in piv:
            w = c[max(0, a - ZZ_ANCHOR): a]
            if len(w) < ZZ_ANCHOR or side * (c[a] - (w.max() if side > 0 else w.min())) <= 0:
                continue
            cand = piv[(piv[:, 0] >= a + ZZ_SPAN) & (side * (c[a] - c[piv[:, 0]]) > 0)]
            later = piv[piv[:, 0] > a]
            best, b_best, rec, ci, li = None, -1, None, 0, 0
            for t in range(conf_a + 1, min(a + ZZ_LIFE, m - 1)):
                changed = False
                while li < len(later) and later[li][1] <= t - 1:
                    li, changed = li + 1, True
                while ci < len(cand) and cand[ci][1] <= t - 1:
                    ci, changed = ci + 1, True
                if changed:
                    best, b_best = None, -1
                    pts = later[:li, 0]
                    for b in cand[:ci, 0]:
                        sl = (c[b] - c[a]) / (b - a)
                        if not np.all(side * (c[pts] - (c[a] + sl * (pts - a))) <= 1e-12):
                            continue
                        seg = np.arange(a + 1, b)
                        if len(seg) and np.any(side * (c[seg] - (c[a] + sl * (seg - a))) > VIS_TOL * atr[b]):
                            continue
                        if best is None or side * sl > side * best:
                            best, b_best = sl, int(b)
                if best is None or not (side * best < 0):
                    continue
                lt, lp = c[a] + best * (t - a), c[a] + best * (t - 1 - a)
                if side * (c[t] - lt) > 0 and side * (c[t - 1] - lp) <= 0:
                    rec = {"side": side, "a": int(a), "b": b_best, "t": t, "line_t": lt}
                    break
            if rec is None and best is not None and side * best < 0:
                rec = {"side": side, "a": int(a), "b": b_best, "t": -1, "line_t": np.nan}
            if rec is not None and (rec["t"] < 0 or rec["t"] not in seen):
                seen.add(rec["t"])
                out.append(rec)
    return out

BASE = "https://fapi.binance.com"
MODES = {"сейчас: зигзаг 3 ATR": lambda d: zz_lines(d), "видимые вершины ±6 свечей": lambda d: vis_lines(d, 6),
         "видимые вершины ±10 свечей": lambda d: vis_lines(d, 10), "видимые вершины ±16 свечей": lambda d: vis_lines(d, 16)}
SHOW = 300


def klines(client: httpx.Client, sym: str, tf: str) -> pd.DataFrame:
    rows = client.get("/fapi/v1/klines", params={"symbol": sym, "interval": tf, "limit": 1500}).json()
    df = pd.DataFrame(rows, columns=["ot", "open", "high", "low", "close", "volume", "ct", "qv", "n", "taker_buy_volume",
                                     "tbq", "ig"])
    df.index = pd.to_datetime(df["ot"], unit="ms", utc=True)
    return df[["open", "high", "low", "close", "volume", "taker_buy_volume"]].astype("float64").iloc[:-1]


def draw(d: pd.DataFrame, sym: str, tf: str, out: Path) -> None:
    v = d.iloc[-SHOW:]
    off = len(d) - SHOW
    fig, axes = plt.subplots(len(MODES), 1, figsize=(18, 5 * len(MODES)))
    for ax, (name, build) in zip(axes, MODES.items()):
        x = range(len(v))
        for i, (o, h, lo, c) in enumerate(v[["open", "high", "low", "close"]].to_numpy()):
            col = "#26a69a" if c >= o else "#ef5350"
            ax.plot([i, i], [lo, h], color=col, lw=0.6)
            ax.plot([i, i], [min(o, c), max(o, c)], color=col, lw=2.2)
        c = d["close"].to_numpy()
        print(f"\n--- {sym} {tf} {name}")
        for r in build(d):
            end = r["t"] if r["t"] > 0 else len(d) - 1
            if end < off or r["b"] < 0:
                continue
            a, b = r["a"], r["b"]
            sl = (c[b] - c[a]) / (b - a)
            xs = [max(a, off) - off, end + 3 - off]
            ys = [c[a] + sl * (max(a, off) - a), c[a] + sl * (end + 3 - a)]
            ax.plot(xs, ys, color="#1e88e5" if r["side"] < 0 else "#fb8c00", lw=1.4)
            if r["t"] > 0:
                ax.annotate("v" if r["side"] < 0 else "^", (r["t"] - off, c[r["t"]]), color="k", fontsize=14,
                            ha="center")
            print(f"  {'шорт' if r['side'] < 0 else 'лонг'}: A {d.index[a]:%d.%m %H:%M} {c[a]:.6g} · "
                  f"B {d.index[b]:%d.%m %H:%M} {c[b]:.6g} · пробой "
                  + (f"{d.index[r['t']]:%d.%m %H:%M} закрытие {c[r['t']]:.6g} линия {r['line_t']:.6g}" if r["t"] > 0
                     else "нет"))
        ticks = list(range(0, len(v), max(1, len(v) // 12)))
        ax.set_xticks(ticks, [f"{v.index[i]:%d.%m %H:%M}" for i in ticks], fontsize=8)
        ax.set_title(f"{sym} {tf} — {name} (UTC)")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / f"{sym}_{tf}.png", dpi=80)
    plt.close(fig)


def coin_kinds(client: httpx.Client) -> None:
    info = client.get("/fapi/v1/exchangeInfo").json()
    usdt = [s for s in info["symbols"] if s.get("quoteAsset") == "USDT" and s.get("contractType") == "PERPETUAL"
            and s.get("status") == "TRADING"]
    print("\n=== exchangeInfo: underlyingType / underlyingSubType ===")
    print(pd.Series([s.get("underlyingType") for s in usdt]).value_counts().to_string())
    subs = pd.Series([x for s in usdt for x in (s.get("underlyingSubType") or ["-"])]).value_counts()
    print(subs.to_string())
    for s in usdt:
        if s.get("underlyingType") != "COIN" or s["symbol"] in ("XAUTUSDT", "PAXGUSDT") or \
                set(s.get("underlyingSubType") or []) & {"RWA", "TradFi", "Commodity", "Stock", "Gold"}:
            print(f"  {s['symbol']}: {s.get('underlyingType')} {s.get('underlyingSubType')}")
    print("не ASCII:", [s["symbol"] for s in usdt if not s["symbol"].isascii()])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out = Path(os.environ.get("OUT", "../out")) / "linecheck"
    out.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=BASE, timeout=30) as client:
        for item in sys.argv[1:]:
            sym, tf = item.split(":")
            draw(klines(client, sym, tf), sym, tf, out)
        coin_kinds(client)


if __name__ == "__main__":
    main()
