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


SHELF_ATR = 0.3             # «полка» у экстремума: закрытия в пределах 0.3 ATR от него


def launch_points(c: np.ndarray, atr: np.ndarray, piv: np.ndarray, side: int) -> np.ndarray:
    """Для каждой вершины — свеча, с которой пошло движение: последняя свеча «полки» у экстремума (закрытия в
    пределах SHELF_ATR ATR от него), после которой цена больше не закрывалась ниже (для вершины — выше) вплоть до
    бара подтверждения."""
    out = piv[:, 0].copy()
    for k, (p, conf) in enumerate(piv):
        end = min(conf, len(c) - 1)
        q = p
        while q + 1 <= end and side * (c[p] - c[q + 1]) <= SHELF_ATR * atr[p]:
            q += 1                                          # конец полки
        while q > p and np.any(side * (c[q + 1: end + 1] - c[q]) > 0):
            q -= 1                                          # после неё были закрытия ниже — берём раньше
        out[k] = q
    return out


def vis_lines(d: pd.DataFrame, n: int, launch: bool = False, minor: int = 0, bos: bool = False) -> list[dict]:
    """Как zz_lines, но вершины — «видимые на этом ТФ»: закрытие — экстремум среди n свечей с каждой стороны
    (известно через n свечей), и между двумя точками линии ни одно закрытие не заходит за линию.
    launch=True — точка линии не самое крайнее закрытие, а последнее закрытие «полки» у экстремума перед движением.
    minor=k — вторая точка из мини-экстремумов (±k свечей): линия от крупного экстремума — касательная к закрытиям,
    как трейдер ведёт её по направлению и прижимает к ближайшему мини-экстремуму, чтобы не резать свечи.
    bos=True — опорная вершина считается экстремумом, только когда цена после неё обновила противоположный
    экстремум (закрылась ниже последнего мини-минимума перед вершиной; для впадины — выше мини-максимума);
    линия известна с этого бара."""
    c = d["close"].to_numpy(dtype="float64")
    atr = _atr(d).to_numpy()
    m = len(c)
    out = []
    for side in (1, -1):
        piv = pivots(c, n, side > 0)
        x = launch_points(c, atr, piv, side) if launch else piv[:, 0]     # где точка линии
        seen = set()
        opp = pivots(c, 3, side < 0)[:, 0]                       # мини-экстремумы противоположной стороны
        for ka, (pa, conf_a) in enumerate(piv):
            w = c[max(0, pa - ZZ_ANCHOR): pa]
            if len(w) < ZZ_ANCHOR or side * (c[pa] - (w.max() if side > 0 else w.min())) <= 0:
                continue
            if bos:
                prev = opp[opp < pa]
                if not len(prev):
                    continue
                lvl, conf_bos = c[prev[-1]], -1
                for j in range(pa + 1, min(pa + ZZ_LIFE, m)):
                    if side * (c[j] - c[pa]) > 0:
                        break                                       # вершину обновили раньше — не экстремум
                    if side * (lvl - c[j]) > 0:
                        conf_bos = j
                        break
                if conf_bos < 0:
                    continue
                conf_a = max(conf_a, conf_bos)
            a = int(x[ka])
            pb_all = pivots(c, minor, side > 0) if minor else piv
            xb_all = pb_all[:, 0] if minor else x
            sel = (pb_all[:, 0] >= pa + ZZ_SPAN) & (side * (c[a] - c[xb_all]) > 0)
            cand, cand_x = pb_all[sel], xb_all[sel]
            lsel = pb_all[:, 0] > a
            later, later_x = pb_all[lsel], xb_all[lsel]
            best, b_best, rec, ci, li = None, -1, None, 0, 0
            for t in range(conf_a + 1, min(pa + ZZ_LIFE, m - 1)):
                changed = False
                while li < len(later) and later[li][1] <= t - 1:
                    li, changed = li + 1, True
                while ci < len(cand) and cand[ci][1] <= t - 1:
                    ci, changed = ci + 1, True
                if changed:
                    best, b_best = None, -1
                    pts = later_x[:li]
                    for pb, b in zip(cand[:ci, 0], cand_x[:ci]):
                        sl = (c[b] - c[a]) / (b - a)
                        if not np.all(side * (c[pts] - (c[a] + sl * (pts - a))) <= 1e-12):
                            continue
                        seg = np.arange(a + 1, pb)                 # до экстремума второй точки
                        if len(seg) and np.any(side * (c[seg] - (c[a] + sl * (seg - a))) > VIS_TOL * atr[b]):
                            continue
                        if best is None or side * sl > side * best:
                            best, b_best = sl, int(b)
                if best is None or not (side * best < 0):
                    continue
                lt, lp = c[a] + best * (t - a), c[a] + best * (t - 1 - a)
                if side * (c[t] - lt) > 0 and side * (c[t - 1] - lp) <= 0:
                    rec = {"side": side, "a": a, "b": b_best, "t": t, "line_t": lt}
                    break
            if rec is None and best is not None and side * best < 0:
                rec = {"side": side, "a": a, "b": b_best, "t": -1, "line_t": np.nan}
            if rec is not None and (rec["t"] < 0 or rec["t"] not in seen):
                seen.add(rec["t"])
                out.append(rec)
    return out

BASE = "https://fapi.binance.com"
SPOT = "https://data-api.binance.vision"
MODES = {"сейчас: зигзаг 3 ATR": lambda d: zz_lines(d),
         "видимые вершины ±10 свечей": lambda d: vis_lines(d, 10),
         "A: видимые ±10 + полка + слом структуры, B: мини-экстремум ±3 (касательная)":
             lambda d: vis_lines(d, 10, launch=True, minor=3, bos=True),
         "видимые ±10 + последнее закрытие перед движением": lambda d: vis_lines(d, 10, launch=True)}
SHOW = 300


def klines(client: httpx.Client, sym: str, tf: str) -> pd.DataFrame | None:
    """Свечи фьючерса; с серверов в США fapi закрыт (HTTP 451) — тогда спот с зеркала data-api.binance.vision."""
    params = {"symbol": sym, "interval": tf, "limit": 1500}
    r = client.get("/fapi/v1/klines", params=params)
    src = "фьючерс"
    if r.status_code != 200:
        r = httpx.get(f"{SPOT}/api/v3/klines", params={**params, "limit": 1000}, timeout=30)
        src = "спот (fapi недоступен)"
    if r.status_code != 200 or not isinstance(r.json(), list) or len(r.json()) < 200:
        print(f"\n--- {sym} {tf}: нет данных (HTTP {r.status_code})")
        return None
    print(f"\n=== {sym} {tf}: {src}")
    df = pd.DataFrame(r.json(), columns=["ot", "open", "high", "low", "close", "volume", "ct", "qv", "n",
                                         "taker_buy_volume", "tbq", "ig"])
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
    r = client.get("/fapi/v1/exchangeInfo")
    if r.status_code != 200:
        print(f"\nexchangeInfo фьючерсов недоступен отсюда (HTTP {r.status_code})")
        return
    info = r.json()
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
            d = klines(client, sym, tf)
            if d is not None:
                draw(d, sym, tf, out)
        coin_kinds(client)


if __name__ == "__main__":
    main()
