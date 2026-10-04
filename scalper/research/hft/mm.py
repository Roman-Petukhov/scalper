"""
Симулятор маркет-мейкинга по историческому стакану Bybit (дельты L2) и тиковым сделкам — с моделью очереди.

Модель исполнения (консервативная):
- наша лимитка встаёт в КОНЕЦ очереди на своей цене: впереди — весь объём уровня в момент активации;
- она исполняется, когда сделки агрессоров по этой цене «съели» объём впереди (частичные исполнения учитываются),
  или когда цена прошла сквозь уровень (сделка хуже нашей цены — значит, наш уровень выбран целиком);
- отмены чужих заявок впереди нас НЕ продвигают очередь (на деле продвигают — мы занижаем число исполнений);
- задержка: новая заявка/перестановка начинает действовать через latency_ms, старая до этого момента живёт;
- позиция ограничена ±max_units котировок; если позиция висит дольше max_hold_s — закрываем рыночным ордером
  по лучшей цене противоположной стороны (taker);
- котируем только при спреде не уже min_spread_bps;
- комиссии: maker на наши исполнения, taker на принудительные выходы; в конце дня позиция оценивается по mid.

    python -m research.hft.mm --symbols A,B --days 2026-09-17,... --out <dir> [--workers 4]
"""
from __future__ import annotations

import argparse
import gzip
import io
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import orjson
import pandas as pd

from .build import BYBIT_TRADES, OB_URL, _get


@dataclass
class Params:
    q_usd: float = 200.0
    latency_ms: float = 100.0
    max_units: int = 1
    max_hold_s: float = 120.0
    min_spread_bps: float = 4.0
    maker_bps: float = 2.0
    taker_bps: float = 5.5


VARIANTS = {
    "base": Params(),
    "latency300": Params(latency_ms=300.0),
    "hold600": Params(max_hold_s=600.0),
    "q1000": Params(q_usd=1000.0),
    "spread8": Params(min_spread_bps=8.0),
}


class Order:
    __slots__ = ("side", "px", "qty", "ahead", "active_ts", "dead_ts")

    def __init__(self, side, px, qty, active_ts):
        self.side, self.px, self.qty, self.active_ts = side, px, qty, active_ts
        self.ahead = None          # объём впереди в очереди; определяется при активации
        self.dead_ts = np.inf      # момент, когда отмена вступит в силу


def ob_stream(ob_blob: bytes):
    """Сообщения стакана по одному (без загрузки всего дня в память)."""
    with zipfile.ZipFile(io.BytesIO(ob_blob)) as z:
        with z.open(z.namelist()[0]) as fh:
            for line in fh:
                yield orjson.loads(line)


def load_trades(trades: pd.DataFrame):
    """Сделки, отсортированные по времени: ts (мс), сторона агрессора (+1 покупка), размер, цена."""
    t_ts = (trades["timestamp"].to_numpy(dtype="float64") * 1000.0)
    t_side = np.where(trades["side"].to_numpy() == "Buy", 1, -1)
    t_sz = trades["size"].to_numpy(dtype="float64")
    t_px = trades["price"].to_numpy(dtype="float64")
    o = np.argsort(t_ts, kind="stable")
    return t_ts[o], t_side[o], t_sz[o], t_px[o]


def simulate(ob_blob: bytes, tr, p: Params) -> dict:
    t_ts, t_side, t_sz, t_px = tr
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    bb, ba = -np.inf, np.inf
    orders = {1: None, -1: None}      # 1 — наш bid, -1 — наш ask
    pending = {1: None, -1: None}     # заявка, ожидающая активации (после задержки)
    inv = 0.0                         # в базовой валюте
    cash = 0.0
    fees = 0.0
    inv_since = None
    fills = 0
    maker_vol = 0.0
    forced = 0
    unit = None
    i_tr, n_tr = 0, len(t_ts)
    lat = p.latency_ms
    last_mid = np.nan

    def activate(side, now):
        o = pending[side]
        if o is None or o.active_ts > now:
            return
        old = orders[side]
        if old is not None:
            old.dead_ts = min(old.dead_ts, now)
        book = bids if side == 1 else asks
        o.ahead = book.get(o.px, 0.0)
        orders[side] = o
        pending[side] = None

    def on_trade(ts, side_taker, sz, px):
        nonlocal inv, cash, fees, fills, maker_vol, inv_since
        # агрессивная продажа бьёт в наш bid, покупка — в наш ask
        my = orders[1] if side_taker < 0 else orders[-1]
        if my is None or my.qty <= 0 or ts < my.active_ts or ts >= my.dead_ts:
            return
        through = (px < my.px) if my.side == 1 else (px > my.px)
        at = px == my.px
        if not (through or at):
            return
        if through:
            fill = my.qty
        else:
            if sz <= my.ahead:
                my.ahead -= sz
                return
            fill = min(my.qty, sz - my.ahead)
            my.ahead = 0.0
        my.qty -= fill
        notional = fill * my.px
        cash -= my.side * notional
        inv_before = inv
        inv += my.side * fill
        fees += notional * p.maker_bps * 1e-4
        fills += 1
        maker_vol += notional
        if inv_before == 0 or np.sign(inv_before) != np.sign(inv):
            inv_since = ts if inv != 0 else None
        elif inv == 0:
            inv_since = None

    for m in ob_stream(ob_blob):
        ts = m["ts"]
        # сделки до этого момента (включительно) — раньше обновления стакана
        while i_tr < n_tr and t_ts[i_tr] <= ts:
            for s in (1, -1):
                activate(s, t_ts[i_tr])
            on_trade(t_ts[i_tr], t_side[i_tr], t_sz[i_tr], t_px[i_tr])
            i_tr += 1
        for s in (1, -1):
            activate(s, ts)
        d = m["data"]
        if m["type"] == "snapshot":
            bids = {float(a): float(b) for a, b in d["b"]}
            asks = {float(a): float(b) for a, b in d["a"]}
            bb = max(bids) if bids else -np.inf
            ba = min(asks) if asks else np.inf
        else:
            for a, b in d["b"]:
                fp, fq = float(a), float(b)
                if fq == 0.0:
                    if bids.pop(fp, None) is not None and fp == bb:
                        bb = max(bids) if bids else -np.inf
                else:
                    bids[fp] = fq
                    if fp > bb:
                        bb = fp
            for a, b in d["a"]:
                fp, fq = float(a), float(b)
                if fq == 0.0:
                    if asks.pop(fp, None) is not None and fp == ba:
                        ba = min(asks) if asks else np.inf
                else:
                    asks[fp] = fq
                    if fp < ba:
                        ba = fp
        if not (np.isfinite(bb) and np.isfinite(ba)) or ba <= bb:
            continue
        mid = (bb + ba) / 2
        last_mid = mid
        if unit is None:
            unit = p.q_usd / mid
        spread_bps = (ba - bb) / mid * 1e4
        # принудительный выход из залежавшейся позиции (рыночный, по лучшей цене против нас)
        if inv != 0 and inv_since is not None and ts - inv_since > p.max_hold_s * 1000.0:
            px = bb if inv > 0 else ba
            notional = abs(inv) * px
            cash += inv * px
            fees += notional * p.taker_bps * 1e-4
            inv = 0.0
            inv_since = None
            forced += 1
            for s in (1, -1):
                if orders[s] is not None:
                    orders[s].dead_ts = ts
        # котировки: стоим на лучших ценах, если спред достаточен и есть лимит позиции
        ok = spread_bps >= p.min_spread_bps
        for s, best in ((1, bb), (-1, ba)):
            room = (p.max_units * unit - s * inv) if unit else 0.0
            want = ok and room > unit * 0.5
            cur = orders[s] if (orders[s] is not None and orders[s].qty > 0 and orders[s].dead_ts == np.inf) else None
            pend = pending[s]
            if not want:
                if cur is not None:
                    cur.dead_ts = ts + lat
                pending[s] = None
                continue
            target = cur.px if cur is not None else (pend.px if pend is not None else None)
            # переставляем, если лучшая цена ушла от нас в нашу сторону (мы больше не первые по цене)
            needs = target is None or (s == 1 and best > target) or (s == -1 and best < target)
            if needs:
                pending[s] = Order(s, best, min(unit, room), ts + lat)
    # оценка остатка по mid
    pnl = cash + inv * last_mid - fees
    return {"fills": fills, "maker_volume_usd": maker_vol, "forced_exits": forced, "fees_usd": fees,
            "pnl_usd": pnl, "pnl_bps_of_volume": pnl / maker_vol * 1e4 if maker_vol > 0 else np.nan,
            "end_inventory_usd": inv * last_mid}


def run_symbol_day(args) -> list[dict]:
    sym, day = args
    ob = None
    for n in (500, 200):
        ob = _get(OB_URL.format(s=sym, d=day, n=n))
        if ob is not None:
            break
    tr = _get(BYBIT_TRADES.format(s=sym, d=day))
    if ob is None or tr is None:
        return [{"symbol": sym, "day": day, "variant": "n/a", "error": "нет данных"}]
    trades = pd.read_csv(io.BytesIO(gzip.decompress(tr)), usecols=["timestamp", "side", "size", "price"])
    t = load_trades(trades)
    out = []
    for name, p in VARIANTS.items():
        r = simulate(ob, t, p)
        out.append({"symbol": sym, "day": day, "variant": name, **r})
    print(f"  {sym} {day}: base pnl ${out[0]['pnl_usd']:.2f}, исполнений {out[0]['fills']}, "
          f"{out[0]['pnl_bps_of_volume']:.2f} б.п. от объёма", flush=True)
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=None, help="папка с mm_results_*.csv: только сводка")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--days", default="")
    ap.add_argument("--out", default=".")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.report:
        report(pd.concat([pd.read_csv(p) for p in sorted(Path(a.report).rglob("mm_results_*.csv"))]))
        sys.exit(0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, d) for s in a.symbols.split(",") for d in a.days.split(",")]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(run_symbol_day, jobs):
            rows += r
    res = pd.DataFrame(rows)
    res.to_csv(out / f"mm_results_{abs(hash(a.symbols)) % 10**8}.csv", index=False)
    report(res)


def report(res: pd.DataFrame) -> None:
    ok = res[res["variant"] != "n/a"]
    agg = ok.groupby(["variant", "symbol"]).agg(days=("day", "nunique"), pnl_usd=("pnl_usd", "sum"),
                                                 fills=("fills", "sum"), volume=("maker_volume_usd", "sum"),
                                                 forced=("forced_exits", "sum"),
                                                 pos_days=("pnl_usd", lambda x: float((x > 0).mean())))
    agg["bps_of_volume"] = agg["pnl_usd"] / agg["volume"] * 1e4
    agg["pnl_per_day"] = agg["pnl_usd"] / agg["days"]
    print("\nИтог по монетам (сумма за дни):")
    print(agg.round(2).sort_values(["variant", "pnl_usd"], ascending=[True, False]).to_string())
    tot = ok.groupby(["variant", "day"])["pnl_usd"].sum().unstack(0)
    print("\nПо дням (сумма по всем монетам, $):")
    print(tot.round(2).to_string())
    print("\nВсего по вариантам: $" + ", ".join(f"{k}: {v:.2f}" for k, v in ok.groupby("variant")["pnl_usd"].sum().items()))
