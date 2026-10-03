"""
Исследование сигналов без исполнения: есть ли у сигнала предсказательная сила?

Для каждого сигнала меряем форвардное движение цены (в б.п., в сторону сигнала)
на нескольких горизонтах. Если среднее не превышает комиссии круга, никакое
исполнение стратегию не спасёт. Это первый фильтр перед подбором параметров.
"""
from __future__ import annotations

import math
from typing import Iterable

from core.features import Features
from core.models import Trade
from core.strategy import FlowScalper

HORIZONS = (5, 15, 30, 60, 120)


def study(cfg, events: Iterable, horizons=HORIZONS) -> dict:
    feats = Features(cfg, cfg.instrument.tick_size)
    strat = FlowScalper(cfg)
    closes: dict[int, float] = {}
    signals = []
    interval = min(float(cfg.strategy.eval_interval_s), 0.25)
    last = 0.0
    for ev in events:
        feats.on_event(ev)
        if isinstance(ev, Trade):
            closes[int(ev.ts)] = ev.price
        if ev.ts - last < interval:
            continue
        last = ev.ts
        snap = feats.snapshot(ev.ts)
        sig = strat.evaluate(snap)
        if sig is not None:
            strat.notify_entry(sig.ts)        # кулдаун, чтобы сигналы не перекрывались
            signals.append((sig, snap.bid, snap.ask))

    def price_at(sec: int) -> float | None:
        for k in range(sec, sec - 10, -1):
            if k in closes:
                return closes[k]
        return None

    rows = []
    for sig, bid, ask in signals:
        mid = (bid + ask) / 2
        fwd = {}
        for h in horizons:
            p = price_at(int(sig.ts) + h)
            if p is not None:
                fwd[h] = sig.side.value * (p - mid) / mid * 1e4
        rows.append((sig, fwd))

    out = {}
    for setup in sorted({s.setup for s, _ in rows}):
        sub = [f for s, f in rows if s.setup == setup]
        stats = {}
        for h in horizons:
            xs = [f[h] for f in sub if h in f]
            if len(xs) < 2:
                continue
            m = sum(xs) / len(xs)
            sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
            stats[h] = {"n": len(xs), "mean_bps": m, "hit": sum(x > 0 for x in xs) / len(xs),
                        "t": m / (sd / math.sqrt(len(xs))) if sd > 0 else 0.0}
        out[setup] = stats
    return {"signals": len(rows), "by_setup": out, "strategy_stats": strat.stats}


def print_study(res: dict, fees) -> None:
    rt_maker = (fees.maker * 2) * 1e4
    rt_mixed = (fees.maker + fees.taker) * 1e4
    rt_taker = (fees.taker * 2) * 1e4
    print("\n" + "=" * 66)
    print(f"  ИССЛЕДОВАНИЕ СИГНАЛОВ: {res['signals']} сигналов")
    print(f"  Комиссия круга: мейкер/мейкер {rt_maker:.1f} б.п. · мейкер/тейкер {rt_mixed:.1f} · тейкер/тейкер {rt_taker:.1f}")
    print("=" * 66)
    for setup, stats in res["by_setup"].items():
        print(f"\n  {setup}:   горизонт   n     среднее, б.п.   hit-rate   t-stat")
        for h, s in stats.items():
            flag = "  <- t>2" if abs(s["t"]) > 2 else ""
            print(f"  {'':12}{h:>5}s  {s['n']:>5}   {s['mean_bps']:>+10.2f}     {s['hit'] * 100:6.1f}%   {s['t']:+6.2f}{flag}")
    print("\n  Сигнал имеет смысл торговать, только если среднее на горизонте удержания")
    print("  заметно больше комиссии круга, а t-stat устойчиво > 2 на разных периодах.")
    print("=" * 66)
