"""Выгрузка журнала для разбора: все сигналы (и те, по которым не вошли, — с причиной) и сделки с итогом на бирже,
в одном JSON. Плоские поля, время — ISO UTC, чтобы файл читался и pandas, и глазами.

К каждому сигналу — свечи Binance вокруг него (на них бот и искал сигнал): от начала линии до выхода из сделки,
а если сделки не было — ещё 60 свечей после сигнала, чтобы было видно, что пропустили. Свечи — по столбцам,
время открытия в секундах UTC: так файл в разы меньше, чем со строкой на свечу."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from datetime import datetime, timedelta
from enum import Enum

import pandas as pd

from ..domain.execution import Trade, TradeStatus
from ..domain.models import Settings, Signal
from .ports import MarketData

log = logging.getLogger(__name__)

EXPORT_VERSION = 2
BARS_BEFORE = 20          # свечей до первой точки линии
MAX_LINE_BARS = 300       # линия старше — показываем только последние 300 свечей до сигнала
AFTER_SIGNAL = 60         # сигнал без сделки: что было потом
AFTER_EXIT = 10           # после выхода из сделки
MAX_BARS = 1500           # одним запросом к Binance
BARS_CONCURRENCY = 8
OPEN_TRADE = {TradeStatus.PLACED, TradeStatus.FILLED}


def bars_window(s: Signal, t: Trade | None, now: datetime) -> tuple[datetime, datetime]:
    """Какие свечи нужны для разбора сигнала: начало линии … выход из сделки (или «что было потом»)."""
    step = timedelta(minutes=s.timeframe.minutes)
    start = max(s.line_points[0][0], s.bar_time - MAX_LINE_BARS * step) - BARS_BEFORE * step
    if t is not None and t.result is not None:
        end = max(t.result.closed_at, s.bar_time) + AFTER_EXIT * step
    elif t is not None and t.status in OPEN_TRADE:
        end = now
    else:
        end = s.bar_time + AFTER_SIGNAL * step
    return start, min(end, now, start + (MAX_BARS - 1) * step)


def _columns(df: pd.DataFrame) -> dict:
    return {"t": [int(x.timestamp()) for x in df.index],
            **{k: df[c].tolist() for k, c in (("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"),
                                              ("v", "volume"), ("tbv", "taker_buy_volume")) if c in df}}


async def collect_bars(market: MarketData, signals: list[Signal], trades: dict[int, Trade],
                       now: datetime) -> dict[int, dict | str]:
    """Свечи по каждому сигналу: id → столбцы или текст ошибки (биржа не ответила — остальное выгружаем)."""
    gate = asyncio.Semaphore(BARS_CONCURRENCY)

    async def one(s: Signal) -> tuple[int, dict | str]:
        start, end = bars_window(s, trades.get(s.id), now)
        async with gate:
            try:
                return s.id, _columns(await market.bars_between(s.symbol, s.timeframe, start, end))
            except Exception as e:
                log.warning("выгрузка свечей %s %s: %s", s.symbol, s.timeframe.value, e)
                return s.id, f"{type(e).__name__}: {str(e)[:120]}"

    return dict(await asyncio.gather(*(one(s) for s in signals if s.id is not None)))


def _plain(v: object) -> object:
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, dict):
        return {str(_plain(k)): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_plain(x) for x in v]
    return v


def _signal_row(s: Signal) -> dict:
    p = s.plan
    return {"signal_id": s.id, "symbol": s.symbol, "tf": s.timeframe.value, "side": int(s.side),
            "bar_time": s.bar_time.isoformat(), "created_at": s.created_at.isoformat() if s.created_at else None,
            "status": s.status.value, "note": s.note, "close": s.close, "line_value": s.line_value,
            "line_a_time": s.line_points[0][0].isoformat(), "line_a": s.line_points[0][1],
            "line_b_time": s.line_points[1][0].isoformat(), "line_b": s.line_points[1][1],
            "aggr": s.aggr, "range_atr": s.range_atr, "plan_entry_kind": p.entry_kind.value, "plan_entry": p.entry,
            "plan_stop": p.stop, "plan_target": p.target, "plan_stop_pct": p.risk_pct_of_price,
            **{f"x_{k}": _plain(v) for k, v in s.extra.items()}}


def _trade_row(t: Trade) -> dict:
    r, h = t.result, t.hedge
    return {"trade_id": t.id, "signal_id": t.signal_id, "network": t.network, "kind": t.kind.value,
            "status": t.status.value, "qty": t.qty, "price": t.price, "stop": t.stop, "target": t.target,
            "notional_usd": t.qty * t.price, "risk_usd": t.qty * t.risk_per_unit,
            "created_at": t.created_at.isoformat() if t.created_at else None,
            "filled_at": t.filled_at.isoformat() if t.filled_at else None,
            "expires_at": t.expires_at.isoformat() if t.expires_at else None,
            "entry_fill": r.entry_fill if r else None, "exit": r.exit if r else None,
            "pnl_usd": r.pnl_usd if r else None, "closed_at": r.closed_at.isoformat() if r else None,
            "r": t.r_multiple, "slippage_r": t.slippage_r, "exit_reason": t.exit_reason if r or t.status.value == "timed_out" else None,
            "hedge_beta": h.beta if h else None, "hedge_btc_in": h.btc_in if h else None,
            "hedge_btc_out": h.btc_out if h else None, "hedge_r": t.hedge_r}


def journal_export(signals: list[Signal], trades: dict[int, Trade], settings: Settings, now: datetime,
                   bars: dict[int, dict | str] | None = None) -> dict:
    """signals — новые первыми; trades — сделка по id сигнала. Сделка лежит в строке своего сигнала.
    bars — из collect_bars; None — выгрузка без свечей."""
    rows = []
    for s in signals:
        row = _signal_row(s)
        t = trades.get(s.id) if s.id is not None else None
        row["trade"] = _trade_row(t) if t is not None else None
        if bars is not None:
            b = bars.get(s.id) if s.id is not None else None
            row["bars"] = b if isinstance(b, dict) else None
            row["bars_error"] = b if isinstance(b, str) else None
        rows.append(row)
    st = asdict(settings)
    st["tf_params"] = {tf.value: _plain(asdict(p)) for tf, p in settings.tf_params.items()}
    return {"version": EXPORT_VERSION, "exported_at": now.isoformat(), "settings": _plain(st),
            "signals": rows, "signals_n": len(rows), "trades_n": sum(r["trade"] is not None for r in rows),
            "bars_source": "binance-usdm" if bars is not None else None}
