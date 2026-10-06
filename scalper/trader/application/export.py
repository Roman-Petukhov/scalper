"""Выгрузка журнала для разбора: все сигналы (и те, по которым не вошли, — с причиной) и сделки с итогом на бирже,
в одном JSON. Плоские поля, время — ISO UTC, чтобы файл читался и pandas, и глазами."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from enum import Enum

from ..domain.execution import Trade
from ..domain.models import Settings, Signal

EXPORT_VERSION = 1


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


def journal_export(signals: list[Signal], trades: dict[int, Trade], settings: Settings, now: datetime) -> dict:
    """signals — новые первыми; trades — сделка по id сигнала. Сделка лежит в строке своего сигнала."""
    rows = []
    for s in signals:
        row = _signal_row(s)
        t = trades.get(s.id) if s.id is not None else None
        row["trade"] = _trade_row(t) if t is not None else None
        rows.append(row)
    st = asdict(settings)
    st["tf_params"] = {tf.value: _plain(asdict(p)) for tf, p in settings.tf_params.items()}
    return {"version": EXPORT_VERSION, "exported_at": now.isoformat(), "settings": _plain(st),
            "signals": rows, "signals_n": len(rows), "trades_n": sum(r["trade"] is not None for r in rows)}
