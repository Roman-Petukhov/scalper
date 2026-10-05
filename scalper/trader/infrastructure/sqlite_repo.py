"""SQLite: сигналы и настройки. Одна база на сервер, доступ из одного процесса."""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ..domain.models import (EntryKind, EntryPolicy, Mode, Settings, Side, Signal, SignalStatus, Timeframe,
                             TradePlan)

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL, timeframe TEXT NOT NULL, side INTEGER NOT NULL, bar_time TEXT NOT NULL,
    created_at TEXT NOT NULL, status TEXT NOT NULL, note TEXT NOT NULL DEFAULT '',
    chart_path TEXT, payload TEXT NOT NULL,
    UNIQUE(symbol, timeframe, bar_time, side)
);
CREATE INDEX IF NOT EXISTS signals_created ON signals(created_at DESC);
CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY CHECK (id = 1), payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS push_subscriptions (
    endpoint TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at TEXT NOT NULL
);
"""


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


class SqliteStore:
    """Реализует SignalRepository и SettingsRepository."""

    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript(SCHEMA)
            self.db.commit()

    # ---------- сигналы ----------
    @staticmethod
    def _payload(s: Signal) -> str:
        return json.dumps({"close": s.close, "line_value": s.line_value,
                           "line_points": [[p[0].isoformat(), p[1]] for p in s.line_points],
                           "aggr": s.aggr, "range_atr": s.range_atr, "plan": {**asdict(s.plan),
                                                                               "entry_kind": s.plan.entry_kind.value},
                           "extra": s.extra})

    @staticmethod
    def _row(r: sqlite3.Row) -> Signal:
        p = json.loads(r["payload"])
        plan = p["plan"]
        return Signal(symbol=r["symbol"], timeframe=Timeframe(r["timeframe"]), side=Side(r["side"]),
                      bar_time=_dt(r["bar_time"]), close=p["close"], line_value=p["line_value"],
                      line_points=tuple((_dt(t), v) for t, v in p["line_points"]),  # type: ignore[arg-type]
                      aggr=p["aggr"], range_atr=p["range_atr"],
                      plan=TradePlan(EntryKind(plan["entry_kind"]), plan["entry"], plan["stop"], plan["target"],
                                     plan["valid_bars"]),
                      status=SignalStatus(r["status"]), id=r["id"], created_at=_dt(r["created_at"]),
                      chart_path=r["chart_path"], note=r["note"], extra=p.get("extra", {}))

    def add(self, signal: Signal) -> Signal | None:
        now = datetime.now(timezone.utc)
        with self.lock:
            try:
                cur = self.db.execute(
                    "INSERT INTO signals(symbol, timeframe, side, bar_time, created_at, status, note, payload)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (signal.symbol, signal.timeframe.value, int(signal.side), signal.bar_time.isoformat(),
                     now.isoformat(), signal.status.value, signal.note, self._payload(signal)))
                self.db.commit()
            except sqlite3.IntegrityError:
                return None
            row = self.db.execute("SELECT * FROM signals WHERE id = ?", (cur.lastrowid,)).fetchone()
        return self._row(row)

    def get(self, signal_id: int) -> Signal | None:
        with self.lock:
            row = self.db.execute("SELECT * FROM signals WHERE id = ?", (signal_id,)).fetchone()
        return self._row(row) if row else None

    def recent(self, limit: int = 100, timeframes: set[Timeframe] | None = None) -> list[Signal]:
        q, args = "SELECT * FROM signals", []
        if timeframes is not None:
            if not timeframes:
                return []
            q += f" WHERE timeframe IN ({','.join('?' * len(timeframes))})"
            args = [t.value for t in timeframes]
        q += " ORDER BY created_at DESC, id DESC LIMIT ?"
        with self.lock:
            rows = self.db.execute(q, (*args, limit)).fetchall()
        return [self._row(r) for r in rows]

    def set_status(self, signal_id: int, status: SignalStatus, note: str = "") -> Signal | None:
        with self.lock:
            self.db.execute("UPDATE signals SET status = ?, note = ? WHERE id = ?", (status.value, note, signal_id))
            self.db.commit()
        return self.get(signal_id)

    def set_chart(self, signal_id: int, path: str) -> None:
        with self.lock:
            self.db.execute("UPDATE signals SET chart_path = ? WHERE id = ?", (path, signal_id))
            self.db.commit()

    # ---------- настройки ----------
    def load(self) -> Settings:
        with self.lock:
            row = self.db.execute("SELECT payload FROM settings WHERE id = 1").fetchone()
        if row is None:
            return Settings()
        p = json.loads(row["payload"])
        p["mode"] = Mode(p["mode"])
        p["entry_policy"] = EntryPolicy(p.get("entry_policy", EntryPolicy.RETEST.value))
        p["timeframes"] = frozenset(Timeframe(x) for x in p["timeframes"])
        known = set(Settings.__dataclass_fields__)
        return Settings(**{k: v for k, v in p.items() if k in known})

    def save(self, settings: Settings) -> None:
        p = asdict(settings)
        p["mode"] = settings.mode.value
        p["entry_policy"] = settings.entry_policy.value
        p["timeframes"] = sorted(t.value for t in settings.timeframes)
        with self.lock:
            self.db.execute("INSERT INTO settings(id, payload) VALUES (1, ?) "
                            "ON CONFLICT(id) DO UPDATE SET payload = excluded.payload", (json.dumps(p),))
            self.db.commit()

    # ---------- подписки на push (устройства) ----------
    def add_subscription(self, sub: dict) -> None:
        with self.lock:
            self.db.execute("INSERT INTO push_subscriptions(endpoint, payload, created_at) VALUES (?,?,?) "
                            "ON CONFLICT(endpoint) DO UPDATE SET payload = excluded.payload",
                            (sub["endpoint"], json.dumps(sub), datetime.now(timezone.utc).isoformat()))
            self.db.commit()

    def remove_subscription(self, endpoint: str) -> None:
        with self.lock:
            self.db.execute("DELETE FROM push_subscriptions WHERE endpoint = ?", (endpoint,))
            self.db.commit()

    def subscriptions(self) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT payload FROM push_subscriptions").fetchall()
        return [json.loads(r["payload"]) for r in rows]
