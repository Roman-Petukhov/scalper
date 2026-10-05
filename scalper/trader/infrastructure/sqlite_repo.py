"""SQLite: сигналы и настройки. Одна база на сервер, доступ из одного процесса."""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ..domain.execution import Trade, TradeStatus
from ..domain.models import (DEFAULT_TF_PARAMS, RETIRED_TIMEFRAMES, SETTINGS_ONCE, STRATEGY_RESETS, TF_FIELDS, EntryKind, EntryPolicy, Mode, Settings,
                             Side, SideFilter, Signal, SignalStatus, TfParams, Timeframe, TradePlan)

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
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER NOT NULL UNIQUE, symbol TEXT NOT NULL, side INTEGER NOT NULL, kind TEXT NOT NULL,
    qty REAL NOT NULL, price REAL NOT NULL, stop REAL NOT NULL, target REAL NOT NULL, order_id TEXT NOT NULL,
    network TEXT NOT NULL, status TEXT NOT NULL, expires_at TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


class SqliteStore:
    """Реализует SignalRepository, SettingsRepository и TradeRepository."""

    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript(SCHEMA)
            self._drop_retired()
            self._reset_strategies(STRATEGY_RESETS)
            self._set_once(SETTINGS_ONCE)
            self.db.commit()

    def _reset_strategies(self, once: dict[str, Timeframe]) -> None:
        """Один раз перевести таймфрейм на новую стратегию в сохранённых настройках: правило — по умолчанию, сигналы
        ищутся, автоторговля выключена. Метка в kv — чтобы дальнейшие правки трейдера не перетирались. Сигналы и сделки
        остаются."""
        for key, tf in once.items():
            if self.db.execute("SELECT 1 FROM kv WHERE key = ?", (key,)).fetchone():
                continue
            row = self.db.execute("SELECT payload FROM settings WHERE id = 1").fetchone()
            if row is not None:
                p = json.loads(row["payload"])
                p["timeframes"] = sorted(set(p.get("timeframes", [])) | {tf.value})
                p["auto_timeframes"] = [x for x in p.get("auto_timeframes", []) if x != tf.value]
                p.setdefault("tf_params", {})[tf.value] = self._tf_payload(DEFAULT_TF_PARAMS[tf])
                self.db.execute("UPDATE settings SET payload = ? WHERE id = 1", (json.dumps(p),))
            self.db.execute("INSERT INTO kv(key, value) VALUES (?, ?)", (key, "done"))

    def _set_once(self, once: dict[str, dict[str, object]]) -> None:
        """Один раз поменять общие поля сохранённых настроек; метка в kv — дальше их меняет только трейдер."""
        for key, fields in once.items():
            if self.db.execute("SELECT 1 FROM kv WHERE key = ?", (key,)).fetchone():
                continue
            row = self.db.execute("SELECT payload FROM settings WHERE id = 1").fetchone()
            if row is not None:
                p = json.loads(row["payload"]) | fields
                self.db.execute("UPDATE settings SET payload = ? WHERE id = 1", (json.dumps(p),))
            self.db.execute("INSERT INTO kv(key, value) VALUES (?, ?)", (key, "done"))

    @staticmethod
    def _tf_payload(v: TfParams) -> dict:
        return {**asdict(v), "entry_policy": v.entry_policy.value, "sides": v.sides.value}

    def _drop_retired(self) -> None:
        """Сигналы убранных таймфреймов (1h) и их записи о сделках: панель их больше не показывает и не ведёт."""
        marks = ",".join("?" * len(RETIRED_TIMEFRAMES))
        ids = [r[0] for r in self.db.execute(f"SELECT id FROM signals WHERE timeframe IN ({marks})", RETIRED_TIMEFRAMES)]
        if ids:
            q = ",".join("?" * len(ids))
            self.db.execute(f"DELETE FROM trades WHERE signal_id IN ({q})", ids)
            self.db.execute(f"DELETE FROM signals WHERE id IN ({q})", ids)

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

    def recent(self, limit: int = 100, timeframes: set[Timeframe] | None = None,
               statuses: set[SignalStatus] | None = None) -> list[Signal]:
        where, args = [], []
        for col, vals in (("timeframe", timeframes), ("status", statuses)):
            if vals is None:
                continue
            if not vals:
                return []
            where.append(f"{col} IN ({','.join('?' * len(vals))})")
            args += [v.value for v in vals]
        q = "SELECT * FROM signals" + (" WHERE " + " AND ".join(where) if where else "")
        q += " ORDER BY created_at DESC, id DESC LIMIT ?"
        with self.lock:
            rows = self.db.execute(q, (*args, limit)).fetchall()
        return [self._row(r) for r in rows]

    def count(self, statuses: set[SignalStatus]) -> int:
        with self.lock:
            return self.db.execute(f"SELECT COUNT(*) FROM signals WHERE status IN ({','.join('?' * len(statuses))})",
                                   [s.value for s in statuses]).fetchone()[0]

    def set_status(self, signal_id: int, status: SignalStatus, note: str = "") -> Signal | None:
        with self.lock:
            self.db.execute("UPDATE signals SET status = ?, note = ? WHERE id = ?", (status.value, note, signal_id))
            self.db.commit()
        return self.get(signal_id)

    def delete_signals(self, statuses: set[SignalStatus], signal_id: int | None = None) -> list[str]:
        """Удалить сигналы с данными статусами (все или один); вернуть пути их PNG-графиков."""
        q = f"WHERE status IN ({','.join('?' * len(statuses))})"
        args: list = [s.value for s in statuses]
        if signal_id is not None:
            q += " AND id = ?"
            args.append(signal_id)
        with self.lock:
            rows = self.db.execute(f"SELECT id, chart_path FROM signals {q}", args).fetchall()
            ids = [r["id"] for r in rows]
            if ids:
                marks = ",".join("?" * len(ids))
                self.db.execute(f"DELETE FROM trades WHERE signal_id IN ({marks})", ids)
                self.db.execute(f"DELETE FROM signals WHERE id IN ({marks})", ids)
                self.db.commit()
        return [r["chart_path"] for r in rows if r["chart_path"]]

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
        known = set(Settings.__dataclass_fields__) - {"tf_params"}
        out = {k: v for k, v in p.items() if k in known}
        out["mode"] = Mode(p["mode"])
        tfs = {t.value for t in Timeframe}                        # убранные ТФ (1h) из старых настроек отбрасываем
        out["timeframes"] = frozenset(Timeframe(x) for x in p["timeframes"] if x in tfs)
        if "auto_timeframes" in p:
            out["auto_timeframes"] = frozenset(Timeframe(x) for x in p["auto_timeframes"] if x in tfs)
        out["tf_params"] = {t: self._tf(p, t) for t in Timeframe}
        return Settings(**out)

    @staticmethod
    def _tf(p: dict, tf: Timeframe) -> TfParams:
        """Параметры ТФ из базы; старый формат (одно правило на все ТФ, риск и фильтр свечи по ТФ) переносится."""
        if "tf_params" in p and tf.value in p["tf_params"]:
            # поля, которых не было в сохранённой версии, — из значений по умолчанию этого ТФ
            d = {**{k: getattr(DEFAULT_TF_PARAMS[tf], k) for k in TF_FIELDS}, **p["tf_params"][tf.value]}
        else:
            sfx = {Timeframe.H4: "", Timeframe.M15: "_15m"}[tf]
            d = {k: p[k] for k in ("min_aggr", "target_r", "min_break_atr", "hybrid_range_atr", "retest_bars",
                                   "entry_policy") if k in p}
            for k in ("risk_pct", "min_close_loc"):
                if k + sfx in p:
                    d[k] = p[k + sfx]
            d = {**{k: getattr(DEFAULT_TF_PARAMS[tf], k) for k in TF_FIELDS}, **d}
        d["entry_policy"] = EntryPolicy(d.get("entry_policy", EntryPolicy.RETEST.value))
        d["sides"] = SideFilter(d.get("sides", SideFilter.BOTH.value))
        return TfParams(**{k: v for k, v in d.items() if k in TF_FIELDS})

    def save(self, settings: Settings) -> None:
        p = {k: getattr(settings, k) for k in Settings.__dataclass_fields__ if k != "tf_params"}
        p["mode"] = settings.mode.value
        p["timeframes"] = sorted(t.value for t in settings.timeframes)
        p["auto_timeframes"] = sorted(t.value for t in settings.auto_timeframes)
        p["tf_params"] = {t.value: self._tf_payload(v) for t, v in settings.tf_params.items()}
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

    # ---------- сделки на бирже ----------
    @staticmethod
    def _trade(r: sqlite3.Row) -> Trade:
        return Trade(signal_id=r["signal_id"], symbol=r["symbol"], side=Side(r["side"]), kind=EntryKind(r["kind"]),
                     qty=r["qty"], price=r["price"], stop=r["stop"], target=r["target"], order_id=r["order_id"],
                     network=r["network"], status=TradeStatus(r["status"]),
                     expires_at=_dt(r["expires_at"]) if r["expires_at"] else None,
                     created_at=_dt(r["created_at"]), id=r["id"])

    def add_trade(self, trade: Trade) -> Trade:
        created = trade.created_at or datetime.now(timezone.utc)
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO trades(signal_id, symbol, side, kind, qty, price, stop, target, order_id, network, status,"
                " expires_at, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (trade.signal_id, trade.symbol, int(trade.side), trade.kind.value, trade.qty, trade.price, trade.stop,
                 trade.target, trade.order_id, trade.network, trade.status.value,
                 trade.expires_at.isoformat() if trade.expires_at else None, created.isoformat()))
            self.db.commit()
            row = self.db.execute("SELECT * FROM trades WHERE id = ?", (cur.lastrowid,)).fetchone()
        return self._trade(row)

    def trades_for(self, signal_ids: list[int]) -> dict[int, Trade]:
        if not signal_ids:
            return {}
        with self.lock:
            rows = self.db.execute(f"SELECT * FROM trades WHERE signal_id IN ({','.join('?' * len(signal_ids))})",
                                   signal_ids).fetchall()
        return {r["signal_id"]: self._trade(r) for r in rows}

    def pending_trades(self) -> list[Trade]:
        with self.lock:
            rows = self.db.execute("SELECT * FROM trades WHERE status = ?", (TradeStatus.PLACED.value,)).fetchall()
        return [self._trade(r) for r in rows]

    def filled_trades(self) -> list[Trade]:
        with self.lock:
            rows = self.db.execute("SELECT * FROM trades WHERE status = ? ORDER BY id DESC",
                                   (TradeStatus.FILLED.value,)).fetchall()
        return [self._trade(r) for r in rows]

    def set_trade_status(self, trade_id: int, status: TradeStatus) -> None:
        with self.lock:
            self.db.execute("UPDATE trades SET status = ? WHERE id = ?", (status.value, trade_id))
            self.db.commit()

    # ---------- ключ-значение: ключи биржи, капитал на начало дня ----------
    def kv_get(self, key: str) -> str | None:
        with self.lock:
            row = self.db.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def kv_set(self, key: str, value: str) -> None:
        with self.lock:
            self.db.execute("INSERT INTO kv(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                            (key, value))
            self.db.commit()

    def kv_delete(self, key: str) -> None:
        with self.lock:
            self.db.execute("DELETE FROM kv WHERE key = ?", (key,))
            self.db.commit()
