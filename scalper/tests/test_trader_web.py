from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from trader.config import AppConfig
from trader.domain.models import Mode, SignalStatus, Timeframe
from trader.scheduler import next_close
from trader.web.app import create_app

from test_trader_app import _signal

HX = {"HX-Request": "true"}


class _Market:
    async def universe(self, min_turnover_usd):
        return []

    async def closed_bars(self, symbol, tf):
        raise AssertionError("не должен вызываться")


@pytest.fixture()
def env(tmp_path):
    cfg = AppConfig(panel_password="correct-horse-battery", session_secret="x" * 40, data_dir=tmp_path,
                    panel_url="http://test", telegram_token=None, telegram_chat_id=None, scan_delay_s=0.0,
                    scheduler=False)
    app = create_app(cfg, market=_Market())
    with TestClient(app) as c:
        yield c, tmp_path


def _login(c):
    r = c.post("/login", data={"password": "correct-horse-battery"}, follow_redirects=False)
    assert r.status_code == 303


def test_login_required_and_wrong_password(env):
    c, _ = env
    assert c.get("/", follow_redirects=False).headers["location"] == "/login"
    assert c.post("/login", data={"password": "nope"}).status_code == 401
    r = c.post("/settings/tf/1h", headers=HX, follow_redirects=False)
    assert r.status_code == 401 and r.headers["HX-Redirect"] == "/login"
    _login(c)
    r = c.get("/")
    assert r.status_code == 200 and "Какие сигналы присылать" in r.text and "без преимущества" in r.text


def test_chips_mode_and_csrf_header(env):
    c, tmp = env
    _login(c)
    assert c.post("/settings/tf/1h").status_code == 403                 # без заголовка HTMX — отказ
    r = c.post("/settings/tf/1h", headers=HX)
    assert r.status_code == 200 and 'hx-post="/settings/tf/1h"' in r.text
    r = c.post("/settings/mode/auto", headers=HX)
    assert "Бот входит сам" in r.text
    from trader.infrastructure.sqlite_repo import SqliteStore
    s = SqliteStore(tmp / "trader.db").load()
    assert Timeframe.H1 in s.timeframes and s.mode is Mode.AUTO


def test_params_validation_message(env):
    c, _ = env
    _login(c)
    form = {"risk_pct": "9", "max_positions": "5", "daily_loss_pct": "4", "min_aggr_pct": "55", "target_r": "3",
            "hybrid_range_atr": "2.5"}
    assert "от 0.05% до 5%" in c.post("/settings/params", data=form, headers=HX).text
    form["risk_pct"] = "0.5"
    assert "Сохранено" in c.post("/settings/params", data=form, headers=HX).text


def test_signal_take_skip_and_feed(env):
    c, tmp = env
    _login(c)
    from trader.infrastructure.sqlite_repo import SqliteStore
    st = SqliteStore(tmp / "trader.db")
    a = st.add(_signal())
    b = st.add(_signal(symbol="ETHUSDT"))
    feed = c.get("/feed").text
    assert "SOLUSDT" in feed and "ETHUSDT" in feed and "Вхожу" in feed
    r = c.post(f"/signals/{a.id}/take", headers=HX)
    assert "в работе" in r.text and "Вхожу" not in r.text
    r = c.post(f"/signals/{a.id}/skip", headers=HX)
    assert "уже обработан" in r.text
    c.post(f"/signals/{b.id}/skip", headers=HX)
    assert st.get(b.id).status is SignalStatus.SKIPPED
    assert c.post("/signals/999/take", headers=HX).status_code == 404


def test_chart_route_blocks_traversal(env):
    c, tmp = env
    _login(c)
    (tmp / "charts").mkdir(exist_ok=True)
    (tmp / "charts" / "1_X_4h.png").write_bytes(b"\x89PNG\r\n")
    (tmp / "secret.png").write_bytes(b"x")
    assert c.get("/charts/1_X_4h.png").status_code == 200
    assert c.get("/charts/..%2Fsecret.png").status_code == 404


def test_scan_now_reports_disabled_timeframe(env):
    c, _ = env
    _login(c)
    assert "выключен" in c.post("/scan/1h", headers=HX).text
    assert "монет 0" in c.post("/scan/4h", headers=HX).text


def test_next_close_on_utc_grid():
    t = datetime(2026, 10, 5, 9, 59, 30, tzinfo=timezone.utc)
    assert next_close(t, Timeframe.M15) == datetime(2026, 10, 5, 10, 0, tzinfo=timezone.utc)
    assert next_close(t, Timeframe.H1) == datetime(2026, 10, 5, 10, 0, tzinfo=timezone.utc)
    assert next_close(t, Timeframe.H4) == datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)
    assert next_close(datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc), Timeframe.H4) == datetime(2026, 10, 5, 16, 0, tzinfo=timezone.utc)
