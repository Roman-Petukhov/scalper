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

    async def live_bars(self, symbol, tf, limit=300):
        idx = pd.date_range("2026-09-30", periods=12, freq="4h", tz="UTC")
        return pd.DataFrame({"open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 1.0,
                             "taker_buy_volume": 0.5}, index=idx)


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
    assert r.status_code == 200 and "Какие сигналы присылать" in r.text and "без преимущества" not in r.text


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


def test_feed_tabs_show_one_timeframe_and_chips_hide_it(env):
    c, tmp = env
    _login(c)
    from trader.infrastructure.sqlite_repo import SqliteStore
    st = SqliteStore(tmp / "trader.db")
    st.add(_signal())                                                    # 4h
    st.add(_signal(symbol="ETHUSDT", tf=Timeframe.H1))
    c.post("/settings/tf/1h", headers=HX)                                # по умолчанию включён только 4h
    feed = c.get("/feed?view=1h").text
    assert "ETHUSDT" in feed and "SOLUSDT" not in feed and 'aria-selected="true"' in feed
    assert "SOLUSDT" not in c.get("/feed").text                         # вкладка запомнилась в сессии
    r = c.post("/settings/tf/1h", headers=HX)                            # выключили 1h — лента обновится
    assert r.headers["HX-Trigger"] == "feed-refresh"
    assert "1h выключен" in c.get("/feed").text
    feed = c.get("/feed?view=all").text
    assert "SOLUSDT" in feed and "ETHUSDT" not in feed


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
    assert "нет данных" in c.post("/scan/4h", headers=HX).text


def test_next_close_on_utc_grid():
    t = datetime(2026, 10, 5, 9, 59, 30, tzinfo=timezone.utc)
    assert next_close(t, Timeframe.M15) == datetime(2026, 10, 5, 10, 0, tzinfo=timezone.utc)
    assert next_close(t, Timeframe.H1) == datetime(2026, 10, 5, 10, 0, tzinfo=timezone.utc)
    assert next_close(t, Timeframe.H4) == datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)
    assert next_close(datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc), Timeframe.H4) == datetime(2026, 10, 5, 16, 0, tzinfo=timezone.utc)


def test_installable_app_assets_are_public(env):
    c, _ = env
    m = c.get("/manifest.webmanifest")
    assert m.status_code == 200 and m.headers["content-type"].startswith("application/manifest+json")
    icons = m.json()["icons"]
    assert {i["purpose"] for i in icons} == {"any", "maskable"}
    for i in icons:
        assert c.get(i["src"]).status_code == 200
    assert c.get("/sw.js").status_code == 200 and c.get("/favicon.ico").status_code == 200
    assert 'rel="manifest"' in c.get("/login").text


def test_new_signals_api_for_app_notifications(env):
    c, tmp = env
    assert c.get("/api/signals/new", follow_redirects=False).status_code in (303, 401)
    _login(c)
    from trader.infrastructure.sqlite_repo import SqliteStore
    st = SqliteStore(tmp / "trader.db")
    a = st.add(_signal())
    b = st.add(_signal(symbol="ETHUSDT"))
    h1 = st.add(_signal(symbol="XRPUSDT", tf=Timeframe.H1))           # 1h выключен — не уведомляем
    r = c.get(f"/api/signals/new?after={a.id}").json()
    assert r["last_id"] == h1.id and [x["symbol"] for x in r["signals"]] == ["ETHUSDT"]
    assert r["signals"][0]["side"] == "лонг" and r["signals"][0]["entry_kind"] == "рынок"


def test_login_remember_me(env, monkeypatch):
    from trader.web import app as web_app
    c, _ = env
    r = c.post("/login", data={"password": "correct-horse-battery", "remember": "1"}, follow_redirects=False)
    assert r.status_code == 303 and "Max-Age=31536000" in r.headers["set-cookie"]
    assert 'name="remember"' in c.get("/login").text
    c.post("/logout")
    c.post("/login", data={"password": "correct-horse-battery"}, follow_redirects=False)
    assert c.get("/", follow_redirects=False).status_code == 200
    now = web_app.time.time()
    monkeypatch.setattr(web_app.time, "time", lambda: now + web_app.SHORT_LOGIN_S + 1)
    assert c.get("/", follow_redirects=False).headers["location"] == "/login"


def test_scan_result_shows_time_and_count(env):
    c, _ = env
    _login(c)
    r = c.post("/scan/4h", headers=HX)
    assert r.headers["HX-Trigger"] == "feed-refresh" and "UTC" in r.text and "scan-result" in r.text


def test_chart_name_handles_windows_paths():
    from trader.web.app import chart_name
    assert chart_name(r"C:\Users\R\scalper\data_panel\charts\ETHUSDT_15m_1.png") == "ETHUSDT_15m_1.png"
    assert chart_name("/srv/data/charts/SOLUSDT_4h_2.png") == "SOLUSDT_4h_2.png"


def test_exchange_wallet_take_and_keys(tmp_path):
    from test_trader_execution import FakeBroker
    from trader.domain.execution import Account, Position
    from trader.domain.models import Side
    from trader.infrastructure.sqlite_repo import SqliteStore
    cfg = AppConfig(panel_password="correct-horse-battery", session_secret="x" * 40, data_dir=tmp_path,
                    panel_url="http://test", telegram_token=None, telegram_chat_id=None, scan_delay_s=0.0,
                    scheduler=False)
    b = FakeBroker(acc=Account(1000.0, 900.0, 4.2, (Position("ETHUSDT", Side.SHORT, 0.5, 2500, 2490, 4.2, 2550, 2350),)))
    app = create_app(cfg, market=_Market(), broker=lambda: b)
    with TestClient(app) as c:
        _login(c)
        w = c.get("/wallet").text
        assert "Кошелёк · Bybit демо" in w and "1 000.00" in w and "+4.20" in w and "ETHUSDT" in w and "2550" in w
        st = SqliteStore(tmp_path / "trader.db")
        s = st.add(_signal())
        b.acc = Account(1000.0, 900.0, 0.0)
        feed = c.get("/feed").text
        assert "отправит ордер на Bybit демо" in feed
        r = c.post(f"/signals/{s.id}/take", headers=HX)
        assert "на бирже" in r.text and "демо · рынок" in r.text and r.headers["HX-Trigger"] == "wallet-refresh"
        assert len(b.placed) == 1


def test_wallet_hidden_without_exchange(env):
    c, _ = env
    _login(c)
    r = c.get("/wallet")
    assert r.status_code == 200 and r.text.strip() == ""
    assert "API Key" in c.get("/").text
    r = c.post("/exchange/keys", data={"api_key": "short", "secret": "x", "network": "demo"}, headers=HX)
    assert "неполными" in r.text
    assert c.post("/exchange/keys", data={"api_key": "k" * 20, "secret": "s" * 30}).status_code == 403


def test_live_chart_api(env):
    c, tmp = env
    _login(c)
    from trader.infrastructure.sqlite_repo import SqliteStore
    s = SqliteStore(tmp / "trader.db").add(_signal())
    d = c.get(f"/api/chart/{s.id}").json()
    assert len(d["candles"]) == 12 and d["candles"][0]["time"] == int(pd.Timestamp("2026-09-30", tz="UTC").timestamp())
    assert d["levels"] == {"entry": 101.0, "stop": 98.0, "target": 110.0} and d["side"] == 1
    a, b = d["line"]
    assert a["time"] == d["candles"][0]["time"] and b["time"] == d["candles"][-1]["time"] + 6 * 4 * 3600
    assert b["value"] < a["value"]                                     # линия по точкам 110 → 105 падает
    assert c.get("/api/chart/999").status_code == 404
    assert 'class="live-chart" data-signal="%d"' % s.id in c.get("/feed").text
