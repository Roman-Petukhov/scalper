from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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

    async def bars_between(self, symbol, tf, start, end):
        if symbol == "ETHUSDT":
            raise RuntimeError("биржа молчит")
        idx = pd.date_range(start, end, freq=f"{tf.minutes}min")
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
    r = c.post("/settings/tf/15m", headers=HX, follow_redirects=False)
    assert r.status_code == 401 and r.headers["HX-Redirect"] == "/login"
    _login(c)
    r = c.get("/")
    assert r.status_code == 200 and "Какие сигналы присылать" in r.text and "без преимущества" not in r.text


def test_chips_mode_and_csrf_header(env):
    c, tmp = env
    _login(c)
    assert c.post("/settings/tf/15m").status_code == 403                 # без заголовка HTMX — отказ
    r = c.post("/settings/tf/15m", headers=HX)
    assert r.status_code == 200 and 'hx-post="/settings/tf/15m"' in r.text
    r = c.post("/settings/mode/auto", headers=HX)
    assert "Бот входит сам" in r.text
    from trader.infrastructure.sqlite_repo import SqliteStore
    s = SqliteStore(tmp / "trader.db").load()
    assert Timeframe.M15 in s.timeframes and s.mode is Mode.AUTO


def _params_form(**over):
    f = {"leverage": "5", "max_positions": "5", "daily_loss_pct": "4"}
    for tf, risk, loc in (("4h", "1", "50"), ("15m", "0.25", "50")):
        f |= {f"{tf}__risk_pct": risk, f"{tf}__entry_policy": "retest", f"{tf}__min_aggr_pct": "55",
              f"{tf}__target_r": "3", f"{tf}__min_close_loc_pct": loc, f"{tf}__min_break_atr": "0",
              f"{tf}__hybrid_range_atr": "2.5", f"{tf}__retest_bars": "12", f"{tf}__sides": "both",
              f"{tf}__max_slope_atr": "0", f"{tf}__top_n": "0", f"{tf}__max_hold_bars": "60",
              f"{tf}__tf_max_positions": "0"}
    return f | over


def test_params_per_timeframe(env):
    c, tmp = env
    _login(c)
    r = c.get("/").text
    assert 'name="15m__entry_policy"' in r and 'name="15m__target_r"' in r and 'name="4h__min_close_loc_pct"' in r
    assert 'name="1h__' not in r
    assert "от 0.05% до 5%" in c.post("/settings/params", data=_params_form(**{"15m__risk_pct": "9"}), headers=HX).text
    assert "плечо — целое от 1× до 10×" in c.post("/settings/params", data=_params_form(leverage="15"), headers=HX).text
    assert 'name="4h__hedge_btc"' in r
    c.post("/settings/params", data=_params_form(**{"4h__hedge_btc": "1"}), headers=HX)
    from trader.infrastructure.sqlite_repo import SqliteStore
    saved = SqliteStore(tmp / "trader.db").load()
    assert saved.p(Timeframe.H4).hedge_btc and not saved.p(Timeframe.M15).hedge_btc and saved.hedging
    form = _params_form(leverage="8", **{"15m__target_r": "2", "15m__entry_policy": "market", "4h__min_close_loc_pct": "70",
                                         "15m__sides": "short", "15m__max_slope_atr": "0.05"})
    r = c.post("/settings/params", data=form, headers=HX).text
    assert "Сохранено" in r and 'name="15m__target_r" type="number" step="0.5" min="1" max="10"' in r
    from trader.infrastructure.sqlite_repo import SqliteStore
    s = SqliteStore(tmp / "trader.db").load()
    assert s.leverage == 8 and s.p(Timeframe.M15).target_r == 2.0 and s.p(Timeframe.H4).target_r == 3.0
    assert s.p(Timeframe.M15).entry_policy.value == "market" and s.p(Timeframe.H4).min_close_loc == 0.7
    assert s.p(Timeframe.M15).sides.value == "short" and s.p(Timeframe.M15).max_slope_atr == 0.05
    assert s.p(Timeframe.H4).sides.value == "both"
    bad = _params_form()
    del bad["4h__target_r"]
    assert "не заполнено поле" in c.post("/settings/params", data=bad, headers=HX).text


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
    now = datetime.now(timezone.utc)
    now = now.replace(minute=now.minute - now.minute % 15, second=0, microsecond=0)
    st.add(replace(_signal(symbol="ETHUSDT", tf=Timeframe.M15), bar_time=now - pd.Timedelta(minutes=15)))  # свежий 15m
    c.post("/settings/tf/15m", headers=HX)                                # по умолчанию включён только 4h
    feed = c.get("/feed?view=15m").text
    assert "ETHUSDT" in feed and "SOLUSDT" not in feed and 'aria-selected="true"' in feed
    assert "SOLUSDT" not in c.get("/feed").text                         # вкладка запомнилась в сессии
    r = c.post("/settings/tf/15m", headers=HX)                            # выключили 15m — лента обновится
    assert r.headers["HX-Trigger"] == "feed-refresh"
    assert "15m выключен" in c.get("/feed").text
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
    assert "выключен" in c.post("/scan/15m", headers=HX).text
    assert "нет данных" in c.post("/scan/4h", headers=HX).text


def test_next_close_on_utc_grid():
    t = datetime(2026, 10, 5, 9, 59, 30, tzinfo=timezone.utc)
    assert next_close(t, Timeframe.M15) == datetime(2026, 10, 5, 10, 0, tzinfo=timezone.utc)
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
    m15 = st.add(_signal(symbol="XRPUSDT", tf=Timeframe.M15))         # 15m выключен — не уведомляем
    r = c.get(f"/api/signals/new?after={a.id}").json()
    assert r["last_id"] == m15.id and [x["symbol"] for x in r["signals"]] == ["ETHUSDT"]
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
        assert 'id="sig-pos-body" hx-swap-oob="innerHTML"' in w and w.count("ETHUSDT") == 2   # и над лентой сигналов
        assert 'id="sig-pos-sum"' in c.get("/").text
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
    assert r.status_code == 200 and "Bybit не подключён" in r.text
    assert '<div id="sig-pos-body" hx-swap-oob="innerHTML"></div>' in r.text     # без биржи блок над лентой пуст
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
    a, b = d["line"][0], d["line"][-1]
    assert a["time"] == d["candles"][0]["time"] and b["time"] == d["candles"][-1]["time"] + 6 * 4 * 3600
    assert b["value"] < a["value"]                                     # линия по точкам 110 → 105 падает
    assert c.get("/api/chart/999").status_code == 404
    assert 'class="live-chart" data-signal="%d"' % s.id in c.get("/feed").text


def test_live_chart_log_line_is_geometric(env):
    c, tmp = env
    _login(c)
    from trader.infrastructure.sqlite_repo import SqliteStore
    s = SqliteStore(tmp / "trader.db").add(replace(_signal(), extra={"log_line": True}))
    d = c.get(f"/api/chart/{s.id}").json()
    assert d["log"] is True and len(d["line"]) > 10
    v = np.array([p["value"] for p in d["line"]])
    t = np.array([p["time"] for p in d["line"]], dtype=float)
    k = np.diff(np.log(v)) / np.diff(t)
    assert np.allclose(k, k[0], rtol=1e-6)                             # прямая в логарифме цены


def test_old_signals_collapse_into_rows(env):
    c, tmp = env
    _login(c)
    from trader.infrastructure.sqlite_repo import SqliteStore
    st = SqliteStore(tmp / "trader.db")
    a = st.add(_signal())
    b = st.add(_signal(symbol="ETHUSDT"))
    r = c.post(f"/signals/{b.id}/skip", headers=HX).text
    assert r.lstrip().startswith('<details class="card signal-row" id="signal-%d"' % b.id) and "пропущен" in r
    feed = c.get("/feed?view=all").text
    assert '<article class="card signal" id="signal-%d"' % a.id in feed and 'id="signal-%d"' % b.id not in feed
    assert '<span class="tab-count num">1</span>' in feed
    arch = c.get("/feed?view=archive").text
    assert 'id="signal-%d"' % b.id in arch and 'id="signal-%d"' % a.id not in arch and "Очистить архив" in arch


def test_delete_archived_signals(env):
    c, tmp = env
    _login(c)
    from trader.infrastructure.sqlite_repo import SqliteStore
    st = SqliteStore(tmp / "trader.db")
    a, b, d = st.add(_signal()), st.add(_signal(symbol="ETHUSDT")), st.add(_signal(symbol="XRPUSDT"))
    chart = tmp / "charts" / "x.png"
    chart.parent.mkdir(exist_ok=True)
    chart.write_bytes(b"png")
    st.set_chart(b.id, str(chart))
    assert c.post(f"/signals/{a.id}/delete", headers=HX).status_code == 409        # новый — удалять нельзя
    c.post(f"/signals/{b.id}/skip", headers=HX)
    c.post(f"/signals/{d.id}/skip", headers=HX)
    assert "Очистить архив" in c.get("/feed?view=archive").text
    r = c.post(f"/signals/{b.id}/delete", headers=HX)
    assert r.status_code == 200 and r.text == "" and st.get(b.id) is None and not chart.exists()
    assert c.post(f"/signals/{b.id}/delete").status_code == 403                       # без заголовка панели
    r = c.post("/signals/archive/clear", headers=HX)
    assert r.headers["HX-Trigger"] == "feed-refresh" and st.get(d.id) is None and st.get(a.id) is not None
    st.set_status(a.id, SignalStatus.CLOSED, "ордер")                                 # закрытая сделка — в архиве
    assert a.symbol in c.get("/feed?view=archive").text and a.symbol not in c.get("/feed?view=all").text
    assert c.post(f"/signals/{a.id}/delete", headers=HX).status_code == 409        # но это история журнала
    c.post("/signals/archive/clear", headers=HX)
    assert st.get(a.id) is not None


def test_reset_to_best_button(env):
    c, tmp = env
    _login(c)
    c.post("/settings/params", data=_params_form(**{"4h__entry_policy": "market"}), headers=HX)
    r = c.post("/settings/reset", headers=HX)
    assert r.status_code == 200 and "Правила как в бэктесте" in r.text and r.headers["HX-Trigger"] == "feed-refresh"
    from trader.domain.models import EntryPolicy
    from trader.infrastructure.sqlite_repo import SqliteStore
    assert SqliteStore(tmp / "trader.db").load().p(Timeframe.H4).entry_policy is EntryPolicy.HYBRID


def test_auto_timeframe_chips(env):
    c, tmp = env
    _login(c)
    r = c.post("/settings/auto-tf/15m", headers=HX)
    assert r.status_code == 200 and 'hx-post="/settings/auto-tf/15m"' in r.text and "нет сигналов" in r.text
    from trader.infrastructure.sqlite_repo import SqliteStore
    assert SqliteStore(tmp / "trader.db").load().auto_timeframes == {Timeframe.H4, Timeframe.M15}


def test_journal_card_shows_live_results_against_backtest(env):
    from trader.domain.execution import Trade, TradeResult, TradeStatus
    from trader.infrastructure.sqlite_repo import SqliteStore
    from test_trader_app import _signal
    c, tmp = env
    st = SqliteStore(tmp / "trader.db")
    s = st.add(_signal())
    t = st.add_trade(Trade(s.id, "SOLUSDT", s.side, s.plan.entry_kind, 2.0, 101.0, 98.0, 110.0, "o1", "demo",
                           TradeStatus.FILLED))
    st.settle_trade(t.id, TradeStatus.CLOSED, TradeResult(101.3, 110.0, 17.4, s.bar_time))
    _login(c)
    r = c.get("/")
    assert "Журнал: биржа против бэктеста" in r.text and "+2.90R" in r.text and "бэктест +0.45R" in r.text
    assert "после 20 сделок (сейчас 1)" in r.text and "цель" in r.text and "проск. +0.10R" in r.text


def test_times_are_rendered_as_utc_for_the_browser_to_localize():
    from datetime import datetime, timezone
    from trader.web.app import local_time
    out = str(local_time(datetime(2026, 10, 6, 0, 0, tzinfo=timezone.utc)))
    assert out == '<time class="lt" datetime="2026-10-06T00:00:00+00:00" data-f="dmhm">06.10 00:00 UTC</time>'
    assert ">06.10<" in str(local_time(datetime(2026, 10, 6, 21, 5), "dm"))      # без пояса — считается UTC


def test_sizing_switch_like_bybit(env):
    from trader.domain.models import Sizing
    from trader.infrastructure.sqlite_repo import SqliteStore
    c, tmp = env
    _login(c)
    assert 'name="4h__risk_pct" type="number"' in c.get("/").text
    r = c.post("/settings/sizing/margin", headers=HX)
    assert r.status_code == 200 and 'name="4h__margin_pct" type="number"' in r.text and "Маржа на сделку" in r.text
    c.post("/settings/params", data=_params_form(**{"4h__margin_pct": "15", "4h__risk_pct": "1"}), headers=HX)
    saved = SqliteStore(tmp / "trader.db").load()
    assert saved.sizing is Sizing.MARGIN and saved.p(Timeframe.H4).margin_pct == 15.0
    assert c.post("/settings/sizing/nope", headers=HX).status_code == 404


def test_journal_export_for_review(env):
    from trader.domain.execution import HedgeLeg, Trade, TradeResult, TradeStatus
    from trader.infrastructure.sqlite_repo import SqliteStore
    from test_trader_app import _signal
    c, tmp = env
    st = SqliteStore(tmp / "trader.db")
    s = st.add(replace(_signal(), extra={"level": True, "break_atr": 0.3}))
    skipped = st.add(_signal(symbol="ETHUSDT"))
    st.set_status(skipped.id, SignalStatus.EXPIRED, "время на вход вышло")
    t = st.add_trade(Trade(s.id, "SOLUSDT", s.side, s.plan.entry_kind, 2.0, 101.0, 98.0, 110.0, "o1", "demo",
                           TradeStatus.FILLED, hedge=HedgeLeg(1.2, 60000, 61000)))
    st.settle_trade(t.id, TradeStatus.CLOSED, TradeResult(101.3, 110.0, 17.4, s.bar_time))
    assert c.get("/export/journal.json", follow_redirects=False).status_code in (302, 303, 401, 403)   # без входа — нельзя
    _login(c)
    assert "/export/journal.json" in c.get("/").text
    r = c.get("/export/journal.json")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    d = r.json()
    assert d["signals_n"] == 2 and d["trades_n"] == 1 and d["settings"]["tf_params"]["4h"]["risk_pct"] == 1.0
    row = next(x for x in d["signals"] if x["symbol"] == "SOLUSDT")
    assert row["x_level"] is True and row["trade"]["r"] == pytest.approx(2.9) and row["trade"]["exit_reason"] == "цель"
    assert row["trade"]["hedge_beta"] == 1.2 and row["trade"]["hedge_r"] is not None
    gone = next(x for x in d["signals"] if x["symbol"] == "ETHUSDT")
    assert gone["status"] == "expired" and gone["note"] == "время на вход вышло" and gone["trade"] is None
    b = row["bars"]                                       # свечи вокруг сигнала: от начала линии
    assert row["bars_error"] is None and len(b["t"]) == len(b["c"]) > 20 and b["t"] == sorted(b["t"])
    assert b["t"][0] <= int(pd.Timestamp(row["line_a_time"]).timestamp())
    assert gone["bars"] is None and "биржа молчит" in gone["bars_error"]      # одна монета не помешала выгрузке
    quick = c.get("/export/journal.json?bars=0").json()
    assert "bars" not in quick["signals"][0] and quick["bars_source"] is None


def test_hedge_leverage_in_params_form(env):
    from trader.infrastructure.sqlite_repo import SqliteStore
    c, tmp = env
    _login(c)
    load = lambda: SqliteStore(tmp / "trader.db").load()                          # noqa: E731
    assert 'name="hedge_leverage"' not in c.get("/").text                         # без хеджа поле не показываем
    c.post("/settings/params", data=_params_form(**{"4h__hedge_btc": "1", "hedge_leverage": "25", "leverage": "10"}),
           headers=HX)
    saved = load()
    assert saved.hedge_leverage == 25 and saved.leverage == 10 and saved.hedging
    assert 'name="hedge_leverage"' in c.get("/").text
    assert "плечо хеджа" in c.post("/settings/params", data=_params_form(**{"4h__hedge_btc": "1", "hedge_leverage": "60"}),
                                   headers=HX).text
    assert load().hedge_leverage == 25                                            # ошибка — ничего не сохранено
    c.post("/settings/params", data=_params_form(**{"4h__hedge_btc": "1"}), headers=HX)   # форма без поля — значение цело
    assert load().hedge_leverage == 25


def test_exchange_aliases_checked_by_price(tmp_path):
    from test_trader_execution import FakeBroker
    from trader.domain.execution import Account
    from trader.infrastructure.bybit import load_aliases
    from trader.infrastructure.sqlite_repo import SqliteStore

    class _B(FakeBroker):
        async def bybit_price(self, bybit_id):
            if bybit_id == "NOPEUSDT":
                raise ValueError("на Bybit нет перпетуала NOPEUSDT")
            return {"PUMPFUNUSDT": 101.5, "OTHERUSDT": 140.0}[bybit_id]

    cfg = AppConfig(panel_password="correct-horse-battery", session_secret="x" * 40, data_dir=tmp_path,
                    panel_url="http://test", telegram_token=None, telegram_chat_id=None, scan_delay_s=0.0,
                    scheduler=False)
    app = create_app(cfg, market=_Market(), broker=lambda: _B(acc=Account(1000.0, 900.0, 0.0)))
    with TestClient(app) as c:
        _login(c)
        assert "Монеты с другим именем на Bybit" in c.get("/").text
        r = c.post("/exchange/alias", data={"binance": "pump usdt", "bybit": "PUMPFUNUSDT"}, headers=HX)
        assert "не похоже на USDT-перпетуал" in r.text
        r = c.post("/exchange/alias", data={"binance": "PUMPUSDT", "bybit": "NOPEUSDT"}, headers=HX)
        assert "нет перпетуала NOPEUSDT" in r.text
        r = c.post("/exchange/alias", data={"binance": "PUMPUSDT", "bybit": "OTHERUSDT"}, headers=HX)
        assert "цены не совпадают" in r.text and "38.6%" in r.text
        st = SqliteStore(tmp_path / "trader.db")
        assert load_aliases(st) == {}
        r = c.post("/exchange/alias", data={"binance": "pumpusdt", "bybit": "pumpfunusdt"}, headers=HX)
        assert "PUMPUSDT → PUMPFUNUSDT" in r.text and "0.50%" in r.text
        assert load_aliases(st) == {"PUMPUSDT": "PUMPFUNUSDT"}
        r = c.post("/exchange/alias/delete", data={"binance": "PUMPUSDT"}, headers=HX)
        assert load_aliases(st) == {} and r.status_code == 200
