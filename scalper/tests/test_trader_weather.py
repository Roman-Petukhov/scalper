import asyncio
import json
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from trader.application.weather import KEY, WeatherService, replay_since
from trader.domain import strategy
from trader.domain.models import EntryPolicy, Settings, Timeframe
from trader.domain.strategy import detect, replay
from trader.domain.weather import Mood, ReplayStats, Weather, alts_vs_btc, ret, share_below_ema
from trader.infrastructure.sentiment import (parse_dominance, parse_dvol, parse_fng, parse_long_short, parse_premium,
                                             parse_series)
from trader.infrastructure.sqlite_repo import SqliteStore
from research.tline import zz_lines

NOW = datetime(2026, 10, 21, 0, 20, tzinfo=timezone.utc)


def _frame(n: int = 1500, seed: int = 5, freq: str = "4h", end: datetime | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = (pd.date_range(end=end, periods=n, freq=freq, tz="UTC") if end is not None
           else pd.date_range("2025-01-01", periods=n, freq=freq, tz="UTC"))
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)) + np.sin(np.arange(n) / 60) * 0.3)
    o = np.r_[c[0], c[:-1]]
    v = rng.uniform(1e3, 2e3, n)
    up = c >= o
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.004, "low": np.minimum(o, c) * 0.996, "close": c,
                         "volume": v, "taker_buy_volume": np.where(up, 0.7, 0.3) * v}, index=idx)


@pytest.mark.parametrize("seed", [6, 7])
def test_replay_finds_the_same_signals_as_detect_bar_by_bar(seed, monkeypatch):
    monkeypatch.setattr(strategy, "STOP_ATR", (0.0, 1e9))  # широкий стоп — больше сигналов на коротком ряду
    d = _frame(seed=seed)
    s = Settings().with_tf(Timeframe.H4, min_aggr=0.5, entry_policy=EntryPolicy.MARKET, min_close_loc=0.0)
    since = d.index[400].to_pydatetime()
    got = replay(d, Timeframe.H4, "X", s, since)
    expected = set()
    for t in sorted({r["t"] for r in zz_lines(d) if 400 <= r["t"] < len(d) - 1}):
        for x in detect(d.iloc[: t + 1], Timeframe.H4, "X", s):
            expected.add((x.bar_time, x.side))
    found = {(t, sd) for t, sd, _ in got}
    assert found and found <= expected                     # всё найденное — сигналы detect
    assert len(expected - found) <= 1                      # кроме сделки, ещё открытой в конце ряда
    assert all(np.isfinite(r) for _, _, r in got)


def test_weather_measures_and_notes():
    up = pd.Series(np.linspace(100, 200, 80))
    down = pd.Series(np.linspace(200, 100, 80))
    assert share_below_ema([up, down, down]) == pytest.approx(2 / 3)
    assert ret(up, 7) == pytest.approx(up.iloc[-1] / up.iloc[-8] - 1) and ret(up[:5], 7) is None
    flat = pd.Series(np.full(80, 50.0))
    assert alts_vs_btc([down, down], flat) == pytest.approx(ret(down, 30))
    w = Weather(date(2026, 10, 21), NOW, 3, 0.7, 0.5, -0.08, 0.01, -0.05,
                Mood(fear_greed=20, fear_greed_label="сильный страх", funding_median=-0.0001),
                (ReplayStats("4h", NOW - timedelta(days=30), 12, 0.4, 0.5, 3, 0.9),))
    notes = " ".join(w.notes())
    assert "на месте" in notes and "слабее BTC" in notes and "ниже дневной EMA50" in notes
    assert "страхе" in notes and "толпа в шортах" in notes
    assert Weather.from_json(json.loads(json.dumps(w.to_json()))) == w


def test_sentiment_parsers():
    assert parse_fng({"data": [{"value": "21", "value_classification": "Extreme Fear"}, {"value": "x"}]}) == \
        [(21, "сильный страх")]
    prem = parse_premium([{"symbol": "ETHUSDT", "lastFundingRate": "0.0001", "markPrice": "2002", "indexPrice": "2000"},
                          {"symbol": "ETHBTC", "lastFundingRate": "0.1", "markPrice": "1", "indexPrice": "1"},
                          {"bad": 1}])
    assert list(prem) == ["ETHUSDT"] and prem["ETHUSDT"][0] == 0.0001 and prem["ETHUSDT"][1] == pytest.approx(0.001)
    assert parse_series([{"sumOpenInterestValue": "5", "timestamp": 2}, {"sumOpenInterestValue": "4", "timestamp": 1}],
                        "sumOpenInterestValue") == [4.0, 5.0]
    assert parse_dominance({"data": {"market_cap_percentage": {"btc": 58.3}}}) == 58.3 and parse_dominance({}) is None
    assert parse_dvol({"result": {"data": [[2, 0, 0, 0, 51.0], [1, 0, 0, 0, 49.0], [3]]}}) == [49.0, 51.0]
    assert parse_long_short([{"longShortRatio": "2.0", "timestamp": 2}, {"longShortRatio": "1.5", "timestamp": 1}]) \
        == [1.5, 2.0]


class _Market:
    def __init__(self) -> None:
        self.daily = {"BTCUSDT": pd.DataFrame({"close": np.linspace(100, 101, 70)}),
                      "AAAUSDT": pd.DataFrame({"close": np.linspace(100, 80, 70)}),
                      "BBBUSDT": pd.DataFrame({"close": np.linspace(100, 90, 70)})}

    async def universe(self, min_turnover_usd):
        return ["BTCUSDT", "AAAUSDT", "BBBUSDT"]

    async def daily_bars(self, symbol, limit):
        return self.daily[symbol]

    async def closed_bars(self, symbol, tf):
        return _frame(seed=hash(symbol) % 50, freq=f"{tf.minutes}min", end=NOW - timedelta(hours=4))


class _Mood:
    async def fear_greed(self, days=8):
        return [(30, "страх")] * 8

    async def premium_now(self):
        return {"BTCUSDT": (0.0001, 0.0002), "AAAUSDT": (0.0002, 0.001), "BBBUSDT": (-0.0001, -0.001)}

    async def btc_long_short(self, days=8):
        raise RuntimeError("Binance не ответил")

    async def btc_top_long_short(self, days=8):
        return [0.9] * 7 + [0.8]

    async def btc_open_interest(self, days=8):
        return [100.0] * 7 + [115.0]

    async def btc_dominance(self):
        return 58.0

    async def btc_dvol(self, days=8):
        return [45.0] * 7 + [62.0]


class _Stables:
    async def supply(self):
        return pd.Series(np.r_[np.full(30, 100.0), 98.0])


class _Settings:
    def load(self):
        return Settings()


def test_service_builds_snapshot_once_a_day(tmp_path):
    st = SqliteStore(tmp_path / "t.db")
    clock = {"t": NOW}
    st.kv_set("weather:dominance", json.dumps({"2026-10-14": 56.5}))
    svc = WeatherService(_Market(), _Mood(), st, _Settings(), clock=lambda: clock["t"], stables=_Stables())
    assert svc.due(NOW)
    w = asyncio.run(svc.run())
    st.kv_set(KEY, json.dumps(w.to_json()))
    assert w.coins == 2 and w.below_ema == 1.0 and w.btc_7d > 0 and w.alts_vs_btc_30d < 0
    m = w.mood
    assert m.fear_greed == 30 and m.funding_median == 0.0001 and m.btc_long_short is None and m.basis_btc == 0.0002
    assert m.oi_btc_7d == pytest.approx(0.15) and m.top_long_short == 0.8 and m.dvol == 62.0 and m.dvol_week == 45.0
    assert m.dominance == 58.0 and m.dominance_week == 56.5 and m.stables_30d == pytest.approx(-0.02)
    notes = " ".join(w.notes())
    assert "деньги уходят из альтов" in notes and "новые деньги" in notes and "плеча в рынке больше" in notes
    assert "сильные движения" in notes
    assert [r.tf for r in w.replay] == ["4h"] and w.replay[0].since == replay_since(Timeframe.H4, NOW)
    assert svc.current() == w and not svc.due(NOW + timedelta(hours=5))
    asyncio.run(svc._guarded())
    assert st.kv_get("weather:2026-10-21") is not None
    assert svc.due(NOW + timedelta(days=1))


def test_15m_replay_window_fits_available_bars():
    assert NOW - replay_since(Timeframe.H4, NOW) == timedelta(days=30)
    assert NOW - replay_since(Timeframe.M15, NOW) == timedelta(minutes=15 * 1100)


def test_liquidation_book_by_hour_and_parser():
    from trader.domain.liquidations import LiqBook, parse_force_order
    from trader.domain.models import Side
    ev = parse_force_order({"e": "forceOrder", "E": 1, "o": {"s": "ETHUSDT", "S": "SELL", "q": "2", "p": "1990",
                                                              "ap": "2000", "z": "2", "T": 1792540800000}})
    assert ev[1] is Side.LONG and ev[2] == 4000.0 and ev[0].tzinfo is not None
    assert parse_force_order({"o": {"S": "BUY"}}) is None
    book = LiqBook(started=NOW - timedelta(hours=30))
    book.add(NOW - timedelta(hours=26), Side.LONG, 1e6)            # вне суток
    book.add(NOW - timedelta(hours=2), Side.LONG, 3000.0)
    book.add(NOW - timedelta(minutes=5), Side.SHORT, 1000.0)
    t = book.totals(NOW)
    assert (t.long_usd, t.short_usd) == (3000.0, 1000.0) and t.long_share == 0.75 and not t.partial
    book.trim(NOW)
    again = LiqBook.from_json(json.loads(json.dumps(book.to_json())))
    assert again.totals(NOW) == t and len(again.buckets) == 2


def test_liquidation_watcher_persists_and_counts_from_start(tmp_path):
    from trader.application.liquidations import LiquidationWatcher
    st = SqliteStore(tmp_path / "t.db")
    clock = {"t": NOW}
    w = LiquidationWatcher(st, clock=lambda: clock["t"])
    assert w.totals().partial and w.totals().long_share is None
    w.on_message({"o": {"S": "BUY", "z": "10", "ap": "5", "T": int(NOW.timestamp() * 1000)}})
    w.on_message({"garbage": 1})
    w.save()
    clock["t"] = NOW + timedelta(minutes=30)
    w2 = LiquidationWatcher(st, clock=lambda: clock["t"])
    assert w2.totals().short_usd == 50.0 and w2.book.started == NOW


def test_assessment_separates_luck_from_a_broken_bot():
    from trader.domain.weather import assess
    base = Weather(date(2026, 10, 21), NOW, 90, 0.7, 0.6, -0.08, 0.01, -0.05)
    rep = lambda live, rule, n=40: replace_rep(base, ReplayStats("4h", NOW, 120, rule, 0.4, n, live))
    a = assess(rep(1.2, 1.1))
    assert a.level == "ok" and "рынок удачный" in a.lines[0] and "попутный" in a.market
    a = assess(rep(-0.5, -0.4))
    assert a.level == "watch" and "не выключать" in a.lines[0]
    a = assess(rep(-0.6, 0.6))
    assert a.level == "warn" and "проверить исполнение" in a.lines[0]
    a = assess(rep(1.5, 0.3))
    assert "скорее везение" in a.lines[0]
    a = assess(rep(1.5, 0.3, n=4))
    assert "для вывода мало" in a.lines[0]
    txt = assess(base).text()
    assert "Настроение:" in txt and "слабее BTC" in txt and "риск" not in txt and "проверенн" not in txt


def replace_rep(w, r):
    from dataclasses import replace as rp
    return rp(w, replay=(r,))


def test_assessment_reports_rule_and_bot_by_side():
    from trader.domain.weather import assess, replay_stats
    r = replay_stats("4h", NOW, [(1, 3.0), (1, -1.0), (-1, -1.0), (-1, -1.0), (-1, 3.0)], [(-1, 2.9), (-1, -1.0)])
    assert (r.long_n, r.short_n, r.live_short_n, r.live_long_n) == (2, 3, 2, 0)
    assert r.long_avg_r == 1.0 and r.short_avg_r == pytest.approx(1 / 3) and r.avg_r == pytest.approx(0.6)
    w = Weather(date(2026, 10, 21), NOW, 90, 0.3, 0.4, 0.08, 0.05, 0.1, replay=(r,))
    a = assess(w)
    assert "встречный для шортов, попутный для лонгов" in a.market
    assert any("лонги +1.00R на 2, шорты +0.33R на 3; бот — шорты +0.95R на 2" in x for x in a.lines)
