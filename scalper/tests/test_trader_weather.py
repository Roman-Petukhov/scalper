import asyncio
import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from trader.application.weather import KEY, WeatherService, replay_since
from trader.domain import strategy
from trader.domain.models import EntryPolicy, Settings, Timeframe
from trader.domain.strategy import detect, replay
from trader.domain.weather import Mood, ReplayStats, Weather, alts_vs_btc, ret, share_below_ema
from trader.infrastructure.sentiment import parse_fng, parse_funding, parse_long_short
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
    assert parse_funding([{"symbol": "ETHUSDT", "lastFundingRate": "0.0001"}, {"symbol": "ETHBTC",
                          "lastFundingRate": "0.1"}, {"bad": 1}]) == {"ETHUSDT": 0.0001}
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

    async def funding_now(self):
        return {"BTCUSDT": 0.0001, "AAAUSDT": 0.0002, "BBBUSDT": -0.0001}

    async def btc_long_short(self, days=8):
        raise RuntimeError("Binance не ответил")


class _Settings:
    def load(self):
        return Settings()


def test_service_builds_snapshot_once_a_day(tmp_path):
    st = SqliteStore(tmp_path / "t.db")
    clock = {"t": NOW}
    svc = WeatherService(_Market(), _Mood(), st, _Settings(), clock=lambda: clock["t"])
    assert svc.due(NOW)
    w = asyncio.run(svc.run())
    st.kv_set(KEY, json.dumps(w.to_json()))
    assert w.coins == 2 and w.below_ema == 1.0 and w.btc_7d > 0 and w.alts_vs_btc_30d < 0
    assert w.mood.fear_greed == 30 and w.mood.funding_median == 0.0001 and w.mood.btc_long_short is None
    assert [r.tf for r in w.replay] == ["4h"] and w.replay[0].since == replay_since(Timeframe.H4, NOW)
    assert svc.current() == w and not svc.due(NOW + timedelta(hours=5))
    assert svc.due(NOW + timedelta(days=1))


def test_15m_replay_window_fits_available_bars():
    assert NOW - replay_since(Timeframe.H4, NOW) == timedelta(days=30)
    assert NOW - replay_since(Timeframe.M15, NOW) == timedelta(minutes=15 * 1100)
