import numpy as np
import pandas as pd

from research.listings import classify, events, measure, tickers


def test_classify_titles():
    assert classify("Binance Will List Pepe (PEPE) with Seed Tag Applied", 48) == "spot_list"
    assert classify("Introducing Lista DAO (LISTA) on Binance HODLer Airdrops!", 48) == "spot_list"
    assert classify("Binance Will Delist ANT, MULTI, VAI on 2023-09-15", 161) == "delist"
    assert classify("Binance Futures Will Delist Multiple USDⓈ-M Perpetual Contracts", 161) == "futures_delist"
    assert classify("Binance Will Extend Monitoring Tag to Include AERGO and AKRO", 49) == "monitoring"
    assert classify("Cross Margin Trading for 1INCH Enabled on Binance", 48) == "other"
    assert classify("Binance Futures Will Launch USDⓈ-Margined CTUSDT Perpetual Contract", 48) == "other"


def test_tickers_skip_quotes_and_numbers():
    assert tickers("Binance Will Add PEPE/USDT and WIF (WIF) on 2024-03-05") == ["PEPE", "WIF"]


def test_events_require_existing_perp():
    ann = pd.DataFrame({"t": pd.to_datetime(["2024-03-05 10:00"], utc=True), "catalog": [48],
                        "title": ["Binance Will List Pepe (PEPE) and Bonk (BONK)"]})
    perps = {"1000PEPEUSDT": pd.Timestamp("2023-05-05", tz="UTC"), "1000BONKUSDT": pd.Timestamp("2024-03-05", tz="UTC")}
    ev = events(ann, perps)
    assert list(ev.symbol) == ["1000PEPEUSDT"]


def _bars(t0, path):
    idx = pd.date_range(t0 - pd.Timedelta(minutes=30), periods=len(path), freq="1min", tz="UTC")
    p = np.array(path, float)
    return pd.DataFrame({"open": p, "high": p, "low": p, "close": p}, index=idx)


def test_measure_uses_only_closed_minutes_before_announcement():
    t0 = pd.Timestamp("2024-03-05 10:00:20", tz="UTC")
    # до 10:00 цена 100, минута 10:00 открывается на 100 и закрывается на 120, дальше 130
    path = [100.0] * 30 + [120.0] + [130.0] * 300
    m = _bars(pd.Timestamp("2024-03-05 10:00", tz="UTC"), path)
    m.iloc[30, m.columns.get_loc("open")] = 100.0
    r = measure(m, t0, 1)
    assert np.isclose(r["i_1m"], 2000) and np.isclose(r["i_5m"], 3000)
    # вход через 1 мин: open минуты 10:02 = 130, выход через 1 ч тоже 130 -> только издержки
    assert np.isclose(r["d1_1h"], -12)
    assert "d1_24h" not in r
