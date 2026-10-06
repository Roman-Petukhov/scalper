import numpy as np
import pandas as pd
import pytest

from research.listings import COST
from research.newlist import HOLDS, STOP, trade_rows


def frame(prices, start="2024-03-01"):
    idx = pd.date_range(start, periods=len(prices), freq="h", tz="UTC")
    p = np.asarray(prices, float)
    return pd.DataFrame({"open": p, "high": p * 1.001, "low": p * 0.999, "close": p, "quote_volume": 1e6}, index=idx)


def test_short_profits_when_price_falls_and_costs_are_deducted():
    n = 24 + 72 + 800
    coin = frame(np.linspace(100, 50, n))
    btc = frame(np.full(n, 60000.0))
    rows = {(r["k"], r["h"]): r for r in trade_rows("X", coin, btc, pd.Series(dtype="float64"))}
    r = rows[(24, 72)]
    e, x = 24, 24 + 72
    assert r["short_nostop"] == pytest.approx(-(coin.close.iloc[x] / coin.open.iloc[e] - 1) - COST)
    assert r["short"] == pytest.approx(r["short_nostop"]) and not r["stopped"]
    assert r["hedged"] == pytest.approx(r["short"] - 0.0 - COST)          # BTC стоит — хедж только добавляет издержки


def test_stop_caps_loss_on_squeeze():
    n = 24 + 72 + 800
    p = np.full(n, 100.0)
    p[40:] = 200.0                                                         # памп в 2 раза после входа
    r = {(x["k"], x["h"]): x for x in trade_rows("X", frame(p), frame(np.full(n, 1.0)), pd.Series(dtype="float64"))}[(24, 72)]
    assert r["stopped"] and r["short"] == pytest.approx(-STOP - COST) and r["worst"] > STOP


def test_funding_is_received_by_short():
    n = 24 + 72 + 800
    coin, btc = frame(np.full(n, 100.0)), frame(np.full(n, 1.0))
    f = pd.Series(0.001, index=pd.date_range("2024-03-02", periods=40, freq="8h", tz="UTC"))
    r = {(x["k"], x["h"]): x for x in trade_rows("X", coin, btc, f)}[(24, 72)]
    assert r["short_nostop"] > -COST + 0.005 and set(HOLDS) >= {72}
