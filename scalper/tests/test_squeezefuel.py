import numpy as np
import pandas as pd

from research import squeezefuel as Q


def _perp(days: int = 40):
    idx = pd.date_range("2024-01-01", periods=24 * days, freq="1h", tz="UTC")
    d = pd.DataFrame({"close": 10.0, "quote_volume": 100.0, "funding": 0.0}, index=idx)
    m = pd.DataFrame({"oi": 1000.0}, index=idx)
    return d, m


def test_fuel_is_oi_value_over_spot_turnover():
    d, m = _perp()
    spot = pd.Series(500.0, index=pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC"))
    x = Q.coin_days(d, m, spot)
    assert abs(x["fuel"].iloc[-1] - 1000 * 10 / 500) < 1e-9 and abs(x["fuel_p"].iloc[-1] - 10000 / 2400) < 1e-9


def test_forward_return_and_funding_paid():
    d, m = _perp()
    d.loc[d.index >= pd.Timestamp("2024-01-11", tz="UTC"), "close"] = 11.0
    d.loc[pd.Timestamp("2024-01-11 08:00", tz="UTC"), "funding"] = 0.001
    x = Q.coin_days(d, m, None)
    day = pd.Timestamp("2024-01-10", tz="UTC")
    assert abs(x.loc[day, "f1"] - 0.1) < 1e-12 and abs(x.loc[day, "ff1"] - 0.001) < 1e-12 and not x["spot"].iloc[0]


def test_long_trade_pays_funding_and_cost():
    x = pd.DataFrame({"day": pd.to_datetime(["2024-01-01"] * 2), "fuel_rk": [1.0, 0.5], "fund3": [-0.001, 0.0],
                      "f1": [0.05, 0.01], "ff1": [0.002, 0.0]})
    z = Q.trades(x, 1, 1)
    assert len(z) == 1 and abs(z.R.iloc[0] - (0.05 - 0.002 - Q.COST)) < 1e-12
    assert abs(z.X.iloc[0] - (0.048 - 0.03 - Q.COST)) < 1e-12
