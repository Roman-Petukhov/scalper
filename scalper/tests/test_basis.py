import numpy as np
import pandas as pd
import pytest

from research.basis import basis_at, daily_rows, portfolio


def _hours(n, start="2024-01-01"):
    return pd.date_range(start, periods=n, freq="1h", tz="UTC")


def test_basis_uses_only_the_previous_24_hours():
    idx = _hours(72)
    prem = pd.Series(np.where(idx < pd.Timestamp("2024-01-02", tz="UTC"), 0.001, 0.003), index=idx)
    b = basis_at(prem, pd.DatetimeIndex([pd.Timestamp("2024-01-02", tz="UTC"), pd.Timestamp("2024-01-03", tz="UTC")]))
    assert b[0] == pytest.approx(0.001) and b[1] == pytest.approx(0.003)   # час 00:00 нового дня ещё не известен


def test_daily_rows_price_and_funding_over_the_next_day():
    idx = _hours(24 * 40)
    close = pd.Series(100.0 + np.arange(len(idx)) * 0.0, index=idx)
    close[idx >= pd.Timestamp("2024-01-10", tz="UTC")] = 110.0
    fund = pd.Series(0.0, index=idx)
    fund[pd.Timestamp("2024-01-10 08:00", tz="UTC")] = 0.001
    d1 = pd.DataFrame({"close": close, "funding": fund})
    days = pd.date_range("2023-12-01", "2024-03-01", freq="D", tz="UTC")
    r = daily_rows(d1, pd.Series(0.0, index=idx), pd.Series(5e7, index=days), "XUSDT").set_index("day")
    d = pd.Timestamp("2024-01-10", tz="UTC")
    assert r.at[d, "ret_24"] == pytest.approx(0.1) and r.at[d, "fund_24"] == pytest.approx(0.001)
    assert r.at[d + pd.Timedelta(days=1), "ret_24"] == pytest.approx(0.0)


def test_portfolio_long_cheap_short_rich():
    day = pd.Timestamp("2024-01-01", tz="UTC")
    n = 25
    df = pd.DataFrame({"symbol": [f"S{i}" for i in range(n)], "day": day, "basis": np.arange(n, dtype=float),
                       "adv": 1e8, "ret_24": np.linspace(0.05, -0.05, n), "fund_24": 0.0,
                       "ret_168": 0.0, "fund_168": 0.0})
    p = portfolio(df, 24, weekly=False)
    assert len(p) == 1 and p.ls.iloc[0] > 0 and p.q1.iloc[0] > p.q5.iloc[0]
    assert p.ls_net.iloc[0] == pytest.approx(p.ls.iloc[0] - 4 * 5.5e-4)   # первая дата — обе ноги новые
