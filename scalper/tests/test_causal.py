import numpy as np
import pandas as pd

from research import causal as C


def _hours(n: int, start: str = "2024-01-05 00:00") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1h")
    return pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "funding": 0.0}, index=idx)


def test_funding_rows_take_hour_before_and_hour_of_settlement():
    d = _hours(10)
    d.iloc[3, d.columns.get_loc("close")] = 99.0          # час перед выплатой
    d.iloc[4, d.columns.get_loc("funding")] = 0.001
    rows = C.funding_rows(d, "X")
    assert len(rows) == 1 and abs(rows[0]["pre"] + 0.01) < 1e-12 and rows[0]["post"] == 0.0


def test_gap_fill_trade_hits_friday_price():
    d = _hours(24 * 10)                                    # пятница 2024-01-05
    sun = pd.Timestamp("2024-01-07 21:00")
    d.loc[pd.Timestamp("2024-01-05 21:00"):sun, ["open", "high", "low", "close"]] = 103.0
    d.loc[pd.Timestamp("2024-01-08 05:00"), "low"] = 99.5   # вернулась к пятничной цене 100
    z = C.gap_rows(d.drop(columns=["funding"]))
    row = z.iloc[0]
    assert row["filled"] and abs(row["g"] - (100 / 103 - 1) * -1) < 1e-12 and abs(row["gap"] - 0.03) < 1e-12


def test_verdict_requires_all_periods_positive():
    t = pd.date_range("2022-01-01", "2026-09-01", freq="7D")
    z = pd.DataFrame({"t": t, "x": 0.01 + 0.001 * np.sin(np.arange(len(t)))})
    good, _ = C.verdict(z, "x", 0.0011)
    assert good
