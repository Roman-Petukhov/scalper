import numpy as np
import pandas as pd

from research import hotcoins as H


def test_signal_hour_uses_only_closed_hour():
    t = pd.Series(pd.to_datetime(["2024-01-01 10:45", "2024-01-01 10:30", "2024-01-01 10:55"], utc=True))
    assert list(H.signal_hour(t, "15m")) == list(pd.to_datetime(["2024-01-01 10:00", "2024-01-01 09:00",
                                                                  "2024-01-01 10:00"], utc=True))
    assert H.signal_hour(t, "5m").iloc[2] == pd.Timestamp("2024-01-01 10:00", tz="UTC")


def test_ranks_by_abs_move_within_gate():
    h = pd.Timestamp("2024-01-01 09:00", tz="UTC")
    hours = pd.DataFrame({"h": [h] * 3, "symbol": ["A", "B", "C"], "ret24": [0.05, -0.30, 0.50],
                          "qv24": [60e6, 80e6, 30e6]})
    r = H.ranks(hours).set_index("symbol")
    assert r.loc["B", "rank"] == 1 and r.loc["A", "rank"] == 2 and np.isnan(r.loc["C", "rank"])
    assert r.loc["C", "rank_low"] == 1 and r.loc["B", "rank_low"] == 2


def test_hourly_ret_and_gate():
    idx = pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC")
    h1 = pd.DataFrame({"close": np.linspace(100, 129, 30), "quote_volume": 1e6}, index=idx)
    x = H.hourly(h1, "A")
    assert len(x) == 6 and abs(x.ret24.iloc[0] - (124 / 100 - 1)) < 1e-6
    assert H.hourly(h1.assign(quote_volume=1e5), "A").empty


def test_label_marks_hot_and_direction():
    h = pd.Timestamp("2024-01-01 09:00", tz="UTC")
    hours = pd.DataFrame({"h": [h] * 12, "symbol": [f"S{i}" for i in range(12)],
                          "ret24": [-(0.5 - 0.01 * i) for i in range(12)], "qv24": [60e6] * 12})
    tr = pd.DataFrame({"symbol": ["S0", "S11", "ZZ"], "tf": "15m", "t": pd.to_datetime(["2024-01-01 09:45"] * 3, utc=True),
                       "side": [-1, -1, 1], "entry": "market", "R3": [1.0, -1.0, 0.0], "risk_pct": 0.01})
    x = H.label(tr, hours).set_index("symbol")
    assert x.loc["S0", "hot"] == "да" and x.loc["S11", "hot"] == "нет" and x.loc["ZZ", "hot"] == "нет"
    assert x.loc["S0", "with_move"] == "по движению"
