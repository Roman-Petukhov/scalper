import numpy as np
import pandas as pd

from research.extreg import gates, parse_dvol, parse_stable


def test_parse_stable_reads_pegged_usd_and_skips_junk():
    raw = [{"date": "1609459200", "totalCirculatingUSD": {"peggedUSD": 2.9e10}},
           {"date": 1609545600, "totalCirculating": {"peggedUSD": 3.0e10}},
           {"date": "x"}, {"oops": 1}]
    s = parse_stable(raw)
    assert list(s.values) == [2.9e10, 3.0e10]
    assert s.index[0] == pd.Timestamp("2021-01-01", tz="UTC")
    assert parse_stable({"error": "x"}).empty


def test_parse_dvol_takes_the_last_candle_of_the_day():
    t0 = int(pd.Timestamp("2022-01-01", tz="UTC").timestamp() * 1000)
    page = {"result": {"data": [[t0, 80, 82, 79, 81], [t0 + 43_200_000, 81, 85, 80, 84]], "continuation": None}}
    s = parse_dvol([page, None, {"result": {}}])
    assert len(s) == 1 and s.iloc[0] == 84


def test_gates_use_only_the_past():
    idx = pd.date_range("2021-01-01", periods=500, freq="1D")
    ext = pd.DataFrame({"stable": np.linspace(1e10, 2e10, 500), "dvol": np.r_[np.full(250, 90.0), np.full(250, 50.0)]},
                       index=idx)
    btc = pd.Series(np.linspace(10, 20, 500), index=idx)
    g = gates(ext, btc, idx)
    assert g["приток"].iloc[40] and not g["приток"].iloc[10]            # первые 30 дней — нет сравнения
    assert g["спокойно"].iloc[400] and not g["спокойно"].iloc[200]
    g2 = gates(ext.assign(stable=ext.stable.where(idx < idx[300], 0.0)), btc, idx)
    assert (g2["приток"].iloc[:300] == g["приток"].iloc[:300]).all()
