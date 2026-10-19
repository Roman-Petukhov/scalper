import pandas as pd

import research.supply as sp
from research.macrocal import FOMC, at, bls_dates, flags


def test_event_time_and_bls_dates():
    assert at("2024-01-31", 14, 0) == pd.Timestamp("2024-01-31 19:00", tz="UTC")      # зима: ET = UTC−5
    assert at("2024-07-31", 14, 0) == pd.Timestamp("2024-07-31 18:00", tz="UTC")      # лето: UTC−4
    html = '<a href="/news.release/archives/cpi_02102022.htm">x</a> cpi_13452022.htm empsit_01052024.htm'
    assert bls_dates(html, "cpi") == ["2022-02-10"] and bls_dates(html, "empsit") == ["2024-01-05"]
    assert len(FOMC) == len(set(FOMC)) and FOMC == sorted(FOMC)


def test_flags():
    t = pd.Timestamp("2024-01-30 08:00", tz="UTC")
    x = pd.DataFrame({"t": [t, t + pd.Timedelta(days=10)], "sig_close": [t + pd.Timedelta(hours=4)] * 2,
                      "fill": [t + pd.Timedelta(hours=8), t + pd.Timedelta(days=10, hours=8)],
                      "exit": [t + pd.Timedelta(days=3), t + pd.Timedelta(days=12)], "R": [1.0, 1.0]})
    x.loc[1, "sig_close"] = x.loc[1, "t"] + pd.Timedelta(hours=4)
    z = flags(x, pd.Series([pd.Timestamp("2024-01-31 19:00", tz="UTC")]))
    assert list(z.M1) == ["нет", "нет"] and list(z.M2) == ["да", "нет"] and list(z.M3) == ["да", "нет"]
    z = flags(x, pd.Series([pd.Timestamp("2024-01-31 06:00", tz="UTC")]))
    assert z.M1[0] == "да"


def test_supply_ticker_and_issued(monkeypatch):
    assert sp.base("1000PEPEUSDT") == "PEPE" and sp.base("ARBUSDT") == "ARB"
    day = 86_400_000
    data = {"market_caps": [[0, 50.0], [day, 60.0]], "prices": [[0, 1.0], [day, 1.0]]}
    monkeypatch.setattr(sp, "_get", lambda url: data)
    monkeypatch.setattr(sp.time, "sleep", lambda s: None)
    s = sp.issued("x", 100.0)
    assert list(s.round(2)) == [0.5, 0.6]
