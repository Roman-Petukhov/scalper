import numpy as np
import pandas as pd

from research.bullongs import label, regimes


def test_regime_is_taken_from_the_last_closed_day():
    days = pd.date_range("2023-01-01", periods=400, freq="1D", tz="UTC")
    btc = pd.Series(np.r_[np.full(300, 10.0), np.full(100, 30.0)], index=days)
    stable = pd.Series(np.linspace(1, 2, 400), index=days)
    reg = regimes(stable, btc)
    assert reg.loc[days[350], "BTC > SMA200"] and not reg.loc[days[250], "BTC > SMA200"]
    assert reg.loc[days[350], "оба"]
    t = pd.Series([days[300] + pd.Timedelta(hours=8), days[300] - pd.Timedelta(hours=8)])
    out = label(pd.DataFrame({"t": t, "R": [1.0, 1.0]}), reg)
    assert out["BTC > SMA200"].tolist() == [False, False]        # день 300 ещё не закрыт — смотрим день 299
    t2 = pd.Series([days[302] + pd.Timedelta(hours=8)])
    assert label(pd.DataFrame({"t": t2, "R": [1.0]}), reg)["BTC > SMA200"].tolist() == [True]
