import numpy as np
import pandas as pd

from research import onchainval as O
from tests.test_spotrev import _panel


def test_month_end_sums_last_30_days():
    days = pd.date_range("2024-01-01", "2024-02-29", freq="D")
    d = pd.DataFrame({"symbol": "AUSDT", "day": days, "fees": 1.0})
    m = O.month_end(d, "fees", "sum30")
    assert m.loc[pd.Timestamp("2024-02-01"), "AUSDT"] == 30.0


def test_cheap_tercile_by_signal():
    c_end, c_d1, adv, meta, regime = _panel(n_coins=30)
    sig = pd.DataFrame(1.0, index=c_end.index, columns=c_end.columns)
    cheap = [f"C{i}USDT" for i in range(10)]
    sig[cheap] = 5.0
    c_d1.loc[pd.Timestamp("2019-08-01"), cheap] = 150.0
    z = O.cohorts(c_end, c_d1, adv, meta, regime, sig, 1)
    row = z[z.t == pd.Timestamp("2019-07-01")].iloc[0]
    assert abs(row["дешёвые"] - 0.5) < 1e-12 and row["n"] == 30


def test_binance_symbol_mapping():
    assert O.binance_symbol("uni") == "UNIUSDT" and O.binance_symbol("ae_eth") is None
    assert O.binance_symbol("avaxc") == "AVAXUSDT"


def test_nmfilter_tercile_uses_previous_month():
    from research import nmfilter as N
    months = pd.date_range("2024-01-01", periods=3, freq="MS")
    cols = [f"C{i}USDT" for i in range(15)]
    nm = pd.DataFrame(np.tile(np.arange(15.0), (3, 1)), index=months, columns=cols)
    z = pd.DataFrame({"t": [pd.Timestamp("2024-02-10")] * 2, "symbol": ["C0USDT", "C14USDT"]})
    t = N.tercile(nm, z)
    assert t.iloc[0] == 2 and t.iloc[1] == 0               # мало адресов на доллар — дорогая
