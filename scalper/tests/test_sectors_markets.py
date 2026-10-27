import numpy as np
import pandas as pd

from research import markets as M
from research import sectors as S


def test_next_day_uses_first_crypto_day_after_us_close():
    days = pd.date_range("2024-01-01", periods=6, freq="D")
    spx = pd.Series([100, 102, 102, 102, 102, 102.0], index=days)       # +2% в день 2024-01-02
    btc = pd.Series([10, 10, 11, 11, 11, 11.0], index=days)              # свеча 01-03 (закрытие 00:00 01-04): 10 → 11
    z = M.next_day(btc, spx)
    assert len(z) == 1 and z.t.iloc[0] == pd.Timestamp("2024-01-03") and abs(z.X.iloc[0] - (0.1 - M.COST)) < 1e-12


def test_sector_map_has_extra_sectors_without_overriding():
    m = S.sector_map()
    assert m["XMRUSDT"] == "приватность" and m["ONDOUSDT"] == "RWA" and m["DOGEUSDT"] == "мемы"


def test_contagion_buys_laggards_of_leader_sector():
    days = pd.date_range("2024-01-01", periods=5, freq="D")
    cols = ["DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "UNIUSDT", "AAVEUSDT", "BTCUSDT"]
    close = pd.DataFrame(100.0, index=days, columns=cols)
    close.loc[days[2]:, "DOGEUSDT"] = 150.0                             # лидер +50% в день 3
    close.loc[days[3]:, ["SHIBUSDT", "PEPEUSDT"]] = 110.0               # отстающие догоняют на следующий день
    r = close.pct_change()
    lag, plc = S.contagion(close, r, r.mean(axis=1), 1)
    assert len(lag) == 1 and lag.n.iloc[0] == 2 and lag.X.iloc[0] > 0.05 and plc.X.iloc[0] < 0
