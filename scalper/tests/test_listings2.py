import numpy as np
import pandas as pd

from research.listings2 import trades


def test_short_pays_negative_funding_and_skips_incomplete_holds():
    t0 = pd.Timestamp("2025-01-01 10:00:30", tz="UTC")
    idx = pd.date_range("2025-01-01 09:00", periods=60 * 30, freq="1min", tz="UTC")
    px = np.full(len(idx), 100.0)
    px[idx >= "2025-01-01 10:01"] = 90.0                            # после анонса цена 90
    m = pd.DataFrame({"open": px, "high": px, "low": px, "close": px}, index=idx)
    m.loc[m.index >= "2025-01-02 00:00", ["open", "high", "low", "close"]] = 81.0
    fund = pd.Series([-0.001, -0.002], index=pd.to_datetime(["2025-01-01 16:00", "2025-01-02 00:00"], utc=True))
    r = trades(m, t0, -1, fund, "")
    # вход через 1 мин по 90, через 24 ч цена 81: шорт +10% - 12 б.п.; funding -0.3% шорт платит
    assert np.isclose(r["g1_24h"], (0.1 - 12e-4) * 1e4)
    assert np.isclose(r["d1_24h"], (0.1 - 12e-4 - 0.003) * 1e4)
    assert "d1_72h" not in r
