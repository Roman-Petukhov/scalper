import numpy as np
import pandas as pd

from research.fundarb import COST, trades
from research.unlocks import measure


def _write(root, sym, perp, spot, rates):
    idx = pd.date_range("2024-01-01", periods=len(perp), freq="1h", tz="UTC")
    ot = np.array([int(x.timestamp() * 1000) for x in idx], dtype="int64")
    pd.DataFrame({"open_time": ot, "close": perp}).to_parquet(root / f"{sym}-1h.parquet")
    pd.DataFrame({"open_time": ot, "close": spot}).to_parquet(root / f"{sym}-spot-1h.parquet")
    ts = [int((pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=8 * (k + 1))).value // 10**6)
          for k in range(len(rates))]
    pd.DataFrame({"ts": ts, "rate": rates}).to_parquet(root / f"{sym}-funding.parquet")


def test_fundarb_collects_funding_until_rate_normalises(tmp_path):
    n = 24 * 5
    _write(tmp_path, "XUSDT", np.full(n, 10.0), np.full(n, 10.0), [0.003, 0.002, 0.0015, 0.00005, 0.003])
    tr = trades(tmp_path, "XUSDT")
    t = tr[tr.thresh == 0.002].iloc[0]
    # вход после первой выплаты (0.3%), получаем 0.2% + 0.15% + 0.005%, выход после выплаты ниже 0.01%
    assert np.isclose(t["funding"], 0.002 + 0.0015 + 0.00005)
    assert np.isclose(t["net"], t["funding"] - COST)


def test_unlock_short_with_funding_and_btc():
    days = pd.date_range("2024-01-01", periods=60, freq="1D", tz="UTC")
    c = pd.Series(100.0, index=days)
    c[days >= "2024-02-01"] = 90.0                    # к T цена упала
    btc = pd.Series(100.0, index=days)
    fund = pd.Series(0.001, index=days)
    t = pd.Timestamp("2024-02-05", tz="UTC")
    m = measure(c, btc, fund, t)
    assert np.isclose(m["short7"], (0.1 - 12e-4) * 1e4)
    assert np.isclose(m["short7_f"], (0.1 - 12e-4 + 0.008) * 1e4)
    assert np.isclose(m["pre7"], -1000)


def test_unlock_grid_short_and_btc_hedge():
    from research.unlocks2 import grid
    days = pd.date_range("2024-01-01", periods=60, freq="1D", tz="UTC")
    c = pd.Series(100.0, index=days)
    c[days >= "2024-02-03"] = 90.0
    btc = pd.Series(100.0, index=days)
    btc[days >= "2024-02-03"] = 95.0
    fund = pd.Series(0.0, index=days)
    g = grid(c, btc, fund, pd.Timestamp("2024-02-05", tz="UTC"))
    assert np.isclose(g["s7_1"], (0.1 - 12e-4) * 1e4)
    assert np.isclose(g["h7_1"], (0.1 - 12e-4 - 0.05 - 12e-4) * 1e4)
