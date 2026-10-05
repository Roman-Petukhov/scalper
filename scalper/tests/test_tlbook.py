import numpy as np
import pandas as pd

from research.hft.tlbook import classify, features


def test_features_signs_and_window_clipped_to_day():
    day = "2025-03-01"
    day0 = pd.Timestamp(day, tz="UTC").value // 10**6
    probe = day0 + 2 * 3_600_000 - 1_000                       # закрытие 02:00 — окно обрезано до 2 ч
    evd = pd.DataFrame({"symbol": ["X"], "day": [day], "probe_ms": [probe], "side": [-1], "R": [1.0], "aggr": [0.6]})
    snaps = pd.DataFrame({"t_ms": [probe], "mid": [100.0], "spread_bps": [1.0],
                          "bid50": [10.0], "ask50": [30.0], "bid100": [20.0], "ask100": [60.0]})
    # продали 30, купили 10 в окне; одна покупка до начала окна не считается
    trades = pd.DataFrame({"timestamp": np.array([day0 - 10_000, day0 + 1_000, day0 + 2_000, day0 + 3_000]) / 1000.0,
                           "side": ["Buy", "Buy", "Sell", "Sell"], "size": [100.0, 10.0, 20.0, 10.0],
                           "price": [100.0] * 4})
    wall_ev = pd.DataFrame({"ts": [day0 + 60_000.0, day0 + 200_000.0], "kind": ["consumed", "absorbed"],
                            "side": [1, -1], "size_x": [20.0, 20.0], "life_ms": [10_000.0, 10_000.0],
                            "dist_bps": [5.0, 5.0]})
    f = features(evd, snaps, wall_ev, trades, day).iloc[0]
    assert np.isclose(f["win_h"], 2 - 1 / 3600)
    assert np.isclose(f["flow_d"], 0.5)                        # (10 − 30) / 40 × (−1)
    assert np.isclose(f["imb100_d"], 0.5)                      # (20 − 60) / 80 × (−1): впереди (bid) пусто
    assert np.isclose(f["thin_100"], 20 / (40 / f["win_h"]))
    # шорт: путь — bid. Съеденная стена bid — на пути; поглощение на ask — за нами (не путь)
    assert f["consumed_path"] > 0 and f["absorbed_path"] == 0 and f["consumed_back"] == 0
    assert classify(pd.DataFrame([f]))[0] == "confirmed"
