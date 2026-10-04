import numpy as np
import pandas as pd

from research.hl_whales import episodes


def _f(rows):
    cols = ["coin", "px", "sz", "side", "time", "startPosition"]
    df = pd.DataFrame(rows, columns=cols)
    df["dir"], df["closedPnl"], df["fee"], df["tid"] = "", 0.0, 0.0, range(len(df))
    return df


H = 3_600_000


def test_long_round_trip_with_scaling():
    ep = episodes(_f([("BTC", 100, 100, "B", 0, 0), ("BTC", 110, 100, "B", H, 100),
                      ("BTC", 120, 200, "A", 2 * H, 200)]))
    assert len(ep) == 1 and ep.side[0] == 1 and np.isclose(ep.own[0], 120 / 105 - 1)


def test_flip_closes_and_opens_short():
    ep = episodes(_f([("ETH", 100, 200, "B", 0, 0), ("ETH", 90, 500, "A", H, 200), ("ETH", 80, 300, "B", 2 * H, -300)]))
    assert list(ep.side) == [1, -1]
    assert np.isclose(ep.own[0], -0.1) and np.isclose(ep.own[1], 1 - 80 / 90)


def test_skip_history_started_before_window_and_open_positions():
    ep = episodes(_f([("SOL", 10, 5000, "A", 0, 3000), ("SOL", 10, 2000, "B", H, -2000)]))
    assert len(ep) == 0


def test_spot_and_small_positions_ignored():
    ep = episodes(_f([("@107", 10, 5000, "B", 0, 0), ("@107", 11, 5000, "A", H, 5000),
                      ("DOGE", 0.1, 100, "B", 0, 0), ("DOGE", 0.2, 100, "A", H, 100)]))
    assert len(ep) == 0
