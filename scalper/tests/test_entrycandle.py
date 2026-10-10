import pandas as pd

from research.entrycandle import per_signal


def test_per_signal_unfilled_retest_is_zero():
    t = pd.Timestamp("2024-01-01", tz="UTC")
    df = pd.DataFrame([("A", t, -1, "market", 1.5, 2.2), ("A", t, -1, "retest", 3.0, 2.2),
                       ("B", t, -1, "market", -1.0, 0.8)], columns=["symbol", "t", "side", "entry", "R3", "rng_atr"])
    x = per_signal(df).set_index("symbol")
    assert x.loc["A", "R_rt"] == 3.0 and x.loc["A", "filled"] and x.loc["A", "R_mk"] == 1.5
    assert x.loc["B", "R_rt"] == 0.0 and not x.loc["B", "filled"]
