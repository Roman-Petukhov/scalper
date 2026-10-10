import numpy as np
import pandas as pd

from research.btcline import direction


def test_direction_uses_only_events_known_by_signal_close():
    ctx = pd.DataFrame({"known": pd.to_datetime(["2024-01-01 04:00", "2024-01-03 08:00"], utc=True), "side": [-1, 1]})
    side, age = direction(ctx, pd.Timestamp("2023-12-31", tz="UTC"))
    assert side == 0 and np.isnan(age)
    assert direction(ctx, pd.Timestamp("2024-01-03 04:00", tz="UTC")) == (-1, 48.0)
    assert direction(ctx, pd.Timestamp("2024-01-03 08:00", tz="UTC")) == (1, 0.0)    # известен на этом закрытии
