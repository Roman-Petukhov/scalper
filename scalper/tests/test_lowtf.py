import numpy as np
import pandas as pd

from research import noline
from research.tline import htf_trend, tf_frame


def _bars(n, freq, seed=1):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-07-01", periods=n, freq=freq, tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    o = np.r_[c[0], c[:-1]]
    v = rng.uniform(1, 10, n)
    return pd.DataFrame({"open_time": t.as_unit("ms").asi8, "open": o, "high": np.maximum(o, c) * 1.001,
                         "low": np.minimum(o, c) * 0.999, "close": c, "volume": v, "quote_volume": v * c, "count": 1,
                         "taker_buy_volume": v * rng.uniform(0.3, 0.7, n), "taker_buy_quote_volume": v * c / 2})


def test_low_tf_files_are_read_and_trend_uses_closed_higher_bars(tmp_path):
    _bars(6000, "1min").to_parquet(tmp_path / "XUSDT-1m.parquet", index=False)
    d = tf_frame(tmp_path, "XUSDT", "1m")
    assert len(d) == 6000 and (d.index[1] - d.index[0]) == pd.Timedelta(minutes=1)
    tr = htf_trend(d, "1m")
    assert set(np.unique(tr[~np.isnan(tr)])) <= {-1.0, 0.0, 1.0}
    # тренд 15-минутки известен только после её закрытия: внутри первой 15-минутки после EMA ничего нового нет
    d2 = d.copy()
    d2.iloc[-5:, d2.columns.get_loc("close")] *= 10
    assert np.array_equal(htf_trend(d2, "1m")[:-15], tr[:-15], equal_nan=True)


def test_lines_only_is_the_line_subset_of_all_triggers():
    d = _bars(8000, "5min").set_index(pd.to_datetime(_bars(8000, "5min")["open_time"], unit="ms", utc=True))
    d["funding"] = 0.0
    noline.TF = "5m"
    liq = np.full(len(d), 1e9)
    full = noline.coin_trades(d, liq, "X")
    only = noline.coin_trades(d, liq, "X", lines_only=True)
    assert len(only) > 0 and set(only.trig) <= {"line"}
    assert len(only) == (full.trig == "line").sum()
    noline.TF = "4h"
