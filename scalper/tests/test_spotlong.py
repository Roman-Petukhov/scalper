import numpy as np
import pandas as pd

from research.spotlong import VARIANTS, spot_on_bars


def test_spot_is_summed_into_4h_bars_and_missing_spot_is_nan(tmp_path):
    idx = pd.date_range("2023-01-01", periods=8, freq="1h", tz="UTC")
    pd.DataFrame({"open_time": idx.as_unit("ms").asi8, "quote_volume": 10.0, "volume": 1.0,
                  "taker_buy_volume": [1, 1, 0, 0, 1, 1, 1, 1.0]}).to_parquet(tmp_path / "X-spot-1h.parquet")
    bars = pd.date_range("2023-01-01", periods=3, freq="4h", tz="UTC")
    buy, q = spot_on_bars(tmp_path, "X", bars)
    assert buy[0] == 0.5 and buy[1] == 1.0 and np.isnan(buy[2])
    assert q[0] == 40.0
    b2, q2 = spot_on_bars(tmp_path, "Y", bars)
    assert np.isnan(b2).all() and np.isnan(q2).all()


def test_variants_select_as_described():
    z = pd.DataFrame({"buy": [0.56, 0.50, 0.50, 0.53], "spot": [0.50, 0.56, 0.52, 0.53]})
    sel = {k: f(z).tolist() for k, f in VARIANTS.items()}
    assert sel["перп 55% (бот)"] == [True, False, False, False]
    assert sel["добавка: спот 55%, перп < 55%"] == [False, True, False, False]
    assert sel["спот сильнее перпа на 3 п.п."] == [False, True, False, False]
    assert sel["оба ≥ 52.5%"] == [False, False, False, True]
