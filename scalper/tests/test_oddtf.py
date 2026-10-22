import numpy as np
import pandas as pd

from research import oddtf as O
from research.tline import tf_frame


def _write(root, sym, base, freq, n, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq=freq, tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    o = np.r_[c[0], c[:-1]]
    v = np.full(n, 1000.0)
    pd.DataFrame({"open_time": ((idx - pd.Timestamp(0, tz="UTC")) // pd.Timedelta("1ms")), "open": o, "high": np.maximum(o, c) * 1.002,
                  "low": np.minimum(o, c) * 0.998, "close": c, "volume": v, "quote_volume": v * c, "count": 10,
                  "taker_buy_volume": v * rng.uniform(0.3, 0.8, n), "taker_buy_quote_volume": v * c / 2}
                 ).to_parquet(root / f"{sym}-{base}.parquet")


def test_mid_and_high_timeframes_are_resampled(tmp_path):
    _write(tmp_path, "AUSDT", "15m", "15min", 4000)
    _write(tmp_path, "AUSDT", "1h", "1h", 4000)
    m30, h8 = tf_frame(tmp_path, "AUSDT", "30m"), tf_frame(tmp_path, "AUSDT", "8h")
    assert len(m30) == 2000 and (m30.index[1] - m30.index[0]) == pd.Timedelta("30min")
    assert (h8.index[1] - h8.index[0]) == pd.Timedelta("8h") and len(h8) == 500


def test_variants_and_verdict():
    rng = np.random.default_rng(0)
    n = 3000
    x = pd.DataFrame({"t": pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 2400, n), unit="D"),
                      "entry": rng.choice(["market", "retest"], n), "side": rng.choice([-1, 1], n),
                      "slope_atr": rng.uniform(0, 0.05, n), "R": rng.normal(0.4, 1, n)})
    flat = O.variant(x, "market_flat_short")
    assert (flat.side < 0).all() and (flat.slope_atr <= O.MAX_SLOPE).all() and (flat.entry == "market").all()
    assert O.verdict(x)[0]
    assert not O.verdict(x.assign(R=x.R - 0.6))[0]
