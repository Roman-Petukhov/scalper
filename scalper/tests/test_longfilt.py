import numpy as np
import pandas as pd

from research.longfilt import VARIANTS, coin_rows, features, masks


def _frame(n=1200, seed=7, drift=0.002):
    idx = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(drift, 0.02, n)))
    o = np.r_[c[0], c[:-1]]
    v = rng.uniform(1e5, 2e5, n)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.005, "low": np.minimum(o, c) * 0.995, "close": c,
                         "volume": v, "taker_buy_volume": v * rng.uniform(0.3, 0.7, n), "funding": np.zeros(n)},
                        index=idx)


def test_features_use_only_the_past_and_the_bar_itself():
    d = _frame()
    a = features(d, d["close"] * 0 + 1)
    d2 = d.copy()
    d2.iloc[700:, d2.columns.get_loc("volume")] *= 10            # будущее меняется — прошлое нет
    b = features(d2, d2["close"] * 0 + 1)
    for k in a:
        np.testing.assert_array_equal(a[k][:700], b[k][:700])
    assert np.nanmax(np.abs(a["rs7"][42:] - (d["close"] / d["close"].shift(42) - 1).to_numpy()[42:])) < 1e-12


def test_rows_are_liquid_uptrend_longs_and_masks_cover_all_variants():
    d = _frame()
    rows = coin_rows(d, np.full(len(d), 1e9), "X", None)
    assert rows and all(np.isfinite(r["R"]) and r["loc"] >= 0.5 for r in rows)
    df = pd.DataFrame(rows)
    mk = masks(df)
    assert set(mk) == set(VARIANTS)
    assert mk["none"].all() and (mk["aggr55"] <= mk["aggr52.5"]).all() and (mk["aggr52.5"] <= mk["aggr50"]).all()
    assert not coin_rows(d, np.zeros(len(d)), "X", None)


def test_absorption_means_price_up_while_sellers_hit_the_bid():
    df = pd.DataFrame({"buy": [0.5] * 3, "rel": [0.0] * 3, "vol_x": [1.0] * 3, "rng_atr": [1.0] * 3,
                       "delta20": [-0.05, 0.05, -0.05], "rs7": [0.0] * 3, "rs30": [0.0] * 3, "loc": [0.6] * 3,
                       "delta60": [-0.03, -0.03, 0.01], "ret20_atr": [1.0, 1.0, -1.0], "ret60_atr": [3.0, 1.0, 3.0]})
    mk = masks(df)
    assert mk["acc20"].tolist() == [True, False, False]
    assert mk["acc60"].tolist() == [True, True, False]
    assert mk["acc60s"].tolist() == [True, False, False]
