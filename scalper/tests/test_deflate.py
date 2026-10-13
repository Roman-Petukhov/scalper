import numpy as np
import pandas as pd

from research.deflate import expected_max_sr, pbo, psr, variant_trades


def test_psr_and_expected_max():
    rng = np.random.default_rng(0)
    good = rng.normal(0.03, 0.05, 120)
    assert psr(good, 0.0) > 0.99 and psr(-good, 0.0) < 0.01
    assert expected_max_sr(0.01, 1000) > expected_max_sr(0.01, 10) > 0
    assert psr(good, expected_max_sr(0.01, 1000)) < psr(good, 0.0)


def test_pbo_low_for_one_real_edge_and_high_for_noise():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, (96, 30))
    p_noise, _ = pbo(noise)
    real = noise.copy()
    real[:, 0] += 1.0
    p_real, w = pbo(real)
    assert p_real < 0.05 and w > 0.9 and 0.3 < p_noise < 0.7


def test_hybrid_picks_market_for_long_candles():
    t = pd.Timestamp("2023-01-01", tz="UTC")
    rows = []
    for i, rng_atr in enumerate((1.0, 3.0)):
        for entry, r in (("market", 1.0), ("retest", -1.0)):
            rows.append(dict(symbol=f"S{i}", t=t, side=-1, adv=1e8, with_trend=True, aggr=0.6, close_loc=0.7,
                             rng_atr=rng_atr, entry=entry, R3=r))
    df = pd.DataFrame(rows)
    x = variant_trades(df, 0.55, 0.5, True, 2.15)
    assert sorted(zip(x.symbol, x.entry)) == [("S0", "retest"), ("S1", "market")]
    assert len(variant_trades(df, 0.65, 0.5, True, "market")) == 0
