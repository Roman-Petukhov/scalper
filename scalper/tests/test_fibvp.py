import numpy as np
import pandas as pd

import research.fibvp as F


def test_value_area_finds_the_volume_peak_and_covers_70pct():
    rng = np.random.default_rng(0)
    price = np.r_[rng.normal(100, 1, 900), rng.uniform(80, 120, 100)]
    vol = np.ones(len(price))
    poc, val, vah = F.value_area(price, vol)
    assert abs(poc - 100) < 1.5 and val < poc < vah
    inside = ((price >= val) & (price <= vah)).mean()
    assert 0.7 <= inside <= 0.85


def _frames(n=3000, seed=3):
    rng = np.random.default_rng(seed)
    t1 = pd.date_range("2022-01-01", periods=n * 4, freq="1h", tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.006, len(t1))))
    o = np.r_[c[0], c[:-1]]
    v = rng.uniform(1, 10, len(t1))
    h1 = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.002, "low": np.minimum(o, c) * 0.998, "close": c,
                       "volume": v, "taker_buy_volume": v * rng.uniform(0.3, 0.7, len(t1)), "funding": 0.0}, index=t1)
    d = h1.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
                              "taker_buy_volume": "sum", "funding": "sum"})
    return d, h1


def test_rows_have_all_variants_and_profile_labels(monkeypatch):
    d, h1 = _frames()
    rows = F.coin_rows(d, h1, np.full(len(d), 1e9), "X")
    assert rows, "на случайном блуждании должны быть пробои с фильтрами"
    z = pd.DataFrame(rows)
    for col in (*F.VARIANTS, "v1", "v2", "ext_k"):
        assert col in z
    assert z.ext_k.between(1, 10).all() and z.vp_ok.mean() > 0.9
    # вход на откате 0.618 глубже, чем 0.382: исполняется не чаще
    assert z["fib0.618"].notna().sum() <= z["fib0.382"].notna().sum()
    p = "/tmp/claude-0/-home-user-scalper/a5f83980-bd41-5a56-bbde-c47eae929e11/scratchpad/fibvp_fake.parquet"
    pd.concat([z.assign(t=z.t + pd.DateOffset(years=k)) for k in range(-2, 4)]).to_parquet(p)
    monkeypatch.setattr(F, "all_parts", lambda name: [p])
    F.report()
