import numpy as np
import pandas as pd

from research import noline
from research import volstop as V


def _rw(seed: int, n: int = 15000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="4h", tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, n)))
    vol = np.full(n, 1000.0)
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": vol,
                         "taker_buy_volume": vol * rng.uniform(0.3, 0.8, n)}, index=idx)


def test_base_matches_bot_and_variants_are_finite():
    d = _rw(1)
    liq = np.full(len(d), 1e9)
    rows = pd.DataFrame(V.coin_rows(d, liq, "RW", "4h"))
    noline.TF = "4h"
    ref = noline.coin_trades(d, liq, "RW", lines_only=True)
    ref = ref[ref.entry == "retest"]
    assert len(rows) == len(ref) > 5
    assert np.allclose(rows["base"].to_numpy(), ref["R3"].to_numpy())
    assert rows[list(V.TITLES)].notna().all().all()


def test_variant_levels():
    v = V.variants(100.0, 104.0, -1, 2.0)                               # шорт, стоп бота 2 ATR
    assert v["atr15"] == (103.0, 3.0) and v["atr3"] == (106.0, 3.0)
    assert v["floor1"] == (104.0, 3.0) and v["cap2"] == (104.0, 3.0)
    assert V.variants(100.0, 101.0, -1, 2.0)["floor1"] == (102.0, 3.0)   # свинг ближе 1 ATR — отодвигаем
    assert V.variants(100.0, 106.0, -1, 2.0)["cap2"] == (104.0, 3.0)     # дальше 2 ATR — придвигаем
    assert abs(v["tp6atr"][1] - 3.0) < 1e-12                            # 6 ATR при стопе 2 ATR = 3R
