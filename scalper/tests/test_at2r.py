import numpy as np
import pandas as pd

from research import at2r as A
from research import exit5 as X
from research.tline import MAKER, TAKER


def _rw(seed: int, n: int = 30000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.001, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.001, n)))
    vol = np.full(n, 1000.0)
    idx = pd.date_range("2023-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": vol,
                         "taker_buy_volume": vol * rng.uniform(0.3, 0.8, n)}, index=idx)


def test_hold_matches_bot_and_rows_reached_2r(monkeypatch):
    monkeypatch.setitem(A.SLICE, "5m", ("market", False, 800))      # все сделки, чтобы на случайном ряду их хватило
    d = _rw(5)
    liq = np.full(len(d), 1e9)
    rows = pd.DataFrame(A.coin_rows(d, liq, "RW", "5m"))
    ref = pd.DataFrame(X.coin_rows(d, liq, "RW", "5m"))
    assert len(rows) > 5
    m = rows.merge(ref, on=["t", "side"])
    assert len(m) == len(rows) and (m.mfe >= 2).all()
    assert np.allclose(m["hold"].to_numpy(), m["base"].to_numpy())
    assert len(rows) <= (ref.mfe >= 2).sum()                          # +2R и 3R в одной свече — решения нет


def test_path_exit_at_close_of_2r_bar():
    # лонг 100, стоп 90: свеча 2 дошла до 121 (+2.1R), закрылась на 118, потом стоп
    c = np.array([100, 110, 118, 100, 85.0])
    o = np.r_[100, c[:-1]]
    hi = np.array([100.5, 110.5, 121.0, 118.5, 100.5])
    lo = np.minimum(o, c) - 0.5
    hold, j2, ex = A.path(o, hi, lo, c, np.zeros(5), 0, 1, 100.0, 90.0, 3.0, 2.0, 100, MAKER)
    assert j2 == 2 and hold < -1
    assert abs(ex - (1.8 - (MAKER + TAKER) * 100 / 10)) < 1e-9


def test_label_extreme_thirds():
    t = pd.date_range("2022-03-01", periods=90, freq="D")
    z = pd.DataFrame({"t": t, "speed": np.arange(90), "aggr": np.arange(90) / 90, "aggr3": 0.5, "vfade": 1.0,
                      "stretch": np.arange(90.0), "wick": "нет", "round": "нет", "swing": "нет"})
    out = A.label(z)
    assert (out.aggr == "да").sum() == 30 and (out.stretch == "да").sum() == 30
    assert out.speed.str.startswith("быстро").sum() == 30
