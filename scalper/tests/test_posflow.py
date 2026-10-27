import numpy as np
import pandas as pd

from research import posflow as P
from research.sectors import sector_map


def _sector_cols(n: int = 4) -> dict[str, list[str]]:
    by: dict[str, list[str]] = {}
    for sym, name in sector_map().items():
        by.setdefault(name, []).append(sym)
    return {k: v[:n] for k, v in by.items() if len(v) >= n}


def test_sector_score_is_relative_and_needs_three_coins():
    secs = _sector_cols()
    names = list(secs)[:2]
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    cols = secs[names[0]] + secs[names[1]]
    coin = pd.DataFrame(0.0, index=idx, columns=cols)
    coin[secs[names[0]]] = 1.0
    live = coin.notna()
    s = P.sector_score(coin, live, names)
    assert np.allclose(s[names[0]], 0.5) and np.allclose(s[names[1]], -0.5)
    live.loc[:, secs[names[1]][:2]] = False                                  # осталось 2 монеты — нет балла
    assert P.sector_score(coin, live, names)[names[1]].isna().all()


def test_oi_funding_top_taker_transforms():
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    oi = pd.DataFrame({"X": np.r_[np.full(60, 100.0), np.full(60, 200.0)]}, index=idx)
    g = P.oi_growth(oi)
    assert np.isclose(g["X"].iloc[63], np.log(2)) and np.isclose(g["X"].iloc[100], 0.0)
    fund = pd.DataFrame({"X": 0.0001}, index=idx)
    assert np.isclose(P.funding_7d(fund, fund).iloc[-1, 0], 0.0007)
    top = pd.DataFrame({"X": np.r_[np.ones(100), np.full(20, np.e)]}, index=idx)
    assert P.top_tilt(top)["X"].iloc[106] > 0.9 and abs(P.top_tilt(top)["X"].iloc[90]) < 1e-9
    qv = pd.DataFrame({"X": 100.0}, index=idx)
    tb = pd.DataFrame({"X": np.r_[np.full(110, 50.0), np.full(10, 70.0)]}, index=idx)
    t = P.taker_tilt(tb, qv)["X"]
    assert abs(t.iloc[100]) < 1e-9 and t.iloc[-1] > 0.5


def test_composite_needs_two_parts():
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    a = pd.DataFrame({"s1": [1.0, 1.0], "s2": [0.0, 0.0], "s3": [-1.0, -1.0]}, index=idx)
    b = pd.DataFrame({"s1": [np.nan, 2.0], "s2": [np.nan, 1.0]}, index=idx)
    c = P.composite([a, b])
    assert c.iloc[0].isna().all()
    assert c.iloc[1].idxmax() == "s1" and np.isnan(c.iloc[1]["s3"])


def test_btc_bull_regime_and_filter():
    idx = pd.date_range("2022-01-03", periods=600, freq="D")
    close = pd.Series(np.r_[np.linspace(100, 50, 400), np.linspace(50, 150, 200)], index=idx)
    bull = P.btc_bull(close)
    assert bull.loc["2022-12-01"] == 0.0 and bull.iloc[-1] == 1.0
    z = pd.DataFrame({"t": [pd.Timestamp("2022-12-05"), idx[-1].normalize() - pd.Timedelta(days=int(idx[-1].dayofweek))],
                      "X": [0.01, 0.02]})
    got = P.in_regime(z, bull > 0.5)
    assert list(got.X) == [0.02]


def test_fear_greed_and_upbit_share():
    raw = {"data": [{"value": "70", "timestamp": "1704153600"}, {"value": "20", "timestamp": "1704067200"}]}
    fg = P.fear_greed(raw)
    assert list(fg) == [20.0, 70.0] and P.fear_greed({"x": 1}).empty
    secs = _sector_cols(2)
    name = next(iter(secs))
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    krw = pd.DataFrame({"BTCUSDT": 1000.0, "ZZZUSDT": 100.0, **{c: 10.0 for c in secs[name]}}, index=idx)
    krw.iloc[-7:, [krw.columns.get_loc(c) for c in secs[name]]] = 40.0
    s = P.upbit_share_score(krw, [name], idx)
    assert s[name].iloc[-1] > 0.5 and abs(s[name].iloc[100]) < 1e-9
