import numpy as np
import pandas as pd

from research import metaroll as R


def test_breadth_counts_other_coins_same_bar():
    t = pd.to_datetime(["2022-01-01 00:00"] * 4 + ["2022-01-01 04:00"])
    z = pd.DataFrame({"t": t, "side": [1, 1, 1, -1, 1], "fund_day": [0.001, 0.001, 0.001, -0.002, 0.0]})
    out = R.add_regime(z)
    assert list(out.breadth) == [1, 1, 1, -3, 0]                       # два других вверх минус один вниз; и т. д.
    assert out.mkt_funding.isna().all()                                # только прошлые дни: в первый день — пусто


def test_roll_trains_only_on_past_with_purge():
    import lightgbm as lgb
    rng = np.random.default_rng(0)
    n = 6000
    t = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.sort(rng.uniform(0, 3 * 365, n)), unit="D")
    z = pd.DataFrame({"t": t, "R": rng.normal(0, 1, n), "bot": rng.random(n) < 0.3, "side": -1})
    for f in R.ALL_FEATS:
        if f not in z:
            z[f] = rng.normal(size=n)
    seen = []
    orig = lgb.train

    def spy(p, ds, num_boost_round):
        seen.append(len(ds.get_label()) if ds.get_label() is not None else 0)
        return orig(p, ds, num_boost_round=num_boost_round)

    lgb_mod = type("L", (), {"train": staticmethod(spy), "Dataset": lgb.Dataset})
    r = R.roll(z, "4h", 6, lgb_mod)
    assert r.modeled.any() and seen
    q = r.q.iloc[-1]
    tr = z[(z.t >= q - pd.DateOffset(months=6)) & (z.t < q - R.PURGE["4h"])]
    assert seen[-1] == len(tr)
    assert r.t.min() >= pd.Timestamp(R.FIRST)
