import numpy as np
import pandas as pd

from research.flatsell import GROUPS, coin_rows, first_break


def test_first_break_reports_which_side_closes_out_first():
    c = np.array([10, 10.5, 9.8, 11.2, 8.0])
    assert first_break(c, 0, 11.0, 9.0) == 1
    assert first_break(np.array([10, 9.5, 8.9, 12.0]), 0, 11.0, 9.0) == -1
    assert first_break(np.array([10, 10.2, 9.9]), 0, 11.0, 9.0) == 0


def _flat(n=900, sell_from=700, seed=1):
    idx = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 0.3, n))
    c[sell_from:] = c[sell_from] + rng.normal(0, 0.2, n - sell_from)    # боковик
    o = np.r_[c[0], c[:-1]]
    v = np.full(n, 1e5)
    buy = rng.uniform(0.48, 0.52, n) * v
    buy[sell_from:] = 0.40 * v[sell_from:]                              # продавцы давят
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + 0.3, "low": np.minimum(o, c) - 0.3, "close": c,
                         "volume": v, "taker_buy_volume": buy}, index=idx)


def test_selling_into_a_flat_range_is_labelled_and_events_do_not_overlap():
    d = _flat()
    rows = pd.DataFrame(coin_rows(d, np.full(len(d), 1e9), "X", d["close"]))
    assert set(rows.group) <= set(GROUPS)
    late = rows[(rows.t >= d.index[760])]
    assert (late.group == "продавцы").any()
    for n, g in rows.groupby("n"):
        gaps = np.diff(d.index.get_indexer(g.t.sort_values()))
        assert (gaps >= n).all()
    assert np.allclose(rows["x1д"].dropna(), 0.0)                       # BTC = сама монета → сверх BTC ноль
    assert not coin_rows(d, np.zeros(len(d)), "X", None)


def test_factor_labels_and_score():
    d = _flat()
    d["funding"] = -1e-4
    rows = pd.DataFrame(coin_rows(d, np.full(len(d), 1e9), "X", d["close"]))
    assert rows["фандинг"].all()
    assert (rows["балл"] == rows[["тренд", "btc", "фандинг", "минимумы"]].sum(axis=1)).all()
