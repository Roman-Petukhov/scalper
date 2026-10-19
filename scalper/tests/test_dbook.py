import numpy as np
import pandas as pd
import pytest

from research.dbook import Policy, simulate


def _tr(rows):
    out = []
    for i, (fill, exit_, r, kw) in enumerate(rows):
        base = {"symbol": f"C{i}", "t": pd.Timestamp(fill, tz="UTC") - pd.Timedelta(days=1), "side": -1, "aggr": 0.5,
                "acct_fill": pd.Timestamp(fill, tz="UTC"), "acct_exit": pd.Timestamp(exit_, tz="UTC"), "acct_r": r,
                "acct_r0": r, "acct_h": 0.0, "acct_add": pd.NaT, "acct_addpx": np.nan, "acct_beta": np.nan,
                "acct_lr": 10.0}
        out.append(base | kw)
    return pd.DataFrame(out)


def test_compounding_risk_and_drawdown():
    t = _tr([("2024-01-01", "2024-01-05", 3.0, {}), ("2024-01-10", "2024-01-12", -1.0, {}),
             ("2024-02-01", "2024-02-03", -1.0, {})])
    r = simulate(t, Policy("x"))
    assert r.taken == 3 and r.final == pytest.approx(1.18 * 0.94 * 0.94)            # риск — от текущего капитала
    assert r.max_dd == pytest.approx(1 - 0.94 * 0.94) and r.worst_month == pytest.approx(0.94 - 1)


def test_cap_margin_and_add():
    same = [("2024-01-01", "2024-01-20", -1.0, {}) for _ in range(5)]
    assert simulate(_tr(same), Policy("x", max_open=3)).skip_cap == 2
    # стоп 1% цены: номинал 6 капиталов, маржа 0.5; вторая урезана до 95% свободной (5.7), третья — до 0.285
    tight = _tr([("2024-01-01", "2024-01-20", -1.0, {"acct_lr": 100.0}) for _ in range(3)])
    r = simulate(tight, Policy("x"))
    assert r.taken == 3 and r.final == pytest.approx(1 - 0.06 - 0.057 - 0.00285)
    full = _tr([("2024-01-01", "2024-01-20", -1.0, {"acct_lr": 100.0})] * 2
               + [("2024-01-02", "2024-01-20", -1.0, {"acct_lr": 100.0, "acct_beta": 1e9})])
    assert simulate(full, Policy("x")).skip_margin == 0                       # бета огромная → номинал крошечный
    add = _tr([("2024-01-01", "2024-01-20", 3.7, {"acct_r0": 3.0, "acct_add": pd.Timestamp("2024-01-05", tz="UTC"),
                                                  "acct_addpx": 1.05})])
    assert simulate(add, Policy("x")).final == pytest.approx(1 + 3.7 * 0.06)
    hedged = _tr([("2024-01-01", "2024-01-20", 1.0, {"acct_h": 0.5, "acct_beta": 1.2})])
    assert simulate(hedged, Policy("x")).final == pytest.approx(1 + 1.5 * 0.06)


def test_dd_risk_cut():
    losses = [(f"2024-01-{d:02d}", f"2024-01-{d:02d}", -1.0, {}) for d in range(1, 5)]
    after = [("2024-01-10", "2024-01-11", -1.0, {})]
    r6 = simulate(_tr(losses + after), Policy("x"))
    r3 = simulate(_tr(losses + after), Policy("x", dd_cut=0.15, dd_risk=0.03))
    assert r6.final == pytest.approx(0.94 ** 5) and r3.final == pytest.approx(0.94 ** 3 * 0.97 ** 2)


def test_side_limits():
    longs = [("2024-01-01", "2024-01-20", -1.0, {"side": 1}) for _ in range(3)]
    shorts = [("2024-01-01", "2024-01-20", -1.0, {}) for _ in range(2)]
    t = _tr(longs + shorts)
    assert simulate(t, Policy("x", side_max=2)).skip_cap == 1                 # третий лонг; шорты не мешают
    assert simulate(t, Policy("x", max_open=2)).skip_cap == 3
    # риск по очереди: лонги 6, 3, 2%, шорты 6, 3%
    assert simulate(t, Policy("x", decay=True)).final == pytest.approx(1 - 0.06 - 0.03 - 0.02 - 0.06 - 0.03)
    # чистый риск ≤ 12%: лонги 6, 6, третий пропущен (остаток 0); шорты 6, 6 — чистый риск падает
    r = simulate(t, Policy("x", net_cap=0.12))
    assert r.skip_cap == 1 and r.final == pytest.approx(1 - 4 * 0.06)
    # ≤ 9%: второй лонг урезан до 3%, третий пропущен (остаток 0 < 1.5%)
    r = simulate(_tr(longs), Policy("x", net_cap=0.09))
    assert r.skip_cap == 1 and r.final == pytest.approx(1 - 0.06 - 0.03)


def test_unhedged_and_month_stats():
    hedged = _tr([("2024-01-01", "2024-01-20", 1.0, {"acct_h": 0.5, "acct_beta": 1.2})])
    assert simulate(hedged, Policy("x", hedge=False)).final == pytest.approx(1.06)
    t = _tr([("2024-01-01", "2024-01-05", -1.0, {}), ("2024-02-01", "2024-02-05", -1.0, {}),
             ("2024-03-01", "2024-03-05", 3.0, {}), ("2024-04-01", "2024-04-05", 1.0, {})])
    r = simulate(t, Policy("x"))
    assert r.neg_months == pytest.approx(0.5) and r.median_month == pytest.approx(0.0)     # −6, −6, +18, +6%
    assert r.underwater_days == (pd.Timestamp("2024-03-05") - pd.Timestamp("2023-12-31")).days
