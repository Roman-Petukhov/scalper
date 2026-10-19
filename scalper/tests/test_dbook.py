import numpy as np
import pandas as pd
import pytest

from research.dbook import Policy, journey, simulate


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


def test_usd_min_orders_and_switch():
    # $100, риск 4%, стоп 10% (lr 10): номинал $40, бета 1 → хедж $40 при BTC $50k = 0.0008 → округляется до 0.001
    h = {"acct_h": 0.5, "acct_beta": 1.0, "acct_btc": 50_000.0}
    r = simulate(_tr([("2024-01-01", "2024-01-05", 1.0, h)]), Policy("x", risk=0.04, usd=100.0))
    assert r.final == pytest.approx(1 + 0.04 * (1 + 0.5 * 50 / 40))
    small = {"acct_h": 0.5, "acct_beta": 1.0, "acct_btc": 200_000.0}       # 0.0002 BTC — меньше шага: без хеджа
    assert simulate(_tr([("2024-01-01", "2024-01-05", 1.0, small)]), Policy("x", risk=0.04, usd=100.0)).final == \
        pytest.approx(1.04)
    tiny = _tr([("2024-01-01", "2024-01-05", 1.0, {})])                     # номинал $4 < $5
    assert simulate(tiny, Policy("x", risk=0.04, usd=10.0)).skip_min == 1
    sw = Policy("x", risk=0.04, usd=100.0, switch_usd=500.0, low_risk=0.03)
    assert simulate(_tr([("2024-01-01", "2024-01-05", 1.0, h)]), sw).final == pytest.approx(1.03)  # 3%, без хеджа


def test_journey():
    rows = [(f"2024-{m:02d}-01", f"2024-{m:02d}-03", 3.0, {}) for m in range(1, 13)]
    rows += [(f"2025-{m:02d}-01", f"2025-{m:02d}-03", 3.0, {}) for m in range(1, 13)]
    j = journey(_tr(rows), Policy("x", risk=0.06, usd=100.0), targets=(150.0,))
    assert j.starts == 12 and j.reach[150.0][0] == 1.0 and j.below_floor == 0.0   # +18% в месяц: $150 за 3 месяца
    assert j.reach[150.0][3] == pytest.approx(64, abs=3)


def test_journey_sparse_history():
    rows = [("2024-01-01", "2024-01-03", 1.0, {}), ("2026-01-01", "2026-01-03", 1.0, {})]
    j = journey(_tr(rows), Policy("x", usd=100.0), targets=(1000.0,))
    assert j.starts == 13 and j.reach[1000.0][1] == 0.0                # старты без сделок за год — счёт $100


def test_journey_horizon_only_counts_paths_with_enough_history():
    rows = [(f"{y}-{m:02d}-01", f"{y}-{m:02d}-03", 3.0, {}) for y in (2024, 2025, 2026) for m in range(1, 13)]
    j = journey(_tr(rows), Policy("x", risk=0.06, usd=100.0), targets=(1e9,))
    assert j.starts == 24 and j.paths_2y == 12 and j.paths_3y == 0 and np.isnan(j.reach[1e9][2])
