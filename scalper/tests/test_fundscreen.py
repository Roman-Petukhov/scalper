import numpy as np
import pandas as pd
import pytest

from research import fundscreen as F


def _ov():
    fees = [{"name": "Uni V3", "parentProtocol": "parent#uniswap", "category": "Dexs", "total30d": 90e6, "total1y": 900e6},
            {"name": "Uni V2", "parentProtocol": "parent#uniswap", "category": "Dexs", "total30d": 10e6, "total1y": 100e6},
            {"name": "Lend", "slug": "lend", "category": "Lending", "total30d": 3e6, "total1y": 24e6},
            {"name": "Junk", "slug": "junk", "category": "Dexs", "total30d": 1e6, "total1y": 12e6}]
    rev = [dict(p, total30d=p["total30d"] * 0.3, total1y=p["total1y"] * 0.3) for p in fees]
    hr = [dict(fees[0], total30d=2e6, total1y=12e6), dict(fees[2], total30d=1e6, total1y=6e6),
          dict(fees[3], total30d=0.5e6, total1y=6e6)]
    return {"dailyFees": fees, "dailyRevenue": rev, "dailyHoldersRevenue": hr}


def test_group_key_merges_parent_versions():
    assert F.group_key({"parentProtocol": "parent#uniswap", "slug": "uniswap-v3"}) == "uniswap"
    assert F.group_key({"name": "Some Lend"}) == "some-lend"


def test_fees_table_sums_versions_of_one_business():
    t = F.fees_table(_ov()).set_index("group")
    assert t.loc["uniswap", "dailyFees_d30"] == pytest.approx(100e6)
    assert t.loc["uniswap", "dailyHoldersRevenue_d30"] == pytest.approx(2e6)
    assert t.loc["uniswap", "name"] == "Uni V3"
    assert t.loc["lend", "dailyRevenue_y1"] == pytest.approx(24e6 * 0.3)


def test_growth_and_unlock_share():
    assert F.growth(2.0, 12.0) == pytest.approx(2.0)
    assert np.isnan(F.growth(1.0, 0.0))
    idx = pd.date_range("2026-01-01", periods=800, freq="D", tz="UTC")
    cum = pd.Series(np.linspace(100, 300, 800), index=idx)
    now = idx[100]
    a, b = cum.iloc[100], cum.iloc[465]
    assert F.unlock_share(cum, now) == pytest.approx((b - a) / a)
    assert np.isnan(F.unlock_share(cum, idx[0] - pd.Timedelta(days=5)))


def test_screen_filters_and_ranks():
    fees = F.fees_table(_ov())
    gecko = {"uniswap": "uniswap", "lend": "lend-token", "junk": "junk-token"}
    mkt = pd.DataFrame({"gecko_id": ["uniswap", "lend-token", "junk-token"], "symbol": ["uni", "lnd", "jnk"],
                        "mcap": [5e9, 100e6, 50e6], "fdv": [6e9, 120e6, 500e6], "volume": [1e8, 5e6, 3e6],
                        "chg_1y": [10.0, -50.0, 0.0], "ath_chg": [-70.0, -90.0, -20.0]})
    z = F.screen(fees, gecko, mkt, {"lend-token": 0.05}).set_index("gecko_id")
    assert z.loc["junk-token", "why"] == "мало в обращении"
    assert z.loc["uniswap", "why"] == "" and z.loc["lend-token", "why"] == ""
    # lend: FDV 120M / (1M*365/30) ≈ 9.9x и рост 2.0 — дешевле и быстрее, чем uniswap (FDV/HR ≈ 246x, рост 2.0)
    assert z.loc["lend-token", "score"] > z.loc["uniswap", "score"]
    assert F.show(z.reset_index()).shape[0] == 3


def test_screen_flags_heavy_unlocks():
    fees = F.fees_table(_ov())
    mkt = pd.DataFrame({"gecko_id": ["lend-token"], "symbol": ["lnd"], "mcap": [100e6], "fdv": [120e6],
                        "volume": [5e6], "chg_1y": [0.0], "ath_chg": [0.0]})
    z = F.screen(fees, {"lend": "lend-token"}, mkt, {"lend-token": 0.4})
    assert z.iloc[0]["why"] == "разлоки" and np.isnan(z.iloc[0]["score"])
