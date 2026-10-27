import json

import numpy as np
import pandas as pd

from research import flows as F


def test_eco_category_ids_and_symbols():
    cats = [{"category_id": "solana-ecosystem", "name": "Solana Ecosystem"},
            {"category_id": "binance-smart-chain", "name": "BNB Chain Ecosystem"}, {"name": "x"}, "junk"]
    assert F.eco_category_ids(cats) == {"SOLUSDT": "solana-ecosystem", "BNBUSDT": "binance-smart-chain"}
    assert F.symbols_of([{"symbol": "jup"}, {"symbol": ""}, {}]) == {"JUPUSDT"}
    assert F.symbols_of({"error": 1}) == set()


def test_ecosystems_drop_multichain_bases_and_missing():
    llama = F.llama_chains([{"symbol": "RAY", "chains": ["Solana"]}, {"symbol": "LINK", "chains": ["Solana", "BSC"]},
                            {"symbol": "-", "chains": ["Solana"]}, {"symbol": "CAKE", "chains": ["BSC"]}])
    assert llama == {"Solana": {"RAYUSDT"}, "BSC": {"CAKEUSDT"}}
    cg = {"SOLUSDT": {"JUPUSDT", "BONKUSDT", "USDTUSDT", "SOLUSDT"}, "BNBUSDT": {"USDTUSDT"},
          "ARBUSDT": {"USDTUSDT", "GMXUSDT"}}
    have = {"SOLUSDT", "BNBUSDT", "ARBUSDT", "JUPUSDT", "BONKUSDT", "RAYUSDT", "USDTUSDT", "GMXUSDT", "CAKEUSDT"}
    eco = F.ecosystems(cg, llama, have)
    assert eco == {"SOLUSDT": ["BONKUSDT", "JUPUSDT", "RAYUSDT"]}            # USDT в 3 сетях, у BNB / ARB < 3


def _panel(days=400, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=days, freq="D")
    cols = ["SOLUSDT", "AUSDT", "BUSDT", "CUSDT", "BNBUSDT", "DUSDT", "EUSDT", "FUSDT"]
    r = pd.DataFrame(rng.normal(0, 0.02, (days, len(cols))), index=idx, columns=cols)
    r.loc["2022-03-01":"2022-03-28", "SOLUSDT"] = 0.02                      # SOL обгоняет рынок
    r.loc["2022-03-29":"2022-04-25", ["AUSDT", "BUSDT", "CUSDT"]] += 0.01   # экосистема догоняет
    return r


def test_spillover_weekly_and_events():
    r = _panel()
    mkt = r.mean(axis=1) * 0
    eco = F.baskets(r, {"SOLUSDT": ["AUSDT", "BUSDT", "CUSDT"], "BNBUSDT": ["DUSDT", "EUSDT", "FUSDT"]})
    ev, plc = F.spill_events(r, mkt, eco)
    first = ev[ev.sector == "SOLUSDT"].iloc[0]
    assert pd.Timestamp("2022-03-20") <= first.t <= pd.Timestamp("2022-03-29") and len(plc) == len(ev)
    assert (ev.groupby("sector").t.diff().dropna().dt.days >= F.HOLD).all()
    assert F.window(pd.Series(0.01, index=r.index), r.index[0], 10) is not None


def test_views_and_attention():
    raw = {"items": [{"timestamp": "2024010100", "views": 10}, {"timestamp": "2024010200", "views": 30}]}
    assert list(F.parse_views(raw)) == [10.0, 30.0]
    assert F.parse_views(None).empty
    idx = pd.date_range("2023-01-01", periods=200, freq="D")
    views = {}
    for sector, pages in F.WIKI_PAGES.items():
        for p in pages[:2]:
            v = pd.Series(100.0, index=idx)
            if sector == "мемы":
                v.iloc[-7:] = 1000.0
            views[p] = v
    att = F.attention_score(views)
    assert att.iloc[-1].idxmax() == "мемы" and abs(att.iloc[100]).max() < 1e-12


def test_fee_frame_categories_and_growth():
    raw = {"protocols": [{"name": "Uni", "displayName": "Uniswap", "category": "Dexs"},
                         {"name": "Aave", "category": "Lending"}, {"name": "Tether", "category": "Stablecoin Issuer"}],
           "totalDataChartBreakdown": [[1700000000 + 86400 * i, {"Uniswap": 1e5 * (1 + (i > 120)), "Aave": 2e4,
                                                                 "Tether": 9e9}] for i in range(160)]}
    df, cat = F.fee_frame(json.loads(json.dumps(raw)))
    assert df.shape == (160, 3) and cat["Uniswap"] == "Dexs"
    cf = F.category_fees(df, cat)
    assert set(cf.columns) == {"Dexs", "Lending"}
    g = F.fee_growth(cf)
    assert g["Dexs"].iloc[-1] > 0.5 and np.isnan(g["Lending"].iloc[-1])        # Lending < $1M за 30 дней
    toks = F.category_tokens([{"category": "Dexs", "symbol": "UNI"}, {"category": "CEX", "symbol": "BNB"},
                              {"category": "Dexs", "symbol": "-"}], {"UNIUSDT", "BNBUSDT"})
    assert toks == {"Dexs": ["UNIUSDT"]}


def test_pick_every_respects_hold():
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    score = pd.DataFrame({"A": 1.0, "B": 0.0, "C": -1.0}, index=idx)
    ex = pd.DataFrame(0.001, index=idx, columns=["A", "B", "C"])
    z = F.pick_every(score, ex, 28)
    assert set(z.sector) == {"A"} and (z.t.diff().dropna().dt.days >= 28).all()
