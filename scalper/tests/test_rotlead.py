import numpy as np
import pandas as pd

from research import rotlead as R
from research.sectors import sector_map


def _sector_cols(n: int = 6) -> dict[str, list[str]]:
    by: dict[str, list[str]] = {}
    for sym, name in sector_map().items():
        by.setdefault(name, []).append(sym)
    return {k: v[:n] for k, v in by.items() if len(v) >= n}


def test_supply_parsing_and_chain_matching():
    raw = [{"date": "1704067200", "totalCirculatingUSD": {"peggedUSD": 100.0, "peggedEUR": 5.0}},
           {"date": "1704153600", "totalCirculatingUSD": {"peggedUSD": 110.0}}, {"junk": 1}, "x"]
    s = R.parse_supply(raw)
    assert list(s.index) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")] and list(s) == [105.0, 110.0]
    assert R.parse_supply(None).empty
    listed = [{"name": "Solana"}, {"name": "Optimism"}, {"name": "BSC"}, {"name": "Hyperliquid"}]
    got = R.match_chains({"SOLUSDT": "Solana", "OPUSDT": "OP Mainnet", "HYPEUSDT": "Hyperliquid L1", "XUSDT": "Nope"}, listed)
    assert got == {"SOLUSDT": "Solana", "OPUSDT": "Optimism", "HYPEUSDT": "Hyperliquid"}
    assert R.match_chains({"SOLUSDT": "Solana"}, None) == {}


def test_supply_growth_needs_base_and_pick_top_takes_growers():
    idx = pd.date_range("2024-01-01", periods=70, freq="D")                     # понедельник
    names = [f"C{i}USDT" for i in range(7)]
    supply = pd.DataFrame({c: 1e9 for c in names}, index=idx)
    supply["C0USDT"] = np.linspace(1e9, 3e9, 70)
    supply["C1USDT"] = np.linspace(1e9, 2e9, 70)
    supply["C2USDT"] = np.linspace(1e9, 1.5e9, 70)
    supply["C6USDT"] = np.linspace(1e6, 9e6, 70)                                # база меньше порога — не считается
    g = R.supply_growth(supply, 28)
    assert g["C6USDT"].isna().all() and g["C0USDT"].iloc[-1] > g["C1USDT"].iloc[-1] > 0
    r = pd.DataFrame(0.0, index=idx, columns=names)
    r[["C0USDT", "C1USDT", "C2USDT"]] = 0.01
    mkt = pd.Series(0.0, index=idx)
    z = R.pick_top(g, r, mkt)
    assert len(z) >= 4 and np.allclose(z.X, 1.01 ** 7 - 1 - R.COST) and set(z.sector) == {"C0"}
    assert R.pick_top(g[names[:4]], r, mkt).empty                              # сетей меньше MIN_CHAINS


def test_residual_removes_beta():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2023-01-01", periods=300, freq="D")
    mkt = pd.Series(rng.normal(0, 0.03, 300), index=idx)
    sec = pd.DataFrame({"hi": 2.0 * mkt + rng.normal(0, 0.002, 300), "lo": 0.5 * mkt + rng.normal(0, 0.002, 300)})
    res = R.residual(sec, mkt)
    assert res.iloc[:60].isna().all().all()
    tail = res.iloc[120:]
    assert abs(tail["hi"].corr(mkt.iloc[120:])) < 0.25 and tail["hi"].std() < 0.01
    rm = R.residual_momentum(res)
    assert rm.iloc[-1].notna().all()


def test_rs_high_and_fresh_breakout():
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    up = pd.Series(0.002, index=idx)
    flat = pd.Series(np.where(np.arange(200) == 150, 0.08, np.where(np.arange(200) < 100, 0.001, -0.001)), index=idx)
    ex = pd.DataFrame({"up": up, "flat": flat})
    high = R.rs_high(ex)
    assert high["up"].iloc[60:].all() and not high["flat"].iloc[120:150].any() and high["flat"].iloc[150]
    fresh = R.fresh_high(high)
    assert not fresh["up"].iloc[100:].any() and fresh["flat"].iloc[150] and not fresh["flat"].iloc[151]
    z = R.pick_flagged(high, ex)
    assert len(z) and (z.X > -R.COST - 0.02).all()


def test_inside_picks_leaders_and_laggards():
    secs = _sector_cols()
    name = next(iter(secs))
    cols = secs[name]
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    step = np.linspace(0.02, -0.02, len(cols))                                  # первая монета — лидер
    close = pd.DataFrame({c: 100 * np.cumprod(np.full(40, 1 + s)) for c, s in zip(cols, step)}, index=idx)
    r = close.pct_change()
    mkt = pd.Series(0.0, index=idx)
    z = pd.DataFrame({"t": [pd.Timestamp("2024-01-22")], "sector": [name]})
    lead, lag = R.inside(z, close, r, mkt, True), R.inside(z, close, r, mkt, False)
    assert len(lead) == 1 and len(lag) == 1 and lead.X.iloc[0] > 0 > lag.X.iloc[0]
    assert R.inside(z, close[cols[:4]], r[cols[:4]], mkt, True).empty          # монет меньше 2n


def test_regime_flags_and_coinbase_parsing():
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    cols = [f"A{i}USDT" for i in range(30)]
    close = pd.DataFrame({c: np.linspace(100, 200, 120) for c in cols}, index=idx)
    for c in cols[:10]:
        close[c] = np.linspace(200, 100, 120)
    b = R.alt_breadth(close, close.notna())
    assert np.isclose(b.iloc[-1], 20 / 30)
    mkt = pd.Series(0.01, index=idx)
    btc = pd.Series(np.linspace(100, 101, 120), index=idx)
    assert R.alts_beat_btc(mkt, btc).iloc[-1]
    s = R.parse_candles([[1704153600, 9, 11, 10, 10.5, 2000.0], [1704067200, 9, 11, 10, 10.0, 1000.0], ["bad"]])
    assert list(s) == [10000.0, 21000.0] and R.parse_candles({"message": "NotFound"}).empty


def test_stable_flag_and_placebo():
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    total = pd.Series(np.r_[np.full(100, 1e11), np.linspace(1e11, 1.2e11, 100)], index=idx)
    f = R.stable_flag(total, 28, 0.01)
    assert not f.iloc[:100].any() and f.iloc[-1]
    mondays = idx[idx.day_of_week == 0]
    z = pd.DataFrame({"t": mondays, "X": np.where(mondays > idx[106], 0.05, -0.01)})
    p, m = R.placebo_p(z, f)
    assert m > 0.03 and p < 0.01


def test_robust_g1_tolerates_no_trades(capsys):
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    total = pd.Series(1e11, index=idx)
    R.robust_g1(pd.DataFrame(), pd.DataFrame(), total, pd.Series(True, index=idx))
    z = pd.DataFrame({"t": [idx[7]], "X": [0.01]})
    R.robust_g1(z, z, total, pd.Series(True, index=idx))
    assert "N7" in capsys.readouterr().out
