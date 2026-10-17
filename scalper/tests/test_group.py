import numpy as np
import pandas as pd

from research.group import known_day, peer_shares, share, trend_state, twins


def _close(days: int = 200, seed: int = 0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=days, freq="1D", tz="UTC")
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(0, 0.03, days))
    cols = {
        "DOGEUSDT": base + rng.normal(0, 0.003, days),        # мем, близнец SHIB
        "SHIBUSDT": base + rng.normal(0, 0.003, days),
        "PEPEUSDT": np.linspace(0, 1.0, days),                # мемы в росте
        "WIFUSDT": np.linspace(0, 0.8, days),
        "BONKUSDT": np.linspace(0, -0.5, days),               # мем в падении
        "UNIUSDT": np.linspace(0, -1.0, days),                # DeFi в падении
        "BTCUSDT": np.linspace(0, 2.0, days),                 # BTC в соседи не берём
    }
    return pd.DataFrame({k: 10 * np.exp(v) for k, v in cols.items()}, index=idx)


def test_known_day_is_last_closed_daily_bar():
    assert known_day(pd.Timestamp("2024-03-05 20:00", tz="UTC")) == pd.Timestamp("2024-03-05", tz="UTC")
    assert known_day(pd.Timestamp("2024-03-05 16:00", tz="UTC")) == pd.Timestamp("2024-03-04", tz="UTC")


def test_trend_state_and_share():
    st = trend_state(_close())
    assert st.iloc[:49].isna().all().all()
    last = st.iloc[-1]
    assert last["PEPEUSDT"] == 1 and last["UNIUSDT"] == -1
    assert share(last, ["PEPEUSDT", "WIFUSDT", "BONKUSDT"], 1) == 2 / 3
    assert share(last, ["PEPEUSDT", "WIFUSDT", "BONKUSDT"], -1) == 1 / 3
    assert np.isnan(share(last, ["PEPEUSDT", "WIFUSDT"], 1))                  # меньше трёх соседей


def test_twins_pick_correlated_coin_only_from_past():
    c = _close()
    ret = c.pct_change(fill_method=None)
    month = pd.Timestamp("2024-06-01", tz="UTC")
    pool = list(c.columns)
    assert twins(ret, month, pool, k=1)["DOGEUSDT"] == ["SHIBUSDT"]
    assert twins(ret, pd.Timestamp("2024-02-01", tz="UTC"), pool) == {}     # истории меньше 60 дней
    c2 = c.copy()
    c2.loc["2024-06-01":, "PEPEUSDT"] = c2.loc["2024-06-01":, "DOGEUSDT"] * 3  # будущее не влияет
    assert twins(c2.pct_change(fill_method=None), month, pool, k=1)["DOGEUSDT"] == ["SHIBUSDT"]


def test_peer_shares_sector_market_and_liquidity():
    c = _close()
    adv = pd.DataFrame(1e8, index=c.index, columns=c.columns)
    adv["WIFUSDT"] = 1e6                                                       # мелкая — не сосед
    t = c.index[-1] - pd.Timedelta(hours=4)                                   # свеча закрывается в полночь
    sig = pd.DataFrame({"symbol": ["DOGEUSDT"], "t": [t], "side": [1], "R": [1.0], "aggr": [True]})
    z = peer_shares(sig, c, adv).iloc[0]
    st = trend_state(c).loc[c.index[-2]]                                   # последний закрытый день
    mem = [s for s in ("SHIBUSDT", "PEPEUSDT", "BONKUSDT")]
    assert z.sector == share(st, mem, 1)
    assert z.market == share(st, [*mem, "UNIUSDT"], 1)                        # без BTC, без самой монеты
