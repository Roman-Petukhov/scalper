import numpy as np
import pandas as pd

from research import spotrev as S


def _daily(prices: np.ndarray, start: str = "2018-01-01", vol: float = 5e6) -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(prices), freq="D", tz="UTC")
    return pd.DataFrame({"close": prices, "quote_volume": vol}, index=idx)


def test_monthly_takes_month_end_and_first_day():
    d = _daily(np.arange(1.0, 61.0))                          # янв 31 день, фев 28
    m = S.monthly(d)
    jan, feb = pd.Timestamp("2018-01-01"), pd.Timestamp("2018-02-01")
    assert m.loc[jan, "c_end"] == 31.0 and m.loc[jan, "c_d1"] == 1.0 and m.loc[feb, "c_d1"] == 32.0


def _panel(n_coins: int = 50, n_months: int = 14):
    months = pd.date_range("2019-01-01", periods=n_months, freq="MS")
    cols = [f"C{i}USDT" for i in range(n_coins)] + ["BTCUSDT"]
    c_end = pd.DataFrame(100.0, index=months, columns=cols)
    c_d1 = pd.DataFrame(100.0, index=months, columns=cols)
    adv = pd.DataFrame(5e6, index=months, columns=cols)
    meta = pd.DataFrame({"first": pd.Timestamp("2018-01-01"), "last": pd.Timestamp("2030-01-01"),
                         "last_close": 100.0}, index=cols)
    regime = pd.Series(True, index=months)
    return c_end, c_d1, adv, meta, regime


def test_losers_basket_buys_past_losers_and_measures_rebound():
    c_end, c_d1, adv, meta, regime = _panel()
    m = pd.Timestamp("2019-06-01")
    c_end.loc[m, ["C0USDT", "C1USDT", "C2USDT", "C3USDT", "C4USDT"]] = 50.0     # упали за месяц
    c_d1.loc[pd.Timestamp("2019-07-01"), ["C0USDT", "C1USDT", "C2USDT", "C3USDT", "C4USDT"]] = 50.0
    c_d1.loc[pd.Timestamp("2019-08-01"), ["C0USDT", "C1USDT", "C2USDT", "C3USDT", "C4USDT"]] = 60.0
    z = S.cohorts(c_end, c_d1, adv, meta, regime, "BTCUSDT", 1, 1, 1e6)
    row = z[z.t == pd.Timestamp("2019-07-01")].iloc[0]
    assert abs(row["проигравшие"] - 0.2) < 1e-12 and row["btc"] == 0.0 and row["n"] == 50


def test_dead_coin_exits_at_last_close_with_haircut():
    c_end, c_d1, adv, meta, _ = _panel()
    meta.loc["C0USDT", ["last", "last_close"]] = [pd.Timestamp("2019-07-15"), 10.0]
    c_d1.loc[pd.Timestamp("2019-08-01"), "C0USDT"] = np.nan
    px = S.exit_prices(c_d1, c_end, meta, pd.Timestamp("2019-08-01"))
    assert abs(px["C0USDT"] - 10.0 * (1 - S.DEAD_HAIRCUT)) < 1e-12 and px["C1USDT"] == 100.0


def test_young_and_illiquid_coins_are_excluded():
    c_end, c_d1, adv, meta, regime = _panel(n_coins=45)
    meta.loc[[f"C{i}USDT" for i in range(10)], "first"] = pd.Timestamp("2019-05-01")
    z = S.cohorts(c_end, c_d1, adv, meta, regime, "BTCUSDT", 1, 1, 1e6)
    assert z[z.t <= pd.Timestamp("2019-10-01")].empty      # 35 отобранных < MIN_ELIG, пока молодые моложе 180 дней
    assert not z[z.t >= pd.Timestamp("2019-12-01")].empty


def test_nw_t_matches_plain_t_without_lag():
    rng = np.random.default_rng(1)
    x = rng.normal(0.01, 0.05, 200)
    plain = x.mean() / (x.std() / np.sqrt(len(x)))
    assert abs(S.nw_t(x, 0) - plain) < 1e-9


def test_low_vol_basket_picks_calm_coins():
    c_end, c_d1, adv, meta, regime = _panel()
    calm = [f"C{i}USDT" for i in range(5)]
    var = pd.DataFrame(0.01, index=c_end.index, columns=c_end.columns)
    var[calm] = 0.0001
    c_d1.loc[pd.Timestamp("2019-08-01"), calm] = 110.0
    z = S.cohorts(c_end, c_d1, adv, meta, regime, "BTCUSDT", 1, 1, 1e6, {"vol6": var, "mx": var})
    row = z[z.t == pd.Timestamp("2019-07-01")].iloc[0]
    assert abs(row["низкая вола"] - 0.1) < 1e-12 and np.isnan(row["высокая вола"])   # остальные — одна связка


def test_monthly_reports_variance_and_best_day():
    p = np.r_[np.full(10, 100.0), 120.0, np.full(50, 120.0)]
    m = S.monthly(_daily(p))
    assert abs(m.iloc[0]["mx"] - 0.2) < 1e-12 and m.iloc[0]["var"] > 0 and m.iloc[1]["var"] == 0
