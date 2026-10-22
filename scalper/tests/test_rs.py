import numpy as np
import pandas as pd
import pytest

from research import rs as S


def _rw(seed: int, n: int = 30000, freq: str = "15min") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq=freq, tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.002, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.002, n)))
    vol = np.full(n, 1000.0)
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": vol,
                         "taker_buy_volume": vol * rng.uniform(0.3, 0.8, n), "funding": 0.0}, index=idx)


def test_rel_strength_is_coin_minus_btc_over_the_window():
    idx = pd.date_range("2023-01-01", periods=10, freq="1h", tz="UTC")
    coin, btc = pd.Series(np.arange(100.0, 110.0), idx), pd.Series(np.full(10, 50.0), idx)
    rs = S.rel_strength(coin, btc, 3)
    assert np.isnan(rs[:3]).all() and rs[5] == pytest.approx(coin.iloc[5] / coin.iloc[2] - 1)
    assert np.isnan(S.rel_strength(coin, btc.iloc[:5], 3)[8])        # у BTC нет данных — NaN


def test_rows_carry_times_and_both_windows(monkeypatch):
    monkeypatch.setattr(S, "MAX_SLOPE", 1e9)
    d = _rw(1)
    rows = pd.DataFrame(S.coin_rows(d, _rw(9)["close"], np.full(len(d), 1e9), "RW", "15m"))
    assert len(rows) > 10 and rows.rs24.notna().any() and rows.rs72.notna().any()
    assert (rows.exit >= rows.fill).all() and (rows.fill >= rows.t).all()


def test_pick_respects_limit_and_order_inside_one_bar():
    t0 = pd.Timestamp("2023-01-01")
    z = pd.DataFrame({"fill": [t0] * 4 + [t0 + pd.Timedelta("15min")], "exit": [t0 + pd.Timedelta("1h")] * 5,
                      "R": [1.0, 2.0, 3.0, 4.0, 5.0], "side": -1})
    got = S.pick(z, np.array([3, 1, 2, 0, 0]), limit=3)
    assert sorted(got.R) == [2.0, 3.0, 4.0]                  # слабейшие по порядку; сигнал следующей свечи не влезает
    assert len(S.pick(z, np.zeros(5), limit=10)) == 5


def test_queue_verdict_rewards_a_real_rs_edge_and_not_noise(monkeypatch):
    monkeypatch.setattr(S, "SHUFFLES", 6)
    rng = np.random.default_rng(0)
    n = 6000
    bars = pd.Timestamp("2020-06-01") + pd.to_timedelta(rng.integers(0, 2300 * 96, n // 6) * 15, unit="min")
    fill = pd.Series(np.repeat(bars.to_numpy(), 6))                  # по шесть сигналов на свечу — выбирать есть из чего
    z = pd.DataFrame({"fill": fill, "exit": fill + pd.Timedelta("6h"), "side": -1, "rs24": rng.normal(0, 0.03, n)})
    z["R"] = rng.normal(0, 1, n) - 8 * z.rs24                 # слабые (RS ниже) платят больше
    assert S.queue_verdict(z)[0]
    z["R"] = rng.normal(0, 1, n)
    assert not S.queue_verdict(z)[0]
