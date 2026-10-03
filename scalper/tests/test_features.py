import math

import numpy as np
import pytest

from core.features import Features, RollingStats
from core.models import Book, Liquidation, Trade


def test_rolling_stats_matches_numpy():
    rng = np.random.default_rng(1)
    xs = rng.normal(5, 2, 500)
    rs = RollingStats(100)
    for x in xs:
        rs.push(float(x))
    tail = xs[-100:]
    assert rs.mean == pytest.approx(tail.mean())
    assert rs.std == pytest.approx(tail.std(ddof=1))
    assert rs.z(10.0) == pytest.approx((10.0 - tail.mean()) / tail.std(ddof=1))


def feed_constant(f: Features, seconds: int, price=100.0, qty=1.0, buyer_maker=False, t0=1000.0):
    for s in range(seconds):
        f.on_trade(Trade(t0 + s + 0.5, price, qty, buyer_maker))


def test_delta_windows_count_aggressive_flow(cfg):
    f = Features(cfg, 0.1)
    feed_constant(f, 120)                       # 1 агрессивная покупка в секунду
    snap = f.snapshot(1000 + 119.9)
    assert snap.delta[5] == pytest.approx(5)
    assert snap.delta[15] == pytest.approx(15)
    assert snap.delta[60] == pytest.approx(60)


def test_delta_sign_for_sells(cfg):
    f = Features(cfg, 0.1)
    feed_constant(f, 30, buyer_maker=True)
    assert f.snapshot(1029.9).delta[5] == pytest.approx(-5)


def test_zscore_spikes_on_burst(cfg):
    f = Features(cfg, 0.1)
    rng = np.random.default_rng(7)
    t = 1000.0
    for s in range(600):                         # шумовой сбалансированный поток
        for _ in range(5):
            f.on_trade(Trade(t + s + rng.random(), 100.0, 1.0, bool(rng.random() < 0.5)))
    for k in range(40):                           # всплеск покупок
        f.on_trade(Trade(t + 600 + k * 0.02, 100.0, 1.0, False))
    snap = f.snapshot(t + 600.9)
    assert snap.z[5] > 3


def test_constant_price_has_zero_vol_and_flat_vwap(cfg):
    f = Features(cfg, 0.1)
    feed_constant(f, 200, price=50000.0)
    s = f.snapshot(1199.9)
    assert s.sigma_1s == 0
    assert s.vwap == pytest.approx(50000.0)
    assert s.vwap_std == pytest.approx(0.0, abs=1e-6)


def test_vwap_is_volume_weighted(cfg):
    f = Features(cfg, 0.1)
    f.on_trade(Trade(1000.1, 100.0, 3.0, False))
    f.on_trade(Trade(1001.1, 110.0, 1.0, False))
    s = f.snapshot(1002.0)
    assert s.vwap == pytest.approx((300 + 110) / 4)
    var = (3 * (100 - 102.5) ** 2 + 1 * (110 - 102.5) ** 2) / 4
    assert s.vwap_std == pytest.approx(math.sqrt(var))


def test_range_excludes_recent_seconds(cfg):
    f = Features(cfg, 0.1)
    for s in range(60):
        f.on_trade(Trade(1000 + s + 0.5, 100.0 + (s % 3) * 0.1, 1.0, False))
    f.on_trade(Trade(1060.5, 105.0, 1.0, False))   # свежий пробой
    snap = f.snapshot(1060.6)
    assert snap.range_hi == pytest.approx(100.2)
    assert snap.price == 105.0


def test_obi_and_microprice(cfg):
    f = Features(cfg, 0.1)
    f.on_trade(Trade(1000.0, 100.0, 1.0, False))
    bk = Book(1000.05, bids=[(99.9, 3.0)] * 10, asks=[(100.0, 1.0)] * 10)
    f.on_book(bk)
    s = f.snapshot(1000.1)
    assert s.obi == pytest.approx(0.5)            # (3-1)/(3+1)
    assert s.microprice == pytest.approx((99.9 * 1 + 100.0 * 3) / 4)


def test_liquidation_window(cfg):
    f = Features(cfg, 0.1)
    f.on_trade(Trade(1000.0, 100.0, 1.0, False))
    f.on_liquidation(Liquidation(1001.0, "SELL", 100.0, 5000.0))
    f.on_liquidation(Liquidation(1002.0, "BUY", 100.0, 1000.0))
    s = f.snapshot(1005.0)
    assert s.liq_long == pytest.approx(500_000)
    assert s.liq_short == pytest.approx(100_000)
    assert f.snapshot(1030.0).liq_long == 0


def test_bid_ask_estimate_without_book(cfg):
    f = Features(cfg, 0.1)
    f.on_trade(Trade(1000.0, 100.0, 1.0, True))   # удар в бид
    assert f.best_bid_ask(1000.0) == (100.0, pytest.approx(100.1))
    f.on_trade(Trade(1000.1, 100.1, 1.0, False))  # удар в аск
    assert f.best_bid_ask(1000.1) == (pytest.approx(100.0), 100.1)
