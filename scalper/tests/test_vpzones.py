import numpy as np

from research.vpzones import vp_zones, zone_signals


def _data():
    # 200 баров торгуются около 100 (зона), потом цена уходит вверх к 105 с малым объёмом
    c = np.concatenate([100 + 0.05 * np.sin(np.arange(200)), np.linspace(100.5, 105, 50)])
    vol = np.concatenate([np.full(200, 10.0), np.full(50, 1.0)])
    return c, c.copy(), vol


def test_zone_below_price_after_leaving_high_volume_area():
    c, vwap, vol = _data()
    z = vp_zones(vwap, vol, c, 240, 10.0, 1.5, 12, 0.15)
    t = 249
    assert np.isnan(z[4, t])                              # цена ушла из зоны
    assert z[3, t] < c[t - 1] and 99.5 < z[3, t] < 100.6  # верхний край зоны около 100 ниже цены
    assert np.isnan(z[:, :240]).all()                     # до заполнения окна зон нет


def test_bounce_from_zone_top_and_break_below_zone():
    n = 5
    z = np.full((6, n), np.nan)
    z[2, :], z[3, :] = 99.0, 100.0                        # зона ниже цены: [99, 100]
    c = np.array([101.0, 101.0, 100.5, 98.5, 98.0])
    h = c + 0.2
    lo = np.array([100.8, 100.8, 99.8, 98.3, 97.8])       # бар 2: тень в зону, закрытие над ней
    osc = np.zeros(n)
    up, dn = zone_signals(h, lo, c, osc, z, -1e9, 0)
    assert up[2] and not dn.any()
    up, dn = zone_signals(h, lo, c, osc, z, -1e9, 1)
    assert dn[3]                                          # закрытие ниже дальнего края зоны
