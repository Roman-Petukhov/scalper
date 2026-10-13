import numpy as np
import pandas as pd

from research.pumpfade import Rule, account, features, signals, trades


def _panel(after, coins=60, n=60, pump_at=40, small_adv=1e7):
    """Монеты C0..C59 с убывающим оборотом; P — мелкая монета (оборот ниже 50-го места), на днях pump_at-2..pump_at
    растёт на 9% в день (рост за 3 дня +29.5% — сигнал только на pump_at), дальше идёт по свечам after: [(open, high, low, close), ...]."""
    days = pd.date_range("2022-01-01", periods=n, freq="D")
    rows = []
    for k in range(coins):
        for d in days:
            rows.append(dict(symbol=f"C{k}", day=d, open=100.0, high=101.0, low=99.0, close=100.0, funding=0.0,
                             adv=1e9 - k * 1e7))
    px = 100.0
    for i, d in enumerate(days):
        if pump_at - 2 <= i <= pump_at:
            o, px = px, px * 1.09
            bar = (o, px, o, px)
        elif i > pump_at and i - pump_at - 1 < len(after):
            bar = after[i - pump_at - 1]
            px = bar[3]
        else:
            bar = (px, px * 1.01, px * 0.99, px)
        rows.append(dict(symbol="P", day=d, open=bar[0], high=bar[1], low=bar[2], close=bar[3], funding=0.0001,
                         adv=small_adv))
    return features(pd.DataFrame(rows))


def test_pump_on_small_coin_is_shorted_and_hits_the_mid_target():
    top = 100 * 1.09 ** 3                                      # ≈ 129.5, стоп ≈ 130.5, середина ≈ 114.8
    f = _panel([(129.0, 130.0, 120.0, 121.0), (121.0, 122.0, 110.0, 112.0)])
    z = trades(f, Rule())
    assert len(z) == 1 and z.symbol.iloc[0] == "P"
    tr = z.iloc[0]
    assert tr.kind == "target" and tr.R > 0
    assert np.isclose(tr.pnl, (129 - (100 + top) / 2) / 129 - 5.5e-4 - 2e-4 + 2 * 0.0001)


def test_squeeze_above_the_pump_high_stops_out_with_slippage():
    f = _panel([(129.0, 170.0, 128.0, 160.0)])
    tr = trades(f, Rule()).iloc[0]
    assert tr.kind == "stop" and tr.R < -1                     # проскальзывание и комиссии — хуже −1R
    assert tr.mae > 0.3


def test_large_coins_are_the_control_group_only():
    f = _panel([(129.0, 130.0, 120.0, 121.0)], small_adv=2e9)      # P — самая ликвидная
    assert trades(f, Rule()).empty
    assert len(trades(f, Rule(top=True))) == 1


def test_filters_and_no_lookahead_in_signal():
    f = _panel([(129.0, 130.0, 120.0, 121.0)])
    assert signals(f, Rule()).sum() == 1
    assert signals(f, Rule(fund=True)).sum() == 0                # funding 0.01% в день — не высокий
    assert signals(f, Rule(exhaust=True)).sum() == 0              # памп закрылся на максимуме


def test_account_caps_open_positions():
    t = pd.DataFrame({"t": pd.to_datetime(["2022-01-01"] * 5), "exit": pd.to_datetime(["2022-01-05"] * 5),
                      "R": [1.0] * 5})
    m = account(t, risk=0.01, max_open=3)
    assert np.isclose(m.sum(), 0.03)
