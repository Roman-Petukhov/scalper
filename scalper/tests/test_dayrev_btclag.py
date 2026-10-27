import numpy as np
import pandas as pd

from research import btclag as B
from research import dayrev as R


def test_dayrev_long_leg_buys_yesterdays_losers():
    days = pd.date_range("2024-01-01", periods=5, freq="D")
    cols = [f"C{i}" for i in range(30)]
    p0 = pd.DataFrame(100.0, index=days, columns=cols)
    p0.iloc[1, :3] = 90.0                                    # три худших вчера
    p0.iloc[2, :3] = 99.0                                    # и отскочили сегодня
    adv = pd.DataFrame(1e7, index=days, columns=cols)
    pf = R.portfolio(p0, p0, adv, (5e6, 50e6), 0)
    assert pf.loc[days[1], "L"] > 0.09 and pf.loc[days[1], "n"] == 30


def test_dayrev_band_excludes_liquid_coins():
    days = pd.date_range("2024-01-01", periods=4, freq="D")
    cols = [f"C{i}" for i in range(25)]
    p0 = pd.DataFrame(100.0, index=days, columns=cols)
    adv = pd.DataFrame(1e9, index=days, columns=cols)
    assert R.portfolio(p0, p0, adv, (5e6, 50e6), 1).empty


def test_btc_events_respect_threshold_and_cooldown():
    idx = pd.date_range("2024-01-01", periods=600, freq="5min", tz="UTC")
    rng = np.random.default_rng(3)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, 600)))
    c[400:] *= 1.02
    c[405:] *= 1.02                                          # второй рывок через 5 свечей — внутри паузы
    ev = B.btc_events(pd.Series(c, index=idx), 4.0)
    assert len(ev) == 1 and ev.t.iloc[0] == idx[400] and ev.side.iloc[0] == 1.0


def test_coin_rows_mark_laggers_and_measure_catch_up():
    idx = pd.date_range("2024-01-01", periods=40, freq="5min", tz="UTC")
    c = pd.Series(100.0, index=idx)
    c.iloc[21:] = 101.5                                      # альт догоняет на следующей свече
    ev = pd.DataFrame({"t": [idx[20]], "side": [1.0], "move": [0.02]})
    adv = pd.Series(1e7, index=pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"))
    rows = B.coin_rows(c, ev, adv, "ALT", 3.0)
    assert rows[0]["lag"] and abs(rows[0]["r3"] - 0.015) < 1e-12
