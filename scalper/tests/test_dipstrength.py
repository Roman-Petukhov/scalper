import numpy as np
import pandas as pd

from research import dipstrength as D


def _index():
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    v = np.r_[np.linspace(100, 120, 60), np.linspace(120, 84, 30), np.linspace(84, 110, 40), np.full(70, 110.0)]
    return pd.Series(v, index=idx)


def test_find_dips_peak_low_turn_and_cooldown():
    x = _index()
    dips = D.find_dips(x)
    assert len(dips) == 1
    d = dips[0]
    assert d.peak == x.index[59] and d.low == x.index[89] and abs(d.depth + 0.30) < 1e-9
    assert x[d.turn] >= 84 * 1.10 and x[x.index[x.index.get_loc(d.turn) - 1]] < 84 * 1.10
    assert D.find_dips(x, dd_min=0.5) == []
    two = pd.concat([x, pd.Series(x.to_numpy(), index=x.index + pd.Timedelta(days=200))])
    assert len(D.find_dips(two)) == 2


def test_strength_universe_and_trades_pick_quintiles():
    x = _index()
    d = D.find_dips(x)[0]
    cols = [f"C{i:02d}USDT" for i in range(40)] + ["BTCUSDT"]
    tilt = np.linspace(-0.3, 0.3, 40)                                  # C39 держится лучше всех и потом растёт сильнее
    close = pd.DataFrame(index=x.index, columns=cols, dtype="float64")
    fall = (x.index > d.peak) & (x.index <= d.low)
    after = x.index > d.turn
    for c, k in zip(cols[:40], tilt):
        lr = np.log(x).diff().fillna(0.0).to_numpy().copy()
        lr[fall] += k / fall.sum()
        lr[after] += k / 100
        close[c] = 10 * np.exp(np.cumsum(lr))
    close["BTCUSDT"] = x
    r = close.pct_change()
    coins = D.universe(r, close, d)
    assert len(coins) == 40 and "BTCUSDT" not in coins
    s = D.fall_strength(close, x, d, coins)
    assert s.idxmax() == "C39USDT" and np.isclose(s["C39USDT"], 0.3) and np.isclose(s["C00USDT"], -0.3)
    top, bottom = D.trades(close, r, x, [d], 28, "top"), D.trades(close, r, x, [d], 28, "bottom")
    assert len(top) == 8 and len(bottom) == 8 and top.X.mean() > 0 > bottom.X.mean()
    assert set(top.sector) == {f"C{i}" for i in range(32, 40)}
    assert "C39USDT" in D.rs_at_high(close, x, d, coins) and "C00USDT" not in D.rs_at_high(close, x, d, coins)
    assert D.trades(close, r, x, [d], 400, "top").empty               # срок ещё не прошёл
    gone = close.copy()
    gone.loc[gone.index > d.turn + pd.Timedelta(days=5), "C39USDT"] = np.nan   # делистинг — по последней цене
    assert "C39USDT" in D.forward(gone, d, coins, 28).index
