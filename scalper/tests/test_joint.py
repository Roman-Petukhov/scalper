import pandas as pd
import pytest

from research.joint import Mix, simulate


def _t(h):
    return pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=h)


def _df(rows):
    return pd.DataFrame(rows, columns=["strategy", "symbol", "t_in", "t_out", "side", "R", "risk_pct"])


def test_simulate_one_position_per_coin_and_risk_sizing():
    df = _df([("bot", "AUSDT", _t(0), _t(10), -1, 3.0, 0.05),
              ("coil", "AUSDT", _t(4), _t(8), -1, -1.0, 0.02),      # монета занята ботом — отказ
              ("coil", "BUSDT", _t(4), _t(20), 1, -1.0, 0.02)])
    b = simulate(df, Mix("x", {"bot": 1.0, "coil": 0.5}))
    assert b.opened == {"bot": 1, "coil": 1} and b.rej_sym == 1
    # +3% от бота (риск 1%), затем −0.5% от coil, считая от капитала после бота
    assert b.eq.iloc[-1] == pytest.approx(1000 * 1.03 - 1000 * 0.005)


def test_simulate_position_limit_and_ignored_strategy():
    df = _df([("bot", f"{c}USDT", _t(0), _t(10), 1, 1.0, 0.05) for c in "ABC"]
             + [("coil", "DUSDT", _t(1), _t(5), 1, 1.0, 0.05)])
    b = simulate(df, Mix("x", {"bot": 1.0}, max_pos=2))
    assert b.opened == {"bot": 2} and b.rej_pos == 1
