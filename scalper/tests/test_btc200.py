import numpy as np
import pandas as pd

from research import btc200 as B
from research import ema200 as E
from research.tline import _to_bars
from tests.test_formation import _rw


def _days(n: int, up: bool) -> pd.DataFrame:
    idx = pd.date_range("2018-01-01", periods=n, freq="D", tz="UTC")
    c = np.linspace(100, 300, n) if up else np.linspace(300, 100, n)
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c}, index=idx)


def test_regimes_need_200_bars_and_follow_the_trend():
    r = B.btc_regimes(_days(2000, True))
    assert r["d200"].iloc[:199].isna().all() and (r["d200"].iloc[199:] > 0).all()
    assert r["w200"].notna().sum() == len(r["w200"]) - 199 and (r["w200"].dropna() > 0).all()
    assert (B.btc_regimes(_days(2000, False))["d200"].dropna() < 0).all()


def test_daily_and_weekly_signs_are_known_only_after_the_close():
    day = pd.Series([1.0, -1.0], index=pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True))
    d = pd.DataFrame(index=pd.date_range("2024-01-01", "2024-01-03 20:00", freq="4h", tz="UTC"))
    v = pd.Series(_to_bars(day.copy(), d, "1D", "4h"), index=d.index)
    assert np.isnan(v["2024-01-01 16:00"]) and v["2024-01-01 20:00"] == 1.0     # дневка 01.01 закрылась в 24:00
    assert v["2024-01-02 16:00"] == 1.0 and v["2024-01-02 20:00"] == -1.0
    wk = pd.Series([1.0], index=pd.to_datetime(["2024-01-01"], utc=True))        # неделя с понедельника 01.01
    dw = pd.DataFrame(index=pd.date_range("2024-01-07 16:00", "2024-01-07 20:00", freq="4h", tz="UTC"))
    w = _to_bars(wk.copy(), dw, "1W", "4h")
    assert np.isnan(w[0]) and w[1] == 1.0                                         # известна после закрытия недели


def test_rows_on_a_random_walk(monkeypatch):
    monkeypatch.setattr(E, "MAX_SLOPE", 1e9)
    d = _rw(5)
    idx = pd.date_range(d.index[0].floor("D") - pd.Timedelta(days=1500), d.index[-1].ceil("D"), freq="D", tz="UTC")
    regs = B.btc_regimes(_days(len(idx), True).set_axis(idx))
    rows = pd.DataFrame(B.coin_rows(d, regs, np.full(len(d), 1e9), "RW", "15m"))
    assert len(rows) > 10 and set(rows.d200.unique()) <= {-1.0, 1.0}
    assert (rows.d200 == rows.side).all() and (rows.w200 == rows.side).all()      # BTC растёт: знак = сторона сделки
