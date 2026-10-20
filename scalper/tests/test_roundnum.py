import numpy as np
import pandas as pd
import pytest

from research import roundnum as R


def test_steps_and_crossing():
    assert R.step(2.3, True) == 0.5 and R.step(2.3, False) == pytest.approx(0.1)
    assert R.step(67000, True) == 5000 and R.step(67000, False) == 1000
    assert R.crossed(1.97, 2.03, True) and not R.crossed(2.03, 2.4, True)
    assert R.crossed(2.03, 2.12, False) and not R.crossed(2.11, 2.19, False)
    assert R.crossed(70100, 69900, True)                         # вниз через 70 000


def test_level_ahead_and_crowd_stop():
    assert R.level_ahead(2.3, 1, 0.25, True) and not R.level_ahead(2.3, 1, 0.15, True)    # 2.5 впереди лонга
    assert R.level_ahead(2.3, -1, 0.35, True) and not R.level_ahead(2.3, -1, 0.25, True)  # 2.0 впереди шорта
    # лонг, стоп 1.96 под круглым 2.0 (мелкий шаг 0.1 → ближайший уровень над стопом 2.0), ATR 0.1: 0.04 ≤ 0.05
    assert R.crowd_stop(1.96, 1, 0.1) == pytest.approx(1.95)
    assert R.crowd_stop(1.93, 1, 0.1) is None                    # до 2.0 далеко
    assert R.crowd_stop(2.04, -1, 0.1) == pytest.approx(2.05)    # шорт: стоп над 2.0 — за уровень вверх


def test_rows_on_random_walk():
    rng = np.random.default_rng(2)
    n = 20000
    c = 3 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    o = np.r_[c[0], c[:-1]]
    vol = np.full(n, 1000.0)
    d = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.004, "low": np.minimum(o, c) * 0.996, "close": c,
                      "volume": vol, "taker_buy_volume": vol * rng.uniform(0.3, 0.8, n)},
                     index=pd.date_range("2018-01-01", periods=n, freq="4h", tz="UTC"))
    x = pd.DataFrame(R.coin_rows(d, np.full(n, 1e9), "RW", "4h"))
    assert len(x) > 10 and set(x.F1) <= {"да", "нет"} and x.R_s1.notna().all()
    assert (x.loc[~x.crowd, "R_s2"] == x.loc[~x.crowd, "R"]).all()
