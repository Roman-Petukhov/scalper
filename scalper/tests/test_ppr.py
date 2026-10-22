import numpy as np
import pandas as pd

from research import ppr as P


def test_find_marks_strict_and_soft_reversals():
    o = np.array([10, 9.0, 8.5, 9.0, 10.0])
    hi = np.array([10.2, 9.2, 8.8, 9.6, 10.5])
    lo = np.array([9.5, 8.6, 8.0, 8.8, 9.2])
    c = np.array([9.6, 8.7, 8.4, 9.0, 10.4])
    got = {(t, s): st for t, s, st in P.find(o, hi, lo, c)}
    assert got[(3, 1)] is True                 # средняя (2) — минимум, третья закрылась выше её максимума 8.8
    hi2 = hi.copy(); hi2[2] = 9.5               # максимум средней выше закрытия третьей — мягкий
    assert {(t, s): st for t, s, st in P.find(o, hi2, lo, c)}[(3, 1)] is False


def _rw(seed=1, n=20000):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.006, n)))
    o = np.r_[c[0], c[:-1]]
    v = np.full(n, 1000.0)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.002, n))),
                         "low": np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.002, n))), "close": c, "volume": v,
                         "taker_buy_volume": v * rng.uniform(0.3, 0.8, n), "funding": 0.0}, index=idx)


def test_rows_have_results_and_flags_on_a_random_walk():
    d = _rw()
    rows = pd.DataFrame(P.coin_rows(d, np.full(len(d), 1e12), "RW", "1h"))
    assert len(rows) > 500 and rows.R3.notna().all()
    assert rows.R3.mean() < 0.15                # на случайном ряде преимущества нет (минус издержки)
    assert rows.strict.any() and (~rows.strict).any() and rows.level.any()


def test_verdict_needs_every_period_and_t3():
    rng = np.random.default_rng(0)
    n = 4000
    z = pd.DataFrame({"t": pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 2400, n), unit="D"),
                      "R3": rng.normal(0.3, 1, n)})
    assert P.verdict(z, "R3")[0]
    assert not P.verdict(z.assign(R3=z.R3 - 0.3), "R3")[0]


def test_verdict_uses_day_clustered_t():
    """Сотни формаций за один день — одно наблюдение: наивный t раздувается, дневной — нет."""
    rng = np.random.default_rng(1)
    days = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.repeat(np.arange(0, 2400, 6), 200), unit="D")
    shift = rng.normal(0.0, 1.0, len(days) // 200)
    day_shift = np.repeat(shift - shift.mean() + 0.03, 200)   # общее среднее +0.03R при разбросе по дням 1R
    z = pd.DataFrame({"t": days, "R3": day_shift + rng.normal(0.05, 0.1, len(days))})
    assert P.tstat(z.R3.to_numpy()) > 3          # наивный t «принял» бы шум
    assert not P.verdict(z, "R3")[0]
