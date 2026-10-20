import numpy as np
import pandas as pd

from research import metalabel as M
from research import noline


def _rw(seed: int, n: int = 6000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, n)))
    vol = np.full(n, 1000.0) * rng.uniform(0.5, 1.5, n)
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": vol,
                         "taker_buy_volume": vol * rng.uniform(0.3, 0.8, n)}, index=idx)


def test_bot_rows_match_bot_rule():
    for seed in range(3):
        d = _rw(seed)
        liq = np.full(len(d), 1e9)
        rows = pd.DataFrame(M.coin_rows(d, liq, "RW", "4h", None))
        noline.TF = "4h"
        ref = noline.coin_trades(d, liq, "RW", lines_only=True)
        ref = ref[(ref.entry == "retest") & (ref.t >= d.index[101])]
        bot = rows[rows.bot]
        assert len(rows) > len(bot)                                    # кандидатов больше, чем сделок бота
        assert list(bot.t) == list(ref.t)
        assert np.allclose(bot.R.to_numpy(), ref.R3.to_numpy())
        assert bot[["aggr", "close_loc", "brk_atr", "risk_atr", "touches", "vol_ratio"]].notna().all().all()
        assert (bot.aggr >= 0.55).all() and (bot.brk_atr > 0).all() and (bot.touches >= 2).all()


def test_evaluate_thresholds_from_training_period():
    bot = pd.DataFrame({"t": pd.date_range("2021-01-01", periods=10, freq="D"), "R": np.arange(10.0)})
    train = np.array([True] * 5 + [False] * 5)
    z = M.evaluate(bot, np.arange(10.0), train)
    assert list(z.keep) == [False, False] + [True] * 8                  # 40-й перцентиль по 0…4 = 1.6
    assert set(z.w[:5]) == {0.5, 1.0, 1.5} and (z.w[5:] == 1.5).all()
