import numpy as np
import pandas as pd

from research.case import bot_rows, donchian_trades


def test_donchian_enters_on_new_high_and_exits_on_low():
    idx = pd.date_range("2026-01-01", periods=60, freq="1D", tz="UTC")
    c = pd.Series(np.r_[np.full(25, 10.0), np.linspace(11, 20, 20), np.linspace(19, 12, 15)], index=idx)
    tr = donchian_trades(c, 20, 10)
    assert len(tr) == 1
    x = tr[0]
    assert x["entry_t"] == idx[25] and x["entry"] == 11.0 and x["exit"] < 20 and x["R"] > 0
    assert donchian_trades(c.iloc[:40], 20, 10)[0]["exit_t"] is pd.NaT          # ещё в позиции


def test_bot_rows_list_every_breakout_with_reasons():
    idx = pd.date_range("2025-01-01", periods=900, freq="4h", tz="UTC")
    rng = np.random.default_rng(5)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 900)))
    o = np.r_[c[0], c[:-1]]
    v = rng.uniform(1e5, 2e5, 900)
    d = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.006, "low": np.minimum(o, c) * 0.994, "close": c,
                      "volume": v, "taker_buy_volume": v * rng.uniform(0.4, 0.6, 900), "funding": 0.0}, index=idx)
    rows = pd.DataFrame(bot_rows(d, np.full(900, 1e9), idx[0], idx[-1]))
    assert len(rows) and set(rows.side) <= {-1, 1}
    assert rows.R.notna().any()
    assert not rows.failed.str.contains("оборот").any()
