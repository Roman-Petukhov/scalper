import json

import numpy as np
import pandas as pd

from research import spx200 as X


def _cycle():
    """Рост, глубокое падение под EMA200, затем восстановление: один эпизод и один запуск."""
    up = np.linspace(100, 300, 120)
    down = np.linspace(300, 120, 24)
    rec = np.linspace(120, 400, 60)
    idx = pd.date_range("1950-01-31", periods=204, freq="ME")
    return pd.Series(np.r_[up, down, rec], index=idx)


def test_frame_warmup_and_ratio():
    w = X.frame(_cycle(), 46, 12)
    assert w["ema200"].iloc[:45].isna().all() and w["ema200"].iloc[45:].notna().all()
    assert np.allclose(w["ratio"].dropna(), (w["close"] / w["ema200"]).dropna())


def test_episode_and_launch_found():
    w = X.frame(_cycle(), 46, 12)
    ep = X.episodes(w)
    assert len(ep) == 1 and ep["ещё вниз"].iloc[0] < 0 and ep["24 мес."].notna().all()
    lt = X.launch_table(w)
    assert len(lt) == 1 and pd.Timestamp(lt["запуск"].iloc[0]) > pd.Timestamp(ep["начало"].iloc[0])


def test_buckets_and_diff():
    st = X.monthly(X.frame(_cycle(), 46, 12))
    rets = {h: X.forward(st["close"], h) for h in X.HS}
    tab = X.bucket_table(st, rets)
    assert list(tab["к EMA200"]) == list(X.BIN_NAMES) and tab["месяцев"].sum() == st["ratio"].notna().sum()
    d, t, n = X.diff_t(st, rets[24], 24)
    assert n > 0 and d > 0


def test_parsers():
    csv = b"Date,SP500,Dividend\n1871-01-01,4.44,0.26\n1871-02-01,4.5,0.26\n2026-10-01,0.0,0\n"
    s = X.parse_shiller(csv)
    assert list(s) == [4.44, 4.5] and s.index[0] == pd.Timestamp("1871-01-31")
    raw = json.dumps({"chart": {"result": [{"timestamp": [0, 604800, 604800 * 2],
                                            "indicators": {"quote": [{"close": [1.0, None, 3.0]}]}}]}}).encode()
    y = X.parse_yahoo(raw)
    assert list(y) == [1.0, 3.0]
