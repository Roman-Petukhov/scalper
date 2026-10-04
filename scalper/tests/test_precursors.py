import numpy as np
import pandas as pd

from research.precursors import auc, candidates, features, hourly


def _hours(start, n):
    return pd.date_range(start, periods=n, freq="1h", tz="UTC")


def test_features_use_only_window_before_t():
    idx = _hours("2024-03-01", 96)
    book = pd.DataFrame({"p-2": 100.0, "p2": 100.0, "p-1": 50.0, "p1": 50.0, "p-5": 300.0, "p5": 300.0}, index=idx)
    t = idx[80]
    book.loc[idx[76]:idx[79], "p2"] = 50.0                    # последние 4 ч до t: продавцов вдвое меньше
    book.loc[t:, ["p2", "p-2"]] = 1e9                         # после t — не должно влиять
    flow = pd.DataFrame({"buy": 10.0, "sell": 10.0, "cnt": 1.0, "bb10": 0.0, "bs10": 0.0, "bb50": 0.0, "bs50": 0.0,
                         "bbr": 0.0, "bsr": 0.0}, index=idx)
    flow.loc[idx[76]:idx[79], ["buy", "bb10", "bbr"]] = [30.0, 20.0, 20.0]
    f = features(book, flow, t)
    assert np.isclose(f["imb2_4"], (100 - 50) / 150)
    assert np.isclose(f["ask_thin2"], np.log(0.5)) and np.isclose(f["bid_thin2"], 0.0)
    assert np.isclose(f["cvd4"], (120 - 40) / 160)
    assert np.isclose(f["big10_net4"], 80 / 160)


def test_auc_orders_groups():
    assert auc(np.arange(20, 40.0), np.arange(0, 20.0)) == 1.0
    assert np.isnan(auc(np.arange(5.0), np.arange(20.0)))


def test_candidates_mark_pump_start(tmp_path):
    n = 24 * 140
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    c = np.full(n, 10.0)
    c[24 * 120:] = 20.0                                        # +100% на 120-й день
    ot = np.array([int(x.timestamp() * 1000) for x in idx], dtype="int64")
    pd.DataFrame({"open_time": ot, "high": c, "low": c, "close": c, "quote_volume": np.full(n, 1e6)}
                 ).to_parquet(tmp_path / "XUSDT-1h.parquet")
    k = hourly(tmp_path, "XUSDT")
    k["adv"] = 5e6
    ev = candidates(k, "XUSDT", np.random.default_rng(0))
    up = ev[ev.kind == "UP"]
    assert len(up) == 1
    # первый час, после которого в 24 ч есть +100%: закрытие часа ровно за 24 ч до скачка
    assert up["t"].iloc[0] == idx[24 * 120 - 24] + pd.Timedelta(hours=1)
