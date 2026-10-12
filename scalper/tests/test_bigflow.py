import io
import zipfile

import numpy as np
import pytest

from research.bigflow import order_features, window_trades


def test_order_features_glues_sweeps_and_splits_by_size():
    # adv $1M: small < $10, big >= $100, whale >= $1000
    price = np.array([10.0, 10.1, 10.2, 10.0, 10.0, 10.0])
    qty = np.array([50.0, 50.0, 1.0, 0.5, 0.5, 20.0])
    ts = np.array([1, 1, 2, 3, 4, 5], dtype="float64")
    maker = np.array([False, False, True, False, True, True])           # True — агрессор продаёт
    f = order_features(price, qty, ts, maker, +1, 1e6)
    assert f["orders"] == 5                                            # два куска одной покупки склеены
    assert f["aggr_whale"] == pytest.approx(1.0)                       # $1005 — одна покупка-кит
    assert f["aggr_big"] == pytest.approx(1005 / (1005 + 200))              # $10.2 — средний, не крупный
    assert f["aggr_small"] == pytest.approx(0.5)                       # $5 купили, $5 продали
    assert f["div"] == pytest.approx(f["aggr_big"] - 0.5)
    short = order_features(price, qty, ts, maker, -1, 1e6)
    assert short["aggr_all"] == pytest.approx(1 - f["aggr_all"])
    assert np.isnan(order_features(price[:0], qty[:0], ts[:0], maker[:0], 1, 1e6)["aggr_all"])


def test_window_trades_keeps_only_the_candle_and_reads_microseconds():
    t0 = 1_700_000_000_000                                              # мс; последняя строка — в мкс
    rows = [f"1,10,1,1,1,{t0 + 1000},false", f"2,11,2,2,2,{t0 + 5000},true", f"3,12,3,3,3,{(t0 + 9000) * 1000},false"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.csv", "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
                   + "\n".join(rows))
    p, q, ts, mk = window_trades(buf.getvalue(), t0 + 2000, t0 + 10000)
    assert list(p) == [11.0, 12.0] and list(ts) == [t0 + 5000.0, t0 + 9000.0] and list(mk) == [True, False]
