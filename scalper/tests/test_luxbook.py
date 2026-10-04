import io
import zipfile

import numpy as np
import orjson
import pandas as pd

from research.hft import luxbook
from research.hft.walls import replay

T0 = 1_790_000_000_000


def _blob(msgs):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("x", b"\n".join(orjson.dumps(m) for m in msgs))
    return b.getvalue()


def test_signals_returns_are_measured_from_signal_close_and_flip_exit():
    n = 400
    rng = np.random.default_rng(1)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    k = pd.DataFrame({"open_time": T0 + np.arange(n) * luxbook.BAR_MS, "open": c, "high": c * 1.001,
                      "low": c * 0.999, "close": c})
    sig = luxbook.signals(k)
    assert len(sig) > 0 and set(sig["dir"]) <= {1.0, -1.0}
    r = sig.iloc[0]
    i = int((r["t_ms"] - luxbook.BAR_MS - T0) // luxbook.BAR_MS)
    assert np.isclose(r["ret_60m"], (c[i + 12] / c[i] - 1) * 1e4)
    if r["hold_bars"] > 0:
        assert np.isclose(r["ret_flip"], (c[i + int(r["hold_bars"])] / c[i] - 1) * 1e4)


def test_replay_snapshots_book_state_at_probe_times():
    bids = [[f"{100 - i * 0.01:.2f}", "10"] for i in range(1, 30)]
    asks = [[f"{100 + i * 0.01:.2f}", "10"] for i in range(1, 30)]
    msgs = [{"type": "snapshot", "ts": T0, "data": {"b": bids, "a": asks}},
            {"type": "delta", "ts": T0 + 2000, "data": {"b": [["99.99", "50"]], "a": []}},
            {"type": "delta", "ts": T0 + 4000, "data": {"b": [], "a": []}}]
    trades = pd.DataFrame({"timestamp": [(T0 + 100) / 1000], "side": ["Buy"], "size": [1.0], "price": [100.01]})
    _, snaps = replay(_blob(msgs), trades, np.array([T0 + 1000, T0 + 3000]))
    assert list(snaps["t_ms"]) == [T0 + 1000, T0 + 3000]
    # 10 б.п. от mid=100 — уровни 99.90..100.10: по 10 уровней на сторону
    assert snaps["bid10"].iloc[0] == 100 and snaps["ask10"].iloc[0] == 100
    assert snaps["bid10"].iloc[1] == 140                       # после delta 99.99 -> 50


def test_features_direction_and_classes():
    t = T0 + 20 * 60_000
    sig = pd.DataFrame({"t_ms": [t, t], "dir": [1.0, -1.0], "ret_15m": [5.0, 5.0], "ret_60m": [5.0, 5.0],
                        "ret_240m": [5.0, 5.0], "ret_flip": [5.0, 5.0], "hold_bars": [3, 3]})
    snaps = pd.DataFrame({"t_ms": [t], "mid": [100.0], "spread_bps": [1.0], "bid10": [30.0], "ask10": [10.0],
                          "bid25": [60.0], "ask25": [20.0]})
    trades = pd.DataFrame({"timestamp": [(t - 60_000) / 1000, (t - 30_000) / 1000, (t - 10 * 60_000) / 1000],
                           "side": ["Buy", "Buy", "Sell"], "size": [3.0, 1.0, 100.0], "price": [100.0] * 3})
    ev = pd.DataFrame({"ts": [float(t - 5 * 60_000)], "kind": ["absorbed"], "side": [-1]})
    f = luxbook.features(sig, snaps, ev, trades)
    assert np.isclose(f["flow_d"].iloc[0], 1.0) and np.isclose(f["flow_d"].iloc[1], -1.0)   # продажа вне 5 мин
    assert np.isclose(f["imb25_d"].iloc[0], 0.5) and np.isclose(f["imb25_d"].iloc[1], -0.5)
    assert list(f["absorbed_against"]) == [1.0, 0.0]          # ask-айсберг против лонга, за шорт
    cls = luxbook.classify(f)
    assert list(cls) == ["fake", "fake"]                       # лонг упёрся в айсберг; шорт против ленты и стакана
