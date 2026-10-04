import io
import zipfile

import orjson
import pandas as pd

from research.hft.walls import detect

T0 = 1_790_000_000_000


def _blob(msgs):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("x", b"\n".join(orjson.dumps(m) for m in msgs))
    return b.getvalue()


def _base_book():
    bids = [[f"{100 - i * 0.01:.2f}", "10"] for i in range(1, 30)]
    asks = [[f"{100 + i * 0.01:.2f}", "10"] for i in range(1, 30)]
    return {"type": "snapshot", "ts": T0, "data": {"b": bids, "a": asks}}


def test_wall_pulled_when_price_approaches_is_spoof_candidate():
    msgs = [_base_book()]
    msgs += [{"type": "delta", "ts": T0 + 1000 + i * 100, "data": {"b": [], "a": []}} for i in range(15)]
    # крупная стена на bid 99.90 (10 б.п. от mid), затем цена подходит (ask/bid сдвигаются вниз), стену снимают
    msgs.append({"type": "delta", "ts": T0 + 3000, "data": {"b": [["99.90", "500"]], "a": []}})
    msgs += [{"type": "delta", "ts": T0 + 4000 + i * 100, "data": {"b": [], "a": []}} for i in range(15)]
    msgs.append({"type": "delta", "ts": T0 + 6000, "data": {"b": [["99.99", "0"], ["99.98", "0"], ["99.97", "0"]],
                                                         "a": [["100.01", "0"], ["100.02", "0"]]}})
    msgs += [{"type": "delta", "ts": T0 + 7000 + i * 100, "data": {"b": [], "a": []}} for i in range(15)]
    msgs.append({"type": "delta", "ts": T0 + 9000, "data": {"b": [["99.90", "0"]], "a": []}})
    msgs += [{"type": "delta", "ts": T0 + 10000 + i * 1000, "data": {"b": [], "a": []}} for i in range(120)]
    trades = pd.DataFrame({"timestamp": [(T0 + 500) / 1000], "side": ["Buy"], "size": [1.0], "price": [100.01]})
    ev = detect(_blob(msgs), trades)
    row = ev[ev["kind"].str.startswith("pulled")].iloc[0]
    assert row["kind"] == "pulled_near" and row["side"] == 1 and row["expect"] == -1


def test_wall_eaten_by_aggressors_is_consumed():
    msgs = [_base_book()]
    msgs += [{"type": "delta", "ts": T0 + 1000 + i * 100, "data": {"b": [], "a": []}} for i in range(15)]
    msgs.append({"type": "delta", "ts": T0 + 3000, "data": {"b": [["99.99", "500"]], "a": []}})
    msgs.append({"type": "delta", "ts": T0 + 9000, "data": {"b": [["99.99", "0"]], "a": []}})
    msgs += [{"type": "delta", "ts": T0 + 10000 + i * 1000, "data": {"b": [], "a": []}} for i in range(10)]
    trades = pd.DataFrame({"timestamp": [(T0 + 8000) / 1000], "side": ["Sell"], "size": [500.0], "price": [99.99]})
    ev = detect(_blob(msgs), trades)
    assert (ev["kind"] == "consumed").any()


def test_short_lived_wall_is_not_recorded():
    msgs = [_base_book()]
    msgs += [{"type": "delta", "ts": T0 + 1000 + i * 100, "data": {"b": [], "a": []}} for i in range(15)]
    msgs.append({"type": "delta", "ts": T0 + 3000, "data": {"b": [["99.99", "500"]], "a": []}})
    msgs.append({"type": "delta", "ts": T0 + 3500, "data": {"b": [["99.99", "0"]], "a": []}})   # мерцание 0.5 с
    msgs += [{"type": "delta", "ts": T0 + 4000 + i * 1000, "data": {"b": [], "a": []}} for i in range(10)]
    trades = pd.DataFrame({"timestamp": [(T0 + 500) / 1000], "side": ["Buy"], "size": [1.0], "price": [100.01]})
    assert len(detect(_blob(msgs), trades)) == 0
