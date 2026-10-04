import io
import zipfile

import orjson
import pandas as pd

from research.hft.mm import Params, load_trades, simulate

T0 = 1_790_000_000_000


def _book(msgs):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("x", b"\n".join(orjson.dumps(m) for m in msgs))
    return b.getvalue()


def _flat(n=3000, bid="99.95", ask="100.05", qty="10"):
    m = [{"type": "snapshot", "ts": T0, "data": {"b": [[bid, qty]], "a": [[ask, qty]]}}]
    return m + [{"type": "delta", "ts": T0 + i * 100, "data": {"b": [], "a": []}} for i in range(1, n)]


def test_no_fill_until_queue_ahead_is_consumed():
    tr = pd.DataFrame({"timestamp": [(T0 + 1000 + i * 1000) / 1000 for i in range(3)], "side": ["Sell"] * 3,
                       "size": [4.0, 4.0, 4.0], "price": [99.95] * 3})
    r = simulate(_book(_flat()), load_trades(tr), Params(q_usd=100, min_spread_bps=4, max_hold_s=1e9))
    # впереди 10, съедено 4+4 -> не исполнено; третья сделка съедает ещё 4 -> исполнено 2 из ~1 единицы -> полностью
    assert r["fills"] == 1 and abs(r["maker_volume_usd"] - 99.95 * (100 / 100.0)) < 1e-6


def test_adverse_move_and_forced_exit_are_charged():
    msgs = _flat(600)
    # после нашей покупки стакан уходит вниз на 50 б.п.
    msgs += [{"type": "snapshot", "ts": T0 + 61_000, "data": {"b": [["99.45", "10"]], "a": [["99.55", "10"]]}}]
    msgs += [{"type": "delta", "ts": T0 + 61_000 + i * 1000, "data": {"b": [], "a": []}} for i in range(1, 200)]
    tr = pd.DataFrame({"timestamp": [(T0 + 2000) / 1000], "side": ["Sell"], "size": [20.0], "price": [99.95]})
    r = simulate(_book(msgs), load_trades(tr), Params(q_usd=100, min_spread_bps=4, max_hold_s=120))
    assert r["fills"] == 1 and r["forced_exits"] == 1
    # купили 1 ед. по 99.95, продали рыночно по 99.45: -0.50 и комиссии 2 + 5.5 б.п.
    assert abs(r["pnl_usd"] - (99.45 - 99.95 - 99.95 * 2e-4 - 99.45 * 5.5e-4)) < 1e-6
