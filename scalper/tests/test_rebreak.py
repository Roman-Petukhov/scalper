import numpy as np
import pandas as pd

from research.rebreak import blocked, events


def test_events_first_fail_and_rebreak_on_the_same_line():
    c = np.full(40, 100.0)
    line = {"t": 10, "side": 1, "a": 0, "slope": 0.0}         # горизонталь 100 для простоты
    c[10] = 101.0                                              # пробой
    c[11] = 100.5
    c[13] = 99.0                                               # возврат за линию через 3 свечи
    c[14:20] = 99.5
    c[20] = 101.5                                              # повторный пробой
    ev = events(c, [line])
    assert [(e["kind"], e["t"], e["fail_lag"]) for e in ev] == [("first", 10, 3), ("rebreak", 20, 3)]
    c2 = c.copy()
    c2[13:] = 100.5                                            # не вернулась — ложного пробоя нет
    assert [(e["kind"], e["fail_lag"]) for e in events(c2, [line])] == [("first", -1)]


def test_rebreak_blocked_while_first_trade_is_open():
    t = pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-05"], utc=True)
    df = pd.DataFrame({"symbol": "X", "kind": ["first", "rebreak", "rebreak"], "t": [t[0], t[1], t[2]],
                       "first_t": t[0], "side": 1, "entry": "market",
                       "exit_t": [pd.Timestamp("2024-01-04", tz="UTC"), t[2], t[2]]})
    assert list(blocked(df)) == [False, True, False]
