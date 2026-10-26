import json

import numpy as np
import pandas as pd

from research import figure as G
from research import formation as F
from tests.test_formation import _rw


def test_classify_by_bulkowski_names_and_touch_rules():
    n, at = 100, 1.0                                       # «горизонтальна» — сдвиг за фигуру <= 0.75 ATR
    assert G.classify(0.0, 0.005, n, 10, 10, at, 3, 2) == "rect"
    assert G.classify(0.0, 0.05, n, 10, 5, at, 3, 2) == "tri_asc"
    assert G.classify(-0.05, 0.0, n, 10, 5, at, 2, 3) == "tri_desc"
    assert G.classify(-0.03, 0.03, n, 10, 4, at, 3, 2) == "tri_sym"
    assert G.classify(0.05, 0.1, n, 10, 5, at, 3, 2) == "wedge_rise"
    assert G.classify(-0.1, -0.05, n, 10, 5, at, 3, 2) == "wedge_fall"
    assert G.classify(0.1, 0.1, n, 10, 10, at, 2, 2) == "chan_up"           # каналу хватает 2 + 2
    assert G.classify(-0.1, -0.1, n, 10, 10, at, 2, 2) == "chan_down"
    assert G.classify(0.05, -0.05, n, 10, 20, at, 3, 2) == "broad"
    assert G.classify(-0.03, 0.03, n, 10, 4, at, 2, 2) == "none"            # треугольнику нужно 5 касаний
    assert G.classify(-0.03, 0.03, n, 10, 4, at, 4, 1) == "none"            # и не меньше 2 на каждой границе


def test_pivots_and_boundaries_on_closes():
    c = np.array([5, 6, 9, 6, 5, 6, 8, 6, 5, 6, 7, 6, 5, 5, 5], dtype=float)
    assert list(G.minor_pivots(c, 0, len(c), True)) == [2, 6, 10]
    assert list(G.minor_pivots(-c, 0, len(c), False)) == [2, 6, 10]
    su, ua, n = G.boundary(c, 0, len(c), True, 0.1)
    assert abs(su + 0.25) < 1e-12 and abs(ua - 9.5) < 1e-12 and n == 3      # верх через 9, 8, 7 — все три касания
    rng = np.random.default_rng(3)
    for _ in range(30):                                     # ни одно закрытие фигуры не за границами
        cl = 100 + np.cumsum(rng.normal(0, 1, 150))
        for top in (True, False):
            b = G.boundary(cl, 0, 150, top, 0.25)
            line = b[1] + b[0] * np.arange(150)
            assert ((cl <= line + 1e-9) if top else (cl >= line - 1e-9)).all()


def test_features_on_a_random_walk(monkeypatch):
    monkeypatch.setattr(F, "MAX_SLOPE", 1e9)
    rows = pd.DataFrame(F.coin_rows(_rw(1), np.full(30000, 1e9), "RW", "15m", G.figure_features))
    assert len(rows) > 20 and set(rows["shape"]) <= set(G.SHAPES) and rows["shape"].nunique() >= 3
    assert rows[["body", "long_share", "wick", "contraction", "touch_gap"]].notna().all().all()
    assert set(rows.classic) <= {-1, 0, 1} and (rows["shape"] != "none").any()
    assert ((rows.long_share >= 0) & (rows.long_share <= 1) & (rows.wick >= 0) & (rows.wick <= 1)).all()


def test_journal_frames_rebuild_the_line(tmp_path):
    d = _rw(2, 200)
    t = [int(x.timestamp()) for x in d.index]
    iso = lambda i: d.index[i].isoformat()
    c = d.close.to_numpy()
    sig = {"symbol": "RW", "tf": "15m", "side": -1, "bar_time": iso(150), "line_a_time": iso(40), "line_a": c[40],
           "line_b_time": iso(90), "line_b": c[90], "trade": {"r": 2.5},
           "bars": {"t": t, "o": d.open.tolist(), "h": d.high.tolist(), "l": d.low.tolist(), "c": c.tolist()}}
    p = tmp_path / "j.json"
    p.write_text(json.dumps({"signals": [sig, {**sig, "bars": None}]}))
    (s, frame, rec, i), = G.journal_frames(p)
    assert (rec["a"], rec["b"], i, rec["side"]) == (40, 90, 150, -1) and rec["slope"] == (c[90] - c[40]) / 50
    G.journal(p, tmp_path / "f.png")
    assert (tmp_path / "f.png").exists()
