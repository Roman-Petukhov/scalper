import json

import numpy as np
import pandas as pd

from research import figure as G
from research import formation as F
from tests.test_formation import _rw


def test_classify_shapes_by_slopes_and_convergence():
    w = 10.0                                    # средняя ширина; n = 100 → «ровная» граница сдвинулась < 2.5
    assert G.classify(0.0, 0.01, 100, w, 1.0, 1) == "box"
    assert G.classify(-0.1, -0.1, 100, w, 1.0, 1) == "channel_counter"       # падающий канал, пробой вверх — флаг
    assert G.classify(-0.1, -0.1, 100, w, 1.0, -1) == "channel_with"
    assert G.classify(0.0, 0.1, 100, w, 0.3, 1) == "tri_flat_break"          # восходящий треугольник вверх
    assert G.classify(-0.1, 0.0, 100, w, 0.3, 1) == "tri_flat_opp"
    assert G.classify(-0.1, 0.1, 100, w, 0.3, 1) == "tri_sym"
    assert G.classify(-0.2, -0.1, 100, w, 0.3, 1) == "wedge_counter"         # падающий клин вверх
    assert G.classify(0.1, 0.2, 100, w, 0.3, 1) == "wedge_with"
    assert G.classify(-0.1, 0.1, 100, w, 2.0, 1) == "expanding"


def test_opposite_boundary_and_touches():
    c = np.array([10.0, 8.0, 9.0, 7.0, 9.5, 8.5, 9.8, 8.6, 9.9])
    j1, j2, so, o1 = G.opposite(c, 0, 8, 1, 0.0)      # пробой вверх, разворотов < 2 — крайние минимумы половин
    assert (j1, j2) == (3, 5) and so == (8.5 - 7.0) / 2 and abs(o1 - 5.6) < 1e-9    # сдвинута под закрытие 8.6
    c = np.r_[10.0, np.tile([9.0, 8.0, 9.0, 9.5, 9.6, 9.5], 5), 10.5]
    j1, j2, so, o1 = G.opposite(c, 0, len(c) - 1, 1, 0.0)     # впадины 8.0 на одной высоте — ровная граница
    assert c[j1] == 8.0 and c[j2] == 8.0 and abs(so) < 1e-12 and abs(o1 - 8.0) < 1e-9
    rng = np.random.default_rng(3)
    for _ in range(50):                                   # ни одно закрытие не выходит за вторую границу
        c = 100 + np.cumsum(rng.normal(0, 1, 120))
        side, sb = int(rng.choice([-1, 1])), float(rng.normal(0, 0.05))
        c[0] = c.max() + 1 if side > 0 else c.min() - 1
        j1, j2, so, o1 = G.opposite(c, 0, 119, side, sb)
        opp = o1 + so * (np.arange(119) - j1)
        assert (side * (c[:119] - opp) >= -1e-9).all() and j1 <= j2
    tp = G.touch_points(np.array([10.0, 9.0, 9.9, 9.0, 9.0, 9.0, 10.0, 9.95]), np.full(8, 1.0), 0, 8, 1, 0.0)
    assert [j for j, _ in tp] == [0, 6] and tp[1][1] == 0.0               # подход 6–7: ближайшая свеча


def test_features_on_a_random_walk(monkeypatch):
    monkeypatch.setattr(F, "MAX_SLOPE", 1e9)
    rows = pd.DataFrame(F.coin_rows(_rw(1), np.full(30000, 1e9), "RW", "15m", G.figure_features))
    assert len(rows) > 20 and set(rows["shape"]) <= set(G.SHAPES) and rows["shape"].nunique() >= 3
    assert rows[["conv", "body", "long_share", "wick", "contraction", "touch_gap"]].notna().all().all()
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
