import numpy as np
import pandas as pd

from research.portfolio_sim2 import CAPITAL, Mode, simulate, watch_rank


def _ev(rows):
    ev = pd.DataFrame(rows, columns=["t", "end", "symbol", "r", "risk", "kind", "wrank"])
    ev["t"] = pd.to_datetime(ev["t"], utc=True)
    ev["end"] = pd.to_datetime(ev["end"], utc=True)
    return ev


def test_watch_rank_uses_only_spikes_before_the_month():
    t = pd.to_datetime(["2025-01-05", "2025-01-06", "2025-01-07", "2025-02-03", "2025-02-04"], utc=True)
    sp = pd.DataFrame({"t": t, "symbol": ["A", "B", "B", "A", "C"]})
    r = watch_rank(sp)
    assert np.isinf(r[:3]).all()                          # в январе прошлого нет — очередь пуста
    assert list(r[3:]) == [1.0, np.inf]                   # в феврале: B (2 прострела) — 0, A — 1, C не было


def test_spike_fills_ignore_leverage_cap_and_others_get_cap_minus_reserve():
    mode = Mode("t", oi_frac=0.3, oi_max=3, sp_frac=0.1, sp_reserve=1.0, coil_risk=0.01, ann_frac=0.6,
                unl_frac=0.1, lev_cap=1.5)
    t0 = "2025-03-01 00:00"
    rows = [(t0, "2025-03-01 01:00", f"S{k}", 0.01, np.nan, "spikes", float(k)) for k in range(10)]
    rows.append(("2025-03-01 00:01", "2025-03-02 00:00", "X", 0.10, np.nan, "announce", np.inf))   # 0.6 > 1.5 − 1.0
    rows.append(("2025-03-01 00:02", "2025-03-02 00:00", "Y", 0.10, np.nan, "unlock", np.inf))     # 0.1 <= 0.5
    eq, ruined, done = simulate(_ev(rows), mode)
    # 10 монет под лимитками (1.0 x 100 / 10), все исполнились сразу, хотя номинал 1.0 + 0.1 > свободного места
    assert done == 11 and not ruined
    assert np.isclose(eq.iloc[-1], CAPITAL + 10 * 10 * 0.01 + 10 * 0.10)


def test_exchange_leverage_frees_margin_for_other_strategies():
    rows = [("2025-03-01 00:01", "2025-03-02 00:00", "X", 0.10, np.nan, "announce", np.inf),
            ("2025-03-01 00:02", "2025-03-02 00:00", "S0", 0.01, np.nan, "spikes", 0.0)]
    base = dict(oi_frac=0.3, oi_max=3, sp_frac=0.1, sp_reserve=4.0, coil_risk=0.01, ann_frac=3.0, unl_frac=0.1,
                lev_cap=5.0)
    # без плеча биржи резерв 4 съедает место: анонсу с номиналом 3 остаётся 1 — сделка пропущена
    _, _, done = simulate(_ev(rows), Mode("a", **base))
    assert done == 1
    # плечо 10: маржа (3 + 4) / 10 = 0.7 капитала, позиции 3 <= 5 — сделка проходит
    _, _, done = simulate(_ev(rows), Mode("b", **base, exch_lev=10.0))
    assert done == 2
