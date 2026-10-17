import numpy as np
import pandas as pd

import research.quarterly as q


def test_verdict_bands():
    assert q.verdict(0.3) == "держится" and q.verdict(-1.0) == "держится"
    assert q.verdict(-1.5) == "под наблюдением" and q.verdict(float("nan")) == "под наблюдением"
    assert q.verdict(-2.01) == "стареет"


def test_mean_se_clusters_same_day_trades():
    r = np.array([1.0, 1.0, -1.0, -1.0])
    m, se_days = q.mean_se(r, np.array(["a", "a", "b", "b"]))           # два дня — две независимые точки
    _, se_each = q.mean_se(r, np.array(["a", "b", "c", "d"]))
    assert m == 0 and se_days > se_each


def _trades(mean_old: float, mean_new: float, n: int = 400, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.Series(pd.date_range("2023-01-01", "2026-09-30", periods=n, tz="UTC"))
    new = t >= pd.Timestamp("2025-10-01", tz="UTC")
    r = np.where(new, mean_new, mean_old) + rng.normal(0, 0.3, n)
    return pd.DataFrame({"t": t, "R": r, "aggr": rng.random(n) < 0.5})


def test_trades_check_flags_decay_only_when_window_is_worse():
    start = pd.Timestamp("2025-10-01", tz="UTC")
    ok = _trades(0.5, 0.45)
    bad = _trades(0.5, -0.4)
    assert q.trades_check("x", ok.t, ok.R, start, 0.22).level == "держится"
    c = q.trades_check("x", bad.t, bad.R, start, 0.22)
    assert c.level == "стареет" and "2026" in c.history


def test_diff_check_uses_half_of_past_gap():
    df = _trades(0.0, 0.0)
    start = pd.Timestamp("2025-10-01", tz="UTC")
    df.loc[df.aggr & (df.t < start), "R"] += 0.6                         # до окна фильтр работал
    assert "+0.30R" in q.diff_check("a", df, start).norm
    assert q.diff_check("a", df, start).level == "стареет"              # в окне разницы нет
    df.loc[df.aggr & (df.t >= start), "R"] += 0.6
    assert q.diff_check("a", df, start).level == "держится"


def test_sharpe_check_and_render():
    idx = pd.date_range("2022-01-01", "2026-09-30", freq="1D")
    rng = np.random.default_rng(0)
    good = pd.Series(rng.normal(0.002, 0.02, len(idx)), index=idx)
    start = idx[-365]
    c_ok = q.sharpe_check("тренд", good, start, 0.635)
    bad = good.copy()
    bad[bad.index >= start] = rng.normal(-0.004, 0.02, 365)
    c_bad = q.sharpe_check("тренд", bad, start, 0.635)
    assert c_ok.level == "держится" and c_bad.level == "стареет"
    text = q.render([c_ok, c_bad], idx[-1])
    assert text.startswith("# Квартальная пересдача: СТАРЕЕТ") and "| тренд |" in text


def test_report_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("PARTS", str(tmp_path / "parts"))
    monkeypatch.setattr(q, "OUT", str(tmp_path / "out" / "quarterly.md"))
    days = pd.date_range("2022-01-01", "2026-09-30", freq="1D", tz="UTC")
    rng = np.random.default_rng(3)
    rows = []
    for j in range(25):
        c = 10 * np.exp(np.cumsum(rng.normal(0.001, 0.04, len(days))))
        rows.append(pd.DataFrame({"symbol": f"C{j}USDT", "day": days, "close": c, "funding": 0.0001,
                                  "adv": 5e7 + j * 1e6}))
    pd.concat(rows).to_parquet(q.part_path("quarterly_daily"), index=False)
    h4 = _trades(0.5, 0.45).assign(symbol="C1USDT", side=1)
    h4.loc[h4.aggr, "R"] += 0.4
    h4.to_parquet(q.part_path("quarterly4h"), index=False)
    q.report()
    text = (tmp_path / "out" / "quarterly.md").read_text(encoding="utf-8")
    assert "бот 4h" in text and "фильтр агрессоров 4h" in text and "тренд (без данных притока" in text


def test_negative_window_is_never_holding():
    assert q.Check("x", "", "", 0.5, -0.01).level == "под наблюдением"
    assert q.Check("x", "", "", 0.5, 0.2).level == "держится"
    assert q.Check("x", "", "", -3.0, 0.2).level == "стареет"
