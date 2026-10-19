import numpy as np
import pandas as pd

from research.htftrend import coin_rows, cot_gate, parse_cot, usable, weekly


def test_usable_skips_stables_fiat_and_leveraged():
    assert usable("SOLUSDT") and usable("BTCUSDT")
    assert not usable("USDCUSDT") and not usable("EURUSDT") and not usable("BTCUPUSDT")
    assert not usable("ETHBTC") and not usable("BNBUSDT_200925")


def test_weekly_starts_monday_and_sums_volume():
    idx = pd.date_range("2024-01-01", periods=14, freq="1D", tz="UTC")              # понедельник
    d = pd.DataFrame({"open": np.arange(14.0), "high": np.arange(14.0) + 1, "low": np.arange(14.0) - 1,
                      "close": np.arange(14.0) + 0.5, "volume": 1.0, "quote_volume": 2.0,
                      "taker_buy_volume": 0.5}, index=idx)
    w = weekly(d)
    assert list(w.index.weekday) == [0, 0] and w.volume.tolist() == [7.0, 7.0]
    assert w.open.iloc[0] == 0 and w.close.iloc[0] == 6.5 and w.high.iloc[1] == 14


def test_cot_known_from_friday_and_compares_four_reports_back():
    raw = [{"report_date_as_yyyy_mm_dd": f"{d:%Y-%m-%d}T00:00:00.000", "asset_mgr_positions_long": str(100 + 10 * i),
            "asset_mgr_positions_short": "50"} for i, d in enumerate(pd.date_range("2024-01-02", periods=8,
                                                                                   freq="7D"))]
    raw.append({"bad": 1})
    cot = parse_cot(raw)
    assert cot.index[0] == pd.Timestamp("2024-01-05") and len(cot) == 8           # вторник + 3 дня
    known = pd.Series(pd.to_datetime(["2024-01-31", "2024-02-02", "2024-02-03"]))
    g = cot_gate(cot, known)
    # 5-й отчёт (вторник 30.01) публикуется 02.02: до этого сравнения с 4 отчётами назад ещё нет
    assert g.tolist() == [False, True, True]


def test_coin_rows_daily_and_weekly():
    rng = np.random.default_rng(6)
    n = 1500
    idx = pd.date_range("2018-01-01", periods=n, freq="1D", tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.04, n)))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.01, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.01, n)))
    d = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": 1.0, "quote_volume": 1e9,
                      "taker_buy_volume": 0.6}, index=idx)
    fn: dict = {}
    rows = coin_rows(d, np.full(n, 1e9), "XUSDT", "1d", fn)
    assert rows and all(r["tf"] == "1d" and r["aggr"] in (0.6, 0.4) and r["slope_atr"] >= 0 for r in rows)
    assert sum(np.isfinite(r["r_3R"]) for r in rows) >= 1
    assert all(k in rows[0] for k in ("m_5R", "r_8R", "r_лестница"))
    assert fn["линий"] >= fn["оборот"] >= fn["тренд"] >= fn["закрытие"] >= fn["свинг"] == len(rows)
    assert coin_rows(d, np.full(n, 1e6), "XUSDT", "1d") == []                  # оборот ниже $5M
    assert all(coin_rows(d.iloc[:0], np.zeros(0), "XUSDT", "1d") == [] for _ in [0])
    w = weekly(d)
    assert all(r["tf"] == "1w" for r in coin_rows(w, np.full(len(w), 1e9), "XUSDT", "1w"))
