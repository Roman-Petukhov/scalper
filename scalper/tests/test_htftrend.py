import numpy as np
import pandas as pd

from research.htftrend import coin_rows, cot_gate, mfe, parse_cot, reentry, usable


def test_usable_skips_stables_fiat_and_leveraged():
    assert usable("SOLUSDT") and usable("BTCUSDT")
    assert not usable("USDCUSDT") and not usable("EURUSDT") and not usable("BTCUPUSDT")
    assert not usable("ETHBTC") and not usable("BNBUSDT_200925")


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


def test_coin_rows_stops_exits_and_funnel():
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
    rows = coin_rows(d, np.full(n, 1e9), "XUSDT", "spot", fn)
    assert rows and all(r["src"] == "spot" and r["aggr"] in (0.6, 0.4) and r["slope_atr"] >= 0 for r in rows)
    assert sum(np.isfinite(r["retest|PIV5|3R"]) for r in rows) >= 1
    assert all(k in rows[0] for k in ("market|1.5 ATR|20R", "retest|PIV3|12R", "mfe|2.5 ATR", "c2|retest|PIV5"))
    assert fn["линий"] >= fn["оборот"] >= fn["тренд"] >= fn["закрытие"] >= fn["свинг"] >= fn["риск в окне"] \
        >= fn["ретест исполнен"]
    m = [r["mfe|PIV5"] for r in rows if np.isfinite(r["mfe|PIV5"])]
    assert m and min(m) >= 0
    assert coin_rows(d, np.full(n, 1e6), "XUSDT", "spot") == []                # оборот ниже $5M


def test_mfe_stops_counting_at_stop_bar():
    hi = np.array([10.0, 11.0, 13.0, 30.0, 40.0])
    lo = np.array([9.5, 10.0, 12.0, 8.0, 30.0])
    assert mfe(hi, lo, 0, 1, 10.0, 9.0, 10) == 3.0                             # бар 3 выбил стоп: его 30 не в зачёт
    assert mfe(hi, lo, 3, 1, 10.0, 9.0, 10) == 0.0                             # стоп на баре входа


def test_reentry_when_line_holds_and_cancel_when_it_breaks():
    # лонг: линия 100 (наклон 0), стоп за свингом 99; бар 2 снимает стоп тенью, закрытия держат линию
    hi = np.array([101.0, 101.5, 101.0, 102.0, 103.0] + [103.0] * 20)
    lo = np.array([100.0, 100.0, 98.5, 100.5, 101.0] + [101.0] * 20)
    c = np.array([100.5, 101.0, 100.2, 101.5, 102.0] + [102.0] * 20)
    e, st = reentry(hi, lo, c, 1, 1, 99.0, 100.0, 0.0, 0, 1.0, 101.5 + 10)
    assert e == 3 and abs(st - (98.5 - 0.1)) < 1e-9
    c2 = c.copy()
    c2[2] = 99.5                                                               # закрытие под линией — отмена
    assert reentry(hi, lo, c2, 1, 1, 99.0, 100.0, 0.0, 0, 1.0, 111.5)[0] == -1
    assert reentry(hi, lo, c, 1, 1, 99.0, 100.0, 0.0, 0, 1.0, 101.2)[0] == -1  # до снятия цена дошла до +1.5R


def test_beta_matches_trader_and_hedge_leg():
    from trader.domain.hedge import beta
    from research.htftrend import beta_at, hedge_r
    rng = np.random.default_rng(3)
    idx = pd.date_range("2024-01-01", periods=400, freq="1D", tz="UTC")
    btc = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, 400)))
    coin = 5 * np.exp(1.4 * np.log(btc / 100) + np.cumsum(rng.normal(0, 0.02, 400)))
    b = beta_at(coin, btc, 300)
    assert abs(b - beta(pd.Series(coin[:300], idx[:300]), pd.Series(btc[:300], idx[:300]), 180)) < 1e-9
    assert np.isnan(beta_at(coin, btc, 100))                                      # меньше 120 доходностей
    btc2 = np.array([100.0, 110.0])
    assert abs(hedge_r(btc2, -1, 1.5, 50.0, 2.0, 0, 1) - (1.5 * 25 * 0.10 - 2 * 5.5e-4 * 1.5 * 25)) < 1e-12
    assert hedge_r(btc2, 1, np.nan, 50.0, 2.0, 0, 1) == 0.0


def test_coin_rows_hedge_columns():
    rng = np.random.default_rng(6)
    n = 1500
    idx = pd.date_range("2018-01-01", periods=n, freq="1D", tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.04, n)))
    o = np.r_[c[0], c[:-1]]
    d = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.01, "low": np.minimum(o, c) * 0.99, "close": c,
                      "volume": 1.0, "quote_volume": 1e9, "taker_buy_volume": 0.6}, index=idx)
    rows = coin_rows(d, np.full(n, 1e9), "XUSDT", "perp", btc=c * 300)
    filled = [r for r in rows if np.isfinite(r["retest|1.5 ATR|3R"])]
    assert filled and all(abs(r["beta|1.5 ATR"] - 1.0) < 1e-9 for r in filled if np.isfinite(r["beta|1.5 ATR"]))
    assert all("h|1.5 ATR|8R" in r for r in filled)
