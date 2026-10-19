"""Правило бота 4h на дневках: спот Binance с 2017 (с делистнутыми) и перпетуалы с 2020, приток и COT.

Правило бота (как research/topcrowd.py coin_rows): пробой линии зигзага 3 ATR на закрытии дневки, неделя по
направлению (закрытие против EMA50 последней закрытой недели), закрытие в своей половине свечи, вход на ретесте линии
(12 свечей) или по рынку, долив +0.5 на +1.5R. Издержки — maker / taker Bybit; funding — на перпетуалах, на споте 0.
Почему спот: архив перпетуалов — с 2020 года, спот — с 2017-08 вместе с делистнутыми парами (без ошибки выжившего).

История: прогон 1 — 264 пробоя (1d + 1w), агрессоры ≥ 55% на дневках почти не встречаются (дневная доля тейкеров
~50%); прогон 2 — оборот на споте в разы меньше, чем на перпетуалах: от $20M — 226 пробоев, от $5M — 606 (+0.46R,
t 3.7, плюс во всех периодах на 3R); 8R лучше 3R в среднем, но не во всех периодах; приток на дневках работает
наоборот, COT — ноль; недели — 38 пробоев за 9 лет, сняты.

Прогон 3 (задан до запуска, по вопросам владельца: «стоп за свингом на дневках не слишком широкий?», «цели и больше
8R», «стоп снимают, но цена держит пробитую линию и уходит»):
    стоп     PIV5 ± 0.1 ATR (бот, риск 0.3–4 ATR) / PIV3 / PIV5 без потолка / 1.5 и 2.5 ATR от линии
    цель     3R (срок 60) / 5R / 8R (180) / 12R / 20R (360)
    MFE      какая доля сделок на ретесте дошла до 1 / 3 / 5 / 8 / 12 / 20R раньше стопа
    перезаход стоп за свингом выбит раньше +1.5R; дальше 12 свечей: закрытие за линией (продолжена по наклону) —
             отмена; закрытие обратно за снятым свингом (не на баре снятия) — вход по рынку, стоп за экстремумом
             снятия ± 0.1 ATR; считаются сам перезаход и «сигнал + перезаход» на один сигнал
    оборот   $5M / $10M / $20M; издержки x2 — ещё один круг комиссий входа и выхода
Прогон 4 (по просьбе владельца, до запуска): хедж BTC как в бумажной книге — обратная позиция BTCUSDT на бету монеты
к BTC за 180 дней до дня входа, по закрытиям дней входа и выхода, taker на вход и выход; только вход на ретесте.
Прогон 5: хеджируется и доливка (+0.5 на +1.5R) — своя нога BTC с закрытия дня доливки.
Хедж годится, если R на сделку с ним не хуже в большинстве периодов, а σ месяца и худший месяц меньше.
Решение: вариант лучше бота (PIV5, 3R), если R за всё время выше и в плюсе во всех периодах с ≥ 10 сделками, на
споте и на перпетуалах, и при издержках x2 знак тот же; перезаход полезен, если «сигнал + перезаход» выше «сигнала»
в большинстве периодов.

    python -m research.htftrend collect --perp <1h-qualified>   (спот качает сам; COT — часть 0)
    python -m research.htftrend report                          (+ python -m research.extreg collect)
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .broad import ADV_MIN, adv30
from .dboost2 import inflow_gate
from .dconf import Exit, simulate, simulate_bar
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import MAKER, STOP_BUF, TAKER, htf_trend, last_confirmed, pivots, retest_fill, tf_frame, zz_lines

START = "2017-08"
COT_URL = ("https://publicreporting.cftc.gov/resource/gpe5-46if.json?cftc_contract_market_code=133741"
           "&$limit=5000&$order=report_date_as_yyyy_mm_dd")     # CFTC TFF, CME Bitcoin (5 BTC), еженедельно с 2017-12
COT_WEEKS = 4
SPOT_LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=/&prefix=data/spot/monthly/klines/"
STABLE = ("USDC", "BUSD", "TUSD", "FDUSD", "DAI", "USDP", "PAX", "USDS", "SUSD", "UST", "USTC", "EUR", "GBP", "AUD",
          "TRY", "BRL", "RUB", "BIDR", "IDRT", "NGN", "UAH", "ZAR", "PLN", "RON", "ARS", "JPY", "BKRW", "EURI", "AEUR",
          "USD1", "XUSD", "BFUSD", "PAXG")
LEVERAGED = re.compile(r"(UP|DOWN|BULL|BEAR)USDT$")
HOLD, RETEST = 60, 12            # дневки: срок цели 3R и окно ретеста, свечей
LIQ = (5e6, 10e6, ADV_MIN)       # пороги оборота: на споте он в разы меньше, чем на перпетуалах
PERP_START = "2020-01"


def exits() -> dict[str, Exit]:
    """Выходы (заданы до третьего прогона): 3R — бот; 5R / 8R со сроком втрое длиннее; 12R / 20R — вшестеро."""
    return {"3R": Exit(3.0, (1.5,), 0.5, HOLD), "5R": Exit(5.0, (1.5,), 0.5, 3 * HOLD),
            "8R": Exit(8.0, (1.5,), 0.5, 3 * HOLD), "12R": Exit(12.0, (1.5,), 0.5, 6 * HOLD),
            "20R": Exit(20.0, (1.5,), 0.5, 6 * HOLD)}


# стоп: (фрактал свинга или 0 — фиксированный от линии, ATR от линии / отступ за свингом, потолок риска в ATR)
STOPS = {"PIV5": (5, STOP_BUF, 4.0), "PIV3": (3, STOP_BUF, 4.0), "PIV5 без потолка": (5, STOP_BUF, np.inf),
         "1.5 ATR": (0, 1.5, np.inf), "2.5 ATR": (0, 2.5, np.inf)}
MFE_R = (1, 3, 5, 8, 12, 20)
FUNNEL = ("линий", "оборот", "тренд", "закрытие", "свинг", "риск в окне", "ретест исполнен")
PER = {"2017–18": ("2017-08-01", "2019-01-01"), "2019": ("2019-01-01", "2020-01-01"),
       "2020": ("2020-01-01", "2021-01-01"), "2021": ("2021-01-01", "2022-01-01"),
       "is": ("2022-01-01", "2024-07-01"), "val": ("2024-07-01", "2025-07-01"), "ho": ("2025-07-01", "2026-10-01")}


def list_spot() -> list[str]:
    syms, marker = [], ""
    while True:
        x = urllib.request.urlopen(SPOT_LIST + (f"&marker={marker}" if marker else ""), timeout=60).read().decode()
        syms += re.findall(r"<Prefix>data/spot/monthly/klines/([^/]+)/</Prefix>", x)
        m = re.search(r"<NextMarker>([^<]+)</NextMarker>", x)
        if not m:
            break
        marker = m.group(1)
    return sorted(s for s in syms if usable(s))


def usable(sym: str) -> bool:
    """USDT-пары монет: без стейблкоинов, фиатов и токенов с плечом."""
    return sym.endswith("USDT") and "_" not in sym and sym[:-4] not in STABLE and not LEVERAGED.search(sym)


def load_daily(sym: str, cache: Path, end: str) -> pd.DataFrame | None:
    parts = [p for p in (D.fetch_klines(sym, "1d", m, cache, "spot") for m in D.months(START, end)) if p is not None]
    if not parts:
        return None
    k = pd.concat([pd.read_parquet(p) for p in parts]).drop_duplicates("open_time").sort_values("open_time")
    k.index = pd.to_datetime(k["open_time"], unit="ms", utc=True)
    return k.drop(columns=["open_time"])


def mfe(hi: np.ndarray, lo: np.ndarray, e: int, side: int, entry: float, stop: float, hold: int) -> float:
    """Максимальный ход в пользу позиции, R, от входа на баре e до стопа или конца срока (бар со стопом не в зачёт)."""
    risk = side * (entry - stop)
    if not risk > 0 or (side > 0 and lo[e] <= stop) or (side < 0 and hi[e] >= stop):
        return 0.0
    best = 0.0
    for j in range(e + 1, min(e + hold, len(hi) - 1) + 1):
        if (side > 0 and lo[j] <= stop) or (side < 0 and hi[j] >= stop):
            break
        best = max(best, side * ((hi[j] if side > 0 else lo[j]) - entry) / risk)
    return best


def _stop(kind: tuple, side: int, level: float, sw_lo: dict, sw_hi: dict, hi, lo, t: int, atr: float) -> float:
    piv, k, _ = kind
    if piv == 0:
        return level - side * k * atr
    sw = (sw_lo if side > 0 else sw_hi)[piv][t]
    if sw < 0:
        return np.nan
    return lo[sw] - k * atr if side > 0 else hi[sw] + k * atr


def reentry(hi, lo, c, fill: int, side: int, stop: float, level: float, slope: float, t: int, atr: float,
            target: float) -> tuple[int, float]:
    """Перезаход после снятия стопа за свингом, если пробитая линия устояла (правило владельца, задано до прогона):
    стоп выбит на баре s (раньше, чем цена дошла до target — ступени, после которой стоп переносится); дальше
    RETEST свечей: закрытие за линией (продолжена по наклону) — отмена; закрытие обратно за снятым свингом — вход
    по закрытию, стоп — за экстремумом снятия ± STOP_BUF ATR. Возвращает (бар входа, стоп) или (−1, nan)."""
    n, s_ = len(c), -1
    for j in range(fill, n):
        if (side > 0 and lo[j] <= stop) or (side < 0 and hi[j] >= stop):
            s_ = j
            break
        if (side > 0 and hi[j] >= target) or (side < 0 and lo[j] <= target):
            return -1, np.nan
    if s_ < 0:
        return -1, np.nan
    ext = lo[s_] if side > 0 else hi[s_]
    for j in range(s_, min(s_ + RETEST, n - 2) + 1):
        ext = min(ext, lo[j]) if side > 0 else max(ext, hi[j])
        line_j = level + slope * (j - t)
        if side * (c[j] - line_j) < 0:
            return -1, np.nan
        if j > s_ and side * (c[j] - stop) > 0:
            return j, ext - side * STOP_BUF * atr
    return -1, np.nan


BETA_DAYS, BETA_MIN, BETA_RANGE = 180, 120, (0.0, 3.0)      # как trader/domain/hedge.beta на дневках бумажной книги
HEDGE_FEE = TAKER


def beta_at(coin: np.ndarray, btc: np.ndarray, e: int) -> float:
    """Бета логдоходностей монеты к BTC по дневкам до бара e (не включая), окно BETA_DAYS; nan — данных мало."""
    x, y = coin[max(0, e - BETA_DAYS - 1): e], btc[max(0, e - BETA_DAYS - 1): e]
    ok = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < BETA_MIN + 1:
        return np.nan
    rx, ry = np.diff(np.log(x)), np.diff(np.log(y))
    var = ry.var(ddof=1)
    if not var > 0:
        return np.nan
    b = np.cov(rx, ry, ddof=1)[0, 1] / var
    return float(min(max(b, BETA_RANGE[0]), BETA_RANGE[1])) if np.isfinite(b) else np.nan


def hedge_r(btc: np.ndarray, side: int, b: float, level: float, rr: float, e: int, j: int,
            adds: list[tuple[int, float]] = (), add_size: float = 0.0) -> float:
    """Нога BTC в R сделки: обратная позиция на бету × номинал (номинал / риск = level / rr) по закрытиям дня
    входа e и выхода j; доливка (бар, цена) — своя нога на бету × add_size × номинал по цене доливки, с закрытия её
    дня; комиссия taker на вход и выход — как бумажная книга trader/domain/daily_break."""
    if not (b > 0) or not (btc[e] > 0) or not (btc[j] > 0):
        return 0.0
    legs = [(e, b * level / rr)] + [(k, b * add_size * px / rr) for k, px in adds if add_size > 0 and btc[k] > 0]
    return sum(-side * h * (btc[j] / btc[k] - 1) - 2 * HEDGE_FEE * h for k, h in legs)


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, src: str, funnel: dict | None = None,
              btc: np.ndarray | None = None) -> list[dict]:
    """Пробои по правилу бота на дневках с оборотом от LIQ[0]. Колонки «вход|стоп|выход» — R; «c2|вход|стоп» —
    добавка к издержкам в R при издержках x2 (ещё один круг комиссий входа и выхода); «mfe|стоп» — на ретесте;
    «h|стоп|выход» — нога хеджа BTC в R для входа на ретесте (btc — закрытия BTC на тех же днях), «beta|стоп».
    funnel — сколько пробоев дошло до каждого шага правила (стоп PIV5, вход на ретесте)."""
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = d["funding"].fillna(0.0).to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    a = _atr(d).to_numpy()
    trend = htf_trend(d, "1d")                                     # неделя против EMA50 (последняя закрытая неделя)
    buy = (d["taker_buy_volume"] / d["volume"].replace(0, np.nan)).to_numpy()
    rng = hi - lo
    loc = (c - lo) / np.where(rng > 0, rng, np.nan)
    pivs = {k[0] for k in STOPS.values() if k[0]}
    sw_lo = {p: last_confirmed(pivots(lo, p, False), len(c)) for p in pivs}
    sw_hi = {p: last_confirmed(pivots(hi, p, True), len(c)) for p in pivs}
    exs = exits()
    fn = funnel if funnel is not None else {}

    def bump(k: str) -> None:
        fn[k] = fn.get(k, 0) + 1

    rows = []
    for r in zz_lines(d):
        t, side = int(r["t"]), int(r["side"])
        if t <= 0 or t >= len(c) - 1 or not (a[t] > 0):
            continue
        bump("линий")
        if not (liq[t] >= LIQ[0]):
            continue
        bump("оборот")
        if trend[t] != side:
            continue
        bump("тренд")
        cl = loc[t] if side > 0 else 1 - loc[t]
        if not (cl >= 0.5):
            continue
        bump("закрытие")
        level = float(r["line_t"])
        row = {"symbol": sym, "src": src, "t": d.index[t], "side": side, "liq": float(liq[t]),
               "aggr": float(buy[t] if side > 0 else 1 - buy[t]), "slope_atr": abs(float(r["slope"])) / a[t]}
        any_ok = False
        for sn, kind in STOPS.items():
            stop = _stop(kind, side, level, sw_lo, sw_hi, hi, lo, t, a[t])
            if sn == "PIV5" and np.isfinite(stop):
                bump("свинг")
            cap = kind[2]
            risk, rr = side * (c[t] - stop), side * (level - stop)
            m_ok = np.isfinite(stop) and 0.3 * a[t] <= risk <= cap * a[t]
            r_ok = np.isfinite(stop) and 0.3 * a[t] <= rr <= cap * a[t]
            if sn == "PIV5" and r_ok:
                bump("риск в окне")
            fill = retest_fill(hi, lo, t, side, level, 3.0 * rr, RETEST) if r_ok else -1
            if sn == "PIV5" and fill >= 0:
                bump("ретест исполнен")
            stopped = fill >= 0 and ((side > 0 and lo[fill] <= stop) or (side < 0 and hi[fill] >= stop))
            row[f"c2|market|{sn}"] = (TAKER + TAKER) * c[t] / risk if m_ok else np.nan
            row[f"c2|retest|{sn}"] = (MAKER + TAKER) * level / rr if fill >= 0 else np.nan
            row[f"mfe|{sn}"] = mfe(hi, lo, fill, side, level, stop, 6 * HOLD) if fill >= 0 else np.nan
            for k, ex in exs.items():
                row[f"market|{sn}|{k}"] = simulate(o, hi, lo, c, f, t, side, c[t], stop, a[t], ex, TAKER) \
                    if m_ok else np.nan
                adds: list[tuple[int, float]] = []
                if fill < 0:
                    row[f"retest|{sn}|{k}"], j = np.nan, -1
                elif stopped:
                    row[f"retest|{sn}|{k}"], j = (side * (stop - level) - (MAKER + TAKER) * level) / rr, fill
                else:
                    row[f"retest|{sn}|{k}"], j = simulate_bar(o, hi, lo, c, f, fill, side, level, stop, a[t], ex, MAKER,
                                                              adds)
                if btc is not None and fill >= 0 and j >= 0:
                    b = row.setdefault(f"beta|{sn}", beta_at(c, btc, fill))
                    row[f"h|{sn}|{k}"] = hedge_r(btc, side, b, level, rr, fill, j, adds, ex.add_size)
            if kind[0] and fill >= 0:
                e2, st2 = reentry(hi, lo, c, fill, side, stop, level, float(r["slope"]), t, a[t], level + side * 1.5 * rr)
                risk2 = side * (c[e2] - st2) if e2 >= 0 else np.nan
                ok2 = e2 >= 0 and 0.3 * a[t] <= risk2 <= kind[2] * a[t]
                row[f"c2|reentry|{sn}"] = 2 * TAKER * c[e2] / risk2 if ok2 else np.nan
                for k, ex in exs.items():
                    row[f"reentry|{sn}|{k}"] = simulate(o, hi, lo, c, f, e2, side, c[e2], st2, a[t], ex, TAKER) \
                        if ok2 else np.nan
            any_ok |= bool(m_ok or fill >= 0)
        if any_ok:
            rows.append(row)
    return rows


def parse_cot(raw: list) -> pd.Series:
    """Чистый лонг управляющих активами по дате публикации (вторник отчёта + 3 дня)."""
    rows = []
    for r in raw if isinstance(raw, list) else []:
        try:
            day = pd.Timestamp(str(r["report_date_as_yyyy_mm_dd"])[:10])
            rows.append((day + pd.Timedelta(days=3),
                         float(r["asset_mgr_positions_long"]) - float(r["asset_mgr_positions_short"])))
        except (KeyError, TypeError, ValueError):
            continue
    s = pd.Series(dict(rows), dtype="float64").sort_index()
    return s[~s.index.duplicated()]


def collect_cot() -> None:
    import json
    try:
        with urllib.request.urlopen(urllib.request.Request(COT_URL, headers={"User-Agent": "Mozilla/5.0"}),
                                    timeout=60) as r:
            raw = json.loads(r.read().decode())
    except Exception as e:
        print(f"  htftrend: COT недоступен ({e})", flush=True)
        return
    cot = parse_cot(raw)
    if not len(cot):
        print(f"  htftrend: COT — не разобрал ответ: {str(raw)[:300]}", flush=True)
        return
    print(f"  htftrend: COT {len(cot)} отчётов ({cot.index.min():%Y-%m-%d} … {cot.index.max():%Y-%m-%d})", flush=True)
    pd.DataFrame({"day": cot.index, "net": cot.to_numpy()}).to_parquet(part_path("htftrend_cot"), index=False)


def collect(root: Path, end: str, perp: list[str]) -> None:
    from .shard import shard_id
    if shard_id()[0] == 0:
        collect_cot()
    syms = mine(list_spot())
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    rows, funnel = [], {"spot": {}, "perp": {}}
    with ThreadPoolExecutor(16) as pool:
        frames = dict(zip(syms, pool.map(lambda s: load_daily(s, cache, end), syms)))
    btc_spot = load_daily("BTCUSDT", cache, end)
    btc_perp = tf_frame(root, "BTCUSDT", "1d") if perp else None

    def on(btc: pd.DataFrame | None, d: pd.DataFrame) -> np.ndarray | None:
        return btc["close"].reindex(d.index).to_numpy(dtype="float64") if btc is not None else None

    print(f"  htftrend: BTC для хеджа — спот {'есть' if btc_spot is not None else 'НЕТ'}, "
          f"перпетуал {'есть' if btc_perp is not None else 'НЕТ'}", flush=True)
    n_spot = n_perp = 0
    for s, d in frames.items():
        try:
            if d is None or len(d) < 120:
                continue
            n_spot += 1
            liq = d["quote_volume"].rolling(30, min_periods=20).mean().shift(1)
            rows += coin_rows(d, liq.to_numpy(), s, "spot", funnel["spot"], on(btc_spot, d) if s != "BTCUSDT" else None)
        except Exception as e:
            print(f"  htftrend {s}: пропуск ({e})", flush=True)
    for s in mine(perp):
        try:
            d = tf_frame(root, s, "1d")
            if d is None or len(d) < 120:
                continue
            n_perp += 1
            liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
            rows += coin_rows(d, liq, s, "perp", funnel["perp"], on(btc_perp, d) if s != "BTCUSDT" else None)
        except Exception as e:
            print(f"  htftrend perp {s}: пропуск ({e})", flush=True)
    print(f"  htftrend: спот {n_spot} из {len(syms)} монет, перпетуалы {n_perp}, пробоев {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("htftrend"), index=False)
    pd.DataFrame([{"src": k, "step": st, "n": v} for k, fn in funnel.items() for st, v in fn.items()]) \
        .to_parquet(part_path("htftrend_funnel"), index=False)


def _gate(df: pd.DataFrame) -> pd.Series | None:
    parts = all_parts("extreg")
    if not parts:
        return None
    ext = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    ext["day"] = pd.to_datetime(ext["day"], utc=True).dt.tz_localize(None)
    st = ext.groupby("day")["stable"].last().dropna()
    idx = pd.date_range(st.index.min(), st.index.max(), freq="1D")
    g = inflow_gate(st, idx)
    return pd.Series(g.reindex(_known(df)).to_numpy(), index=df.index)


def cot_gate(cot: pd.Series, known: pd.Series) -> pd.Series:
    """True — на этот день последний опубликованный чистый лонг управляющих выше, чем COT_WEEKS отчётов назад."""
    up = (cot > cot.shift(COT_WEEKS)).astype(float).where(cot.shift(COT_WEEKS).notna())
    days = pd.date_range(cot.index.min(), max(cot.index.max(), known.max()), freq="1D")
    daily = up.reindex(days, method="ffill")
    return pd.Series(daily.reindex(known.to_numpy()).to_numpy() == 1.0, index=known.index)


def _known(df: pd.DataFrame) -> pd.Series:
    return df.t.dt.tz_localize(None).dt.normalize()                 # дневка закрыта — её день уже известен


ALL = ("всё время", (PER["2017–18"][0], PER["ho"][1]))
SRC = {"spot": "спот Binance с 2017", "perp": "перпетуалы Binance с 2020 (с funding)"}


def _stats(z: pd.DataFrame, col: str, c2: str | None = None) -> tuple[str, list[bool]]:
    cells, ok = [], []
    for p, (a, b) in (*PER.items(), ALL):
        m = z[(z.t >= a) & (z.t < b) & z[col].notna()]
        if p in ("2017–18", "2019") and not len(m):
            continue
        cells.append(f"{p}: {_cell(m.rename(columns={col: 'R'}))}")
        if p != ALL[0] and len(m) >= 10:
            ok.append(m[col].mean() > 0)
    if c2 is not None:
        m = z[z[col].notna()]
        if len(m) >= 10:
            cells.append(f"x2: {(m[col] - m[c2].fillna(0)).mean():+.3f}R")
    return "  ".join(cells), ok


def _verdict(ok: list[bool]) -> str:
    return f"плюс в {sum(ok)} из {len(ok)}" if ok else "мало сделок"


def _funnel() -> None:
    parts = all_parts("htftrend_funnel")
    if not parts:
        return
    f = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).groupby(["src", "step"]).n.sum()
    for src in SRC:
        if src in f.index.get_level_values(0):
            steps = "  →  ".join(f"{k} {int(f.get((src, k), 0))}" for k in FUNNEL)
            print(f"  воронка {src} (оборот от ${LIQ[0] / 1e6:.0f}M, стоп PIV5, ретест): {steps}")


def report() -> None:
    parts = all_parts("htftrend")
    print("\n===== HTFTREND: правило бота 4h на дневках — стопы, цели до 20R, MFE, перезаход, спот с 2017 и перпетуалы =====")
    if not parts:
        print("частей нет")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    g = _gate(df)
    df["inflow"] = g.fillna(False).astype(bool) if g is not None else False
    cp = all_parts("htftrend_cot")
    if cp:
        c = pd.concat([pd.read_parquet(p) for p in cp], ignore_index=True)
        cot = pd.Series(c["net"].to_numpy(), index=pd.to_datetime(c["day"])).sort_index()
        df["cot"] = cot_gate(cot, _known(df))
        df["cot_known"] = _known(df) >= cot.index.min() + pd.Timedelta(weeks=COT_WEEKS)
    else:
        df["cot"], df["cot_known"] = False, False
    print(f"пробоев {len(df)} (спот {int((df.src == 'spot').sum())}, перпетуалы {int((df.src == 'perp').sum())}), "
          f"монет {df.symbol.nunique()}; приток {'есть' if g is not None else 'НЕТ'}, COT {'есть' if cp else 'НЕТ'}")
    _funnel()
    ex = list(exits())
    for src, title in SRC.items():
        x0 = df[df.src == src]
        if not len(x0):
            continue
        print(f"\n  ===== {title} =====")
        print("  1) порог оборота (стоп PIV5, вход на ретесте):")
        for lq in LIQ:
            x = x0[x0.liq >= lq]
            for k in ("3R", "8R", "20R"):
                line, ok = _stats(x, f"retest|PIV5|{k}", "c2|retest|PIV5")
                print(f"     от ${lq / 1e6:.0f}M, {k}: {line}  → {_verdict(ok)}")
        base = LIQ[0] if src == "spot" else ADV_MIN
        x = x0[x0.liq >= base]
        print(f"  2) стоп × цель (оборот от ${base / 1e6:.0f}M; итог за всё время, x2 — издержки вдвое):")
        for entry, nm in (("retest", "ретест"), ("market", "по рынку")):
            for sn in STOPS:
                for k in ex:
                    line, ok = _stats(x, f"{entry}|{sn}|{k}", f"c2|{entry}|{sn}")
                    print(f"     {nm}, {sn}, {k}: {line}  → {_verdict(ok)}")
        print("  3) максимальный ход сделки на ретесте (доля сделок, дошедших до N R до стопа, срок 360 дней):")
        for sn in STOPS:
            m = x[f"mfe|{sn}"].dropna()
            if len(m):
                print(f"     {sn} (n={len(m)}): " + "  ".join(f"≥{r}R {np.mean(m >= r):.0%}" for r in MFE_R))
        print("  6) хедж BTC (ретест; нога BTC на бету за 180 дней, по закрытиям дней входа и выхода, taker x2):")
        for sn in ("1.5 ATR", "PIV5"):
            for k in ("3R", "5R", "8R"):
                col, hc = f"retest|{sn}|{k}", f"h|{sn}|{k}"
                if hc not in x:
                    continue
                z = x[x[col].notna() & x[hc].notna()].copy()
                z["hedged"] = z[col] + z[hc]
                for c_, nm in ((col, "без хеджа"), ("hedged", "с хеджем"), (hc, "сама нога BTC")):
                    line, ok = _stats(z, c_)
                    print(f"     {sn}, {k}, {nm}: {line}" + (f"  → {_verdict(ok)}" if c_ != hc else ""))
                for c_, nm in ((col, "без хеджа"), ("hedged", "с хеджем")):
                    mon = z.groupby(z.t.dt.tz_localize(None).dt.to_period("M"))[c_].sum()
                    if len(mon) >= 6:
                        print(f"       {nm}: R в месяц {mon.mean():+.2f}, σ месяца {mon.std():.2f}, худший месяц "
                              f"{mon.min():+.1f}R, Sharpe месяцев {mon.mean() / mon.std() * np.sqrt(12):.2f}")
                if sn == "1.5 ATR" and k == "3R":
                    b = z[f"beta|{sn}"].dropna()
                    print(f"       бета: медиана {b.median():.2f}, доля шортов {(z.side < 0).mean():.0%}")
                    for side, nm in ((1, "лонги"), (-1, "шорты")):
                        for c_, hn in ((col, "без хеджа"), ("hedged", "с хеджем")):
                            line, _ = _stats(z[z.side == side], c_)
                            print(f"       {nm}, {hn}: {line}")
        print("  4) перезаход после снятия стопа за свингом, пока линия держится:")
        for sn in (k for k, v in STOPS.items() if v[0]):
            for k in ("3R", "8R"):
                col = f"reentry|{sn}|{k}"
                if col not in x:
                    continue
                line, ok = _stats(x, col, f"c2|reentry|{sn}")
                print(f"     {sn}, {k}, сам перезаход: {line}  → {_verdict(ok)}")
                both = x[x[f"retest|{sn}|{k}"].notna()].copy()
                both["sum"] = both[f"retest|{sn}|{k}"] + both[col].fillna(0)
                line2, ok2 = _stats(both, "sum")
                line1, _ = _stats(both, f"retest|{sn}|{k}")
                print(f"     {sn}, {k}, сигнал без перезахода: {line1}")
                print(f"     {sn}, {k}, сигнал + перезаход: {line2}  → {_verdict(ok2)}")
        print("  5) разрезы (стоп PIV5, ретест, 3R / 8R):")
        flow = ((x.side > 0) & x.inflow) | ((x.side < 0) & ~x.inflow)      # лонг — при притоке, шорт — при оттоке
        cotd = x.cot_known & (((x.side > 0) & x.cot) | ((x.side < 0) & ~x.cot))
        is_ = x[(x.t >= PER["is"][0]) & (x.t < PER["is"][1])]
        ref = is_ if len(is_) >= 30 else x
        q1, q2 = ref.slope_atr.quantile([1 / 3, 2 / 3])
        groups = {"лонги": x[x.side > 0], "шорты": x[x.side < 0], "по притоку": x[flow],
                  "против притока": x[~flow], "по COT": x[cotd], "против COT": x[x.cot_known & ~cotd],
                  f"агрессоры выше медианы IS ({ref.aggr.median():.3f})": x[x.aggr > ref.aggr.median()],
                  f"пологие (≤ {q1:.3f} ATR/свечу)": x[x.slope_atr <= q1], f"крутые (> {q2:.3f})": x[x.slope_atr > q2]}
        for nm, z in groups.items():
            for k in ("3R", "8R"):
                line, ok = _stats(z, f"retest|PIV5|{k}")
                print(f"     {nm}, {k}: {line}  → {_verdict(ok)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--end", default=__import__("os").environ.get("RESEARCH_END", "2026-09"))
    ap.add_argument("--perp", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), a.end, [x for x in a.perp.split(",") if x])
    else:
        report()
