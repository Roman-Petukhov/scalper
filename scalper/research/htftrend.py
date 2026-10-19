"""Правило бота 4h на дневках и неделях + включатель «приток стейблкоинов», история со спота Binance с 2017 года.

Правило бота (как research/topcrowd.py coin_rows): пробой линии зигзага 3 ATR на закрытии свечи, тренд старшего ТФ
по направлению (1d — неделя против EMA50, как в боте; 1w — сама неделя против EMA50 своих закрытий ≈ год), закрытие
в своей половине свечи, стоп за подтверждённым свингом PIV5 ± 0.1 ATR (0.3–4 ATR), вход на ретесте линии (1d — 12
свечей, 1w — 4), цель 3R, долив +0.5 на +1.5R, срок 1d — 60 свечей, 1w — 26; оборот от $20M в день. Издержки как у
бота (maker / taker Bybit); funding — 0 (спот; на перпетуалах он был бы).
Почему спот: архив перпетуалов — с 2020 года, а спот Binance — с 2017-08, вместе с делистнутыми парами (без ошибки
выжившего): добавляются пузырь 2017, зима 2018 и боковик 2019 — проверка медвежьего цикла, которого нет в данных бота.
Разрезы (заданы до запуска): агрессоры ≥ 55% (как бот) и все пробои; включатель «по притоку» — лонг, если предложение
стейблкоинов за 30 дней выросло на последний закрытый день, шорт — если снизилось (DefiLlama, research/extreg.py); наклон линии в ATR за свечу — трети по IS
(пологие / средние / крутые), как фильтр не обязателен — для справки.
Включатель COT (добавлен по просьбе владельца до запуска): отчёт CFTC «Traders in Financial Futures» по биткоину
на CME — лонг, если управляющие активами (asset managers) нарастили чистый лонг за 4 отчёта, шорт — если сократили; данные за вторник выходят в пятницу,
поэтому действуют с субботы. Оговорка: с 2024 позиции фондов с плечом искажены базисной сделкой (шорт CME против
спотовых ETF), поэтому берём управляющих, а не фонды с плечом.
Решение: годится как вторая нога, если R на сделку > 0 во всех периодах, где ≥ 10 сделок, и приток не хуже без него
в каждом периоде (отсекаемые не лучше оставшихся); при < 10 сделок в периоде — вывода нет.
Второй прогон (после первого: 264 пробоя за 9 лет, агрессоры ≥ 55% на дневках почти не встречаются — дневная доля
тейкеров усредняется к 50%): основа — все пробои с оборотом от $20M (агрессоры — разрез «выше медианы IS»); выходы
3R / 5R / 8R / лестница без цели (по просьбе владельца: 3R мало для дневок и недель), сроки 5R / 8R / лестницы втрое
длиннее; воронка правила (сколько пробоев отсекает каждый шаг); справка — оборот от $5M и итог за всё время.
Длинная цель лучше 3R, если R на сделку выше в большинстве периодов с ≥ 10 сделками и за всё время.

    python -m research.htftrend collect          (сам скачивает дневки спота своей части монет)
    python -m research.htftrend report           (+ python -m research.extreg collect)
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
from .broad import ADV_MIN
from .dboost2 import inflow_gate
from .dconf import Exit, simulate
from .shard import all_parts, mine, part_path
from .smc import _atr, _cell
from .tline import MAKER, PIV, STOP_BUF, TAKER, htf_trend, last_confirmed, pivots, retest_fill, zz_lines

START = "2017-08"
COT_URL = ("https://publicreporting.cftc.gov/resource/gpe5-46if.json?cftc_contract_market_code=133741"
           "&$limit=5000&$order=report_date_as_yyyy_mm_dd")     # CFTC TFF, CME Bitcoin (5 BTC), еженедельно с 2017-12
COT_WEEKS = 4
SPOT_LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=/&prefix=data/spot/monthly/klines/"
STABLE = ("USDC", "BUSD", "TUSD", "FDUSD", "DAI", "USDP", "PAX", "USDS", "SUSD", "UST", "USTC", "EUR", "GBP", "AUD",
          "TRY", "BRL", "RUB", "BIDR", "IDRT", "NGN", "UAH", "ZAR", "PLN", "RON", "ARS", "JPY", "BKRW", "EURI", "AEUR",
          "USD1", "XUSD", "BFUSD", "PAXG")
LEVERAGED = re.compile(r"(UP|DOWN|BULL|BEAR)USDT$")
HOLD = {"1d": 60, "1w": 26}
RETEST = {"1d": 12, "1w": 4}
LIQ_WIDE = 5e6                 # широкая выборка для справки (основное правило — ADV_MIN, как у бота)


def exits(tf: str) -> dict[str, Exit]:
    """Выходы (заданы до второго прогона): 3R — бот; 5R и 8R; «лестница» — без цели, доливы на +1.5 / 3 / 4.5R,
    после последней — стоп в 3 ATR за лучшим закрытием (как dconf). Длинным целям — срок втрое больше."""
    h = HOLD[tf]
    return {"3R": Exit(3.0, (1.5,), 0.5, h), "5R": Exit(5.0, (1.5,), 0.5, 3 * h),
            "8R": Exit(8.0, (1.5,), 0.5, 3 * h), "лестница": Exit(None, (1.5, 3.0, 4.5), 0.5, 3 * h, 3.0)}


FUNNEL = ("линий", "оборот", "тренд", "закрытие", "свинг", "в отчёте")
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


def weekly(d: pd.DataFrame) -> pd.DataFrame:
    w = d.resample("W-MON", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "quote_volume": "sum",
         "taker_buy_volume": "sum"}).dropna(subset=["close"])
    return w


def trend_of(d: pd.DataFrame, tf: str) -> np.ndarray:
    if tf == "1d":
        return htf_trend(d, "1d")                                 # неделя против EMA50 (последняя закрытая неделя)
    c = d["close"]
    return np.sign(c - c.ewm(span=50, adjust=False).mean()).to_numpy()


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, tf: str, funnel: dict | None = None) -> list[dict]:
    """Пробои по правилу бота с оборотом от LIQ_WIDE; колонки m_<выход> / r_<выход> — R по рынку / на ретесте.
    funnel — счётчик, сколько пробоев дошло до каждого шага правила."""
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = np.zeros(len(c))
    a = _atr(d).to_numpy()
    trend = trend_of(d, tf)
    buy = (d["taker_buy_volume"] / d["volume"].replace(0, np.nan)).to_numpy()
    rng = hi - lo
    loc = (c - lo) / np.where(rng > 0, rng, np.nan)
    sw_lo, sw_hi = last_confirmed(pivots(lo, PIV, False), len(c)), last_confirmed(pivots(hi, PIV, True), len(c))
    exs = exits(tf)
    fn = funnel if funnel is not None else {}
    rows = []
    for r in zz_lines(d):
        t, side = int(r["t"]), int(r["side"])
        if t <= 0 or t >= len(c) - 1 or not (a[t] > 0):
            continue
        fn["линий"] = fn.get("линий", 0) + 1
        if not (liq[t] >= LIQ_WIDE):
            continue
        fn["оборот"] = fn.get("оборот", 0) + 1
        if trend[t] != side:
            continue
        fn["тренд"] = fn.get("тренд", 0) + 1
        cl = loc[t] if side > 0 else 1 - loc[t]
        sw = sw_lo[t] if side > 0 else sw_hi[t]
        if not (cl >= 0.5):
            continue
        fn["закрытие"] = fn.get("закрытие", 0) + 1
        if sw < 0:
            continue
        fn["свинг"] = fn.get("свинг", 0) + 1
        level = float(r["line_t"])
        stop = (lo[sw] - STOP_BUF * a[t]) if side > 0 else (hi[sw] + STOP_BUF * a[t])
        row = {"symbol": sym, "tf": tf, "t": d.index[t], "side": side, "liq": float(liq[t]),
               "aggr": float(buy[t] if side > 0 else 1 - buy[t]), "slope_atr": abs(float(r["slope"])) / a[t]}
        risk = side * (c[t] - stop)
        rr = side * (level - stop)
        fill = retest_fill(hi, lo, t, side, level, 3.0 * rr, RETEST[tf]) if 0.3 * a[t] <= rr <= 4 * a[t] else -1
        stopped = fill >= 0 and ((side > 0 and lo[fill] <= stop) or (side < 0 and hi[fill] >= stop))
        for k, ex in exs.items():
            row[f"m_{k}"] = simulate(o, hi, lo, c, f, t, side, c[t], stop, a[t], ex, TAKER) \
                if 0.3 * a[t] <= risk <= 4 * a[t] else np.nan
            row[f"r_{k}"] = np.nan if fill < 0 else \
                (side * (stop - level) - (MAKER + TAKER) * level) / rr if stopped else \
                simulate(o, hi, lo, c, f, fill, side, level, stop, a[t], ex, MAKER)
        if np.isfinite(row["m_3R"]) or np.isfinite(row["r_3R"]):
            fn["в отчёте"] = fn.get("в отчёте", 0) + 1
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


def collect(root: Path, end: str) -> None:
    from .shard import shard_id
    if shard_id()[0] == 0:
        collect_cot()
    syms = mine(list_spot())
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    rows, n, funnel = [], 0, {"1d": {}, "1w": {}}
    with ThreadPoolExecutor(16) as pool:
        frames = dict(zip(syms, pool.map(lambda s: load_daily(s, cache, end), syms)))
    for s, d in frames.items():
        try:
            if d is None or len(d) < 120:
                continue
            n += 1
            liq_d = d["quote_volume"].rolling(30, min_periods=20).mean().shift(1)
            rows += coin_rows(d, liq_d.to_numpy(), s, "1d", funnel["1d"])
            w = weekly(d)
            if len(w) >= 60:
                liq_w = liq_d.reindex(w.index).to_numpy()            # оборот на начало недели — известен
                rows += coin_rows(w, liq_w, s, "1w", funnel["1w"])
        except Exception as e:
            print(f"  htftrend {s}: пропуск ({e})", flush=True)
    print(f"  htftrend: монет со свечами {n} из {len(syms)}, пробоев {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("htftrend"), index=False)
    pd.DataFrame([{"tf": tf, "step": k, "n": v} for tf, fn in funnel.items() for k, v in fn.items()]) \
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
    bar = df.tf.map({"1d": pd.Timedelta(days=1), "1w": pd.Timedelta(days=7)})
    return (df.t + bar).dt.tz_localize(None).dt.normalize() - pd.Timedelta(days=1)   # последний закрытый день


ALL = ("всё время", (PER["2017–18"][0], PER["ho"][1]))


def _line(z: pd.DataFrame, col: str) -> tuple[str, list[bool]]:
    cells, ok = [], []
    for p, (a, b) in (*PER.items(), ALL):
        m = z[(z.t >= a) & (z.t < b) & z[col].notna()]
        cells.append(f"{p}: {_cell(m.rename(columns={col: 'R'}))}")
        if p != ALL[0] and len(m) >= 10:
            ok.append(m[col].mean() > 0)
    return "  ".join(cells), ok


def _funnel() -> None:
    parts = all_parts("htftrend_funnel")
    if not parts:
        return
    f = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).groupby(["tf", "step"]).n.sum()
    for tf in ("1d", "1w"):
        if tf in f.index.get_level_values(0):
            steps = "  →  ".join(f"{k} {int(f.get((tf, k), 0))}" for k in FUNNEL)
            print(f"  воронка {tf} (оборот от ${LIQ_WIDE / 1e6:.0f}M): {steps}")


def report() -> None:
    parts = all_parts("htftrend")
    print("\n===== HTFTREND: правило бота 4h на 1d и 1w, спот Binance с 2017 (с делистнутыми), приток, COT, длинные цели =====")
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
    print(f"пробоев {len(df)}, монет {df.symbol.nunique()}; данных притока {'есть' if g is not None else 'НЕТ'}, "
          f"COT {'есть' if cp else 'НЕТ'} (включатель COT — только на днях, когда отчёты уже есть)")
    _funnel()
    names = list(exits("1d"))
    for tf in ("1d", "1w"):
        wide = df[df.tf == tf]
        x = wide[wide.liq >= ADV_MIN]
        if not len(x):
            continue
        is_ = x[(x.t >= PER["is"][0]) & (x.t < PER["is"][1])]
        ref = is_ if len(is_) >= 30 else x
        q1, q2 = ref.slope_atr.quantile([1 / 3, 2 / 3])
        am = ref.aggr.median()
        print(f"\n  {tf}: пробоев с оборотом от ${ADV_MIN / 1e6:.0f}M — {len(x)} (лонг {int((x.side > 0).sum())}, "
              f"шорт {int((x.side < 0).sum())}), от ${LIQ_WIDE / 1e6:.0f}M — {len(wide)}; агрессоры: медиана "
              f"{am:.3f}, ≥ 55% — {int((x.aggr >= 0.55).sum())}; наклон: пологие ≤ {q1:.3f}, крутые > {q2:.3f} ATR/свечу")
        flow = ((x.side > 0) & x.inflow) | ((x.side < 0) & ~x.inflow)      # лонг — при притоке, шорт — при оттоке
        cotd = x.cot_known & (((x.side > 0) & x.cot) | ((x.side < 0) & ~x.cot))
        groups = {
            "все пробои": x,
            "по притоку": x[flow],
            "против притока (отсекаемые)": x[~flow],
            "по COT": x[cotd],
            "против COT (отсекаемые)": x[x.cot_known & ~cotd],
            "по притоку и COT": x[flow & cotd],
            "агрессоры выше медианы IS": x[x.aggr > am],
            "пологие": x[x.slope_atr <= q1],
            "средние": x[(x.slope_atr > q1) & (x.slope_atr <= q2)],
            "крутые": x[x.slope_atr > q2],
            "лонги": x[x.side > 0],
            "шорты": x[x.side < 0],
            f"широкая выборка (от ${LIQ_WIDE / 1e6:.0f}M)": wide,
        }
        for k in names:
            print(f"    выход {k}, вход на ретесте:")
            for nm, z in groups.items():
                line, ok = _line(z, f"r_{k}")
                verdict = (f"плюс в {sum(ok)} из {len(ok)} периодов с ≥ 10 сделками" if ok else "мало сделок")
                print(f"      {nm}: {line}  → {verdict}")
            for nm in ("все пробои", "по притоку"):
                line, ok = _line(groups[nm], f"m_{k}")
                print(f"      по рынку, {nm}: {line}  → "
                      f"{f'плюс в {sum(ok)} из {len(ok)}' if ok else 'мало сделок'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--end", default=__import__("os").environ.get("RESEARCH_END", "2026-09"))
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), a.end)
    else:
        report()
