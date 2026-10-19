"""Четыре системы известных трейдеров на дневках крипты (по просьбе владельца «проверь то, что ещё не проверяли»).

Данные — как research/htftrend.py: спот Binance с 2017 (с делистнутыми) и перпетуалы Binance с 2020 (с funding).
Монета в сделке — при среднем обороте за 30 прошлых дней от $20M (перпетуалы) / $5M (спот). Вход и выход — taker
(вход по стоп-ордеру или на закрытии), тейк — maker; стоп проверяется раньше тейка; если свеча входа по стоп-ордеру
дошла и до стопа, первым считается уровень, ближний к её открытию. Одна позиция на монету в системе:
пока сделка открыта, новые сигналы этой системы по монете пропускаются. R — в риске сделки (вход − стоп).

Правила заданы до запуска, по истории не подбираются; оригинал — лонг, зеркальный шорт считается отдельно.

  Q   Qullamaggie / Minervini — пробой лидера из узкой базы.
      Сила: доходность за 63 дня (на вчерашнем закрытии) от +30% и в верхних 10% монет того же рынка с тем же
      порогом оборота; база — 10 прошлых дней, высота не больше 3 ATR, вчера закрытие выше SMA20 и SMA10 > SMA20.
      Пробой — первое закрытие выше максимума базы на объёме от 1.5× среднего за 20 дней. Вход на закрытии, стоп —
      минимум дня пробоя (не ближе 0.3 ATR; дальше 1.5 ATR — пропуск: «стоп не шире ATR» у Qullamaggie).
      Выходы: «по Qullamaggie» — половина на закрытии 5-го дня, если в плюсе, и стоп в безубыток, остаток — первое
      закрытие ниже SMA10 (вариант SMA20), до 120 дней; «3R» — цель 3R, до 60 дней.
      Шорт: нижние 10%, за 63 дня от −30%, зеркально.
  C   Connors RSI(2) — откат в тренде. Лонг: закрытие выше SMA200, RSI(2) ниже 10; вход на закрытии; выход —
      первое закрытие выше SMA5 или через 10 дней. У Коннорса стопа нет: «аварийный» стоп 3 ATR (R = 3 ATR) и
      вариант со стопом 1.5 ATR. Шорт: ниже SMA200, RSI(2) выше 90, выход — закрытие ниже SMA5.
  H   Linda Raschke «Holy Grail». Вчера ADX(14) выше 30 и +DI > −DI; сегодня минимум впервые коснулся EMA20 (вчера был
      выше). Завтра — покупка стоп-ордером выше максимума дня касания (только один день), стоп — минимум дня касания.
      Выходы: «к экстремуму» — тейк на максимуме 20 дней до касания (не ближе 1R), до 20 дней; «3R», до 20 дней.
      Шорт зеркально (−DI > +DI, касание EMA20 снизу).
  T   Turtle Soup Plus One (Raschke, Connors) — ложный пробой 20-дневного канала. Сегодня минимум и закрытие ниже
      минимума 20 прошлых дней, причём тот минимум поставлен не позже чем за 4 дня; завтра — покупка стоп-ордером на
      уровне старого минимума; стоп — минимум сегодняшнего дня. Цель 2R / 3R, до 20 дней. Шорт — по максимумам.
  N   Toby Crabel NR7 и ID/NR4 — сжатие перед пробоем. NR7: диапазон дня — самый узкий из 7; ID/NR4: внутренний день и
      самый узкий из 4. Завтра — вход стоп-ордером за максимумом (лонг) или минимумом (шорт) этого дня; если завтра
      пробиты оба края — вход с ближнего к открытию и стоп на той же свече. Стоп — противоположный край дня
      сжатия. Цель 2R / 3R, до 20 дней. Отдельно — только по тренду недели (закрытие недели против EMA50, как у бота).

Принятие (как у прочих проверок): R на сделку за всё время выше нуля с t ≥ 2, плюс в большинстве периодов с ≥ 10
сделками (2020 / 2021 / IS / VAL / HO) на перпетуалах, тот же знак при издержках x2 и плюс на споте. Ориентир — пробой
линии на дневках (research/htftrend.py): перпетуалы +0.58R, спот +0.41R при стопе 1.5 ATR и 3R.

    python -m research.masters collect --perp <1h-qualified>   (спот качает сам)
    python -m research.masters report
"""
from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .broad import ADV_MIN, adv30
from .htftrend import PER, _stats, _verdict, list_spot, load_daily
from .shard import all_parts, mine, part_path
from .smc import _atr
from .tline import MAKER, TAKER, htf_trend, tf_frame

LIQ = {"perp": ADV_MIN, "spot": 5e6}
RS_DAYS, RS_MIN, RS_TOP = 63, 0.30, 0.10
BASE, BASE_ATR, VOL_X = 10, 3.0, 1.5
SYSTEMS = {"Q": "Qullamaggie: пробой лидера из узкой базы", "C": "Connors RSI(2): откат в тренде",
           "H": "Raschke Holy Grail: откат к EMA20 при ADX > 30", "T": "Turtle Soup Plus One: ложный пробой 20 дней",
           "N": "Crabel: пробой NR7 / ID-NR4"}


def sma(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).mean().to_numpy()


def ema(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).ewm(span=n, adjust=False, min_periods=n).mean().to_numpy()


def rsi(c: np.ndarray, n: int = 2) -> np.ndarray:
    """RSI Уайлдера."""
    d = np.diff(c, prepend=np.nan)
    up = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100 - 100 / (1 + up.to_numpy() / dn.to_numpy())
    out[(dn.to_numpy() == 0) & (up.to_numpy() > 0)] = 100.0
    out[np.isnan(d)] = np.nan
    return out


def adx(hi: np.ndarray, lo: np.ndarray, c: np.ndarray, n: int = 14) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ADX, +DI, −DI Уайлдера."""
    up, dn = np.diff(hi, prepend=np.nan), -np.diff(lo, prepend=np.nan)
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = np.roll(c, 1)
    tr = np.maximum.reduce([hi - lo, np.abs(hi - pc), np.abs(lo - pc)])
    tr[0] = np.nan

    def w(x: np.ndarray) -> np.ndarray:
        return pd.Series(x).ewm(alpha=1 / n, adjust=False, min_periods=n).mean().to_numpy()

    atr_ = w(tr)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi, mdi = 100 * w(pdm) / atr_, 100 * w(mdm) / atr_
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    return w(dx), pdi, mdi


def walk(o, hi, lo, c, f, e: int, side: int, entry: float, stop: float, hold: int, *, tp: float | None = None,
         close_exit: Callable[[int], bool] | None = None, half_at: int | None = None, stop_entry_bar: bool = False,
         entry_fee: float = TAKER) -> tuple[float, int]:
    """Сделка с входа на баре e по цене entry: (R, бар выхода); R с комиссиями и funding. stop_entry_bar — вход
    стоп-ордером внутри бара e: если свеча дошла и до стопа, порядок неизвестен — первым считается уровень, ближний
    к открытию (вход первым — убыток на этой же свече; стоп первым — сделка идёт дальше). close_exit(j) — выход на закрытии j; half_at — половина
    на закрытии этого дня от входа, если в плюсе, и стоп остатка в безубыток."""
    risk = side * (entry - stop)
    if not risk > 0:
        return np.nan, -1
    units, done, paid, sp = 1.0, 0.0, 0.0, stop

    def leg(px: float, q: float, fee: float) -> float:
        return q * (side * (px - entry) - entry_fee * entry - fee * px)

    if stop_entry_bar and ((side > 0 and lo[e] <= sp) or (side < 0 and hi[e] >= sp)) \
            and abs(o[e] - entry) <= abs(o[e] - sp):
        return leg(sp, 1.0, TAKER) / risk, e        # вход и стоп на одной свече: первым — ближний к открытию уровень
    n = len(c)
    for j in range(e + 1, n):
        paid += f[j] * units
        if (side > 0 and lo[j] <= sp) or (side < 0 and hi[j] >= sp):
            px = min(sp, o[j]) if side > 0 else max(sp, o[j])
            return (done + leg(px, units, TAKER) - side * paid * entry) / risk, j
        if tp is not None and ((side > 0 and hi[j] >= tp) or (side < 0 and lo[j] <= tp)):
            return (done + leg(tp, units, MAKER) - side * paid * entry) / risk, j
        if half_at is not None and j - e == half_at and units == 1.0 and side * (c[j] - entry) > 0:
            done += leg(c[j], 0.5, TAKER)
            units, sp = 0.5, entry
        if (close_exit is not None and close_exit(j)) or j - e >= hold or j == n - 1:
            return (done + leg(c[j], units, TAKER) - side * paid * entry) / risk, j
    return np.nan, -1


def coin_rows(d: pd.DataFrame, liq: np.ndarray, sym: str, src: str) -> tuple[list[dict], pd.DataFrame]:
    """Сигналы пяти систем на одной монете (без отбора по силе — он в отчёте, по всем монетам сразу) и ряд
    доходности за RS_DAYS для этого отбора."""
    o, hi, lo, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    v = d["volume"].to_numpy(dtype="float64")
    f = d["funding"].fillna(0.0).to_numpy(dtype="float64") if "funding" in d else np.zeros(len(c))
    a = _atr(d).to_numpy()
    s5, s10, s20, s200 = sma(c, 5), sma(c, 10), sma(c, 20), sma(c, 200)
    e20, r2 = ema(c, 20), rsi(c, 2)
    ax, pdi, mdi = adx(hi, lo, c)
    week = htf_trend(d, "1d")
    vavg = sma(v, 20)
    ret = c / np.roll(c, RS_DAYS) - 1
    ret[:RS_DAYS] = np.nan
    rng = hi - lo
    n = len(c)
    rows: list[dict] = []
    busy: dict[str, int] = {}

    def add(sysk: str, t: int, side: int, e: int, entry: float, stop: float, exits: dict[str, tuple[float, int]],
            **extra) -> None:
        risk = side * (entry - stop)
        for name, (r, j) in exits.items():
            if np.isfinite(r):
                rows.append({"sys": sysk, "exit": name, "symbol": sym, "src": src, "t": d.index[t], "side": side,
                             "liq": float(liq[t]), "R": float(r), "c2": 2 * TAKER * entry / risk, "j": j, **extra})
        ends = [j for r, j in exits.values() if j >= 0]
        busy[f"{sysk}{extra.get('kind', '')}"] = max(ends) if ends else e

    def free(key: str, t: int) -> bool:
        return busy.get(key, -1) < t

    for t in range(max(RS_DAYS + 1, 30), n - 1):
        if not (a[t] > 0) or not (liq[t] >= min(LIQ.values())):
            continue
        # Q — пробой лидера из узкой базы (сила по монетам — в отчёте)
        bh, bl = hi[t - BASE:t].max(), lo[t - BASE:t].min()
        if free("Q", t) and bh - bl <= BASE_ATR * a[t - 1] and v[t] >= VOL_X * vavg[t - 1]:
            for side in (1, -1):
                trend_ok = side * (c[t - 1] - s20[t - 1]) > 0 and side * (s10[t - 1] - s20[t - 1]) > 0
                brk = (c[t] > bh >= c[t - 1]) if side > 0 else (c[t] < bl <= c[t - 1])
                if not (trend_ok and brk and side * ret[t - 1] >= RS_MIN):
                    continue
                stop = lo[t] if side > 0 else hi[t]
                stop = min(stop, c[t] - 0.3 * a[t]) if side > 0 else max(stop, c[t] + 0.3 * a[t])
                if side * (c[t] - stop) > 1.5 * a[t]:
                    continue
                ma10 = (lambda j, s=side: s * (c[j] - s10[j]) < 0)
                ma20 = (lambda j, s=side: s * (c[j] - s20[j]) < 0)
                add("Q", t, side, t, c[t], stop, {
                    "по Qullamaggie, SMA10": walk(o, hi, lo, c, f, t, side, c[t], stop, 120, close_exit=ma10, half_at=5),
                    "по Qullamaggie, SMA20": walk(o, hi, lo, c, f, t, side, c[t], stop, 120, close_exit=ma20, half_at=5),
                    "3R": walk(o, hi, lo, c, f, t, side, c[t], stop, 60, tp=c[t] + side * 3 * (side * (c[t] - stop)))},
                    rs=float(ret[t - 1]), rs_day=d.index[t - 1])
                break
        # C — RSI(2)
        if free("C", t) and np.isfinite(s200[t]) and np.isfinite(r2[t]):
            side = 1 if (c[t] > s200[t] and r2[t] < 10) else -1 if (c[t] < s200[t] and r2[t] > 90) else 0
            if side:
                out = (lambda j, s=side: s * (c[j] - s5[j]) > 0)
                add("C", t, side, t, c[t], c[t] - side * 3 * a[t], {
                    "стоп 3 ATR": walk(o, hi, lo, c, f, t, side, c[t], c[t] - side * 3 * a[t], 10, close_exit=out),
                    "стоп 1.5 ATR": walk(o, hi, lo, c, f, t, side, c[t], c[t] - side * 1.5 * a[t], 10, close_exit=out)})
        # H — Holy Grail: касание EMA20 сегодня, вход завтра стоп-ордером
        if free("H", t + 1) and np.isfinite(ax[t - 1]) and ax[t - 1] > 30 and np.isfinite(e20[t - 1]):
            side = 1 if (pdi[t - 1] > mdi[t - 1] and lo[t] <= e20[t] and lo[t - 1] > e20[t - 1]) else \
                -1 if (mdi[t - 1] > pdi[t - 1] and hi[t] >= e20[t] and hi[t - 1] < e20[t - 1]) else 0
            if side and side * ((hi[t + 1] if side > 0 else lo[t + 1]) - (hi[t] if side > 0 else lo[t])) > 0:
                lvl, stop = (hi[t], lo[t]) if side > 0 else (lo[t], hi[t])
                entry = max(lvl, o[t + 1]) if side > 0 else min(lvl, o[t + 1])
                risk = side * (entry - stop)
                if 0.3 * a[t] <= risk <= 4 * a[t]:
                    ext = hi[t - 20:t].max() if side > 0 else lo[t - 20:t].min()
                    tp = ext if side * (ext - entry) >= risk else entry + side * risk
                    add("H", t, side, t + 1, entry, stop, {
                        "к экстремуму 20 дней": walk(o, hi, lo, c, f, t + 1, side, entry, stop, 20, tp=tp,
                                                     stop_entry_bar=True),
                        "3R": walk(o, hi, lo, c, f, t + 1, side, entry, stop, 20, tp=entry + side * 3 * risk,
                                   stop_entry_bar=True)})
        # T — Turtle Soup Plus One
        if free("T", t + 1) and t >= 21 and t + 1 < n:
            for side in (1, -1):
                win = lo[t - 20:t] if side > 0 else hi[t - 20:t]
                k = int(np.argmin(win) if side > 0 else np.argmax(win))
                old = win[k]
                if t - (t - 20 + k) < 4:
                    continue
                new_ext = (lo[t] < old and c[t] < old) if side > 0 else (hi[t] > old and c[t] > old)
                back = (hi[t + 1] >= old) if side > 0 else (lo[t + 1] <= old)
                if not (new_ext and back):
                    continue
                entry = max(old, o[t + 1]) if side > 0 else min(old, o[t + 1])
                stop = lo[t] if side > 0 else hi[t]
                risk = side * (entry - stop)
                if 0.3 * a[t] <= risk <= 4 * a[t]:
                    add("T", t, side, t + 1, entry, stop, {
                        f"{m}R": walk(o, hi, lo, c, f, t + 1, side, entry, stop, 20, tp=entry + side * m * risk,
                                      stop_entry_bar=True) for m in (2, 3)})
                break
        # N — NR7 и ID/NR4
        if t >= 7 and t + 1 < n:
            for kind, ok in (("NR7", rng[t] < rng[t - 6:t].min()),
                             ("ID/NR4", hi[t] < hi[t - 1] and lo[t] > lo[t - 1] and rng[t] < rng[t - 3:t].min())):
                if not ok or not free(f"N{kind}", t + 1):
                    continue
                up, dn = hi[t + 1] > hi[t], lo[t + 1] < lo[t]
                if not (up or dn):
                    continue
                # пробиты оба края: первым, скорее всего, — ближний к открытию; тогда стоп на той же свече (убыток),
                # пропускать такие дни нельзя — это отбор по будущему
                side = (1 if hi[t] - o[t + 1] <= o[t + 1] - lo[t] else -1) if (up and dn) else (1 if up else -1)
                entry = max(hi[t], o[t + 1]) if side > 0 else min(lo[t], o[t + 1])
                stop = lo[t] if side > 0 else hi[t]
                risk = side * (entry - stop)
                if 0.3 * a[t] <= risk <= 4 * a[t]:
                    add("N", t, side, t + 1, entry, stop, {
                        f"{m}R": walk(o, hi, lo, c, f, t + 1, side, entry, stop, 20, tp=entry + side * m * risk,
                                      stop_entry_bar=True) for m in (2, 3)},
                        kind=kind, week=int(week[t]) if np.isfinite(week[t]) else 0)
    panel = pd.DataFrame({"symbol": sym, "src": src, "day": d.index, "ret": ret, "liq": liq})
    return rows, panel.dropna(subset=["ret"])


def collect(root: Path, end: str, perp: list[str]) -> None:
    syms = mine(list_spot())
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    rows, panels = [], []
    with ThreadPoolExecutor(16) as pool:
        frames = dict(zip(syms, pool.map(lambda s: load_daily(s, cache, end), syms)))
    n_spot = n_perp = 0
    for s, d in frames.items():
        try:
            if d is None or len(d) < 120:
                continue
            n_spot += 1
            liq = d["quote_volume"].rolling(30, min_periods=20).mean().shift(1).to_numpy()
            r, p = coin_rows(d, liq, s, "spot")
            rows += r
            panels.append(p)
        except Exception as e:
            print(f"  masters {s}: пропуск ({e})", flush=True)
    for s in mine(perp):
        try:
            d = tf_frame(root, s, "1d")
            if d is None or len(d) < 120:
                continue
            n_perp += 1
            liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
            r, p = coin_rows(d, liq, s, "perp")
            rows += r
            panels.append(p)
        except Exception as e:
            print(f"  masters perp {s}: пропуск ({e})", flush=True)
    print(f"  masters: спот {n_spot} из {len(syms)} монет, перпетуалы {n_perp}, строк {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("masters"), index=False)
    if panels:
        pd.concat(panels, ignore_index=True).to_parquet(part_path("masters_rs"), index=False)


def leaders(panel: pd.DataFrame) -> pd.DataFrame:
    """Перцентиль доходности за RS_DAYS среди монет того же рынка с оборотом от его порога, по дням."""
    p = panel[panel.liq >= panel.src.map(LIQ)].copy()
    p["pct"] = p.groupby(["src", "day"])["ret"].rank(pct=True)
    return p[["src", "symbol", "day", "pct"]]


def select(x: pd.DataFrame, rs: pd.DataFrame | None) -> pd.DataFrame:
    """Сделки с оборотом от порога рынка; для Q — только лидеры (лонг) и аутсайдеры (шорт) по RS_TOP."""
    x = x[x.liq >= x.src.map(LIQ)]
    if rs is None or "rs_day" not in x:
        return x[x.sys != "Q"]
    q = x[x.sys == "Q"].merge(rs, left_on=["src", "symbol", "rs_day"], right_on=["src", "symbol", "day"], how="left")
    q = q[((q.side > 0) & (q.pct >= 1 - RS_TOP)) | ((q.side < 0) & (q.pct <= RS_TOP))].drop(columns=["day", "pct"])
    return pd.concat([x[x.sys != "Q"], q], ignore_index=True)


def report() -> None:
    parts = all_parts("masters")
    if not parts:
        print("  masters: нет данных")
        return
    x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    x["t"] = pd.to_datetime(x["t"], utc=True).dt.tz_localize(None)
    if "rs_day" in x:
        x["rs_day"] = pd.to_datetime(x["rs_day"], utc=True)
    rp = all_parts("masters_rs")
    rs = None
    if rp:
        panel = pd.concat([pd.read_parquet(p) for p in rp], ignore_index=True)
        panel["day"] = pd.to_datetime(panel["day"], utc=True)
        rs = leaders(panel)
    x = select(x, rs)
    print("===== MASTERS: системы известных трейдеров на дневках (правила до запуска; R на сделку по периодам) =====")
    print(f"     периоды: {', '.join(PER)}; принятие — плюс за всё время с t ≥ 2, в большинстве периодов, x2 и спот"
          " (шорт на споте — оценка: в споте его не открыть)")
    for src, title in (("perp", "перпетуалы Binance с 2020, оборот от $20M"), ("spot", "спот Binance с 2017, от $5M")):
        z0 = x[x.src == src]
        print(f"  {title}:")
        for sysk, name in SYSTEMS.items():
            z1 = z0[z0.sys == sysk]
            if not len(z1):
                print(f"   {name}: сигналов нет")
                continue
            print(f"   {name}:")
            kinds = sorted(z1["kind"].dropna().unique()) if "kind" in z1 and sysk == "N" else [None]
            for kind in kinds:
                z2 = z1 if kind is None else z1[z1.kind == kind]
                for ex in z2.exit.unique():
                    for side, sn in ((1, "лонг"), (-1, "шорт")):
                        z = z2[(z2.exit == ex) & (z2.side == side)]
                        line, ok = _stats(z, "R", "c2")
                        lab = f"{kind + ', ' if kind else ''}{ex}, {sn}"
                        print(f"     {lab}: {line}  → {_verdict(ok)}")
                    if sysk == "N":
                        z = z2[(z2.exit == ex) & (z2.side == z2.week)]
                        line, ok = _stats(z, "R", "c2")
                        print(f"     {kind}, {ex}, по тренду недели: {line}  → {_verdict(ok)}")


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
