"""
Уровни «умных денег» на 1h и вход сеткой на ретесте: ордер-блоки (OB), имбалансы (FVG) и, для сравнения, простой
пробой 20-часового диапазона; ML-фильтр «настоящий / ложный» на признаках нескольких таймфреймов (1h / 4h / 1d / 1w).

Определения (всё известно на закрытии бара t, без заглядывания вперёд):
    OB+SWEEP      OB, у которого дно импульса ниже предыдущего подтверждённого свинг-лоу (снятие ликвидности —
                  пролив за чужие стопы перед сломом структуры вверх; шорт — зеркально)
    свинги        фрактал 3 бара: вершина подтверждается через 3 бара после неё
    OB (лонг)     закрытие t впервые выше последнего подтверждённого свинг-хая (слом структуры, BOS); импульс —
                  от минимума последних 6 баров до close t, не меньше 1 ATR; OB — последняя медвежья свеча на минимуме
                  импульса или до 5 баров раньше; зона — [low, high] этой свечи. Шорт — зеркально.
    FVG (лонг)    low t > high t-2, средняя свеча — бычья, тело >= 1 ATR, ширина разрыва >= 0.3 ATR;
                  зона — [high t-2, low t]. Шорт — зеркально.
    BRK (лонг)    первое закрытие выше максимума 20 прошлых баров; «зона» — [уровень − 0.5 ATR, уровень].
Вход: 3 равные лимитки — верх, середина и низ зоны (для лонга), живут 48 ч (BRK — 12 ч); стоп — за зоной на 0.25 ATR
(BRK — уровень − 1 ATR); тейк — 5R от средней цены трёх лимиток; держим до 120 ч. Лимитка исполняется, только если
цена прошла сквозь её уровень; в баре исполнения тейк не засчитывается, а стоп — засчитывается (консервативно);
если цена дошла до тейка раньше первой лимитки — сетап пропущен. Издержки: вход maker 2 б.п., тейк maker 2 б.п.,
стоп / таймаут taker 5.5 б.п., funding. Результат — в R (риск полной позиции из трёх лимиток).

Против SMC: FADE — цена дошла до зоны, входим в обратную сторону стоп-ордером, стоп и тейк 1:1 / 1:2 от расстояния
до стопа SMC-трейдера; TRAP — цена пробила стоп SMC-трейдера за зоной и за 6 баров закрылась обратно в зоне, входим
по исходному направлению, стоп за экстремумом пролива, тейк 3R.

Признаки сетапа (~45): импульс и зона в ATR, объём и поток агрессоров импульса, OI за 4 / 24 / 72 ч, funding 8 / 24 /
72 ч, LSR толпы и топов; на 1h / 4h / 1d / 1w — положение к EMA20 / EMA50 и их наклон, место в диапазоне 20 баров,
перцентиль волатильности, направление последнего слома структуры; совпадение зоны с OB старших таймфреймов (4h, 1d);
BTC на 4h / 1d / 1w; час суток. Монеты — 725 с метриками, оборот за 30 дней >= $20M.
LightGBM (регрессия на R, обрезанный в [−1.5, 6]) учится на IS (внефолдовые прогнозы), отбор — верхние 10 / 20%
прогнозов IS; проверка на VAL / HOLDOUT и на монетах вне подбора (ext54 + fresh).

    python -m research.smc collect --symbols <725 монет>   (по частям)
    python -m research.smc report
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from .broad import ADV_MIN, adv30, group_of
from .shard import all_parts, mine, part_path
from .wave2 import Data2

PER = {"is": ("2022-01-01", "2024-07-01"), "val": ("2024-07-01", "2025-07-01"), "ho": ("2025-07-01", "2026-10-01")}
MAKER, TAKER = 2e-4, 5.5e-4
K_R = 5.0
MAX_HOLD = 120
VALID = {"OB": 48, "OB+SWEEP": 48, "FVG": 48, "BRK": 12}
TOPS = (0.10, 0.20)
TFS = {"1h": "1h", "4h": "4h", "1d": "1D", "1w": "7D"}


# ---------- движок сетапа ----------

@njit(cache=True)
def ladder(o, h, lo, c, fund, t0, side, l1, l2, l3, stop, valid, max_hold):
    """R сделки (NaN — ни одна лимитка не исполнилась), число исполненных лимиток, бар выхода."""
    n = len(c)
    legs = np.array([l1, l2, l3])
    filled = np.zeros(3, np.bool_)
    avg = (l1 + l2 + l3) / 3.0
    risk = side * (avg - stop)
    if not (risk > 0):
        return np.nan, 0, t0
    tp = avg + side * K_R * risk
    first = -1
    paid = 0.0
    j = t0 + 1
    while j < n:
        new = False
        if j <= t0 + valid:
            for k in range(3):
                if not filled[k] and ((side > 0 and lo[j] < legs[k]) or (side < 0 and h[j] > legs[k])):
                    filled[k] = True
                    new = True
                    if first < 0:
                        first = j
        nf = filled.sum()
        if nf == 0:
            if (side > 0 and h[j] >= tp) or (side < 0 and lo[j] <= tp) or j >= t0 + valid:
                return np.nan, 0, j                       # ушла к цели без нас или ретеста не было
            j += 1
            continue
        paid += fund[j]
        ex, fee = np.nan, TAKER
        if (side > 0 and lo[j] <= stop) or (side < 0 and h[j] >= stop):
            ex = min(stop, o[j]) if side > 0 else max(stop, o[j])
            if new and first == j:
                ex = stop                                # вход и стоп в одном баре: по стопу
        elif not new and ((side > 0 and h[j] >= tp) or (side < 0 and lo[j] <= tp)):
            ex, fee = tp, MAKER
        elif j - first >= max_hold or j == n - 1:
            ex = c[j]
        if not np.isnan(ex):
            r = 0.0
            for k in range(3):
                if filled[k]:
                    r += (side * (ex - legs[k]) - (MAKER + fee) * legs[k]) / risk / 3.0
            r -= side * paid * avg * nf / 3.0 / risk
            return r, nf, j
        j += 1
    return np.nan, 0, n - 1


@njit(cache=True)
def fade(o, h, lo, c, fund, t0, side, zone_top, smc_stop, valid, k_r, max_hold):
    """Против SMC-входа: цена дошла до зоны — входим в обратную сторону стоп-ордером (taker) по краю зоны;
    риск = расстояние от края зоны до стопа SMC-трейдера, стоп на том же расстоянии с другой стороны, тейк k_r R."""
    n = len(c)
    d = side * (zone_top - smc_stop)
    if not (d > 0):
        return np.nan
    s = -side
    j = t0 + 1
    while j <= min(t0 + valid, n - 1):
        if (side > 0 and lo[j] <= zone_top) or (side < 0 and h[j] >= zone_top):
            entry = min(zone_top, o[j]) if side > 0 else max(zone_top, o[j])
            stop, tp = entry - s * d, entry + s * k_r * d
            paid = 0.0
            q = j
            while q < n:
                if q > j:
                    paid += fund[q]
                if (s > 0 and lo[q] <= stop) or (s < 0 and h[q] >= stop):
                    ex, fee = stop, TAKER
                    if q > j:
                        ex = min(stop, o[q]) if s > 0 else max(stop, o[q])
                    return (s * (ex - entry) - (TAKER + fee) * entry - s * paid * entry) / d
                if q > j and ((s > 0 and h[q] >= tp) or (s < 0 and lo[q] <= tp)):
                    return (s * (tp - entry) - (TAKER + MAKER) * entry - s * paid * entry) / d
                if q - j >= max_hold or q == n - 1:
                    return (s * (c[q] - entry) - 2 * TAKER * entry - s * paid * entry) / d
                q += 1
            return np.nan
        j += 1
    return np.nan


@njit(cache=True)
def trap(o, h, lo, c, fund, t0, side, zone_edge, smc_stop, atr, valid, reclaim, k_r, max_hold):
    """Охота на стопы SMC-трейдеров: цена пробила их стоп за зоной и за `reclaim` баров закрылась обратно в зоне —
    входим по закрытию в сторону исходного сетапа; стоп за экстремумом пролива (−0.1 ATR), тейк k_r R."""
    n = len(c)
    j = t0 + 1
    while j <= min(t0 + valid, n - 1):
        if (side > 0 and lo[j] < smc_stop) or (side < 0 and h[j] > smc_stop):
            ext = lo[j] if side > 0 else h[j]
            for q in range(j, min(j + reclaim, n - 1) + 1):
                ext = min(ext, lo[q]) if side > 0 else max(ext, h[q])
                if (side > 0 and c[q] > zone_edge) or (side < 0 and c[q] < zone_edge):
                    entry = c[q]
                    stop = ext - side * 0.1 * atr
                    d = side * (entry - stop)
                    if not (d > 0):
                        return np.nan
                    tp = entry + side * k_r * d
                    paid = 0.0
                    for z in range(q + 1, n):
                        paid += fund[z]
                        if (side > 0 and lo[z] <= stop) or (side < 0 and h[z] >= stop):
                            ex = min(stop, o[z]) if side > 0 else max(stop, o[z])
                            return (side * (ex - entry) - 2 * TAKER * entry - side * paid * entry) / d
                        if (side > 0 and h[z] >= tp) or (side < 0 and lo[z] <= tp):
                            return (side * (tp - entry) - (TAKER + MAKER) * entry - side * paid * entry) / d
                        if z - q >= max_hold or z == n - 1:
                            return (side * (c[z] - entry) - 2 * TAKER * entry - side * paid * entry) / d
                    return np.nan
            return np.nan
        j += 1
    return np.nan


# ---------- признаки ----------

def _atr(d: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = d["close"].shift()
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def _bars(h: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule == "1h":
        return h
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return h[list(agg)].resample(rule, label="left", closed="left").agg(agg).dropna(subset=["close"])


def swings(hi: np.ndarray, lo: np.ndarray, n: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Последний подтверждённый свинг-хай / лоу на каждом баре (вершина видна через n баров)."""
    m = len(hi)
    sh, sl = np.full(m, np.nan), np.full(m, np.nan)
    cur_h = cur_l = np.nan
    for i in range(m):
        p = i - n
        if p >= n:
            if hi[p] == hi[p - n: p + n + 1].max():
                cur_h = hi[p]
            if lo[p] == lo[p - n: p + n + 1].min():
                cur_l = lo[p]
        sh[i], sl[i] = cur_h, cur_l
    return sh, sl


def order_blocks(d: pd.DataFrame, atr: np.ndarray, with_sweep: bool = False) -> list[tuple]:
    """(бар слома, сторона, низ зоны, верх зоны[, снята ли ликвидность]) для OB на этом таймфрейме.
    Снятие ликвидности: дно импульса (для лонга) ниже предпоследнего подтверждённого свинг-лоу — цену пролили
    за чужие стопы, и только потом она сломала структуру в обратную сторону."""
    o, h, l, c = (d[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    sh, sl = swings(h, l)
    out = []
    for t in range(8, len(c)):
        a = atr[t]
        if not (a > 0):
            continue
        for side in (1, -1):
            lvl, prev_lvl = (sh[t - 1], sh[t - 1]) if side > 0 else (sl[t - 1], sl[t - 1])
            if np.isnan(lvl):
                continue
            broke = c[t] > lvl and c[t - 1] <= prev_lvl if side > 0 else c[t] < lvl and c[t - 1] >= prev_lvl
            if not broke:
                continue
            seg = slice(t - 6, t + 1)
            j = (t - 6) + (int(np.argmin(l[seg])) if side > 0 else int(np.argmax(h[seg])))
            if side * (c[t] - (l[j] if side > 0 else h[j])) < a:
                continue
            ob = -1
            for q in range(j, max(j - 6, 0), -1):
                if (side > 0 and c[q] < o[q]) or (side < 0 and c[q] > o[q]):
                    ob = q
                    break
            if ob >= 0:
                if with_sweep:
                    ref = sl[j - 1] if side > 0 else sh[j - 1]
                    swept = (not np.isnan(ref)) and ((side > 0 and l[j] < ref) or (side < 0 and h[j] > ref))
                    out.append((t, side, l[ob], h[ob], bool(swept)))
                else:
                    out.append((t, side, l[ob], h[ob]))
    return out


def tf_state(d: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Состояние таймфрейма на закрытии каждого его бара (сдвинуто: используется только закрытый бар)."""
    c = d["close"]
    e20, e50 = c.ewm(span=20, adjust=False).mean(), c.ewm(span=50, adjust=False).mean()
    a = _atr(d)
    rng_hi, rng_lo = d["high"].rolling(20).max(), d["low"].rolling(20).min()
    rv = np.log(c).diff().rolling(20).std()
    sh, sl = swings(d["high"].to_numpy(), d["low"].to_numpy())
    bos = np.zeros(len(c))
    last = 0.0
    cc = c.to_numpy()
    for i in range(1, len(cc)):
        if not np.isnan(sh[i - 1]) and cc[i] > sh[i - 1]:
            last = 1.0
        elif not np.isnan(sl[i - 1]) and cc[i] < sl[i - 1]:
            last = -1.0
        bos[i] = last
    st = pd.DataFrame({f"{tf}_e20": (c - e20) / a, f"{tf}_e50": (c - e50) / a,
                       f"{tf}_e20s": e20.diff(5) / a, f"{tf}_e50s": e50.diff(5) / a,
                       f"{tf}_pos": (c - rng_lo) / (rng_hi - rng_lo).replace(0, np.nan),
                       f"{tf}_rvp": rv.rolling(120, min_periods=40).rank(pct=True),
                       f"{tf}_bos": bos}, index=d.index)
    return st


def _known_at(state: pd.DataFrame, rule: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Состояние старшего таймфрейма, известное на закрытии часового бара: только закрытые бары."""
    if rule == "1h":
        return state.reindex(index)
    closed = state.copy()
    closed.index = closed.index + pd.tseries.frequencies.to_offset(rule)       # бар известен после закрытия
    return closed.reindex(index + pd.Timedelta(hours=1), method="ffill").set_axis(index)


def htf_ob_zones(h: pd.DataFrame, rule: str, index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Последняя бычья и медвежья зона OB старшего таймфрейма, известная на каждом часовом баре."""
    d = _bars(h, rule)
    a = _atr(d).to_numpy()
    obs = order_blocks(d, a)
    off = pd.tseries.frequencies.to_offset(rule)
    bull_lo = pd.Series(np.nan, index=index)
    bull_hi, bear_lo, bear_hi = bull_lo.copy(), bull_lo.copy(), bull_lo.copy()
    for t, side, zlo, zhi in obs:
        known = d.index[t] + off - pd.Timedelta(hours=1)          # часовой бар, на закрытии которого зона видна
        if side > 0:
            bull_lo[bull_lo.index >= known] = zlo
            bull_hi[bull_hi.index >= known] = zhi
        else:
            bear_lo[bear_lo.index >= known] = zlo
            bear_hi[bear_hi.index >= known] = zhi
    return bull_lo.to_numpy(), bull_hi.to_numpy(), np.vstack([bear_lo.to_numpy(), bear_hi.to_numpy()])


def setups(h: pd.DataFrame) -> list[tuple[str, int, int, float, float]]:
    """(тип, бар, сторона, низ зоны, верх зоны) на часовых барах."""
    a = _atr(h).to_numpy()
    out = [("OB+SWEEP" if sw else "OB", t, s, zl, zh) for t, s, zl, zh, sw in order_blocks(h, a, with_sweep=True)]
    o, hi, lo, c = (h[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    for t in range(2, len(c)):
        if not (a[t] > 0):
            continue
        body = c[t - 1] - o[t - 1]
        if lo[t] > hi[t - 2] and body >= a[t] and lo[t] - hi[t - 2] >= 0.3 * a[t]:
            out.append(("FVG", t, 1, hi[t - 2], lo[t]))
        if hi[t] < lo[t - 2] and -body >= a[t] and lo[t - 2] - hi[t] >= 0.3 * a[t]:
            out.append(("FVG", t, -1, hi[t], lo[t - 2]))
    up20 = pd.Series(hi).shift(1).rolling(20).max().to_numpy()
    dn20 = pd.Series(lo).shift(1).rolling(20).min().to_numpy()
    for t in range(21, len(c)):
        if not (a[t] > 0):
            continue
        if c[t] > up20[t] and c[t - 1] <= up20[t - 1]:
            out.append(("BRK", t, 1, up20[t] - 0.5 * a[t], up20[t]))
        if c[t] < dn20[t] and c[t - 1] >= dn20[t - 1]:
            out.append(("BRK", t, -1, dn20[t], dn20[t] + 0.5 * a[t]))
    return out


def coin_setups(h: pd.DataFrame, btc_state: pd.DataFrame, adv: pd.Series, sym: str) -> pd.DataFrame:
    idx = h.index
    a = _atr(h)
    av = a.to_numpy()
    o, hi, lo, c = (h[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    f = h["funding"].fillna(0.0).to_numpy(dtype="float64")
    v = h["volume"]
    tb = h["taker_buy_volume"]
    flow = (2 * tb - v)
    oi = np.log(h["oi"].replace(0, np.nan)) if "oi" in h else pd.Series(np.nan, index=idx)
    base = pd.DataFrame({
        "vol_r3": np.log((v.rolling(3).sum() + 1) / (v.rolling(480, min_periods=120).mean() * 3 + 1)),
        "flow3": flow.rolling(3).sum() / v.rolling(3).sum().replace(0, np.nan),
        "flow24": flow.rolling(24).sum() / v.rolling(24).sum().replace(0, np.nan),
        "oi4": oi.diff(4), "oi24": oi.diff(24), "oi72": oi.diff(72),
        "fund8": h["funding"].fillna(0).rolling(8).sum(), "fund24": h["funding"].fillna(0).rolling(24).sum(),
        "fund72": h["funding"].fillna(0).rolling(72).sum(),
        "lsr": h["lsr"] if "lsr" in h else np.nan, "top_lsr": h["top_lsr"] if "top_lsr" in h else np.nan,
        "atr_pct": a / h["close"], "hour": idx.hour.astype("float64"),
    }, index=idx)
    states = [_known_at(tf_state(_bars(h, rule), tf), rule, idx) for tf, rule in TFS.items()]
    feats = pd.concat([base, *states, btc_state.reindex(idx)], axis=1)
    conf = {}
    for rule in ("4h", "1D"):
        bl, bh, bear = htf_ob_zones(h, rule, idx)
        conf[rule] = (bl, bh, bear[0], bear[1])
    # поглощение на экстремуме импульса: объём за 6 баров против среднего, делённый на их диапазон в ATR
    rng6 = (h["high"].rolling(6).max() - h["low"].rolling(6).min()) / a
    absorb = (np.log((v.rolling(6).sum() + 1) / (v.rolling(480, min_periods=120).mean() * 6 + 1)) - np.log(rng6.clip(lower=0.1))
              ).shift(6).to_numpy()
    ok_adv = (adv.reindex(idx.floor("D")).to_numpy() >= ADV_MIN)
    rows = []
    for kind, t, side, zl, zh in setups(h):
        if not ok_adv[t] or not (av[t] > 0):
            continue
        width = zh - zl
        if side > 0:
            l1, l2, l3, stop = zh, (zl + zh) / 2, zl, zl - (1.0 if kind == "BRK" else 0.25) * av[t]
            if kind == "BRK":
                stop = zh - 1.0 * av[t]
        else:
            l1, l2, l3, stop = zl, (zl + zh) / 2, zh, zh + (1.0 if kind == "BRK" else 0.25) * av[t]
            if kind == "BRK":
                stop = zl + 1.0 * av[t]
        r, nf, ex = ladder(o, hi, lo, c, f, t, side, l1, l2, l3, stop, VALID[kind], MAX_HOLD)
        edge, smc_stop = (zh, zl - 0.25 * av[t]) if side > 0 else (zl, zh + 0.25 * av[t])
        inner = zl if side > 0 else zh
        r_f1 = fade(o, hi, lo, c, f, t, side, edge, smc_stop, VALID[kind], 1.0, MAX_HOLD)
        r_f2 = fade(o, hi, lo, c, f, t, side, edge, smc_stop, VALID[kind], 2.0, MAX_HOLD)
        r_tr = trap(o, hi, lo, c, f, t, side, inner, smc_stop, av[t], VALID[kind], 6, 3.0, MAX_HOLD)
        row = {"symbol": sym, "t": idx[t], "kind": kind, "side": side, "R": r, "fills": nf,
               "R_fade1": r_f1, "R_fade2": r_f2, "R_trap": r_tr,
               "exit": idx[min(ex, len(idx) - 1)],
               "zone_atr": width / av[t], "dist_atr": side * (c[t] - (zh if side > 0 else zl)) / av[t],
               "imp_atr": side * (c[t] - c[max(t - 6, 0)]) / av[t], "absorb": absorb[t]}
        for rule, (bl, bh, rl, rh) in conf.items():
            if side > 0:
                row[f"conf_{rule}"] = float(not np.isnan(bl[t]) and zl <= bh[t] and zh >= bl[t])
            else:
                row[f"conf_{rule}"] = float(not np.isnan(rl[t]) and zl <= rh[t] and zh >= rl[t])
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    fx = feats.reindex(out["t"]).reset_index(drop=True)
    # направленные признаки: умножаем на сторону, чтобы «по направлению сделки» имело один знак
    for col in [c_ for c_ in fx.columns if c_.endswith(("_e20", "_e50", "_e20s", "_e50s", "_bos"))
                or c_ in ("flow3", "flow24", "oi4", "oi24", "oi72", "fund8", "fund24", "fund72")]:
        fx[col] = fx[col] * out["side"].to_numpy()
    for col in [c_ for c_ in fx.columns if c_.endswith("_pos")]:
        fx[col] = np.where(out["side"].to_numpy() > 0, fx[col], 1 - fx[col])
    return pd.concat([out, fx.astype("float32")], axis=1)


def btc_states(h: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for tf, rule in (("4h", "4h"), ("1d", "1D"), ("1w", "7D")):
        st = tf_state(_bars(h, rule), tf)[[f"{tf}_e50", f"{tf}_e50s", f"{tf}_bos"]]
        st.columns = [f"btc_{c_}" for c_ in st.columns]
        parts.append(_known_at(st, rule, h.index))
    return pd.concat(parts, axis=1)


def collect(root: Path, syms: list[str]) -> None:
    data = Data2(root, ["BTCUSDT", *syms])
    btc = btc_states(data.get("BTCUSDT", "1h", "full"))
    parts = []
    for s in mine(syms):
        try:
            h = data.get(s, "1h", "full")
            if len(h) < 24 * 60:
                continue
            x = coin_setups(h, btc, adv30(root, s), s)
            if len(x):
                parts.append(x)
        except Exception as e:
            print(f"  {s}: пропуск ({e})", flush=True)
        finally:
            data.cache.pop((s, "1h", "m"), None)
            data.cache.pop((s, "1h"), None)
    print(f"  монет с сетапами: {len(parts)}", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path("smc"), index=False)


# ---------- отчёт ----------

FEATS_X = ["zone_atr", "dist_atr", "imp_atr", "absorb", "conf_4h", "conf_1D", "vol_r3", "flow3", "flow24", "oi4", "oi24",
           "oi72", "fund8", "fund24", "fund72", "lsr", "top_lsr", "atr_pct", "hour",
           *[f"{tf}_{k}" for tf in TFS for k in ("e20", "e50", "e20s", "e50s", "pos", "rvp", "bos")],
           *[f"btc_{tf}_{k}" for tf in ("4h", "1d", "1w") for k in ("e50", "e50s", "bos")], "side"]


def _cell(x: pd.DataFrame) -> str:
    r = x["R"].dropna().to_numpy(dtype="float64")
    if len(r) < 10:
        return f"n={len(r)}"
    day = x.loc[x["R"].notna(), "t"].dt.floor("D").to_numpy()
    g = pd.Series(r - r.mean()).groupby(day).sum()
    t_day = r.sum() / np.sqrt((g ** 2).sum()) if (g ** 2).sum() > 0 else np.nan
    months = max(1, x["t"].dt.to_period("M").nunique())
    return f"{r.mean():+.3f}R (t {t_day:.1f}, {np.mean(r > 0):.0%}, {len(r) / months:.0f}/мес)"


def series_number(df: pd.DataFrame) -> np.ndarray:
    """Номер сетапа в серии: подряд идущие сетапы одного типа и стороны на монете; серия обнуляется после OB+SWEEP
    той же стороны (вытряхивание) или любого сетапа (кроме BRK) в обратную сторону. OB+SWEEP сам — №1."""
    d = df[df.kind != "BRK"].sort_values(["symbol", "t"])
    out = pd.Series(0, index=df.index, dtype="int64")
    for _, g in d.groupby("symbol", sort=False):
        cnt: dict[tuple[str, int], int] = {}
        for i, k, sd in zip(g.index, g["kind"].to_numpy(), g["side"].to_numpy()):
            for key in list(cnt):
                if key[1] != sd:
                    cnt[key] = 0
            if k == "OB+SWEEP":
                for key in list(cnt):
                    if key[1] == sd:
                        cnt[key] = 0
            cnt[(k, sd)] = cnt.get((k, sd), 0) + 1
            out[i] = cnt[(k, sd)]
    return out.to_numpy()


def report() -> None:
    parts = all_parts("smc")
    if not parts:
        print("частей нет")
        return
    import lightgbm as lgb
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["group"] = [group_of(s) for s in df["symbol"]]
    df["per"] = ""
    for p, (a, b) in PER.items():
        df.loc[(df.t >= a) & (df.t < b), "per"] = p
    print(f"===== SMC: сетапов {len(df):,}, монет {df.symbol.nunique()}, частей {len(parts)}; "
          f"с исполнением {int(df.R.notna().sum()):,} =====")
    print("ячейка: средний R на сделку (t по дням, прибыльных, сделок в месяц на весь рынок); тейк 5R, стоп за зоной")

    print("\n=== 1. Все сетапы без фильтра ===")
    rows = []
    for (k, sd), g in df.groupby(["kind", "side"]):
        rows.append({"тип": k, "сторона": "лонг" if sd > 0 else "шорт",
                     **{p: _cell(g[g.per == p]) for p in PER},
                     "совпадение с OB 4h": _cell(g[(g.conf_4h == 1) & (g.per != "is")]),
                     "совпадение с OB 1d": _cell(g[(g.conf_1D == 1) & (g.per != "is")])})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== 1b. Связки SMC (правила заданы до запуска), VAL и HOLDOUT ===")
    absorb_hi = df.loc[df.per == "is", "absorb"].quantile(0.8)
    combos = {
        "старшие ТФ по направлению (4h и 1d слом структуры в нашу сторону)": (df["4h_bos"] > 0) & (df["1d_bos"] > 0),
        "против старших ТФ": (df["4h_bos"] < 0) & (df["1d_bos"] < 0),
        "зона совпала с OB 4h": df["conf_4h"] == 1,
        "зона совпала с OB 4h и 1d": (df["conf_4h"] == 1) & (df["conf_1D"] == 1),
        "толпа против нас (funding 24 ч против стороны)": df["fund24"] < 0,
        "поглощение на дне (верхние 20% IS)": df["absorb"] >= absorb_hi,
        "OI рос в импульс (новые позиции)": df["oi4"] > 0,
        "старшие ТФ + OB 4h + толпа против": (df["4h_bos"] > 0) & (df["1d_bos"] > 0) & (df["conf_4h"] == 1)
                                            & (df["fund24"] < 0),
    }
    rows = []
    for k in ("OB+SWEEP", "OB", "FVG", "BRK"):
        g0 = df[df.kind == k]
        for name, mask in combos.items():
            g = g0[mask.loc[g0.index]]
            rows.append({"тип": k, "связка": name, "val": _cell(g[g.per == "val"]), "ho": _cell(g[g.per == "ho"]),
                         "is": _cell(g[g.per == "is"])})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== 1c. Против SMC: FADE 1:1 / 1:2 (вход против зоны стоп-ордером) и TRAP (стопы SMC сняты, цена вернулась "
          "в зону — входим по исходному направлению, тейк 3R); сторона — исходного SMC-сетапа ===")
    rows = []
    for (k, sd), g in df[df.kind != "BRK"].groupby(["kind", "side"]):
        for col, nm in (("R_fade1", "FADE 1:1"), ("R_fade2", "FADE 1:2"), ("R_trap", "TRAP 1:3")):
            x = g.assign(R=g[col])
            rows.append({"тип": k, "сетап": "лонг" if sd > 0 else "шорт", "вариант": nm,
                         **{p: _cell(x[x.per == p]) for p in PER}})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== 1d. Номер сигнала в серии: №1 / №2 / №3+ одного типа в одну сторону; серия обнуляется после "
          "вытряхивания (OB+SWEEP той же стороны) или сигнала в обратную сторону ===")
    df["nth"] = series_number(df)
    rows = []
    for (k, sd), g in df[df.kind != "BRK"].groupby(["kind", "side"]):
        for col, nm in (("R", "по SMC, 5R"), ("R_fade1", "FADE 1:1"), ("R_trap", "TRAP 1:3")):
            for nth, lab in ((1, "№1"), (2, "№2"), (3, "№3+")):
                x = g[g.nth == nth if nth < 3 else g.nth >= 3].assign(R=lambda d: d[col])
                rows.append({"тип": k, "сетап": "лонг" if sd > 0 else "шорт", "вариант": nm, "номер": lab,
                             **{p: _cell(x[x.per == p]) for p in PER}})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== 2. ML-фильтр (LightGBM на IS, признаки 1h / 4h / 1d / 1w), отбор по прогнозу R ===")
    feats = [c_ for c_ in FEATS_X if c_ in df.columns]
    df["y"] = df["R"].fillna(0.0).clip(-1.5, 6.0)
    params = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 400,
              "bagging_fraction": 0.7, "bagging_freq": 1, "feature_fraction": 0.7, "verbose": -1}
    rows = []
    for k, g in df.groupby("kind"):
        is_ = g[g.per == "is"].sort_values("t")
        folds = np.array_split(np.arange(len(is_)), 4)
        oof = np.full(len(is_), np.nan)
        for q in range(1, 4):
            tr = np.concatenate(folds[:q])
            m_ = lgb.train(params, lgb.Dataset(is_[feats].iloc[tr], is_["y"].iloc[tr]), 300)
            oof[folds[q]] = m_.predict(is_[feats].iloc[folds[q]])
        m = lgb.train(params, lgb.Dataset(is_[feats], is_["y"]), 300)
        imp = pd.Series(m.feature_importance("gain"), index=feats).sort_values(ascending=False)
        print(f"  {k}: важные признаки — " + ", ".join(f"{a} {v / imp.sum():.0%}" for a, v in imp.head(8).items()))
        pred = pd.Series(m.predict(g[feats]), index=g.index)
        pred.loc[is_.index] = oof
        for top in TOPS:
            thr = np.nanquantile(oof, 1 - top)
            sel = g[pred >= thr]
            rows.append({"тип": k, "отбор": f"верх {top:.0%}",
                         **{p: _cell(sel[sel.per == p]) for p in PER},
                         "VAL+HO вне подбора": _cell(sel[(sel.per != "is") & sel.group.isin(["ext54", "fresh"])])})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 340)
    pd.set_option("display.max_columns", 30)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [x for x in a.symbols.split(",") if x])
    else:
        report()
