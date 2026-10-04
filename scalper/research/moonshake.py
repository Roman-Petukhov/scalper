"""
Вход после «вытряхивания»: на сигналах модели research.moonshots (монету скоро понесёт) ждём пролива на −b от цены
среза в течение 24 ч (раньше пролива цена не должна уйти на +b вверх) и покупаем, когда цена вернулась к уровню
среза в течение R часов после пролива (по закрытию часа). Стоп — под минимумом пролива (−0.5%), трейлинг trail от
лучшей цены без тейка, до 30 дней. Издержки 12 б.п. + проскальзывание 0.25% на входе и на выходе по стопу, funding.
Размер позиции в счёте — от расстояния до стопа (риск 1% / 2% капитала, номинал не больше капитала).

Отборы — как в research.moonbracket (модель, vol_ratio, случайные срезы — проверка, что дело не в самом паттерне).

    python -m research.moonshake collect --symbols <все перпетуалы>   (по частям)
    python -m research.moonshake report
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from .broad import adv30
from .moonbracket import COST, MAX_HOLD, SLIP, WINDOW, account, dedupe, load, selectors, trade_stats
from .moonshots import PER, coin_rows
from .shard import mine, part_path
from .wave2 import Data2

DIPS = (0.03, 0.05, 0.10)
RECLAIM_H = (6, 24)
TRAILS = (0.30, 0.50)
CONFIGS = [(b, rh, tr) for b in DIPS for rh in RECLAIM_H for tr in TRAILS]
STOP_PAD = 0.005


def cfg_name(b: float, rh: int, tr: float) -> str:
    return f"d{int(b * 100)}_r{rh}_t{int(tr * 100)}"


@njit(cache=True)
def shakeout(o, h, lo, c, fund, starts, dip, reclaim_h, trail, window, max_hold, cost, slip, pad):
    """Результат лонга после пролива и возврата (NaN — паттерна не было), бар выхода (или конца ожидания),
    расстояние до стопа на входе (доля) и лучший ход (доля от входа)."""
    n, m = len(c), len(starts)
    res = np.full(m, np.nan)
    end_out = np.zeros(m, np.int64)
    risk = np.full(m, np.nan)
    mfe = np.full(m, np.nan)
    for k in range(m):
        i = starts[k]
        ref = c[i]
        lv, up = ref * (1 - dip), ref * (1 + dip)
        j = i + 1
        last_wait = min(i + window, n - 1)
        d = -1
        while j <= last_wait:
            if h[j] >= up:
                break                                      # сначала ушла вверх — это не вытряхивание
            if lo[j] <= lv:
                d = j
                break
            j += 1
        if d < 0:
            end_out[k] = min(j, last_wait)
            continue
        dip_low = lo[d]
        e = -1
        q = d
        last_rc = min(d + reclaim_h, n - 1)
        while q <= last_rc:
            dip_low = min(dip_low, lo[q])
            if c[q] >= ref:
                e = q
                break
            q += 1
        if e < 0:
            end_out[k] = last_rc
            continue
        entry = c[e] * (1 + slip)
        sp = dip_low * (1 - pad)
        risk[k] = 1 - sp / entry
        ext = c[e]
        paid = 0.0
        ex = np.nan
        q = e + 1
        last = min(e + max_hold, n - 1)
        while q <= last:
            paid += fund[q]
            if o[q] <= sp:
                ex = o[q]
            elif lo[q] <= sp:
                ex = sp
            else:
                ext = max(ext, h[q])
                sp = max(sp, ext * (1 - trail))
            if not np.isnan(ex):
                break
            q += 1
        if np.isnan(ex):
            q = last
            ex = c[last]
        res[k] = ex / entry - 1 - cost - slip - paid
        end_out[k] = q
        mfe[k] = ext / entry - 1
    return res, end_out, risk, mfe


def collect(root: Path, syms: list[str]) -> None:
    data = Data2(root, ["BTCUSDT", *syms])
    btc = data.get("BTCUSDT", "1h", "full")["close"]
    parts = []
    for s in mine(syms):
        try:
            h = data.get(s, "1h", "full")
            if len(h) < 24 * 3:
                continue
            rows = coin_rows(h, adv30(root, s), btc, s, False)
            if not len(rows):
                continue
            idx = h.index.get_indexer(pd.to_datetime(rows["t"], utc=True)).astype(np.int64)
            arr = {c_: h[c_].to_numpy(dtype="float64") for c_ in ("open", "high", "low", "close")}
            f = h["funding"].fillna(0.0).to_numpy(dtype="float64")
            for b, rh, tr in CONFIGS:
                r, end, rk, mf = shakeout(arr["open"], arr["high"], arr["low"], arr["close"], f, idx, b, rh, tr,
                                          WINDOW, MAX_HOLD, COST, SLIP, STOP_PAD)
                nm = cfg_name(b, rh, tr)
                rows[f"r_{nm}"] = r.astype("float32")
                rows[f"end_{nm}"] = h.index[end].to_numpy()
                rows[f"risk_{nm}"] = rk.astype("float32")
                rows[f"mfe_{nm}"] = mf.astype("float32")
                rows[f"side_{nm}"] = np.where(np.isnan(r), 0, 1).astype(np.int8)
            parts.append(rows.drop(columns=[c_ for c_ in rows.columns if c_.startswith("tr_")]))
        except Exception as e:
            print(f"  {s}: пропуск ({e})", flush=True)
        finally:
            data.cache.pop((s, "1h", "m"), None)
            data.cache.pop((s, "1h"), None)
    print(f"  монет со срезами: {len(parts)}", flush=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(part_path("moonshake"), index=False)


def report() -> None:
    df = load("moonshake")
    if df is None:
        print("частей нет")
        return
    print(f"===== MOONSHAKE: срезов {len(df):,}, монет {df['symbol'].nunique()}, частей {df.attrs['parts']} =====")
    print(f"пролив на −d за {WINDOW} ч, возврат к цене среза за R ч → лонг; стоп под минимумом пролива, трейлинг t, "
          f"до {MAX_HOLD // 24} дней; издержки {COST * 1e4:.0f} б.п. + проскальзывание {SLIP:.2%} x2")
    sel = selectors(df)
    for b, rh, tr in CONFIGS:
        nm = cfg_name(b, rh, tr)
        print(f"\n=== пролив −{b:.0%}, возврат за {rh} ч, трейлинг {tr:.0%} ===")
        rows = []
        for sname, mask in sel.items():
            for p, (a, z) in PER.items():
                x = dedupe(df[mask & (df.t >= a) & (df.t < z)], nm)
                st = trade_stats(x, nm)
                st.pop("лонг", None)
                st.pop("шорт", None)
                rk = x[f"risk_{nm}"].dropna()
                rows.append({"отбор": sname, "период": p, **st,
                             "стоп (медиана)": f"{rk.median():.1%}" if len(rk) else "—"})
        print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== Счёт: риск 1% / 2% капитала на сделку (номинал от расстояния до стопа), до 10 позиций ===")
    rows = []
    for sname in ("модель, верх 1.0%", "модель, верх 2.0%", "все срезы (5% случайных)"):
        for b, rh, tr in CONFIGS:
            nm = cfg_name(b, rh, tr)
            for lab, a in (("VAL+HO", PER["val"][0]), ("HO", PER["ho"][0])):
                x = dedupe(df[sel[sname] & (df.t >= a) & (df.t < PER["ho"][1])], nm)
                for risk in (0.01, 0.02):
                    rows.append({"отбор": sname, "вариант": nm, "период": lab, "риск": f"{risk:.0%}",
                                 **account(x, nm, f"risk_{nm}", risk)})
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
