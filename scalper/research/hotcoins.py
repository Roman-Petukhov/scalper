"""«Горячие» монеты дня и правило бота на 15m / 5m — по разбору канала Digahka (скальпинг 1m/5m только по топу
растущих и падающих за сутки, с ликвидностью от $10–50M или 1M сделок за 24 ч).

Идея: наши 15m и 5m берут крупные спокойные монеты, где стоп 0.8–1.5% цены и комиссии съедают ощутимую долю R; у топа
движения дня волатильность в разы выше — может быть, пробои там доходят до 3R чаще. Сделки — правило бота (линия
зигзага 3 ATR, агрессоры от 55%, закрытие в своей половине, тренд старшего ТФ, стоп за свингом, 3R — research/noline.py,
как в research/lowtf.py), меняется только отбор монет. Монеты — все из symbols_qualified (~725), с 2022.

Задано до запуска:
  «горячая» монета — на последнем закрытом часе к закрытию свечи сигнала оборот за 24 ч ≥ $50M и она в топ-10 таких
  монет по |изменению цены за 24 ч| (по часовым свечам; закрытие часа ≤ закрытия свечи сигнала — без подглядывания).
  Принятие для каждого ТФ, вход по рынку (как у бота), обе стороны: «горячие» лучше остальных в каждом периоде
  (IS / VAL / HO), где в обеих группах ≥ 20 сделок, разница за всё время с t ≥ 2, и сами «горячие» в плюсе в каждом
  таком периоде (лучше, но в минусе — не нужно). Справочно: только шорты (бот 15m / 5m торгует шорты), вход на ретесте,
  порог оборота $20M, топ-30, сторона по движению (лонг у растущей, шорт у падающей) против встречной.

    python -m research.hotcoins collect --symbols <hot-wide>    (по частям)
    python -m research.hotcoins report
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import noline
from .broad import adv30
from .footprint import verdict
from .oos import PER
from .shard import all_parts, mine, part_path
from .smc import _cell
from .tline import tf_frame

TFS = tuple(os.environ.get("HOT_TFS", "15m,5m").split(","))
GATE, GATE_LOW, TOP, TOP_WIDE = 50e6, 20e6, 10, 30
STEP = {"15m": pd.Timedelta("15min"), "5m": pd.Timedelta("5min")}


def hourly(h1: pd.DataFrame, sym: str) -> pd.DataFrame:
    """Изменение цены и оборот за 24 ч на каждом часе (индекс — открытие часа); только часы с оборотом ≥ GATE_LOW."""
    c = h1["close"].astype("float64")
    qv = h1["quote_volume"].astype("float64").rolling(24, min_periods=24).sum()
    x = pd.DataFrame({"h": h1.index, "symbol": sym, "ret24": (c / c.shift(24) - 1).to_numpy(), "qv24": qv.to_numpy()})
    x = x[(x.qv24 >= GATE_LOW) & x.ret24.notna()]
    return x.astype({"ret24": "float32", "qv24": "float32"})


def ranks(hours: pd.DataFrame) -> pd.DataFrame:
    """Место монеты по |ret24| в своём часе среди монет с оборотом ≥ GATE (rank) и ≥ GATE_LOW (rank_low); 1 — сильнее."""
    x = hours.copy()
    mv = x.ret24.abs()
    x["rank_low"] = mv.groupby(x.h).rank(ascending=False, method="first")
    big = x.qv24 >= GATE
    x["rank"] = np.nan
    x.loc[big, "rank"] = mv[big].groupby(x.h[big]).rank(ascending=False, method="first")
    return x


def signal_hour(t: pd.Series, tf: str) -> pd.Series:
    """Открытие последнего часа, закрывшегося не позже закрытия свечи сигнала (t — открытие свечи)."""
    return (t + STEP[tf]).dt.floor("h") - pd.Timedelta(hours=1)


def label(trades: pd.DataFrame, hours: pd.DataFrame) -> pd.DataFrame:
    """Признаки «горячести» для сделок по рангам часов."""
    r = ranks(hours)[["h", "symbol", "ret24", "rank", "rank_low"]]
    out = []
    for tf, z in trades.groupby("tf"):
        z = z.assign(h=signal_hour(z.t, str(tf))).merge(r, on=["h", "symbol"], how="left")
        out.append(z)
    x = pd.concat(out, ignore_index=True)
    x["hot"] = np.where(x["rank"] <= TOP, "да", "нет")
    x["hot_low"] = np.where(x["rank_low"] <= TOP, "да", "нет")
    x["hot30"] = np.where(x["rank"] <= TOP_WIDE, "да", "нет")
    x["with_move"] = np.where(np.sign(x.ret24.fillna(0)) == x.side, "по движению", "против")
    return x


def collect(root: Path, syms: list[str]) -> None:
    trades, hours = [], []
    for s in mine(syms):
        try:
            h1 = tf_frame(root, s, "1h")
            if h1 is None or len(h1) < 500:
                continue
            hours.append(hourly(h1, s))
            liq_d = adv30(root, s)
            for tf in TFS:
                d = tf_frame(root, s, tf)
                if d is None or len(d) < 2000:
                    continue
                noline.TF = tf
                x = noline.coin_trades(d, liq_d.reindex(d.index.floor("D")).to_numpy(), s, lines_only=True)
                if len(x):
                    trades.append(x.assign(tf=tf)[["symbol", "tf", "t", "side", "entry", "R3", "risk_pct"]])
        except Exception as e:
            print(f"  hotcoins {s}: пропуск ({e})", flush=True)
    print(f"  hotcoins: монет с часами {len(hours)}, частей сделок {len(trades)}", flush=True)
    if trades:
        pd.concat(trades, ignore_index=True).to_parquet(part_path("hotcoins"), index=False)
    if hours:
        pd.concat(hours, ignore_index=True).to_parquet(part_path("hotcoins_hours"), index=False)


def _split(z: pd.DataFrame, key: str, good: str) -> None:
    for g in sorted(z[key].dropna().unique()):
        m = z[z[key] == g]
        cells = "  ".join(f"{p}: {_cell(m[(m.t >= a) & (m.t < b)])}" for p, (a, b) in PER.items())
        print(f"       {g}: доля {len(m) / max(1, len(z)):.0%}  {cells}  всё время: {_cell(m)}")
    line, _ = verdict(z, key, good)
    print(f"       «{good}» минус остальные: {line}")


def accepted(z: pd.DataFrame) -> tuple[bool, str]:
    """Основной критерий: verdict по «hot» плюс «горячие» в плюсе в каждом периоде с ≥ 20 сделками в обеих группах."""
    _, ok = verdict(z, "hot", "да")
    pos = []
    for a, b in PER.values():
        m = z[(z.t >= a) & (z.t < b)]
        g, r = m[m.hot == "да"], m[m.hot != "да"]
        if len(g) >= 20 and len(r) >= 20:
            pos.append(g.R.mean() > 0)
    plus = bool(pos) and all(pos)
    return ok and plus, f"лучше остальных: {'да' if ok else 'нет'}; в плюсе во всех периодах: {'да' if plus else 'нет'}"


def report() -> None:
    tp, hp = all_parts("hotcoins"), all_parts("hotcoins_hours")
    print("\n===== HOTCOINS: правило бота на 15m / 5m только по топ-10 движения дня (оборот ≥ $50M) против остальных; "
          "ячейка — R на сделку (t по дням, прибыльных, сделок в месяц) =====")
    if not tp or not hp:
        print("  hotcoins: нет данных")
        return
    tr = pd.concat([pd.read_parquet(p) for p in tp], ignore_index=True)
    hr = pd.concat([pd.read_parquet(p) for p in hp], ignore_index=True)
    tr["t"] = pd.to_datetime(tr["t"], utc=True)
    hr["h"] = pd.to_datetime(hr["h"], utc=True)
    x = label(tr, hr)
    x["R"] = x["R3"]
    x["t"] = x["t"].dt.tz_localize(None)
    print(f"     сделок {len(x)} на {x.symbol.nunique()} монетах; часов в рейтинге {hr.h.nunique()}, "
          f"монет с оборотом ≥ $50M в часе (медиана) {hr[hr.qv24 >= GATE].groupby('h').size().median():.0f}")
    for tf in TFS:
        for entry in ("market", "retest"):
            z = x[(x.tf == tf) & (x.entry == entry)]
            if z.empty:
                continue
            hot = z[z.hot == "да"]
            print(f"\n  --- {tf}, вход {entry}: сделок {len(z)}, «горячих» {len(hot)} ({len(hot) / len(z):.0%}); стоп "
                  f"медиана: горячие {hot.risk_pct.median() * 100:.2f}%, остальные "
                  f"{z[z.hot != 'да'].risk_pct.median() * 100:.2f}% цены ---")
            for nm, zz in (("обе стороны", z), ("только шорты", z[z.side < 0])):
                print(f"    {nm}:")
                print("     топ-10, оборот ≥ $50M (основной):")
                _split(zz, "hot", "да")
                if entry == "market" and nm == "обе стороны":
                    ok, why = accepted(zz)
                    print(f"     ИТОГ {tf}: {'принято' if ok else 'не принято'} ({why})")
                print("     топ-10, оборот ≥ $20M:")
                _split(zz, "hot_low", "да")
                print("     топ-30, оборот ≥ $50M:")
                _split(zz, "hot30", "да")
            h = z[z.hot == "да"]
            if len(h):
                print("    «горячие»: сторона по движению за 24 ч против встречной:")
                _split(h, "with_move", "по движению")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--root", default="~/bn")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(Path(a.root).expanduser(), [s for s in a.symbols.split(",") if s])
    else:
        report()
