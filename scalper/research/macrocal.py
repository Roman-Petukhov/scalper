"""Макрособытия США и сделки бота 4h (по вопросу владельца «стоит ли добавлять макро-фундаментал»).

События и время выхода (America/New_York): FOMC — решение по ставке в 14:00; CPI и отчёт о занятости (NFP) — 8:30.
Даты: FOMC — список ниже (заседания 2020–2026, включая внеплановые 3 и 15 марта 2020), сверяется со страницей ФРС;
CPI и NFP — архивы релизов BLS (имя файла архива содержит дату выхода). Не скачалось — событие в отчёте не участвует.
Календарь известен заранее, поэтому признаки ниже не заглядывают вперёд.

Сделки — правило бота 4h без изменений (research/noline.py: линия, агрессоры, закрытие, тренд, ретест, стоп, 3R).
Признаки и ожидаемое направление заданы до запуска («событие» — любое из трёх, и каждое отдельно):
  M1 сигнал у события    свеча пробоя закрылась в пределах 24 ч до события или содержит его; ждём, что хуже
  M2 событие впереди     событие в ближайшие 72 ч после исполнения ретеста; ждём, что хуже
  M3 позиция пережила    событие пришлось на время позиции (от исполнения до выхода) — справочно, не фильтр:
                         при входе выход неизвестен
Принятие (как research/footprint.py): группа «без события» лучше группы «с событием» в каждом периоде, где в обеих
≥ 20 сделок, и разница за всё время с t ≥ 2 — тогда кандидат в фильтр «не входить перед событием».

    python -m research.macrocal collect --symbols <1h-qualified>    (календарь — часть 0)
    python -m research.macrocal report
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.request
import warnings
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .broad import adv30
from .footprint import verdict
from .noline import TF, coin_trades
from .oos import PER_ALL
from .shard import all_parts, mine, part_path, shard_id
from .smc import _cell
from .tline import tf_frame

NY = ZoneInfo("America/New_York")
FOMC = (  # день решения (второй день заседания); 2020-03-03 и 2020-03-15 — внеплановые
    "2020-01-29 2020-03-03 2020-03-15 2020-04-29 2020-06-10 2020-07-29 2020-09-16 2020-11-05 2020-12-16 "
    "2021-01-27 2021-03-17 2021-04-28 2021-06-16 2021-07-28 2021-09-22 2021-11-03 2021-12-15 "
    "2022-01-26 2022-03-16 2022-05-04 2022-06-15 2022-07-27 2022-09-21 2022-11-02 2022-12-14 "
    "2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20 2023-11-01 2023-12-13 "
    "2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18 "
    "2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10 "
    "2026-01-28 2026-03-18 2026-04-29 2026-06-17 2026-07-29 2026-09-16").split()
FED_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
BLS = {"CPI": "https://www.bls.gov/bls/news-release/cpi.htm",
       "NFP": "https://www.bls.gov/bls/news-release/empsit.htm"}
BLS_FILE = {"CPI": "cpi", "NFP": "empsit"}
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
           "Accept": "text/html,application/xhtml+xml", "Accept-Language": "en-US,en;q=0.9"}
BEFORE, AHEAD = pd.Timedelta(hours=24), pd.Timedelta(hours=72)
SPLITS = {"M1": "сигнал за 24 ч до события или на нём", "M2": "событие в 72 ч после входа",
          "M3": "событие во время позиции (справочно)"}


def _get(url: str) -> str | None:
    for k in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception as e:
            print(f"  macrocal: {url} — {e}", flush=True)
            time.sleep(2 ** k)
    return None


def at(day: str, hh: int, mm: int) -> pd.Timestamp:
    """Момент события в UTC по нью-йоркскому времени."""
    d = pd.Timestamp(day)
    return pd.Timestamp(d.year, d.month, d.day, hh, mm, tz=NY).tz_convert("UTC")


def bls_dates(html: str, stem: str) -> list[str]:
    """Даты выхода из ссылок архива вида /news.release/archives/cpi_02102022.htm (MMDDYYYY)."""
    out = set()
    for m in re.findall(rf"{stem}_(\d{{8}})\.htm", html):
        try:
            out.add(pd.Timestamp(f"{m[4:]}-{m[:2]}-{m[2:4]}").strftime("%Y-%m-%d"))
        except ValueError:
            continue
    return sorted(out)


def fed_check(html: str) -> list[str]:
    """Годы, в которых число заседаний на странице ФРС не совпало со списком FOMC (грубая сверка)."""
    bad = []
    for y in range(2021, 2027):
        block = re.search(rf"{y} FOMC Meetings(.*?)(?:{y - 1} FOMC Meetings|$)", html, re.S)
        if not block:
            continue
        n_page = len(re.findall(r'fomc-meeting__month', block.group(1)))
        n_list = sum(d.startswith(str(y)) for d in FOMC)
        if n_page and n_page != n_list and not (y == 2026 and n_page == 8):
            bad.append(f"{y}: на странице {n_page}, в списке {n_list}")
    return bad


def calendar() -> pd.DataFrame:
    rows = [{"kind": "FOMC", "at": at(d, 14, 0)} for d in FOMC]
    for kind, url in BLS.items():
        html = _get(url)
        days = bls_dates(html, BLS_FILE[kind]) if html else []
        print(f"  macrocal: {kind} — дат {len(days)}" + (f" ({days[0]} … {days[-1]})" if days else ""), flush=True)
        rows += [{"kind": kind, "at": at(d, 8, 30)} for d in days if "2019-12-01" <= d <= "2026-12-31"]
    html = _get(FED_URL)
    if html:
        bad = fed_check(html)
        print("  macrocal: сверка FOMC с ФРС — " + ("расхождения: " + "; ".join(bad) if bad else "совпадает"), flush=True)
    return pd.DataFrame(rows)


def collect(root: Path, syms: list[str]) -> None:
    if shard_id()[0] == 0:
        calendar().to_parquet(part_path("macrocal_events"), index=False)
    rows = []
    for s in mine(syms):
        try:
            d = tf_frame(root, s, TF)
            if d is None or len(d) < 500:
                continue
            liq = adv30(root, s).reindex(d.index.floor("D")).to_numpy()
            x = coin_trades(d, liq, s)
            if not len(x):
                continue
            x = x[(x.trig == "line") & (x.entry == "retest")]
            step = pd.Timedelta(TF)
            for r in x.itertuples(index=False):
                rows.append({"symbol": s, "t": r.t, "side": int(r.side), "R": float(r.R3),
                             "sig_close": r.t + step, "fill": d.index[int(r.fill_i)],
                             "exit": d.index[int(r.exit_i)] + step})
        except Exception as e:
            print(f"  macrocal {s}: пропуск ({e})", flush=True)
    print(f"  macrocal: сделок {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(part_path("macrocal"), index=False)


def flags(x: pd.DataFrame, ev: pd.Series) -> pd.DataFrame:
    """M1–M3 по моментам событий ev (UTC)."""
    e = np.sort(pd.to_datetime(ev, utc=True).to_numpy(dtype="datetime64[ns]"))
    out = x.copy()

    def any_in(lo: pd.Series, hi: pd.Series) -> np.ndarray:
        lo_, hi_ = lo.to_numpy(dtype="datetime64[ns]"), hi.to_numpy(dtype="datetime64[ns]")
        return np.searchsorted(e, hi_, side="right") > np.searchsorted(e, lo_, side="left")

    start = out.t
    out["M1"] = np.where(any_in(start, out.sig_close + BEFORE), "да", "нет")
    out["M2"] = np.where(any_in(out.fill, out.fill + AHEAD), "да", "нет")
    out["M3"] = np.where(any_in(out.fill, out.exit), "да", "нет")
    return out


def report() -> None:
    parts, ep = all_parts("macrocal"), all_parts("macrocal_events")
    if not parts or not ep:
        print("  macrocal: нет данных")
        return
    x = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    for c in ("t", "sig_close", "fill", "exit"):
        x[c] = pd.to_datetime(x[c], utc=True)
    ev = pd.concat([pd.read_parquet(p) for p in ep], ignore_index=True)
    print("===== MACROCAL: FOMC, CPI, NFP и сделки бота 4h (признаки и направление до запуска) =====")
    print("     событий: " + ", ".join(f"{k} {n}" for k, n in ev.kind.value_counts().items()) + f"; сделок {len(x)}")
    for kinds in (("FOMC", "CPI", "NFP"), ("FOMC",), ("CPI",), ("NFP",)):
        e = ev[ev.kind.isin(kinds)]["at"]
        if not len(e):
            print(f"  {' + '.join(kinds)}: дат нет")
            continue
        z = flags(x, e)
        z["t"] = z["t"].dt.tz_localize(None)
        print(f"  {' + '.join(kinds)}:")
        for key, title in SPLITS.items():
            print(f"   {key} {title}:")
            for g in ("да", "нет"):
                m = z[z[key] == g]
                cells = "  ".join(f"{p}: {_cell(m[(m.t >= a) & (m.t < b)])}" for p, (a, b) in PER_ALL.items())
                print(f"     {g}: доля {len(m) / max(1, len(z)):.0%}  {cells}  всё время: {_cell(m)}")
            line, _ = verdict(z, key, "нет")
            print(f"     «нет» минус «да»: {line}")


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
