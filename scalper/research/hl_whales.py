"""
Hyperliquid: можно ли зарабатывать, повторяя сделки прибыльных кошельков (все позиции на HL публичны).

Данные — публичный API (api.hyperliquid.xyz/info, лимит 1200 «весов» в минуту на IP):
    leaderboard                кошельки с оборотом и PnL (stats-data.hyperliquid.xyz)
    userFillsByTime            сделки кошелька (не больше 10 000 последних, по 2000 за запрос)
    candleSnapshot 1h          часовые свечи монеты (последние 5000 — около 208 дней)
Кошельки берутся СЛУЧАЙНО из активных (оборот за всё время $20M–$5B), а не из топа по PnL: рейтинг знает будущее.

Сделки кошелька склеиваются в эпизоды «от нуля до нуля» по каждой монете (переворот — закрытие и новое открытие).
Эпизоды, начатые до доступной истории или не закрытые, отбрасываются; позиция не меньше $10k, удержание >= 5 мин.

Протокол: граница T. Отбор кошельков — только по эпизодам, закрытым до T (их собственные цены):
    S_all      все кошельки (база)
    S_skill    >= 15 эпизодов до T, средний результат > 0 и t > 2
    S_pnl      >= 10 эпизодов до T и сумма PnL до T >= $1M
    S_losers   >= 15 эпизодов до T и t < -2 — проверяем, есть ли смысл торговать ПРОТИВ них
После T — эпизоды, открытые после T:
    own        результат по их собственным ценам (потолок: копия без задержки)
    copy_1h    копия с задержкой: вход по open первого часового бара после их первой сделки, выход — после
               закрытия ими позиции; издержки 12 б.п. (Bybit тейкер туда-обратно)
t — кластеризованный по дням открытия; также разбивка по длительности удержания.

    python -m research.hl_whales collect [--cache <dir>] [--wallets 500]   (по частям, см. research.shard)
    python -m research.hl_whales report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .shard import all_parts, mine, part_path

INFO = "https://api.hyperliquid.xyz/info"
LEADERBOARD = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"
T_SPLIT = pd.Timestamp("2026-04-01", tz="UTC")
FILLS_SINCE = pd.Timestamp("2025-07-01", tz="UTC")
MAX_PAGES = 3
MIN_NOTIONAL = 10_000.0
MIN_HOLD = pd.Timedelta(minutes=5)
COST = 12e-4
WEIGHT_PER_MIN = 1000


class Limiter:
    """Скользящее окно 60 с по весам запросов (запас до лимита 1200)."""

    def __init__(self, per_min: int) -> None:
        self.per_min, self.log = per_min, []

    def take(self, w: int) -> None:
        while True:
            now = time.time()
            self.log = [(t, x) for t, x in self.log if now - t < 60]
            if sum(x for _, x in self.log) + w <= self.per_min:
                self.log.append((now, w))
                return
            time.sleep(1.0)


LIM = Limiter(WEIGHT_PER_MIN)


def _post(body: dict, weight: int) -> list | dict | None:
    LIM.take(weight)
    data = json.dumps(body).encode()
    for attempt in range(6):
        try:
            req = urllib.request.Request(INFO, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            if 400 <= e.code < 500:
                return None                                    # монеты/кошелька нет (например, делистинг)
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"HL info: не удалось выполнить {body.get('type')}")


def leaderboard(cache: Path) -> pd.DataFrame:
    p = cache / "leaderboard.json"
    if not p.exists():
        with urllib.request.urlopen(urllib.request.Request(LEADERBOARD, headers={"User-Agent": "Mozilla/5.0"}),
                                    timeout=120) as r:
            p.write_bytes(r.read())
    rows = []
    for x in json.loads(p.read_text())["leaderboardRows"]:
        w = {k: v for k, v in x["windowPerformances"]}
        rows.append({"user": x["ethAddress"], "account": float(x["accountValue"]),
                     "vlm_all": float(w["allTime"]["vlm"]), "pnl_all": float(w["allTime"]["pnl"]),
                     "vlm_month": float(w["month"]["vlm"])})
    return pd.DataFrame(rows)


def fills(user: str, cache: Path) -> pd.DataFrame:
    p = cache / f"fills-{user}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    out, start = [], int(FILLS_SINCE.timestamp() * 1000)
    for _ in range(MAX_PAGES):
        page = _post({"type": "userFillsByTime", "user": user, "startTime": start, "aggregateByTime": True},
                     20 + 100)
        if not page:
            break
        out += page
        if len(page) < 2000:
            break
        start = max(f["time"] for f in page) + 1
    df = pd.DataFrame(out)
    if len(df):
        df = df[["coin", "px", "sz", "side", "time", "startPosition", "dir", "closedPnl", "fee", "tid"]]
        df = df.drop_duplicates("tid").astype({"px": "float64", "sz": "float64", "startPosition": "float64",
                                               "closedPnl": "float64", "fee": "float64"})
    df.to_parquet(p, index=False)
    return df


def episodes(f: pd.DataFrame) -> pd.DataFrame:
    """Эпизоды позиции «от нуля до нуля» по каждой перп-монете; переворот закрывает эпизод и открывает новый."""
    if not len(f):
        return pd.DataFrame()
    f = f[~f["coin"].str.startswith("@") & ~f["coin"].str.contains(":")].sort_values(["coin", "time", "tid"])
    rows = []
    for coin, g in f.groupby("coin"):
        cur = None
        for r in g.itertuples():
            signed = r.sz if r.side == "B" else -r.sz
            before, after = r.startPosition, r.startPosition + signed
            if abs(after) < 1e-12:
                after = 0.0
            if cur is None:
                if abs(before) > 1e-12:
                    continue                                       # начало позиции раньше доступной истории
                if after == 0.0:
                    continue
            if cur is not None:
                adds = np.sign(signed) == cur["side"]
                q = abs(signed) if adds else min(abs(signed), abs(before))
                if adds:
                    cur["in_q"] += q
                    cur["in_v"] += q * r.px
                else:
                    cur["out_q"] += q
                    cur["out_v"] += q * r.px
                cur["pnl"] += r.closedPnl - r.fee
                cur["peak"] = max(cur["peak"], abs(after) * r.px)
                if after == 0.0 or np.sign(after) != cur["side"]:
                    cur["t_close"] = r.time
                    rows.append(cur)
                    cur = None
                    if after == 0.0:
                        continue
                    before = 0.0                                   # переворот: остаток — новый эпизод
                    signed = after
                else:
                    continue
            cur = {"coin": coin, "side": float(np.sign(signed)), "t_open": r.time, "in_q": abs(signed),
                   "in_v": abs(signed) * r.px, "out_q": 0.0, "out_v": 0.0, "pnl": 0.0,
                   "peak": abs(signed) * r.px}
    ep = pd.DataFrame(rows)
    if not len(ep):
        return ep
    ep["t_open"] = pd.to_datetime(ep["t_open"], unit="ms", utc=True)
    ep["t_close"] = pd.to_datetime(ep["t_close"], unit="ms", utc=True)
    ep["own"] = ep["side"] * (ep["out_v"] / ep["out_q"] / (ep["in_v"] / ep["in_q"]) - 1)
    ep = ep[(ep["peak"] >= MIN_NOTIONAL) & (ep["t_close"] - ep["t_open"] >= MIN_HOLD)]
    return ep[["coin", "side", "t_open", "t_close", "own", "pnl", "peak"]].reset_index(drop=True)


def candles(coin: str, cache: Path) -> pd.DataFrame | None:
    p = cache / f"c1h-{coin}.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        return df if len(df) else None
    end = int(pd.Timestamp("2026-10-01", tz="UTC").timestamp() * 1000)
    try:
        raw = _post({"type": "candleSnapshot", "req": {"coin": coin, "interval": "1h",
                                                       "startTime": end - 5000 * 3_600_000, "endTime": end}}, 20 + 84)
    except RuntimeError as e:
        print(f"  свечи {coin}: пропуск ({e})", flush=True)
        return None
    df = pd.DataFrame([{"t": c["t"], "o": float(c["o"])} for c in raw]) if raw else pd.DataFrame(columns=["t", "o"])
    df.to_parquet(p, index=False)
    return df if len(df) else None


def copy_returns(ep: pd.DataFrame, cache: Path) -> pd.Series:
    out = pd.Series(np.nan, index=ep.index)
    for coin, g in ep.groupby("coin"):
        c = candles(coin, cache)
        if c is None:
            continue
        t = pd.to_datetime(c["t"], unit="ms", utc=True).to_numpy()
        o = c["o"].to_numpy()
        i_in = np.searchsorted(t, g["t_open"].to_numpy(), side="right")
        i_out = np.searchsorted(t, g["t_close"].to_numpy(), side="right")
        ok = (i_in < len(t)) & (i_out < len(t)) & (i_in > 0)
        r = np.full(len(g), np.nan)
        r[ok] = g["side"].to_numpy()[ok] * (o[i_out[ok]] / o[i_in[ok]] - 1) - COST
        out.loc[g.index] = r
    return out


def _stat(x: pd.Series, t: pd.Series) -> dict:
    ok = x.notna()
    r, d = x[ok].to_numpy(dtype="float64"), t[ok].dt.floor("D").to_numpy()
    if len(r) < 5:
        return {"n": len(r)}
    g = pd.Series(r - r.mean()).groupby(d).sum()
    return {"n": len(r), "mean": r.mean(), "median": float(np.median(r)), "win": float((r > 0).mean()),
            "t_day": r.sum() / np.sqrt((g ** 2).sum()) if (g ** 2).sum() > 0 else np.nan}


def wallet_is(ep: pd.DataFrame) -> dict:
    pre = ep[ep["t_close"] < T_SPLIT]
    if len(pre) < 3:
        return {"n_is": len(pre)}
    r = pre["own"].to_numpy()
    return {"n_is": len(pre), "mean_is": r.mean(), "t_is": r.mean() / (r.std(ddof=1) / np.sqrt(len(r)) + 1e-12),
            "pnl_is": pre["pnl"].sum()}


def report(all_ep: pd.DataFrame, wal: pd.DataFrame) -> None:
    pre = all_ep[all_ep["t_close"] < T_SPLIT]
    hold = ((pre["t_close"] - pre["t_open"]).dt.total_seconds() / 3600).groupby(pre["user"]).median()
    wal = wal.assign(hold_is=wal["user"].map(hold))
    groups = {
        "S_all (все)": wal["user"],
        "S_swing (медиана удержания до T >= 24ч)": wal.query("n_is >= 10 and hold_is >= 24")["user"],
        "S_swing_skill (swing и средний результат до T > 0)": wal.query("n_is >= 10 and hold_is >= 24 and mean_is > 0")["user"],
        "S_skill (t>2 до T)": wal.query("n_is >= 15 and mean_is > 0 and t_is > 2")["user"],
        "S_pnl (PnL до T >= $1M)": wal.query("n_is >= 10 and pnl_is >= 1e6")["user"],
        "S_losers (t<-2 до T)": wal.query("n_is >= 15 and t_is < -2")["user"],
    }
    post = all_ep[all_ep["t_open"] >= T_SPLIT]
    print(f"\n=== После T ({T_SPLIT.date()}): эпизодов {len(post)}, кошельков {post.user.nunique()} ===")
    rows = []
    for name, users in groups.items():
        g = post[post["user"].isin(set(users))]
        for col in ("own", "copy_1h"):
            s = _stat(g[col], g["t_open"])
            rows.append({"группа": name, "кошельков": len(set(users)), "результат": col, **s})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("\n=== copy_1h по длительности удержания (все кошельки / S_skill) ===")
    bins = pd.cut((post["t_close"] - post["t_open"]).dt.total_seconds() / 3600, [0, 1, 4, 24, 168, np.inf],
                  labels=["<1ч", "1-4ч", "4-24ч", "1-7д", ">7д"])
    rows = []
    for b, g in post.groupby(bins, observed=True):
        sk = g[g["user"].isin(set(groups["S_skill (t>2 до T)"]))]
        rows.append({"удержание": b, **{f"все_{k}": v for k, v in _stat(g["copy_1h"], g["t_open"]).items()},
                     **{f"skill_{k}": v for k, v in _stat(sk["copy_1h"], sk["t_open"]).items()}})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    sk = wal[wal["user"].isin(set(groups["S_skill (t>2 до T)"]))]
    print("\n=== Устойчивость навыка: корреляция среднего результата кошелька до и после T (own) ===")
    per = post.groupby("user")["own"].agg(["mean", "size"]).rename(columns={"mean": "mean_oos", "size": "n_oos"})
    j = wal.set_index("user").join(per, how="inner").query("n_is >= 10 and n_oos >= 10")
    if len(j) > 5:
        print(f"  кошельков {len(j)}: Spearman(mean_is, mean_oos) = {j['mean_is'].rank().corr(j['mean_oos'].rank()):+.3f}; "
              f"из прибыльных до T прибыльны и после: {((j.mean_is > 0) & (j.mean_oos > 0)).sum()} из {(j.mean_is > 0).sum()}")
    print(f"  S_skill: {len(sk)} кошельков; примеры: " +
          ", ".join(f"{u[:8]}… n={int(n)} t={t:.1f}" for u, n, t in sk[["user", "n_is", "t_is"]].head(8).itertuples(index=False)))


def collect(cache: Path, n_wallets: int) -> None:
    lb = leaderboard(cache)
    active = lb[(lb.vlm_all >= 2e7) & (lb.vlm_all <= 5e9) & (lb.vlm_month > 0)]
    pick = mine(active.sample(min(n_wallets, len(active)), random_state=7)["user"])
    print(f"leaderboard: {len(lb)} кошельков, активных с оборотом $20M–$5B: {len(active)}, "
          f"случайных {n_wallets}, в этой части {len(pick)}", flush=True)
    eps, wal = [], []
    for k, u in enumerate(pick):
        try:
            ep = episodes(fills(u, cache))
        except Exception as e:
            print(f"  {u}: пропуск ({e})", flush=True)
            continue
        if len(ep):
            ep["user"] = u
            eps.append(ep)
        wal.append({"user": u, **wallet_is(ep if len(ep) else pd.DataFrame(columns=["t_close", "own", "pnl"]))})
        if k % 10 == 0:
            print(f"  {k}/{len(pick)} кошельков", flush=True)
    wal = pd.DataFrame(wal)
    for c in ("n_is", "mean_is", "t_is", "pnl_is"):
        if c not in wal:
            wal[c] = np.nan
    wal["n_is"] = wal["n_is"].fillna(0)
    wal.to_parquet(part_path("hl_wallets"), index=False)
    if not eps:
        return
    all_ep = pd.concat(eps, ignore_index=True)
    post = all_ep["t_open"] >= T_SPLIT
    all_ep["copy_1h"] = np.nan
    all_ep.loc[post, "copy_1h"] = copy_returns(all_ep[post], cache)
    all_ep.to_parquet(part_path("hl_episodes"), index=False)


def final_report() -> None:
    wal = pd.concat([pd.read_parquet(p) for p in all_parts("hl_wallets")], ignore_index=True)
    eps = [pd.read_parquet(p) for p in all_parts("hl_episodes")]
    if not eps:
        print("эпизодов нет")
        return
    all_ep = pd.concat(eps, ignore_index=True)
    print(f"===== HL WHALES: частей {len(all_parts('hl_wallets'))}, кошельков {len(wal)}, эпизодов {len(all_ep)}, "
          f"кошельков с эпизодами {all_ep.user.nunique()}, монет {all_ep.coin.nunique()} =====")
    print(f"  медиана удержания {(all_ep.t_close - all_ep.t_open).median()}, медиана позиции ${all_ep.peak.median():,.0f}; "
          f"история с {all_ep.t_open.min().date()}")
    report(all_ep, wal)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "report"])
    ap.add_argument("--cache", default="~/bn/cache/hl")
    ap.add_argument("--wallets", type=int, default=500)
    a = ap.parse_args()
    if a.mode == "collect":
        cache = Path(a.cache).expanduser()
        cache.mkdir(parents=True, exist_ok=True)
        collect(cache, a.wallets)
    else:
        final_report()
