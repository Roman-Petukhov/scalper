"""Метрики бэктеста: чистый PnL после комиссий, PF, expectancy, просадка, Sharpe и т.д."""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.models import TradeRecord


def trades_df(trades: list[TradeRecord]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame([asdict(t) for t in trades]).drop(columns=["extra"])
    df["entry_time"] = pd.to_datetime(df["entry_ts"], unit="s", utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_ts"], unit="s", utc=True)
    df["hold_s"] = df["exit_ts"] - df["entry_ts"]
    df["notional"] = df["qty"] * df["entry_price"]
    df["net_bps"] = df["net_pnl"] / df["notional"] * 1e4
    return df


def _pf(s: pd.Series) -> float:
    gains, losses = s[s > 0].sum(), -s[s < 0].sum()
    return gains / losses if losses > 0 else (math.inf if gains > 0 else 0.0)


def compute(trades: list[TradeRecord], start_equity: float, days: float) -> dict:
    df = trades_df(trades)
    if df.empty:
        return {"trades": 0, "note": "сделок нет: смягчите пороги или проверьте фильтры"}
    eq = start_equity + df["net_pnl"].cumsum()
    peak = eq.cummax().clip(lower=start_equity)
    dd = (eq - peak) / peak
    daily = df.groupby(df["exit_time"].dt.date)["net_pnl"].sum()
    daily_ret = daily / start_equity
    # Sharpe по <5 дням статистически бессмыслен
    sharpe = (daily_ret.mean() / daily_ret.std() * math.sqrt(365)
              if days >= 5 and len(daily_ret) > 1 and daily_ret.std() > 0 else None)
    wins, losses = df[df["net_pnl"] > 0], df[df["net_pnl"] <= 0]

    def group(col):
        g = df.groupby(col)
        return {k: {"trades": int(len(v)), "net": round(v["net_pnl"].sum(), 2),
                    "winrate": round((v["net_pnl"] > 0).mean() * 100, 1),
                    "avg_bps": round(v["net_bps"].mean(), 2)} for k, v in g}

    return {
        "trades": int(len(df)),
        "trades_per_day": round(len(df) / max(days, 1e-9), 1),
        "net_pnl": round(df["net_pnl"].sum(), 2),
        "gross_pnl": round(df["gross_pnl"].sum(), 2),
        "fees": round(df["fees"].sum(), 2),
        "return_pct": round(df["net_pnl"].sum() / start_equity * 100, 2),
        "winrate_pct": round(len(wins) / len(df) * 100, 1),
        "avg_win": round(wins["net_pnl"].mean(), 4) if len(wins) else 0.0,
        "avg_loss": round(losses["net_pnl"].mean(), 4) if len(losses) else 0.0,
        "profit_factor": round(_pf(df["net_pnl"]), 2),
        "expectancy_usdt": round(df["net_pnl"].mean(), 4),
        "expectancy_bps": round(df["net_bps"].mean(), 2),
        "max_drawdown_pct": round(dd.min() * 100, 2),
        "sharpe_daily_ann": round(sharpe, 2) if sharpe is not None else "n/a (<5 дней)",
        "avg_hold_s": round(df["hold_s"].mean(), 1),
        "by_setup": group("setup"),
        "by_exit": group("exit_reason"),
        "by_side": group("side"),
    }


def print_summary(m: dict, extra: dict | None = None) -> None:
    print("\n" + "=" * 60)
    print("  РЕЗУЛЬТАТ БЭКТЕСТА")
    print("=" * 60)
    if not m.get("trades"):
        print("  " + m.get("note", "нет сделок"))
    else:
        rows = [
            ("Сделок", f"{m['trades']}  ({m['trades_per_day']}/день)"),
            ("Чистый PnL", f"{m['net_pnl']:+.2f} USDT  ({m['return_pct']:+.2f}%)"),
            ("Валовый PnL / комиссии", f"{m['gross_pnl']:+.2f} / {m['fees']:.2f}"),
            ("Winrate", f"{m['winrate_pct']}%"),
            ("Ср. прибыль / убыток", f"{m['avg_win']:+.4f} / {m['avg_loss']:+.4f}"),
            ("Profit factor", f"{m['profit_factor']}"),
            ("Expectancy", f"{m['expectancy_usdt']:+.4f} USDT  ({m['expectancy_bps']:+.2f} б.п.)"),
            ("Макс. просадка", f"{m['max_drawdown_pct']}%"),
            ("Sharpe (дн., годовой)", f"{m['sharpe_daily_ann']}"),
            ("Ср. удержание", f"{m['avg_hold_s']} c"),
        ]
        for k, v in rows:
            print(f"  {k:<26}{v}")
        for title, key in (("По сетапам", "by_setup"), ("По выходам", "by_exit"), ("По сторонам", "by_side")):
            print(f"\n  {title}:")
            for name, s in m[key].items():
                print(f"    {name:<16} n={s['trades']:<5} net={s['net']:+9.2f}  wr={s['winrate']:5.1f}%  avg={s['avg_bps']:+6.2f}bp")
    for k, v in (extra or {}).items():
        print(f"\n  {k}: {v}")
    print("=" * 60)


def save(trades: list[TradeRecord], metrics: dict, start_equity: float, out_dir: Path, tag: str,
         extra: dict | None = None) -> Path:
    out = Path(out_dir) / f"{tag}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    df = trades_df(trades)
    if not df.empty:
        df.to_csv(out / "trades.csv", index=False)
        _plot(df, start_equity, out / "equity.png", tag)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, **(extra or {})}, f, ensure_ascii=False, indent=2, default=str)
    return out


def _plot(df: pd.DataFrame, start_equity: float, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eq = start_equity + df["net_pnl"].cumsum()
    gross = start_equity + df["gross_pnl"].cumsum()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(df["exit_time"], gross, color="#9aa5b1", lw=1, label="до комиссий")
    ax1.plot(df["exit_time"], eq, color="#1f6feb", lw=1.6, label="после комиссий")
    ax1.axhline(start_equity, color="#666", lw=0.6, ls="--")
    ax1.set_ylabel("Equity, USDT")
    ax1.set_title(title)
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.2)
    peak = eq.cummax().clip(lower=start_equity)
    ax2.fill_between(df["exit_time"], (eq - peak) / peak * 100, 0, color="#d1242f", alpha=0.35, lw=0)
    ax2.set_ylabel("Просадка, %")
    ax2.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
