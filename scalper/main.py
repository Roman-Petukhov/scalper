"""
Flow Scalper — CLI.

  python main.py download --days 7               скачать aggTrades
  python main.py backtest --days 7               бэктест на скачанных днях
  python main.py backtest --recorded             бэктест на записях со стаканом
  python main.py study --days 7                  предсказательная сила сигналов (без исполнения)
  python main.py record                          записывать живые потоки
  python main.py paper                           симуляция на живых котировках
  python main.py live --live                     реальная торговля (по умолчанию demo)

Любой параметр конфига можно переопределить: --set strategy.momentum.z_fast=2.5
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import yaml

from core.config import ROOT, load_config, to_dict


def setup_logging(cfg, name: str) -> None:
    log_dir = ROOT / cfg.logging.dir
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)-9s %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(cfg.logging.level)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = RotatingFileHandler(log_dir / f"{name}.log", maxBytes=20_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    root.handlers = [sh, fh]
    for noisy in ("websockets", "asyncio", "ccxt"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def parse_overrides(items: list[str] | None) -> dict:
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k.strip()] = yaml.safe_load(v)
    return out


def cmd_download(cfg, a) -> None:
    from backtest.data_loader import date_range, download
    dates = date_range(a.days, a.start, a.end)
    print(f"Скачиваю {cfg.exchange.symbol} aggTrades: {dates[0]} .. {dates[-1]}")
    paths = download(cfg.exchange.symbol, dates, ROOT / cfg.backtest.data_dir)
    print(f"Готово: {len(paths)} из {len(dates)} дней")


def _events(cfg, a):
    """Источник событий для backtest/study: скачанные aggTrades или записи со стаканом."""
    from backtest.data_loader import (count_rows, date_range, day_path, download,
                                      iter_agg_trades, iter_recorded)
    data_dir = ROOT / cfg.backtest.data_dir
    if a.recorded:
        paths = sorted((ROOT / cfg.recorder.dir / cfg.exchange.symbol).glob("*.jsonl.gz"))
        if not paths:
            raise SystemExit("Нет записей. Сначала: python main.py record")
        span = f"{len(paths)} файлов записи"
        print(f"Данные: записи со стаканом, {span}")
        return iter_recorded(paths), None, span, paths
    dates = date_range(a.days, a.start, a.end)
    paths = [day_path(data_dir, cfg.exchange.symbol, d) for d in dates]
    missing = [d for d, p in zip(dates, paths) if not p.exists()]
    if missing:
        print(f"Докачиваю {len(missing)} дней...")
        download(cfg.exchange.symbol, missing, data_dir)
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise SystemExit("Нет данных за выбранный период.")
    if getattr(cfg, "need_infer", False):
        from backtest.data_loader import infer_instrument
        spec = infer_instrument(cfg.exchange.symbol, paths)
        for k, v in spec.items():
            setattr(cfg.instrument, k, v)
        print(f"Параметры инструмента выведены из данных: {spec}")
    total = count_rows(paths)
    span = f"{dates[0]} .. {dates[-1]}"
    print(f"Данные: {cfg.exchange.symbol} {span}, {total:,} сделок (стакан недоступен, OBI отключён)")
    return iter_agg_trades(paths), total, span, paths


def cmd_study(cfg, a) -> None:
    from backtest.signal_study import print_study, study
    events, _, span, _ = _events(cfg, a)
    res = study(cfg, events)
    print_study(res, cfg.fees)


def cmd_backtest(cfg, a) -> None:
    from backtest import report
    from backtest.engine import run_backtest

    logging.getLogger("position").setLevel(logging.WARNING)
    events, total, span, paths = _events(cfg, a)
    s = run_backtest(cfg, events, total=total)
    trades = s.manager.trades
    days = (trades[-1].exit_ts - trades[0].entry_ts) / 86400 if len(trades) > 1 else max(len(paths), 1)
    days = max(days, len(paths) if not a.recorded else days)
    m = report.compute(trades, float(cfg.paper.start_equity), days)
    extra = {
        "Период": span,
        "Сигналов": s.manager.signals_seen,
        "Пропуски входа": s.manager.skips,
        "Счётчики стратегии": dict(sorted(s.strategy.stats.items(), key=lambda kv: -kv[1])[:10]),
        "Скорость": f"{s.events:,} событий за {s.elapsed:.0f}с",
    }
    report.print_summary(m, extra)
    out = report.save(trades, m, float(cfg.paper.start_equity), ROOT / cfg.backtest.results_dir,
                      f"{cfg.exchange.symbol}", {"period": span, "config": to_dict(cfg),
                                                 "skips": s.manager.skips, "strategy_stats": s.strategy.stats})
    print(f"\nРезультаты: {out}")


def cmd_realtime(cfg, a, live: bool) -> None:
    from runner import run_realtime
    if live:
        if not a.live or cfg.mode != "live":
            raise SystemExit("Live требует И флаг --live, И mode: live в config.yaml (защита от случайного запуска).")
        if cfg.exchange.env == "mainnet":
            print("\n!!! ВНИМАНИЕ: торговля РЕАЛЬНЫМИ деньгами на Binance mainnet !!!")
            if input("Введите YES для подтверждения: ").strip() != "YES":
                raise SystemExit("Отменено.")
    try:
        asyncio.run(run_realtime(cfg, live=live, record=getattr(a, "record", False)))
    except KeyboardInterrupt:
        pass


def cmd_record(cfg, a) -> None:
    from runner import run_recorder
    try:
        asyncio.run(run_recorder(cfg, a.minutes))
    except KeyboardInterrupt:
        pass


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="Flow Scalper — order-flow скальпер для Binance USDⓈ-M")
    ap.add_argument("--config", default=None)
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", help="переопределить параметр конфига")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("download", "backtest", "study"):
        p = sub.add_parser(name)
        p.add_argument("--days", type=int, default=7)
        p.add_argument("--start")
        p.add_argument("--end")
        if name != "download":
            p.add_argument("--recorded", action="store_true", help="использовать записи recorder (со стаканом)")
    p = sub.add_parser("record")
    p.add_argument("--minutes", type=float, default=None)
    p = sub.add_parser("paper")
    p.add_argument("--record", action="store_true", help="параллельно записывать потоки")
    p = sub.add_parser("live")
    p.add_argument("--live", action="store_true", help="подтверждение реальной торговли")
    p.add_argument("--record", action="store_true")
    a = ap.parse_args()

    overrides = parse_overrides(a.set)
    if a.symbol:
        overrides["exchange.symbol"] = a.symbol.upper()
    cfg = load_config(a.config, overrides)
    setup_logging(cfg, a.cmd)
    if a.cmd in ("backtest", "study", "paper"):
        from backtest.data_loader import fetch_instrument
        spec = fetch_instrument(cfg.exchange.symbol, ROOT / cfg.backtest.data_dir)
        if spec:
            for k, v in spec.items():
                setattr(cfg.instrument, k, v)
        cfg.need_infer = spec is None
        logging.getLogger("main").info("инструмент %s: %s", cfg.exchange.symbol, vars(cfg.instrument))

    if a.cmd == "download":
        cmd_download(cfg, a)
    elif a.cmd == "backtest":
        cmd_backtest(cfg, a)
    elif a.cmd == "study":
        cmd_study(cfg, a)
    elif a.cmd == "record":
        cmd_record(cfg, a)
    elif a.cmd == "paper":
        cmd_realtime(cfg, a, live=False)
    elif a.cmd == "live":
        cmd_realtime(cfg, a, live=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
