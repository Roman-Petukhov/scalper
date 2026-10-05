"""Проверка, что с этого компьютера / сервера доступны биржи: запускать перед арендой VPS на месяц.

    python -m trader.check

Проверяет: откуда выходим в интернет (IP, страна), свечи Binance USDⓈ-M (данные для сигналов), публичный API
Bybit (обычный и демо) и, если ключи уже сохранены в панели, — что Bybit принимает их и отдаёт баланс.
Код выхода 0 — всё работает, 1 — что-то недоступно (причина в выводе)."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import ccxt.async_support as ccxt
import httpx

from .infrastructure.binance_data import BASE as BINANCE
from .infrastructure.bybit import KV_KEY, BybitBroker, BybitCredentials

OK, FAIL, SKIP = "✓", "✗", "–"


def _data_dir() -> Path:
    env = Path(__file__).resolve().parent.parent / "panel.env"
    if "DATA_DIR" not in os.environ and env.exists():
        from .local import _read_env
        os.environ.update({k: v for k, v in _read_env(env).items() if k == "DATA_DIR"})
    return Path(os.environ.get("DATA_DIR", "data_panel"))


def _short(e: Exception) -> str:
    msg = str(e) or type(e).__name__
    return msg if len(msg) <= 200 else msg[:197] + "…"


async def _where(client: httpx.AsyncClient) -> tuple[str, str]:
    try:
        r = await client.get("https://ipinfo.io/json", timeout=10)
        d = r.json()
        return OK, f"IP {d.get('ip')} · {d.get('country')} · {d.get('org', '')}"
    except Exception as e:
        return SKIP, f"не удалось определить ({_short(e)})"


async def _binance(client: httpx.AsyncClient) -> tuple[str, str]:
    try:
        r = await client.get(f"{BINANCE}/fapi/v1/klines", params={"symbol": "BTCUSDT", "interval": "4h", "limit": 2},
                             timeout=15)
        if r.status_code == 451:
            return FAIL, "HTTP 451: Binance закрыт для этой страны / дата-центра — сигналы работать не будут"
        if r.status_code in (403, 418, 429):
            return FAIL, f"HTTP {r.status_code}: Binance режет этот IP"
        r.raise_for_status()
        return OK, f"свечи BTCUSDT 4h получены (закрытие {float(r.json()[-1][4]):,.1f})".replace(",", " ")
    except Exception as e:
        return FAIL, _short(e)


async def _bybit_public(demo: bool) -> tuple[str, str]:
    ex = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    if demo:
        ex.enable_demo_trading(True)
    try:
        await ex.fetch_time()
        t = await ex.fetch_ticker("BTC/USDT:USDT")
        return OK, f"отвечает, BTCUSDT {t['last']:,.1f}".replace(",", " ")
    except ccxt.ExchangeNotAvailable as e:
        return FAIL, f"недоступен из этой страны / сети: {_short(e)}"
    except Exception as e:
        return FAIL, _short(e)
    finally:
        await ex.close()


async def _bybit_keys(data_dir: Path) -> tuple[str, str]:
    db = data_dir / "trader.db"
    if not db.exists():
        return SKIP, f"ключи ещё не сохранены (нет {db}) — подключите Bybit в панели и запустите проверку снова"
    from .infrastructure.sqlite_repo import SqliteStore
    creds = BybitCredentials.loads(SqliteStore(db).kv_get(KV_KEY))
    if creds is None:
        return SKIP, "ключи Bybit в панели не подключены"
    br = BybitBroker(creds)
    try:
        acc = await br.account()
        net = "демо" if creds.network == "demo" else "реальный"
        return OK, f"{net} счёт, ключ {creds.masked_key}: капитал {acc.equity:,.2f} USDT, позиций {len(acc.positions)}".replace(",", " ")
    except ccxt.AuthenticationError as e:
        return FAIL, f"ключи не приняты: {_short(e)}"
    except Exception as e:
        return FAIL, _short(e)
    finally:
        await br.close()


async def run() -> int:
    async with httpx.AsyncClient() as client:
        rows = [("Откуда выходим", await _where(client)),
                ("Binance (данные для сигналов)", await _binance(client)),
                ("Bybit, публичный API", await _bybit_public(False)),
                ("Bybit демо, публичный API", await _bybit_public(True)),
                ("Bybit, ваши ключи", await _bybit_keys(_data_dir()))]
    width = max(len(name) for name, _ in rows)
    for name, (mark, text) in rows:
        print(f" {mark}  {name.ljust(width)}  {text}")
    failed = [name for name, (mark, _) in rows if mark == FAIL]
    print("\nИтог: " + ("всё доступно — бот здесь будет работать." if not failed else
                        "недоступно: " + ", ".join(failed) + ". Этот сервер / сеть не подходит."))
    return 1 if failed else 0


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(asyncio.run(run()))


if __name__ == "__main__":
    main()
