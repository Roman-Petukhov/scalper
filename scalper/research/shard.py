"""
Разбиение тяжёлых прогонов на части для параллельных машин GitHub Actions (.github/workflows/research.yml).

Окружение задаёт workflow:
    SHARD, NSHARDS   номер части и число частей (по умолчанию 0 и 1 — весь прогон на одной машине)
    SMOKE=1          пробный запуск: из своей части берутся первые SMOKE_N элементов (по умолчанию 3)
    PARTS            куда collect складывает свои результаты и откуда report их читает
Монета или кошелёк закрепляется за частью по crc32 имени, поэтому кеш каждой машины стабилен между прогонами.

    python -m research.shard prune <cache dir>   удалить из кеша файлы чужих частей перед сохранением
"""
from __future__ import annotations

import os
import sys
import zlib
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")
ALWAYS = {"BTCUSDT"}                                  # нужна всем частям (бета, режим рынка)


def shard_id() -> tuple[int, int]:
    return int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))


def smoke() -> bool:
    return os.environ.get("SMOKE", "") not in ("", "0")


def owner(key: str) -> int:
    return zlib.crc32(key.encode()) % shard_id()[1]


def mine(items: Iterable[T], key: Callable[[T], str] = str) -> list[T]:
    i, n = shard_id()
    out = [x for x in items if zlib.crc32(key(x).encode()) % n == i]
    if smoke():
        out = out[: int(os.environ.get("SMOKE_N", "3"))]
    return out


def parts_dir() -> Path:
    p = Path(os.environ.get("PARTS", "parts"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def part_path(name: str) -> Path:
    i, n = shard_id()
    return parts_dir() / f"{name}.{i:02d}of{n:02d}.parquet"


def all_parts(name: str) -> list[Path]:
    return sorted(Path(os.environ.get("PARTS", "parts")).rglob(f"{name}.*of*.parquet"))


def _cache_key(f: Path) -> str | None:
    """Ключ закрепления файла кеша: символ (BTCUSDT-1h-2024-01.parquet, 1m/BTCUSDT-1m-...) или кошелёк
    (hl/fills-0x....parquet). None — общий файл, его не трогаем."""
    name = f.name
    if name.startswith("fills-0x"):
        return name[len("fills-"):].split(".")[0]
    head = name.split("-")[0]
    if head.isupper() and head.endswith("USDT"):
        return head
    return None


def prune(cache: Path) -> tuple[int, int]:
    i, n = shard_id()
    removed = kept = 0
    if n <= 1:
        return 0, 0
    for f in cache.rglob("*"):
        if not f.is_file():
            continue
        k = _cache_key(f)
        if k is None or k in ALWAYS or zlib.crc32(k.encode()) % n == i:
            kept += 1
            continue
        f.unlink()
        removed += 1
    return removed, kept


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "prune":
        r, k = prune(Path(sys.argv[2]).expanduser())
        print(f"кеш части {shard_id()[0]}/{shard_id()[1]}: удалено чужих файлов {r}, оставлено {k}")
    else:
        sys.exit("usage: python -m research.shard prune <cache dir>")
