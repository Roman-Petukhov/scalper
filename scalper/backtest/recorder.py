"""Запись живых потоков (сделки, стакан, mark, ликвидации) для будущих бэктестов со стаканом."""
from __future__ import annotations

import gzip
import json
import logging
import time
from pathlib import Path

log = logging.getLogger("recorder")


class Recorder:
    def __init__(self, out_dir: Path, symbol: str, rotate_minutes: int = 60):
        self.dir = Path(out_dir) / symbol
        self.dir.mkdir(parents=True, exist_ok=True)
        self.symbol = symbol
        self.rotate_s = rotate_minutes * 60
        self.f = None
        self.opened = 0.0
        self.count = 0

    def _rotate(self) -> None:
        if self.f:
            self.f.close()
        name = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        path = self.dir / f"{self.symbol}-{name}.jsonl.gz"
        self.f = gzip.open(path, "at", encoding="utf-8", compresslevel=5)
        self.opened = time.time()
        log.info("recording -> %s", path)

    def write(self, stream: str, data: dict) -> None:
        if self.f is None or time.time() - self.opened >= self.rotate_s:
            self._rotate()
        self.f.write(json.dumps({"s": stream, "d": data}, separators=(",", ":")) + "\n")
        self.count += 1

    def close(self) -> None:
        if self.f:
            self.f.close()
            self.f = None
