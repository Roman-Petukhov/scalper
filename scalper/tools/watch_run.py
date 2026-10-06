"""Дождаться прогона GitHub Actions и вывести итог: python -m tools.watch_run <run_id> [<run_id> …]
или python -m tools.watch_run --latest <branch> (последний прогон research на ветке). Без токена: репозиторий публичный."""
from __future__ import annotations

import json
import sys
import time
import urllib.request

API = "https://api.github.com/repos/Roman-Petukhov/scalper/actions/runs"


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def latest(branch: str) -> str:
    runs = _get(f"{API}?branch={branch}&per_page=5")["workflow_runs"]
    return str(next(r["id"] for r in runs if r["name"] == "research"))


def main(ids: list[str]) -> int:
    while True:
        done = []
        for rid in ids:
            try:
                run, jobs = _get(f"{API}/{rid}"), _get(f"{API}/{rid}/jobs?per_page=50")["jobs"]
            except Exception:
                continue
            bad = [j for j in jobs if j["conclusion"] in ("failure", "cancelled", "timed_out")]
            if bad:
                done.append(f"{rid}: ПРОБЛЕМА в " + ", ".join(f"{j['name']}={j['conclusion']}" for j in bad))
            elif run["status"] == "completed":
                done.append(f"{rid}: завершён, {run['conclusion']}")
        if len(done) == len(ids):
            print("\n".join(done))
            return 0 if all("ПРОБЛЕМА" not in d for d in done) else 1
        time.sleep(60)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--latest"]:
        args = [latest(args[1])]
        print(f"прогон {args[0]}", flush=True)
    sys.exit(main(args))
