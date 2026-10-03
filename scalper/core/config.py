"""Загрузка config.yaml в объект с доступом через точку."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def _ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(v) for v in obj]
    return obj


def to_dict(ns):
    if isinstance(ns, SimpleNamespace):
        return {k: to_dict(v) for k, v in vars(ns).items()}
    if isinstance(ns, list):
        return [to_dict(v) for v in ns]
    return ns


def load_config(path: str | Path | None = None, overrides: dict | None = None):
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    for dotted, value in (overrides or {}).items():
        node = raw
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    load_dotenv(ROOT / ".env")
    return _ns(raw)


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()
