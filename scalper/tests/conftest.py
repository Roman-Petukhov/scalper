import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_config  # noqa: E402


@pytest.fixture
def cfg():
    return load_config(overrides={
        "features.warmup_s": 0,
        "paper.latency_ms": 0,
        "paper.slippage_bps": 0,
        "telegram.enabled": False,
    })
