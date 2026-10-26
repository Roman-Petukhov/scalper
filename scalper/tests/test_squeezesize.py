import numpy as np
import pandas as pd

from research import squeezesize as Q
from tests.test_crisk import _z


def test_weights_mark_only_squeezed_trades():
    z = _z([1.0, 2.0, -1.0]).assign(squeeze=[0.4, 0.9, 0.5])
    assert list(Q.weights(z, 1.5, 0.5)) == [1.5, 1.0, 1.5]


def test_bootstrap_keeps_outcome_and_multiplier_together():
    z = _z([-1.0] * 40, gap_h=8, hold_h=4).assign(squeeze=0.3)
    w = Q.weights(z, 2.0, 0.5)
    rng = np.random.default_rng(0)
    assert Q.dd50w(z, Q.Policy("f", flat=0.05), w, 5, rng) == 1.0         # 40 убытков по 10% — просадка > 50%
    assert Q.dd50w(z, Q.Policy("f", flat=0.005), w, 5, rng) == 0.0
