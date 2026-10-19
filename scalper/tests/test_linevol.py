import numpy as np
import pytest

from research.linevol import features


def test_features_drying_spike_and_delta():
    vol = np.r_[np.full(5, 100.0), np.linspace(100, 20, 30), 200.0]     # опора на 4, консолидация 5…34, пробой 35
    buy = vol * 0.4                                                       # покупатели 40% — для шорта своя сторона 60%
    f = features(vol, buy, 4, 35, -1)
    assert f["V1"] == "затухает" and f["V3"] == "да" and f["V4"] == "да" and f["V2"] == "да"
    assert f["own_span"] == pytest.approx(0.6)
    g = features(vol, buy, 4, 35, 1)
    assert g["V4"] == "нет"


def test_features_rising_and_short_span():
    vol = np.r_[1.0, np.linspace(10, 50, 20), 40.0]
    f = features(vol, vol * 0.5, 0, 21, 1)
    assert f["V1"] == "растёт" and f["V3"] == "нет" and f["V2"] == "нет"
    assert "V1" not in features(vol, vol * 0.5, 0, 4, 1)
