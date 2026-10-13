import numpy as np
import pandas as pd

from research.oilong import VARIANTS, oi_features


def test_oi_features_use_the_past_and_the_bar_itself():
    idx = pd.date_range("2023-01-01", periods=300, freq="4h", tz="UTC")
    d = pd.DataFrame({"close": np.r_[np.full(150, 10.0), np.full(150, 9.0)], "funding": -1e-4}, index=idx)
    oi = np.r_[np.full(200, 100.0), np.linspace(100, 120, 100)]
    m = pd.DataFrame({"oi": oi, "top_lsr": 1.0}, index=idx)
    f = oi_features(d, m)
    assert np.isclose(f["oi_chg"][250], oi[250] / oi[232] - 1)
    assert f["ret_before"][160] < 0 and f["fund"][50] < 0
    assert np.isclose(f["doi_bar"][250], oi[250] / oi[249] - 1)


def test_variants_select_as_described():
    z = pd.DataFrame({"buy": [0.5, 0.6], "oi_chg": [0.06, 0.01], "ret_before": [-0.01, 0.02], "fund": [-1e-4, 1e-4],
                      "top_dev": [-0.1, 0.1], "doi_bar": [-0.01, 0.02]})
    sel = {k: f(z).tolist() for k, f in VARIANTS.items()}
    assert sel["шорты копятся + фандинг < 0"] == [True, False]
    assert sel["агрессоры ≥ 55% (бот)"] == [False, True]
    assert sel["вынос пошёл: OI на свече пробоя упал"] == [True, False]
