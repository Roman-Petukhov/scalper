import numpy as np
import pandas as pd

from research import sidegate as G


def _daily(alt_trend: float, btc_trend: float, n: int = 120, coins: int = 30) -> pd.DataFrame:
    days = pd.date_range("2023-01-01", periods=n, freq="D")
    rows = [pd.DataFrame({"symbol": "BTCUSDT", "day": days, "close": 100 * (1 + btc_trend) ** np.arange(n),
                          "adv": 1e10})]
    for i in range(coins):
        rows.append(pd.DataFrame({"symbol": f"A{i}USDT", "day": days, "close": 10 * (1 + alt_trend) ** np.arange(n),
                                  "adv": 1e8 - i}))
    return pd.concat(rows, ignore_index=True)


def test_regimes_follow_the_panel_rule():
    assert G.regimes(_daily(-0.01, 0.0)).iloc[-1] == "short"       # альты падают, BTC на месте
    assert G.regimes(_daily(0.01, 0.005)).iloc[-1] == "long"       # альты растут быстрее, BTC +3.5% за неделю
    mixed = G.regimes(_daily(-0.01, -0.02)).iloc[-1]               # альты падают, но медленнее BTC
    assert mixed == "mixed"
