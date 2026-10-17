import gzip

import pandas as pd

from research.bybitflow import buy_share, parse_trades


def _gz(text: str) -> bytes:
    return gzip.compress(text.encode())


def test_parse_and_share_in_bar():
    csv = ("timestamp,symbol,side,size,price,tickDirection,trdMatchID,grossValue,homeNotional,foreignNotional\n"
           "1700000000.1,NEARUSDT,Buy,3,1.5,PlusTick,a,1,3,4.5\n"
           "1700000100.2,NEARUSDT,Sell,1,1.5,MinusTick,b,1,1,1.5\n"
           "1700020000.0,NEARUSDT,Sell,5,1.5,MinusTick,c,1,5,7.5\n")          # следующая 4h-свеча
    tr = parse_trades(_gz(csv))
    start = pd.Timestamp(1700000000, unit="s", tz="UTC").floor("4h")
    assert len(tr) == 3 and tr.buy.tolist() == [True, False, False]
    assert buy_share(tr, start) == 0.75


def test_millisecond_timestamps_and_unknown_format():
    tr = parse_trades(_gz("timestamp,side,size\n1700000000100,Sell,2\n1700000000200,Buy,2\n"))
    assert tr.t.iloc[0].year == 2023 and buy_share(tr, tr.t.iloc[0].floor("4h")) == 0.5
    assert parse_trades(_gz("a,b\n1,2\n")) is None
