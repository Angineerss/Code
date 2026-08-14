from datetime import date
from io import BytesIO

import pandas as pd

from src.download import months_from_to, parse_aggtrades_csv


def test_parse_aggtrades_ms_and_aggressor_side():
    csv = b"\n".join(
        [
            b"1,100.0,2.0,1,1,1498793709153,false",
            b"2,99.5,1.0,2,2,1498793709154,true",
        ]
    )
    df = parse_aggtrades_csv(BytesIO(csv))
    assert list(df["side"]) == [1, -1]
    assert df["quote_qty"].iloc[0] == 200.0
    assert df["timestamp"].dt.tz is not None


def test_parse_aggtrades_microseconds():
    csv = b"0,0.2,50.0,0,0,1735689600010866,False"
    df = parse_aggtrades_csv(BytesIO(csv))
    ts = df["timestamp"].iloc[0]
    assert ts.floor("s") == pd.Timestamp("2025-01-01", tz="UTC")
    assert df["side"].iloc[0] == 1


def test_months_from_to_includes_endpoints():
    months = months_from_to(date(2017, 8, 17), date(2018, 1, 5))
    assert months[0] == (2017, 8)
    assert months[-1] == (2018, 1)
    assert len(months) == 6
