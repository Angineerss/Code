from datetime import date
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from src.download import (
    filter_ticks_day,
    load_day_from_archive,
    months_from_to,
    parse_aggtrades_csv,
)


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


def test_filter_ticks_day_keeps_utc_half_open():
    ticks = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01T23:59:59.999Z",
                    "2024-01-02T00:00:00.000Z",
                    "2024-01-02T12:00:00.000Z",
                    "2024-01-03T00:00:00.000Z",
                ],
                utc=True,
                format="ISO8601",
            ),
            "price": [1.0, 2.0, 3.0, 4.0],
        }
    )
    out = filter_ticks_day(ticks, date(2024, 1, 2))
    assert list(out["price"]) == [2.0, 3.0]


def test_load_day_from_archive_prefers_daily_zip(tmp_path: Path):
    daily_dir = tmp_path / "daily"
    monthly_dir = tmp_path / "monthly"
    daily_dir.mkdir()
    monthly_dir.mkdir()
    day = date(2024, 1, 2)
    csv = b"\n".join(
        [
            b"1,100.0,1.0,1,1,1704153600000,false",  # 2024-01-02 00:00:00 UTC
            b"2,101.0,1.0,2,2,1704239999000,true",  # still Jan 2
        ]
    )
    with ZipFile(daily_dir / "BTCUSDT-aggTrades-2024-01-02.zip", "w") as zf:
        zf.writestr("BTCUSDT-aggTrades-2024-01-02.csv", csv)
    # Monthly present but must be ignored when daily exists.
    with ZipFile(monthly_dir / "BTCUSDT-aggTrades-2024-01.zip", "w") as zf:
        zf.writestr("BTCUSDT-aggTrades-2024-01.csv", b"9,1.0,1.0,9,9,1704153600000,false")
    ticks = load_day_from_archive("BTCUSDT", day, tmp_path)
    assert len(ticks) == 2
    assert list(ticks["trade_id"]) == [1, 2]


def test_load_day_from_archive_uses_monthly_cache(tmp_path: Path):
    monthly_dir = tmp_path / "monthly"
    monthly_dir.mkdir()
    csv = b"\n".join(
        [
            b"1,100.0,1.0,1,1,1704067200000,false",  # 2024-01-01
            b"2,101.0,1.0,2,2,1704153600000,true",  # 2024-01-02
        ]
    )
    with ZipFile(monthly_dir / "BTCUSDT-aggTrades-2024-01.zip", "w") as zf:
        zf.writestr("BTCUSDT-aggTrades-2024-01.csv", csv)
    cache: dict[tuple[int, int], dict[date, pd.DataFrame]] = {}
    d1 = load_day_from_archive("BTCUSDT", date(2024, 1, 1), tmp_path, month_cache=cache)
    assert date(2024, 1, 2) in cache[(2024, 1)]
    d2 = load_day_from_archive("BTCUSDT", date(2024, 1, 2), tmp_path, month_cache=cache)
    assert len(d1) == 1 and len(d2) == 1
    assert (2024, 1) not in cache
    assert list(d1["trade_id"]) == [1]
    assert list(d2["trade_id"]) == [2]
