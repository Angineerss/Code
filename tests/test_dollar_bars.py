from datetime import date

from src.daily_notional import (
    dollar_threshold_from_average,
    parse_klines,
    prior_year_window,
)
from src.imbalance import build_dollar_bars, dollar_bar_threshold
from tests.helpers import make_ticks, tight_config


def test_prior_year_window_excludes_as_of_day():
    start, end = prior_year_window(date(2024, 1, 15), lookback_days=365)
    assert start == date(2023, 1, 15)
    assert end == date(2024, 1, 14)


def test_dollar_threshold_is_one_fiftieth_of_prior_year_average():
    config = tight_config(bar_type="dollar", dollar_bar_divisor=50)
    average = 1_000_000.0
    assert dollar_bar_threshold(average, config) == 20_000.0
    assert dollar_threshold_from_average(average, 50) == 20_000.0


def test_parse_klines_csv_quote_volume():
    from io import BytesIO

    from src.daily_notional import parse_klines_csv

    csv = b"1700000000000,1,2,0.5,1.5,10,1700086399999,123.5,4,1,2,0"
    df = parse_klines_csv(BytesIO(csv))
    assert float(df["quote_volume"].iloc[0]) == 123.5


def test_months_covering_span():
    from src.daily_notional import months_covering

    months = months_covering(date(2023, 1, 15), date(2024, 1, 14))
    assert months[0] == (2023, 1)
    assert months[-1] == (2024, 1)
    assert len(months) == 13


def test_parse_klines_quote_volume():
    payload = [
        [1_700_000_000_000, "0", "0", "0", "0", "0", 1_700_086_399_999, "123.5", 1, "0", "0", "0"],
    ]
    df = parse_klines(payload)
    assert float(df["quote_volume"].iloc[0]) == 123.5
    assert df["open_time"].dt.tz is not None


def test_dollar_bars_with_same_day_fallback_target_about_fifty_bars():
    ticks = make_ticks(n=500, buy_prob=0.6, qty=1.0)
    config = tight_config(bar_type="dollar", dollar_bar_divisor=50)
    bars = build_dollar_bars(ticks, config)
    d = float(ticks["quote_qty"].sum()) / 50
    completed = bars[bars["close_reason"] == "dollar"]
    assert 45 <= len(bars) <= 52
    assert (completed["quote_volume"] >= d).all()
    assert abs(bars["quote_volume"].sum() - ticks["quote_qty"].sum()) < 1e-6


def test_dollar_bars_honor_explicit_prior_year_threshold():
    ticks = make_ticks(n=500, buy_prob=0.6, qty=1.0)
    config = tight_config(bar_type="dollar", dollar_bar_divisor=50)
    same_day_d = float(ticks["quote_qty"].sum()) / 50
    bars = build_dollar_bars(ticks, config, threshold=same_day_d * 2)
    assert len(bars) < 40
