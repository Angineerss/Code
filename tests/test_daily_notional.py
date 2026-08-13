from datetime import date

from src.daily_notional import (
    parse_klines,
    parse_klines_csv,
    prior_year_window,
    threshold_from_average,
)
from src.imbalance import ImbalanceSeed, build_imbalance_bars
from tests.helpers import make_ticks, tight_config


def test_prior_year_window_excludes_as_of_day():
    start, end = prior_year_window(date(2024, 1, 15), lookback_days=365)
    assert start == date(2023, 1, 15)
    assert end == date(2024, 1, 14)


def test_imbalance_threshold_is_one_fiftieth_of_prior_year_average():
    assert threshold_from_average(1_000_000.0, 50) == 20_000.0


def test_parse_klines_csv_quote_volume():
    from io import BytesIO

    csv = b"1700000000000,1,2,0.5,1.5,10,1700086399999,123.5,4,1,2,0"
    df = parse_klines_csv(BytesIO(csv))
    assert float(df["quote_volume"].iloc[0]) == 123.5
    assert float(df["n_trades"].iloc[0]) == 4


def test_months_covering_span():
    from src.daily_notional import months_covering

    months = months_covering(date(2023, 1, 15), date(2024, 1, 14))
    assert months[0] == (2023, 1)
    assert months[-1] == (2024, 1)
    assert len(months) == 13


def test_parse_klines_quote_volume():
    payload = [
        [1_700_000_000_000, "0", "0", "0", "0", "0", 1_700_086_399_999, "123.5", 4, "0", "0", "0"],
    ]
    df = parse_klines(payload)
    assert float(df["quote_volume"].iloc[0]) == 123.5
    assert float(df["n_trades"].iloc[0]) == 4


def test_dollar_imbalance_uses_seeded_prior_year_threshold():
    ticks = make_ticks(n=400, buy_prob=1.0, qty=1.0)
    d = float(ticks["quote_qty"].sum()) / 8
    config = tight_config(bar_type="dollar_imbalance", initial_expected_ticks=15)
    bars, _state = build_imbalance_bars(
        ticks,
        config,
        seed=ImbalanceSeed(expected_imbalance=d, expected_size=100.0),
    )
    assert len(bars) >= 2
    assert set(bars["close_reason"]).issubset({"warmup", "imbalance", "max_ticks"})
