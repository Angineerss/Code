from src.imbalance import build_dollar_bars, dollar_bar_threshold
from tests.helpers import make_ticks, tight_config


def test_dollar_threshold_is_one_fiftieth_of_daily_notional():
    ticks = make_ticks(n=400, buy_prob=0.6, qty=2.0)
    config = tight_config(bar_type="dollar", dollar_bar_divisor=50)
    daily = float(ticks["quote_qty"].sum())
    d = dollar_bar_threshold(ticks, config)
    assert d == daily / 50


def test_dollar_bars_target_about_fifty_bars():
    ticks = make_ticks(n=500, buy_prob=0.6, qty=1.0)
    config = tight_config(bar_type="dollar", dollar_bar_divisor=50)
    bars = build_dollar_bars(ticks, config)
    d = dollar_bar_threshold(ticks, config)
    completed = bars[bars["close_reason"] == "dollar"]
    assert 45 <= len(bars) <= 52
    assert (completed["quote_volume"] >= d).all()
    assert abs(bars["quote_volume"].sum() - ticks["quote_qty"].sum()) < 1e-6
