from src.imbalance import build_imbalance_bars
from tests.helpers import make_ticks, tight_config


def test_imbalance_bars_close_after_signed_flow_run():
    ticks = make_ticks(n=250, buy_prob=1.0)
    config = tight_config(bar_type="tick_imbalance", initial_expected_ticks=25)
    bars = build_imbalance_bars(ticks, config)
    assert len(bars) >= 5
    assert (bars["tick_count"] > 0).all()
    assert (bars["end_ts"] >= bars["start_ts"]).all()
    assert bars["close"].iloc[-1] == ticks["price"].iloc[int(bars["tick_count"].sum()) - 1]


def test_dollar_imbalance_uses_quote_flow():
    ticks = make_ticks(n=120, buy_prob=1.0, qty=2.0)
    config = tight_config(bar_type="dollar_imbalance", initial_expected_ticks=15)
    bars = build_imbalance_bars(ticks, config)
    assert len(bars) >= 2
    assert (bars["quote_volume"] > 0).all()
    assert (bars["signed_flow"] > 0).all()
