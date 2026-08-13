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


def test_balanced_flow_does_not_inflate_bar_size():
    ticks = make_ticks(n=800, buy_prob=0.52, seed=1)
    config = tight_config(
        bar_type="tick_imbalance",
        initial_expected_ticks=20,
        max_ticks_mult=4.0,
        max_abs_2p1=0.15,
    )
    bars = build_imbalance_bars(ticks, config)
    assert len(bars) >= 20
    assert bars["tick_count"].median() <= 80


def test_max_ticks_force_closes_when_flow_cancels():
    ticks = make_ticks(n=200, buy_prob=0.5, seed=2)
    config = tight_config(
        bar_type="tick_imbalance",
        initial_expected_ticks=10,
        max_ticks_mult=3.0,
        min_abs_2p1=1.0,
        max_abs_2p1=1.0,
    )
    bars = build_imbalance_bars(ticks, config)
    assert "max_ticks" in set(bars["close_reason"])


def test_expected_ticks_do_not_run_away_after_max_tick_bars():
    ticks = make_ticks(n=2000, buy_prob=0.5, seed=3)
    config = tight_config(
        bar_type="tick_imbalance",
        initial_expected_ticks=20,
        max_ticks_mult=4.0,
        expected_ticks_max_mult=2.0,
        min_abs_2p1=1.0,
        max_abs_2p1=1.0,
    )
    bars = build_imbalance_bars(ticks, config)
    assert bars["tick_count"].max() <= 20 * 2 * 4
    assert len(bars) >= 20
