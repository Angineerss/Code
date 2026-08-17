import numpy as np
from src.imbalance import EwmaState, ImbalanceSeed, build_dollar_bars, build_imbalance_bars
from tests.helpers import make_ticks, tight_config


def test_imbalance_bars_close_after_signed_flow_run():
    ticks = make_ticks(n=250, buy_prob=1.0)
    config = tight_config(bar_type="tick_imbalance", initial_expected_ticks=25)
    bars, _state = build_imbalance_bars(ticks, config)
    assert len(bars) >= 5
    assert (bars["tick_count"] > 0).all()
    assert (bars["end_ts"] >= bars["start_ts"]).all()
    assert bars["close"].iloc[-1] == ticks["price"].iloc[int(bars["tick_count"].sum()) - 1]


def test_dollar_imbalance_uses_quote_flow():
    ticks = make_ticks(n=120, buy_prob=1.0, qty=2.0)
    config = tight_config(bar_type="dollar_imbalance", initial_expected_ticks=15)
    bars, _state = build_imbalance_bars(ticks, config)
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
    bars, _state = build_imbalance_bars(ticks, config)
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
    bars, _state = build_imbalance_bars(ticks, config)
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
    bars, _state = build_imbalance_bars(ticks, config)
    assert bars["tick_count"].max() <= 20 * 2 * 4
    assert len(bars) >= 20


def test_hard_max_ticks_caps_bar_length():
    ticks = make_ticks(n=400, buy_prob=0.5, seed=4)
    config = tight_config(
        bar_type="tick_imbalance",
        initial_expected_ticks=20,
        max_ticks=50,
        max_ticks_mult=10.0,
        min_abs_2p1=1.0,
        max_abs_2p1=1.0,
    )
    bars, _state = build_imbalance_bars(ticks, config)
    assert bars["tick_count"].max() <= 50


def test_warmup_bar_keeps_init_b():
    ticks = make_ticks(n=30, buy_prob=1.0)
    config = tight_config(bar_type="tick_imbalance", initial_expected_ticks=30, init_b=0.5)
    bars, state = build_imbalance_bars(ticks, config)
    assert (bars["close_reason"] == "warmup").any()
    assert state.b == 0.5
    assert bars.iloc[0]["tick_count"] == 30


def test_initial_state_skips_warmup_and_continues_ewma_theta():
    ticks = make_ticks(n=80, buy_prob=1.0, qty=1.0)
    config = tight_config(bar_type="dollar_imbalance", initial_expected_ticks=20, max_ticks=25)
    first, state = build_imbalance_bars(
        ticks.iloc[:40],
        config,
        seed=ImbalanceSeed(expected_imbalance=50.0, expected_size=100.0),
    )
    assert (first["close_reason"] == "warmup").any()
    next_seed = ImbalanceSeed(expected_imbalance=80.0)
    second, _ = build_imbalance_bars(
        ticks.iloc[40:],
        config,
        seed=next_seed,
        initial_state=state,
    )
    assert "warmup" not in set(second["close_reason"])
    lo, hi = 80.0 * 0.5, 80.0 * 2.0
    continued = min(max(state.expected_imbalance, lo), hi)
    assert second.iloc[0]["threshold"] == continued
    assert state.expected_size > 0


def test_continued_theta_is_clipped_to_todays_d_band():
    ticks = make_ticks(n=80, buy_prob=1.0, qty=1.0)
    config = tight_config(bar_type="dollar_imbalance", initial_expected_ticks=20, max_ticks=25)
    state = EwmaState(expected_ticks=20.0, b=0.5, expected_size=100.0, expected_imbalance=10.0)
    bars, _ = build_imbalance_bars(
        ticks,
        config,
        seed=ImbalanceSeed(expected_imbalance=100.0),
        initial_state=state,
    )
    assert "warmup" not in set(bars["close_reason"])
    assert bars.iloc[0]["threshold"] == 50.0


def test_dollar_bars_close_on_quote_not_theta():
    ticks = make_ticks(n=400, buy_prob=0.5, seed=5)
    config = tight_config(bar_type="dollar", initial_expected_ticks=20, max_ticks=200)
    bars, _state = build_dollar_bars(
        ticks,
        config,
        seed=ImbalanceSeed(expected_imbalance=150.0, expected_size=100.0),
    )
    usable = bars.loc[bars["close_reason"] != "warmup"]
    assert not usable.empty
    assert set(usable["close_reason"]).issubset({"dollar", "max_ticks"})
    assert (usable["close_reason"] == "dollar").any()
    assert (usable["quote_volume"] >= 150.0 * 0.5).any()
    assert "dollar_threshold" in bars.columns


def test_dollar_bar_theta_is_signed_quote_flow():
    ticks = make_ticks(n=80, buy_prob=1.0, qty=1.0)
    config = tight_config(bar_type="dollar", initial_expected_ticks=20)
    bars, _state = build_dollar_bars(
        ticks,
        config,
        seed=ImbalanceSeed(expected_imbalance=200.0, expected_size=100.0),
    )
    usable = bars.loc[bars["close_reason"] != "warmup"]
    assert not usable.empty
    assert (usable["signed_flow"] > 0).all()
    assert (usable["signed_flow"].abs() <= usable["quote_volume"] + 1e-6).all()


def test_dollar_bar_ewma_continues_t_dollar():
    ticks = make_ticks(n=120, buy_prob=1.0, qty=1.0)
    config = tight_config(bar_type="dollar", initial_expected_ticks=20, max_ticks=40)
    first, state = build_dollar_bars(
        ticks.iloc[:50],
        config,
        seed=ImbalanceSeed(expected_imbalance=80.0, expected_size=100.0),
    )
    assert (first["close_reason"] == "warmup").any()
    assert np.isfinite(state.expected_dollar)
    second, _ = build_dollar_bars(
        ticks.iloc[50:],
        config,
        seed=ImbalanceSeed(expected_imbalance=80.0),
        initial_state=state,
    )
    assert "warmup" not in set(second["close_reason"])
    assert abs(second.iloc[0]["dollar_threshold"] - state.expected_dollar) < 1e-9


def test_dollar_clock_differs_from_imbalance_control():
    ticks = make_ticks(n=800, buy_prob=0.5, seed=1)
    seed = ImbalanceSeed(expected_imbalance=200.0, expected_size=100.0)
    dollar_cfg = tight_config(bar_type="dollar", initial_expected_ticks=20, max_ticks=200)
    imb_cfg = tight_config(bar_type="dollar_imbalance", initial_expected_ticks=20, max_ticks=200)
    dollar, _ = build_dollar_bars(ticks, dollar_cfg, seed=seed)
    imb, _ = build_imbalance_bars(ticks, imb_cfg, seed=seed)
    d_u = dollar.loc[dollar["close_reason"] != "warmup"]
    i_u = imb.loc[imb["close_reason"] != "warmup"]
    assert not d_u.empty and not i_u.empty
    assert "imbalance" not in set(d_u["close_reason"])
    assert (d_u["close_reason"] == "dollar").any()
    assert set(i_u["close_reason"]).issubset({"imbalance", "max_ticks"})
    # Treatment can close on T$ while |θ| is still below E[θ]; control waits for θ.
    weak_dollar = d_u.loc[d_u["signed_flow"].abs() < d_u["threshold"].abs()]
    assert not weak_dollar.empty
    strong_imb = i_u.loc[i_u["close_reason"] == "imbalance"]
    assert not strong_imb.empty
    assert (strong_imb["signed_flow"].abs() + 1e-9 >= strong_imb["threshold"].abs()).all()
